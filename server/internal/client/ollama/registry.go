// Package ollama provides a client for interacting with an Ollama registry
// which pushes and pulls model manifests and layers as defined by the
// [ollama.com/manifest].
package ollama

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/internal/names"

	_ "embed"
)

// Errors
var (
	// ErrModelNotFound is returned when a manifest is not found in the
	// cache or registry.
	ErrModelNotFound = errors.New("model not found")

	// ErrManifestInvalid is returned when a manifest found in a local or
	// remote cache is invalid.
	ErrManifestInvalid = errors.New("invalid manifest")

	// ErrMissingModel is returned when the model part of a name is missing
	// or invalid.
	ErrNameInvalid = errors.New("invalid or missing name")

	// ErrCached is passed to [Trace.PushUpdate] when a layer already
	// exists. It is a non-fatal error and is never returned by [Registry.Push].
	ErrCached = errors.New("cached")

	// ErrIncomplete is returned by [Registry.Pull] when a model pull was
	// incomplete due to one or more layer download failures. Users that
	// want specific errors should use [WithTrace].
	ErrIncomplete = errors.New("incomplete")
)

// Defaults
const (
	// DefaultChunkingThreshold is the threshold at which a layer should be
	// split up into chunks when downloading.
	DefaultChunkingThreshold = 64 << 20
)

var defaultCache = sync.OnceValues(func() (*blob.DiskCache, error) {
	dir := os.Getenv("OLLAMA_MODELS")
	if dir == "" {
		home, _ := os.UserHomeDir()
		home = cmp.Or(home, ".")
		dir = filepath.Join(home, ".ollama", "models")
	}
	return blob.Open(dir)
})

// DefaultCache returns the default cache used by the registry. It is
// configured from the OLLAMA_MODELS environment variable, or defaults to
// $HOME/.ollama/models, or, if an error occurs obtaining the home directory,
// it uses the current working directory.
func DefaultCache() (*blob.DiskCache, error) {
	return defaultCache()
}

// Error is the standard error returned by Ollama APIs. It can represent a
// single or multiple error response.
//
// Single error responses have the following format:
//
//	{"code": "optional_code","error":"error message"}
//
// Multiple error responses have the following format:
//
//	{"errors": [{"code": "optional_code","message":"error message"}]}
//
// Note, that the error field is used in single error responses, while the
// message field is used in multiple error responses.
//
// In both cases, the code field is optional and may be empty.
type Error struct {
	status  int    `json:"-"` // TODO(bmizerany): remove this
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Temporary reports if the error is temporary (e.g. 5xx status code).
func (e *Error) Temporary() bool {
	return e.status >= 500
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("registry responded with status ")
	b.WriteString(strconv.Itoa(e.status))
	if e.Code != "" {
		b.WriteString(": code ")
		b.WriteString(e.Code)
	}
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	return b.String()
}

func (e *Error) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Int("status", e.status),
		slog.String("code", e.Code),
		slog.String("message", e.Message),
	)
}

// UnmarshalJSON implements json.Unmarshaler.
func (e *Error) UnmarshalJSON(b []byte) error {
	type E Error
	var v struct {
		// Single error
		Code  string
		Error string

		// Multiple errors
		Errors []E
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	if v.Error != "" {
		// Single error case
		e.Code = v.Code
		e.Message = v.Error
		return nil
	}
	if len(v.Errors) == 0 {
		return fmt.Errorf("no messages in error response: %s", string(b))
	}
	*e = Error(v.Errors[0]) // our registry only returns one error.
	return nil
}

const DefaultMask = "registry.ollama.ai/library/_:latest"

var defaultMask = func() names.Name {
	n := names.Parse(DefaultMask)
	if !n.IsFullyQualified() {
		panic("default mask is not fully qualified")
	}
	return n
}()

// CompleteName returns a fully qualified name by merging the given name with
// the default mask. If the name is already fully qualified, it is returned
// unchanged.
func CompleteName(name string) string {
	return names.Merge(names.Parse(name), defaultMask).String()
}

// Registry is a client for performing push and pull operations against an
// Ollama registry.
type Registry struct {
	// Cache is the cache used to store models. If nil, [DefaultCache] is
	// used.
	Cache *blob.DiskCache

	// UserAgent is the User-Agent header to send with requests to the
	// registry. If empty, the User-Agent is determined by HTTPClient.
	UserAgent string

	// Key is the key used to authenticate with the registry.
	//
	// Currently, only Ed25519 keys are supported.
	Key crypto.PrivateKey

	// HTTPClient is the HTTP client used to make requests to the registry.
	//
	// If nil, [http.DefaultClient] is used.
	//
	// As a quick note: If a Registry function that makes a call to a URL
	// with the "https+insecure" scheme, the client will be cloned and the
	// transport will be set to skip TLS verification, unless the client's
	// Transport done not have a Clone method with the same signature as
	// [http.Transport.Clone], which case, the call will fail.
	HTTPClient *http.Client

	// MaxStreams is the maximum number of concurrent streams to use when
	// pushing or pulling models. If zero, the number of streams is
	// determined by [runtime.GOMAXPROCS].
	//
	// A negative value means no limit.
	MaxStreams int

	// ChunkingThreshold is the maximum size of a layer to download in a single
	// request. If zero, [DefaultChunkingThreshold] is used.
	ChunkingThreshold int64

	// Mask, if set, is the name used to convert non-fully qualified names
	// to fully qualified names.
	// If empty, [DefaultMask] is used.
	Mask string

	// ReadTimeout is the maximum duration for reading the entire request,
	// including the body.
	// A zero or negative value means there will be no timeout.
	ReadTimeout time.Duration
}

func (r *Registry) readTimeout() time.Duration {
	if r.ReadTimeout > 0 {
		return r.ReadTimeout
	}
	return 1<<63 - 1 // no timeout, max int
}

func (r *Registry) cache() (*blob.DiskCache, error) {
	if r.Cache != nil {
		return r.Cache, nil
	}
	return defaultCache()
}

func (r *Registry) parseName(name string) (names.Name, error) {
	mask := defaultMask
	if r.Mask != "" {
		mask = names.Parse(r.Mask)
	}
	n := names.Merge(names.Parse(name), mask)
	if !n.IsFullyQualified() {
		return names.Name{}, fmt.Errorf("%w: %q", ErrNameInvalid, name)
	}
	return n, nil
}

// DefaultRegistry returns a new Registry configured from the environment. The
// key is read from $HOME/.ollama/id_ed25519, MaxStreams is set to the
// value of OLLAMA_REGISTRY_MAXSTREAMS, and ReadTimeout is set to 30 seconds.
//
// It returns an error if any configuration in the environment is invalid.
func DefaultRegistry() (*Registry, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	keyPEM, err := os.ReadFile(filepath.Join(home, ".ollama/id_ed25519"))
	if err != nil && errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}

	var rc Registry
	rc.ReadTimeout = 30 * time.Second
	rc.UserAgent = UserAgent()
	rc.Key, err = ssh.ParseRawPrivateKey(keyPEM)
	if err != nil {
		return nil, err
	}
	maxStreams := os.Getenv("OLLAMA_REGISTRY_MAXSTREAMS")
	if maxStreams != "" {
		var err error
		rc.MaxStreams, err = strconv.Atoi(maxStreams)
		if err != nil {
			return nil, fmt.Errorf("invalid OLLAMA_REGISTRY_MAXSTREAMS: %w", err)
		}
	}
	return &rc, nil
}

func UserAgent() string {
	buildinfo, _ := debug.ReadBuildInfo()

	version := buildinfo.Main.Version
	if version == "(devel)" {
		// When using `go run .` the version is "(devel)". This is seen
		// as an invalid version by ollama.com and so it defaults to
		// "needs upgrade" for some requests, such as pulls. These
		// checks can be skipped by using the special version "v0.0.0",
		// so we set it to that here.
		version = "v0.0.0"
	}

	return fmt.Sprintf("ollama/%s (%s %s) Go/%s",
		version,
		runtime.GOARCH,
		runtime.GOOS,
		runtime.Version(),
	)
}

func (r *Registry) maxStreams() int {
	return cmp.Or(r.MaxStreams, runtime.GOMAXPROCS(0))
}

func (r *Registry) maxChunkingThreshold() int64 {
	return cmp.Or(r.ChunkingThreshold, DefaultChunkingThreshold)
}

type PushParams struct {
	// From is an optional destination name for the model. If empty, the
	// destination name is the same as the source name.
	From string
}

// Push pushes the model with the name in the cache to the remote registry.
func (r *Registry) Push(ctx context.Context, name string, p *PushParams) error {
	if p == nil {
		p = &PushParams{}
	}

	c, err := r.cache()
	if err != nil {
		return err
	}

	m, err := r.ResolveLocal(cmp.Or(p.From, name))
	if err != nil {
		return err
	}

	// Before much else happens, check layers at not null, the blobs exist,
	// and the sizes match. This prevents long uploads followed by
	// disappointment.
	for _, l := range m.Layers {
		if l == nil {
			return fmt.Errorf("%w: null layer", ErrManifestInvalid)
		}
		info, err := c.Get(l.Digest)
		if err != nil {
			return fmt.Errorf("error getting %s: %w", l.Digest.Short(), err)
		}
		if info.Size != l.Size {
			return fmt.Errorf("size mismatch for %s: %d != %d", l.Digest.Short(), info.Size, l.Size)
		}
	}

	t := traceFromContext(ctx)

	scheme, n, _, err := r.parseNameExtended(name)
	if err != nil {
		// This should never happen since ResolveLocal should have
		// already validated the name.
		panic(err)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var g errgroup.Group
	g.SetLimit(r.maxStreams())
	for _, l := range m.Layers {
		var progress atomic.Int64
		g.Go(func() (err error) {
			defer func() { t.update(l, progress.Load(), err) }()

			t.update(l, 0, nil)

			startURL := fmt.Sprintf("%s://%s/v2/%s/%s/blobs/uploads/?digest=%s",
				scheme,
				n.Host(),
				n.Namespace(),
				n.Model(),
				l.Digest,
			)
			res, err := r.send(ctx, "POST", startURL, nil)
			if err != nil {
				return err
			}
			res.Body.Close()

			f, err := os.Open(c.GetFile(l.Digest))
			if err != nil {
				return err
			}
			defer f.Close()

			uploadURL := res.Header.Get("Location")
			if uploadURL == "" {
				t.update(l, l.Size, ErrCached)
				return nil
			}

			req, err := r.newRequest(ctx, "PUT", uploadURL, f)
			if err != nil {
				return fmt.Errorf("invalid upload URL returned from registry: %q: %w", uploadURL, err)
			}
			req.ContentLength = l.Size

			res, err = sendRequest(r.client(), req)
			if err == nil {
				res.Body.Close()
			}
			return err
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// Commit
	path := fmt.Sprintf("%s://%s/v2/%s/%s/manifests/%s",
		scheme,
		n.Host(),
		n.Namespace(),
		n.Model(),
		n.Tag(),
	)
	res, err := r.send(ctx, "PUT", path, bytes.NewReader(m.Data))
	if err == nil {
		res.Body.Close()
	}
	// TODO(bmizerany): add a "commit" trace event
	return err
}

// trackingReader is an io.Reader that tracks the number of bytes read and
// calls the update function with the layer, the number of bytes read.
//
// It always calls update with a nil error.
type trackingReader struct {
	r      io.Reader
	update func(n int64, err error) // err is always nil
}

func (r *trackingReader) Read(p []byte) (n int, err error) {
	n, err = r.r.Read(p)
	r.update(int64(n), nil)
	return
}

// Pull pulls the model with the given name from the remote registry into the
// cache.
//
// For layers larger then [Registry.MaxChunkSize], the layer is downloaded in
// chunks of the specified size, and then reassembled and verified. This is
// typically slower than splitting the model up across layers, and is mostly
// utilized for layers of type equal to "application/vnd.ollama.image".
func (r *Registry) Pull(ctx context.Context, name string) error {
	m, err := r.Resolve(ctx, name)
	if err != nil {
		return err
	}

	// TODO(bmizerany): decide if this should be considered valid. Maybe
	// server-side we special case '{}' to have some special meaning? Maybe
	// "archiving" a tag (which is how we reason about it in the registry
	// already, just with a different twist).
	if len(m.Layers) == 0 {
		return fmt.Errorf("%w: no layers", ErrManifestInvalid)
	}

	c, err := r.cache()
	if err != nil {
		return err
	}

	// TODO(bmizerany): work to remove the need to do this
	layers := m.Layers
	if m.Config != nil && m.Config.Digest.IsValid() {
		layers = append(layers, m.Config)
	}

	// Send initial layer trace events to allow clients to have an
	// understanding of work to be done before work starts.
	var expected int64
	t := traceFromContext(ctx)
	for _, l := range layers {
		t.update(l, 0, nil)
		expected += l.Size
	}

	var g errgroup.Group
	g.SetLimit(r.maxStreams())

	var completed atomic.Int64
	for _, l := range layers {
		var received atomic.Int64
		update := func(n int64, err error) {
			if n == 0 && err == nil {
				// Clients expect an update with no progress and no error to mean "starting download".
				// This is not the case here,
				// so we don't want to send an update in this case.
				return
			}
			completed.Add(n)
			t.update(l, received.Add(n), err)
		}

		info, err := c.Get(l.Digest)
		if err == nil && info.Size == l.Size {
			update(l.Size, ErrCached)
			continue
		}

		func() (err error) {
			defer func() {
				if err != nil {
					update(0, err)
				}
			}()

			var wg sync.WaitGroup
			chunked, err := c.Chunked(l.Digest, l.Size)
			if err != nil {
				return err
			}
			defer func() {
				// Close the chunked writer when all chunks are
				// downloaded.
				//
				// This is done as a background task in the
				// group to allow the next layer to start while
				// we wait for the final chunk in this layer to
				// complete. It also ensures this is done
				// before we exit Pull.
				g.Go(func() error {
					wg.Wait()
					chunked.Close()
					return nil
				})
			}()

			for cs, err := range r.chunksums(ctx, name, l) {
				if err != nil {
					// Note the chunksum stream
					// interruption, but do not cancel
					// in-flight downloads. We can still
					// make progress on them. Once they are
					// done, ErrIncomplete will be returned
					// below.
					update(0, err)
					break
				}

				cacheKey := fmt.Sprintf(
					"v1 pull chunksum %s %s %d-%d",
					l.Digest,
					cs.Digest,
					cs.Chunk.Start,
					cs.Chunk.End,
				)
				cacheKeyDigest := blob.DigestFromBytes(cacheKey)
				_, err := c.Get(cacheKeyDigest)
				if err == nil {
					update(cs.Chunk.Size(), ErrCached)
					continue
				}

				wg.Add(1)
				g.Go(func() (err error) {
					defer func() {
						defer wg.Done()
						if err != nil {
							update(0, err)
						}
					}()

					ctx, cancel := context.WithCancelCause(ctx)
					defer cancel(nil)

					timer := time.AfterFunc(r.readTimeout(), func() {
						cancel(fmt.Errorf("%w: downloading %s %d-%d/%d",
							context.DeadlineExceeded,
							cs.Digest.Short(),
							cs.Chunk.Start,
							cs.Chunk.End,
							l.Size,
						))
					})
					defer timer.Stop()

					req, err := http.NewRequestWithContext(ctx, "GET", cs.URL, nil)
					if err != nil {
						return err
					}
					req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", cs.Chunk.Start, cs.Chunk.End))
					res, err := sendRequest(r.client(), req)
					if err != nil {
						return err
					}
					defer res.Body.Close()

					tr := &trackingReader{
						r: res.Body,
						update: func(n int64, err error) {
							timer.Reset(r.readTimeout())
							update(n, err)
						},
					}
					if err := chunked.Put(cs.Chunk, cs.Digest, tr); err != nil {
						return err
					}

					// Record the downloading of this chunk.
					return blob.PutBytes(c, cacheKeyDigest, cacheKey)
				})
			}

			return nil
		}()
	}
	if err := g.Wait(); err != nil {
		return err
	}
	if recv := completed.Load(); recv != expected {
		return fmt.Errorf("%w: received %d/%d bytes", ErrIncomplete, recv, expected)
	}

	md := blob.DigestFromBytes(m.Data)
	if err := blob.PutBytes(c, md, m.Data); err != nil {
		return err
	}
	return c.Link(m.Name, md)
}

// Unlink is like [blob.DiskCache.Unlink], but makes name fully qualified
// before attempting to unlink the model.
func (r *Registry) Unlink(name string) (ok bool, _ error) {
	n, err := r.parseName(name)
	if err != nil {
		return false, err
	}
	c, err := r.cache()
	if err != nil {
		return false, err
	}
	return c.Unlink(n.String())
}

// Manifest represents a [ollama.com/manifest].
type Manifest struct {
	Name   string   `json:"-"` // the canonical name of the model
	Data   []byte   `json:"-"` // the raw data of the manifest
	Layers []*Layer `json:"layers"`

	// For legacy reasons, we still have to download the config layer.
	Config *Layer `json:"config"`
}

// Layer returns the layer with the given
// digest, or nil if not found.
func (m *Manifest) Layer(d blob.Digest) *Layer {
	for _, l := range m.Layers {
		if l.Digest == d {
			return l
		}
	}
	return nil
}

func (m *Manifest) All() iter.Seq[*Layer] {
	return func(yield func(*Layer) bool) {
		if !yield(m.Config) {
			return
		}
		for _, l := range m.Layers {
			if !yield(l) {
				return
			}
		}
	}
}

func (m *Manifest) Size() int64 {
	var size int64
	if m.Config != nil {
		size += m.Config.Size
	}
	for _, l := range m.Layers {
		size += l.Size
	}
	return size
}

// MarshalJSON implements json.Marshaler.
//
// NOTE: It adds an empty config object to the manifest, which is required by
// the registry, but not used by the client. In the future, the config object
// will not be required by the registry and this will should be removed.
func (m Manifest) MarshalJSON() ([]byte, error) {
	type M Manifest
	v := struct {
		M

		// This is ignored, mostly, by the registry But, if not
		// present, it will cause an error to be returned during the
		// last phase of the commit which expects it, but does nothing
		// with it. This will be fixed in a future release of
		// ollama.com.
		Config Layer `json:"config"`
	}{
		M: M(m),
	}
	return json.Marshal(v)
}

// unmarshalManifest unmarshals the data into a manifest, and sets the name
// field to the string representation of the name.
//
// It panics if the name is not fully qualified. Callers should ensure the name
// is fully qualified before calling this function.
func unmarshalManifest(n names.Name, data []byte) (*Manifest, error) {
	if !n.IsFullyQualified() {
		panic(fmt.Sprintf("unmarshalManifest: name is not fully qualified: %s", n.String()))
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	m.Name = n.String()
	m.Data = data
	return &m, nil
}

// Layer is a layer in a model.
type Layer struct {
	Digest    blob.Digest `json:"digest"`
	MediaType string      `json:"mediaType"`
	Size      int64       `json:"size"`
}

// ResolveLocal resolves a name to a Manifest in the local cache.
func (r *Registry) ResolveLocal(name string) (*Manifest, error) {
	_, n, d, err := r.parseNameExtended(name)
	if err != nil {
		return nil, err
	}
	c, err := r.cache()
	if err != nil {
		return nil, err
	}
	if !d.IsValid() {
		// No digest, so resolve the manifest by name.
		d, err = c.Resolve(n.String())
		if err != nil {
			return nil, err
		}
	}
	data, err := os.ReadFile(c.GetFile(d))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", ErrModelNotFound, name)
		}
		return nil, err
	}
	m, err := unmarshalManifest(n, data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, errors.Join(ErrManifestInvalid, err))
	}
	return m, nil
}

// Resolve resolves a name to a Manifest in the remote registry.
func (r *Registry) Resolve(ctx context.Context, name string) (*Manifest, error) {
	scheme, n, d, err := r.parseNameExtended(name)
	if err != nil {
		return nil, err
	}

	manifestURL := fmt.Sprintf("%s://%s/v2/%s/%s/manifests/%s", scheme, n.Host(), n.Namespace(), n.Model(), n.Tag())
	if d.IsValid() {
		manifestURL = fmt.Sprintf("%s://%s/v2/%s/%s/blobs/%s", scheme, n.Host(), n.Namespace(), n.Model(), d)
	}

	res, err := r.send(ctx, "GET", manifestURL, nil)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	// TODO(bmizerany): return digest here
	m, err := unmarshalManifest(n, data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, errors.Join(ErrManifestInvalid, err))
	}
	return m, nil
}

type chunksum struct {
	URL    string
	Chunk  blob.Chunk
	Digest blob.Digest
}

// chunksums returns a sequence of chunksums for the given layer. If the layer is under the
// chunking threshold, a single chunksum is returned that covers the entire layer. If the layer
// is over the chunking threshold, the chunksums are read from the chunksums endpoint.
func (r *Registry) chunksums(ctx context.Context, name string, l *Layer) iter.Seq2[chunksum, error] {
	return func(yield func(chunksum, error) bool) {
		scheme, n, _, err := r.parseNameExtended(name)
		if err != nil {
			yield(chunksum{}, err)
			return
		}

		if l.Size < r.maxChunkingThreshold() {
			// any layer under the threshold should be downloaded
			// in one go.
			cs := chunksum{
				URL: fmt.Sprintf("%s://%s/v2/%s/%s/blobs/%s",
					scheme,
					n.Host(),
					n.Namespace(),
					n.Model(),
					l.Digest,
				),
				Chunk:  blob.Chunk{Start: 0, End: l.Size - 1},
				Digest: l.Digest,
			}
			yield(cs, nil)
			return
		}

		// The response is a sequence of chunksums.
		//
		// Chunksums are chunks of a larger blob that can be
		// downloaded and verified independently.
		//
		// The chunksums endpoint is a GET request that returns a
		// sequence of chunksums in the following format:
		//
		//     > GET /v2/<namespace>/<model>/chunksums/<digest>
		//
		//     < HTTP/1.1 200 OK
		//     < Content-Location: <blobURL>
		//     <
		//     < <digest> <start>-<end>
		//     < ...
		//
		// The <blobURL> is the URL to download the chunks from and
		// each <digest> is the digest of the chunk, and <start>-<end>
		// is the range the chunk in the blob.
		//
		// Ranges may be used directly in Range headers like
		// "bytes=<start>-<end>".
		//
		// The chunksums returned are guaranteed to be contiguous and
		// include all bytes of the layer. If the stream is cut short,
		// clients should retry.

		chunksumsURL := fmt.Sprintf("%s://%s/v2/%s/%s/chunksums/%s",
			scheme,
			n.Host(),
			n.Namespace(),
			n.Model(),
			l.Digest,
		)

		req, err := r.newRequest(ctx, "GET", chunksumsURL, nil)
		if err != nil {
			yield(chunksum{}, err)
			return
		}
		res, err := sendRequest(r.client(), req)
		if err != nil {
			yield(chunksum{}, err)
			return
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			err := fmt.Errorf("chunksums: unexpected status code %d", res.StatusCode)
			yield(chunksum{}, err)
			return
		}
		blobURL := res.Header.Get("Content-Location")

		s := bufio.NewScanner(res.Body)
		s.Split(bufio.ScanWords)
		for {
			if !s.Scan() {
				if s.Err() != nil {
					yield(chunksum{}, s.Err())
				}
				return
			}
			d, err := blob.ParseDigest(s.Bytes())
			if err != nil {
				yield(chunksum{}, fmt.Errorf("invalid digest: %q", s.Bytes()))
				return
			}

			if !s.Scan() {
				err := s.Err()
				if err == nil {
					err = fmt.Errorf("missing chunk range for digest %s", d)
				}
				yield(chunksum{}, err)
				return
			}
			chunk, err := parseChunk(s.Bytes())
			if err != nil {
				yield(chunksum{}, fmt.Errorf("invalid chunk range for digest %s: %q", d, s.Bytes()))
				return
			}

			cs := chunksum{
				URL:    blobURL,
				Chunk:  chunk,
				Digest: d,
			}
			if !yield(cs, nil) {
				return
			}
		}
	}
}

func (r *Registry) client() *http.Client {
	if r.HTTPClient != nil {
		return r.HTTPClient
	}
	return http.DefaultClient
}

// newRequest constructs a new request, ready to use, with the given method,
// url, and body, pre-signed with client [Key] and [UserAgent].
func (r *Registry) newRequest(ctx context.Context, method, url string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, err
	}
	if r.UserAgent != "" {
		req.Header.Set("User-Agent", r.UserAgent)
	}
	if r.Key != nil {
		token, err := makeAuthToken(r.Key)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req, nil
}

// sendRequest makes a request with the given client and request, and returns the
// response if the status code is 200. If the status code is not 200, an Error
// is parsed from the response body and returned. If any other error occurs, it
// is returned.
func sendRequest(c *http.Client, r *http.Request) (_ *http.Response, err error) {
	if r.URL.Scheme == "https+insecure" {
		// TODO(bmizerany): clone client.Transport, set
		// InsecureSkipVerify, etc.

		type cloner interface {
			Clone() *http.Transport
		}

		// Attempt to configure the transport to skip TLS verification
		// if we can clone it, otherwise fall through and let the http
		// client complain and the scheme being invalid.
		x, ok := cmp.Or(c.Transport, http.DefaultTransport).(cloner)
		if ok {
			tr := x.Clone()
			tr.TLSClientConfig = cmp.Or(tr.TLSClientConfig, &tls.Config{})
			tr.TLSClientConfig.InsecureSkipVerify = true

			cc := *c // shallow copy
			cc.Transport = tr
			c = &cc

			r = r.Clone(r.Context())
			r.URL.Scheme = "https"

			// fall through
		}
	}

	res, err := c.Do(r)
	if err != nil {
		return nil, err
	}
	if res.StatusCode/100 != 2 {
		out, err := io.ReadAll(res.Body)
		if err != nil {
			return nil, err
		}
		var re Error
		if err := json.Unmarshal(out, &re); err != nil {
			// Use the raw body if we can't parse it as an error object.
			re.Message = string(out)
		}

		// coerce MANIFEST_UNKNOWN to ErrManifestNotFound
		if strings.EqualFold(re.Code, "MANIFEST_UNKNOWN") {
			return nil, ErrModelNotFound
		}

		re.status = res.StatusCode
		return nil, &re
	}
	return res, nil
}

// send is a convenience method for making a request with newRequest and
// passing it to send with r.client().
func (r *Registry) send(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	req, err := r.newRequest(ctx, method, path, body)
	if err != nil {
		return nil, err
	}
	return sendRequest(r.client(), req)
}

// makeAuthToken creates an Ollama auth token for the given private key.
//
// NOTE: This format is OLD, overly complex, and should be replaced. We're
// inheriting it from the original Ollama client and ollama.com
// implementations, so we need to support it for now.
func makeAuthToken(key crypto.PrivateKey) (string, error) {
	privKey, _ := key.(*ed25519.PrivateKey)
	if privKey == nil {
		return "", fmt.Errorf("unsupported private key type: %T", key)
	}

	url := fmt.Sprintf("https://ollama.com?ts=%d", time.Now().Unix())
	// Part 1: the checkData (e.g. the URL with a timestamp)

	// Part 2: the public key
	pubKeyShort, err := func() ([]byte, error) {
		sshPubKey, err := ssh.NewPublicKey(privKey.Public())
		if err != nil {
			return nil, err
		}
		pubKeyParts := bytes.Fields(ssh.MarshalAuthorizedKey(sshPubKey))
		if len(pubKeyParts) < 2 {
			return nil, fmt.Errorf("malformed public key: %q", pubKeyParts)
		}
		pubKeyShort := pubKeyParts[1]
		return pubKeyShort, nil
	}()
	if err != nil {
		return "", err
	}

	// Part 3: the signature
	sig := ed25519.Sign(*privKey, []byte(checkData(url)))

	// Assemble the token: <checkData>:<pubKey>:<signature>
	var b strings.Builder
	io.WriteString(&b, base64.StdEncoding.EncodeToString([]byte(url)))
	b.WriteByte(':')
	b.Write(pubKeyShort)
	b.WriteByte(':')
	io.WriteString(&b, base64.StdEncoding.EncodeToString(sig))

	return b.String(), nil
}

// The original spec for Ollama tokens was to use the SHA256 of the zero
// string as part of the signature. I'm not sure why that was, but we still
// need it to verify the signature.
var zeroSum = func() string {
	sha256sum := sha256.Sum256(nil)
	x := base64.StdEncoding.EncodeToString([]byte(hex.EncodeToString(sha256sum[:])))
	return x
}()

// checkData takes a URL and creates the original string format of the
// data signature that is used by the ollama client to sign requests
func checkData(url string) string {
	return fmt.Sprintf("GET,%s,%s", url, zeroSum)
}

type publicError struct {
	wrapped error
	message string
}

func withPublicMessagef(err error, message string, args ...any) error {
	return publicError{wrapped: err, message: fmt.Sprintf(message, args...)}
}

func (e publicError) Error() string { return e.message }
func (e publicError) Unwrap() error { return e.wrapped }

var supportedSchemes = []string{
	"http",
	"https",
	"https+insecure",
}

var supportedSchemesMessage = fmt.Sprintf("supported schemes are %v", strings.Join(supportedSchemes, ", "))

// parseNameExtended parses and validates an extended name, returning the scheme, name,
// and digest.
//
// If the scheme is empty, scheme will be "https". If an unsupported scheme is
// given, [ErrNameInvalid] wrapped with a display friendly message is returned.
//
// If the digest is invalid, [ErrNameInvalid] wrapped with a display friendly
// message is returned.
//
// If the name is not, once merged with the mask, fully qualified,
// [ErrNameInvalid] wrapped with a display friendly message is returned.
func (r *Registry) parseNameExtended(s string) (scheme string, _ names.Name, _ blob.Digest, _ error) {
	scheme, name, digest := splitExtended(s)
	scheme = cmp.Or(scheme, "https")
	if !slices.Contains(supportedSchemes, scheme) {
		err := withPublicMessagef(ErrNameInvalid, "unsupported scheme: %q: %s", scheme, supportedSchemesMessage)
		return "", names.Name{}, blob.Digest{}, err
	}

	var d blob.Digest
	if digest != "" {
		var err error
		d, err = blob.ParseDigest(digest)
		if err != nil {
			err = withPublicMessagef(ErrNameInvalid, "invalid digest: %q", digest)
			return "", names.Name{}, blob.Digest{}, err
		}
		if name == "" {
			// We have can resolve a manifest from a digest only,
			// so skip name validation and return the scheme and
			// digest.
			return scheme, names.Name{}, d, nil
		}
	}

	n, err := r.parseName(name)
	if err != nil {
		return "", names.Name{}, blob.Digest{}, err
	}
	return scheme, n, d, nil
}

// splitExtended splits an extended name string into its scheme, name, and digest
// parts.
//
// Examples:
//
//	http://ollama.com/bmizerany/smol:latest@digest
//	https://ollama.com/bmizerany/smol:latest
//	ollama.com/bmizerany/smol:latest@digest // returns "https" scheme.
//	model@digest
//	@digest
func splitExtended(s string) (scheme, name, digest string) {
	i := strings.Index(s, "://")
	if i >= 0 {
		scheme = s[:i]
		s = s[i+3:]
	}
	i = strings.LastIndex(s, "@")
	if i >= 0 {
		digest = s[i+1:]
		s = s[:i]
	}
	return scheme, s, digest
}

// parseChunk parses a string in the form "start-end" and returns the Chunk.
func parseChunk[S ~string | ~[]byte](s S) (blob.Chunk, error) {
	startPart, endPart, found := strings.Cut(string(s), "-")
	if !found {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid range %q: missing '-'", s)
	}
	start, err := strconv.ParseInt(startPart, 10, 64)
	if err != nil {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid start to %q: %v", s, err)
	}
	end, err := strconv.ParseInt(endPart, 10, 64)
	if err != nil {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid end to %q: %v", s, err)
	}
	if start > end {
		return blob.Chunk{}, fmt.Errorf("chunks: invalid range %q: start > end", s)
	}
	return blob.Chunk{Start: start, End: end}, nil
}
