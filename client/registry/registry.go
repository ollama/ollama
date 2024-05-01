package registry

import (
	"cmp"
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"sync"

	"github.com/ollama/ollama/client/ollama"
	"github.com/ollama/ollama/client/registry/apitype"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/exp/constraints"
	"golang.org/x/sync/errgroup"
)

// Errors
var (
	ErrLayerNotFound = errors.New("layer not found")
)

type Client struct {
	BaseURL string

	Logger *slog.Logger

	// NameFill is a string that is used to fill in the missing parts of
	// a name when it is not fully qualified. It is used to make a name
	// fully qualified before pushing or pulling it. The default is
	// "registry.ollama.ai/library/_:latest".
	//
	// Most users can ignore this field. It is intended for use by
	// clients that need to push or pull names to registries other than
	// registry.ollama.ai, and for testing.
	NameFill string
}

func (c *Client) log() *slog.Logger {
	return cmp.Or(c.Logger, slog.Default())
}

func (c *Client) oclient() *ollama.Client {
	return &ollama.Client{
		BaseURL: c.BaseURL,
	}
}

type ReadAtSeekCloser interface {
	io.ReaderAt
	io.Seeker
	io.Closer
}

type Cache interface {
	// LayerFile returns the absolute file path to the layer file for
	// the given model digest.
	//
	// If the digest is invalid, or the layer does not exist, the empty
	// string is returned.
	LayerFile(model.Digest) string

	// OpenLayer opens the layer file for the given model digest and
	// returns it, or an if any. The caller is responsible for closing
	// the returned file.
	OpenLayer(model.Digest) (ReadAtSeekCloser, error)

	// PutLayerFile moves the layer file at fromPath to the cache for
	// the given model digest. It is a hack intended to short circuit a
	// file copy operation.
	//
	// The file returned is expected to exist for the lifetime of the
	// cache.
	//
	// TODO(bmizerany): remove this; find a better way. Once we move
	// this into a build package, we should be able to get rid of this.
	PutLayerFile(_ model.Digest, fromPath string) error

	// SetManifestData sets the provided manifest data for the given
	// model name. If the manifest data is empty, the manifest is
	// removed. If the manifeest exists, it is overwritten.
	//
	// It is an error to call SetManifestData with a name that is not
	// complete.
	SetManifestData(model.Name, []byte) error

	// ManifestData returns the manifest data for the given model name.
	//
	// If the name incomplete, or the manifest does not exist, the empty
	// string is returned.
	ManifestData(name model.Name) []byte
}

// Pull pulls the manifest for name, and downloads any of its required
// layers that are not already in the cache. It returns an error if any part
// of the process fails, specifically:
func (c *Client) Pull(ctx context.Context, cache Cache, name string) error {
	mn := parseNameFill(name, c.NameFill)
	if !mn.IsFullyQualified() {
		return fmt.Errorf("ollama: pull: invalid name: %s", name)
	}

	log := c.log().With("name", name)

	pr, err := ollama.Do[*apitype.PullResponse](ctx, c.oclient(), "GET", "/v1/pull/"+name, nil)
	if err != nil {
		return fmt.Errorf("ollama: pull: %w: %s", err, name)
	}

	if pr.Manifest == nil || len(pr.Manifest.Layers) == 0 {
		return fmt.Errorf("ollama: pull: invalid manifest: %s: no layers found", name)
	}

	// download required layers we do not already have
	for _, l := range pr.Manifest.Layers {
		d, err := model.ParseDigest(l.Digest)
		if err != nil {
			return fmt.Errorf("ollama: reading manifest: %w: %s", err, l.Digest)
		}
		if cache.LayerFile(d) != "" {
			continue
		}
		err = func() error {
			log := log.With("digest", l.Digest, "mediaType", l.MediaType, "size", l.Size)
			log.Debug("starting download")

			// TODO(bmizerany): stop using temp which might not
			// be on same device as cache.... instead let cache
			// give us a place to store parts...
			tmpFile, err := os.CreateTemp("", "ollama-download-")
			if err != nil {
				return err
			}
			defer func() {
				tmpFile.Close()
				os.Remove(tmpFile.Name()) // in case we fail before committing
			}()

			g, ctx := errgroup.WithContext(ctx)
			g.SetLimit(8) // TODO(bmizerany): make this configurable

			// TODO(bmizerany): make chunk size configurable
			const chunkSize = 50 * 1024 * 1024 // 50MB
			chunks(l.Size, chunkSize)(func(_ int, rng chunkRange[int64]) bool {
				g.Go(func() (err error) {
					defer func() {
						if err == nil {
							return
						}
						safeURL := redactAmzSignature(l.URL)
						err = fmt.Errorf("%w: %s %s bytes=%s: %s", err, pr.Name, l.Digest, rng, safeURL)
					}()

					log.Debug("downloading", "range", rng)

					// TODO(bmizerany): retry
					// TODO(bmizerany): use real http client
					// TODO(bmizerany): resumable
					// TODO(bmizerany): multipart download
					req, err := http.NewRequestWithContext(ctx, "GET", l.URL, nil)
					if err != nil {
						return err
					}
					req.Header.Set("Range", "bytes="+rng.String())

					res, err := http.DefaultClient.Do(req)
					if err != nil {
						return err
					}
					defer res.Body.Close()
					if res.StatusCode/100 != 2 {
						log.Debug("unexpected non-2XX status code", "status", res.StatusCode)
						return fmt.Errorf("unexpected status code fetching layer: %d", res.StatusCode)
					}
					if res.ContentLength != rng.Size() {
						return fmt.Errorf("unexpected content length: %d", res.ContentLength)
					}
					w := io.NewOffsetWriter(tmpFile, rng.Start)
					_, err = io.Copy(w, res.Body)
					return err
				})
				return true
			})
			if err := g.Wait(); err != nil {
				return err
			}

			tmpFile.Close() // release our hold on the file before moving it
			return cache.PutLayerFile(d, tmpFile.Name())
		}()
		if err != nil {
			return fmt.Errorf("ollama: pull: %w", err)
		}
	}

	// do not store the presigned URLs in the cache
	for i := range pr.Manifest.Layers {
		pr.Manifest.Layers[i].URL = ""
	}
	data, err := json.Marshal(pr.Manifest)
	if err != nil {
		return err
	}

	// TODO(bmizerany): remove dep on model.Name
	return cache.SetManifestData(mn, data)
}

type nopSeeker struct {
	io.Reader
}

func (nopSeeker) Seek(int64, int) (int64, error) {
	return 0, nil
}

func parseNameFill(name, fill string) model.Name {
	fill = cmp.Or(fill, "bllamo.com/library/_:latest")
	f := model.ParseNameBare(fill)
	if !f.IsFullyQualified() {
		panic(fmt.Errorf("invalid fill: %q", fill))
	}
	return model.Merge(model.ParseNameBare(name), f)
}

// Push pushes a manifest to the server and responds to the server's
// requests for layer uploads, if any, and finally commits the manifest for
// name. It returns an error if any part of the process fails, specifically:
//
// If the server requests layers not found in the cache, ErrLayerNotFound is
// returned.
func (c *Client) Push(ctx context.Context, cache Cache, name string) error {
	mn := parseNameFill(name, c.NameFill)
	if !mn.IsFullyQualified() {
		return fmt.Errorf("ollama: push: invalid name: %s", name)
	}
	manifest := cache.ManifestData(mn)
	if len(manifest) == 0 {
		return fmt.Errorf("manifest not found: %s", name)
	}

	var mu sync.Mutex
	var completed []*apitype.CompletePart
	push := func() (*apitype.PushResponse, error) {
		v, err := ollama.Do[*apitype.PushResponse](ctx, c.oclient(), "POST", "/v1/push", &apitype.PushRequest{
			Name:          name,
			Manifest:      manifest,
			CompleteParts: completed,
		})
		if err != nil {
			return nil, fmt.Errorf("Do: %w", err)
		}
		return v, nil
	}

	pr, err := push()
	if err != nil {
		return err
	}

	var g errgroup.Group
	for _, need := range pr.Needs {
		g.Go(func() error {
			nd, err := model.ParseDigest(need.Digest)
			if err != nil {
				return fmt.Errorf("ParseDigest: %w: %s", err, need.Digest)
			}
			f, err := cache.OpenLayer(nd)
			if err != nil {
				return fmt.Errorf("OpenLayer: %w: %s", err, need.Digest)
			}
			defer f.Close()

			c.log().Info("pushing layer", "digest", need.Digest, "start", need.Start, "end", need.End)
			cp, err := PushLayer(ctx, f, need.URL, need.Start, need.End)
			if err != nil {
				return fmt.Errorf("PushLayer: %w: %s", err, need.Digest)
			}
			mu.Lock()
			completed = append(completed, cp)
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return fmt.Errorf("Push: Required: %w", err)
	}

	if len(completed) > 0 {
		pr, err := push()
		if err != nil {
			return err
		}
		if len(pr.Needs) > 0 {
			var errs []error
			for _, r := range pr.Needs {
				errs = append(errs, fmt.Errorf("Push: server failed to find part: %q", r.Digest))
			}
			return errors.Join(errs...)
		}
	}

	return cache.SetManifestData(mn, manifest)
}

func PushLayer(ctx context.Context, body io.ReaderAt, url string, start, end int64) (*apitype.CompletePart, error) {
	if start < 0 || end < start {
		return nil, errors.New("start must satisfy 0 <= start <= end")
	}

	file := io.NewSectionReader(body, start, end-start+1)
	req, err := http.NewRequest("PUT", url, file)
	if err != nil {
		return nil, err
	}
	req.ContentLength = end - start + 1

	// TODO(bmizerany): take content type param
	req.Header.Set("Content-Type", "text/plain")

	if start != 0 || end != 0 {
		req.Header.Set("x-amz-copy-source-range", fmt.Sprintf("bytes=%d-%d", start, end))
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		e := parseS3Error(res)
		return nil, fmt.Errorf("unexpected status code: %d; %w", res.StatusCode, e)
	}
	cp := &apitype.CompletePart{
		URL:  url,
		ETag: res.Header.Get("ETag"),
		// TODO(bmizerany): checksum
	}
	return cp, nil
}

type s3Error struct {
	XMLName   xml.Name `xml:"Error"`
	Code      string   `xml:"Code"`
	Message   string   `xml:"Message"`
	Resource  string   `xml:"Resource"`
	RequestId string   `xml:"RequestId"`
}

func (e *s3Error) Error() string {
	return fmt.Sprintf("S3 (%s): %s: %s: %s", e.RequestId, e.Resource, e.Code, e.Message)
}

// parseS3Error parses an XML error response from S3.
func parseS3Error(res *http.Response) error {
	var se *s3Error
	if err := xml.NewDecoder(res.Body).Decode(&se); err != nil {
		return err
	}
	return se
}

// TODO: replace below by using upload pkg after we have rangefunc; until
// then, we need to keep this free of rangefunc for now.
type chunkRange[I constraints.Integer] struct {
	// Start is the byte offset of the chunk.
	Start I

	// End is the byte offset of the last byte in the chunk.
	End I
}

func (c chunkRange[I]) Size() I {
	return c.End - c.Start + 1
}

func (c chunkRange[I]) String() string {
	return fmt.Sprintf("%d-%d", c.Start, c.End)
}

func (c chunkRange[I]) LogValue() slog.Value {
	return slog.StringValue(c.String())
}

// Chunks yields a sequence of a part number and a Chunk. The Chunk is the offset
// and size of the chunk. The last chunk may be smaller than chunkSize if size is
// not a multiple of chunkSize.
//
// The first part number is 1 and increases monotonically.
func chunks[I constraints.Integer](size, chunkSize I) iter.Seq2[int, chunkRange[I]] {
	return func(yield func(int, chunkRange[I]) bool) {
		var n int
		for off := I(0); off < size; off += chunkSize {
			n++
			if !yield(n, chunkRange[I]{
				Start: off,
				End:   off + min(chunkSize, size-off) - 1,
			}) {
				return
			}
		}
	}
}

func redactAmzSignature(s string) string {
	u, err := url.Parse(s)
	if err != nil {
		return ""
	}
	q := u.Query()
	q.Set("X-Amz-Signature", "REDACTED")
	u.RawQuery = q.Encode()
	return u.String()
}
