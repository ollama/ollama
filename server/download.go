package server

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const maxRetries = 6

var (
	errMaxRetriesExceeded   = errors.New("max retries exceeded")
	errPartStalled          = errors.New("part stalled")
	errMaxRedirectsExceeded = errors.New("maximum redirects exceeded (10) for directURL")
)

var blobDownloadManager sync.Map

type blobDownload struct {
	Name   string
	Digest string

	Total     int64
	Completed atomic.Int64

	Parts []*blobDownloadPart

	context.CancelFunc

	done       chan struct{}
	err        error
	references atomic.Int32

	// inlineHash computes the SHA256 of the blob bytes as they flow in
	// from the network. Used instead of re-reading the file after download
	// to avoid a Linux kernel read-path bug that intermittently returns
	// corrupted bytes when reading very large files. Only safe because we
	// force single-part sequential downloads (no concurrent writers) in
	// Prepare below.
	inlineHash hash.Hash
}

type blobDownloadPart struct {
	N         int
	Offset    int64
	Size      int64
	Completed atomic.Int64

	lastUpdatedMu sync.Mutex
	lastUpdated   time.Time

	*blobDownload `json:"-"`
}

type jsonBlobDownloadPart struct {
	N         int
	Offset    int64
	Size      int64
	Completed int64
}

func (p *blobDownloadPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(jsonBlobDownloadPart{
		N:         p.N,
		Offset:    p.Offset,
		Size:      p.Size,
		Completed: p.Completed.Load(),
	})
}

func (p *blobDownloadPart) UnmarshalJSON(b []byte) error {
	var j jsonBlobDownloadPart
	if err := json.Unmarshal(b, &j); err != nil {
		return err
	}
	*p = blobDownloadPart{
		N:      j.N,
		Offset: j.Offset,
		Size:   j.Size,
	}
	p.Completed.Store(j.Completed)
	return nil
}

const (
	numDownloadParts          = 16
	minDownloadPartSize int64 = 100 * format.MegaByte
	maxDownloadPartSize int64 = 1000 * format.MegaByte
)

func (p *blobDownloadPart) Name() string {
	return strings.Join([]string{
		p.blobDownload.Name, "partial", strconv.Itoa(p.N),
	}, "-")
}

func (p *blobDownloadPart) StartsAt() int64 {
	return p.Offset + p.Completed.Load()
}

func (p *blobDownloadPart) StopsAt() int64 {
	return p.Offset + p.Size
}

func (p *blobDownloadPart) Write(b []byte) (n int, err error) {
	n = len(b)
	p.blobDownload.Completed.Add(int64(n))
	p.lastUpdatedMu.Lock()
	p.lastUpdated = time.Now()
	p.lastUpdatedMu.Unlock()
	return n, nil
}

func (b *blobDownload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	if envconfig.VerifyInlineHash() {
		return b.prepareInline(ctx, requestURL, opts)
	}

	partFilePaths, err := filepath.Glob(b.Name + "-partial-*")
	if err != nil {
		return err
	}

	b.done = make(chan struct{})

	for _, partFilePath := range partFilePaths {
		part, err := b.readPart(partFilePath)
		if err != nil {
			return err
		}

		b.Total += part.Size
		b.Completed.Add(part.Completed.Load())
		b.Parts = append(b.Parts, part)
	}

	if len(b.Parts) == 0 {
		resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		b.Total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

		size := b.Total / numDownloadParts
		switch {
		case size < minDownloadPartSize:
			size = minDownloadPartSize
		case size > maxDownloadPartSize:
			size = maxDownloadPartSize
		}

		var offset int64
		for offset < b.Total {
			if offset+size > b.Total {
				size = b.Total - offset
			}

			if err := b.newPart(offset, size); err != nil {
				return err
			}

			offset += size
		}
	}

	if len(b.Parts) > 0 {
		slog.Info(fmt.Sprintf("downloading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size)))
	}

	return nil
}

// prepareInline sets up an opt-in single-part download with an inline
// sha256 hasher. This avoids re-reading the file from disk to verify the
// digest after download, which is necessary on Linux kernels where the
// read path returns corrupted bytes for sequential reads larger than a
// few GiB even though the bytes on disk are correct. Enabled by setting
// OLLAMA_VERIFY_INLINE_HASH=1. Trades multi-part parallelism and
// cross-invocation resume for correctness on affected systems.
func (b *blobDownload) prepareInline(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	// Fresh start every invocation. Existing partial bytes can't be
	// incorporated into the inline hasher after the fact, so resume is
	// not supported in this mode.
	_ = os.Remove(b.Name + "-partial")
	if oldParts, _ := filepath.Glob(b.Name + "-partial-*"); len(oldParts) > 0 {
		for _, p := range oldParts {
			_ = os.Remove(p)
		}
	}

	b.done = make(chan struct{})

	resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	b.Total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	b.inlineHash = sha256.New()

	// Single part covering the entire blob. Only one goroutine writes to
	// b.inlineHash, sequentially, so the final digest is authoritative
	// without ever re-reading the file.
	if err := b.newPart(0, b.Total); err != nil {
		return err
	}

	slog.Info(fmt.Sprintf("downloading %s (%s, inline-hash single stream)", b.Digest[7:19], format.HumanBytes(b.Total)))
	return nil
}

func (b *blobDownload) Run(ctx context.Context, requestURL *url.URL, opts *registryOptions) {
	defer close(b.done)
	b.err = b.run(ctx, requestURL, opts)
}

func newBackoff(maxBackoff time.Duration) func(ctx context.Context) error {
	var n int
	return func(ctx context.Context) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		n++

		// n^2 backoff timer is a little smoother than the
		// common choice of 2^n.
		d := min(time.Duration(n*n)*10*time.Millisecond, maxBackoff)
		// Randomize the delay between 0.5-1.5 x msec, in order
		// to prevent accidental "thundering herd" problems.
		d = time.Duration(float64(d) * (rand.Float64() + 0.5))
		t := time.NewTimer(d)
		defer t.Stop()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			return nil
		}
	}
}

func (b *blobDownload) run(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	defer blobDownloadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	file, err := os.OpenFile(b.Name+"-partial", os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	setSparse(file)

	_ = file.Truncate(b.Total)

	directURL, err := func() (*url.URL, error) {
		ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		backoff := newBackoff(10 * time.Second)
		for {
			// shallow clone opts to be used in the closure
			// without affecting the outer opts.
			newOpts := new(registryOptions)
			*newOpts = *opts

			newOpts.CheckRedirect = func(req *http.Request, via []*http.Request) error {
				if len(via) > 10 {
					return errMaxRedirectsExceeded
				}

				// if the hostname is the same, allow the redirect
				if req.URL.Hostname() == requestURL.Hostname() {
					return nil
				}

				// stop at the first redirect that is not
				// the same hostname as the original
				// request.
				return http.ErrUseLastResponse
			}

			resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, nil, nil, newOpts)
			if err != nil {
				slog.Warn("failed to get direct URL; backing off and retrying", "err", err)
				if err := backoff(ctx); err != nil {
					return nil, err
				}
				continue
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusTemporaryRedirect && resp.StatusCode != http.StatusOK {
				return nil, fmt.Errorf("unexpected status code %d", resp.StatusCode)
			}
			return resp.Location()
		}
	}()
	if err != nil {
		return err
	}

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numDownloadParts)
	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed.Load() == part.Size {
			continue
		}

		// In inline-hash mode a mid-stream error leaves the hasher with
		// partial state that can't be cleanly resumed, so we fail fast
		// and let the next pull invocation start over from scratch.
		maxTries := maxRetries
		if b.inlineHash != nil {
			maxTries = 1
		}

		g.Go(func() error {
			var err error
			for try := 0; try < maxTries; try++ {
				w := io.NewOffsetWriter(file, part.StartsAt())
				err = b.downloadChunk(inner, directURL, w, part)
				switch {
				case errors.Is(err, context.Canceled), errors.Is(err, syscall.ENOSPC):
					return err
				case errors.Is(err, errPartStalled):
					if b.inlineHash != nil {
						return err
					}
					try--
					continue
				case err != nil:
					if b.inlineHash != nil {
						return err
					}
					sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
					slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
					time.Sleep(sleep)
					continue
				default:
					return nil
				}
			}

			return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
		})
	}

	cleanupPartials := func() {
		_ = os.Remove(file.Name())
		for i := range b.Parts {
			_ = os.Remove(file.Name() + "-" + strconv.Itoa(i))
		}
	}

	if err := g.Wait(); err != nil {
		if b.inlineHash != nil {
			_ = file.Close()
			cleanupPartials()
		}
		return err
	}

	// Verify the inline hash before promoting the partial file to its
	// final blob path. This replaces the previous re-read-from-disk check
	// in verifyBlob() which is unreliable on kernels where the page cache
	// returns inconsistent bytes for large files.
	if b.inlineHash != nil {
		got := fmt.Sprintf("sha256:%x", b.inlineHash.Sum(nil))
		if got != b.Digest {
			_ = file.Close()
			cleanupPartials()
			return fmt.Errorf("%w: want %s, got %s (inline)", errDigestMismatch, b.Digest, got)
		}
	}

	// explicitly close the file so we can rename it
	if err := file.Close(); err != nil {
		return err
	}

	for i := range b.Parts {
		if err := os.Remove(file.Name() + "-" + strconv.Itoa(i)); err != nil {
			return err
		}
	}

	if err := os.Rename(file.Name(), b.Name); err != nil {
		return err
	}

	return nil
}

func (b *blobDownload) downloadChunk(ctx context.Context, requestURL *url.URL, w io.Writer, part *blobDownloadPart) error {
	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", part.StartsAt(), part.StopsAt()-1))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		// Reject anything that isn't a real body response. Catches cases
		// where an expired CDN signed URL returns 403, a gateway returns
		// an error page, or a proxy injects a redirect - any of which
		// would otherwise be silently written into the blob file.
		if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
			return fmt.Errorf("unexpected status code %d for range request", resp.StatusCode)
		}

		// Bytes flow: HTTP body -> inline hasher -> progress writer -> file.
		// Nested TeeReaders guarantee each byte is seen by every sink in
		// the same order, so the inline hash represents exactly what was
		// written to disk.
		src := io.Reader(resp.Body)
		if b.inlineHash != nil {
			src = io.TeeReader(src, b.inlineHash)
		}
		src = io.TeeReader(src, part)

		n, err := io.CopyN(w, src, part.Size-part.Completed.Load())
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, io.ErrUnexpectedEOF) {
			// rollback progress
			b.Completed.Add(-n)
			return err
		}

		part.Completed.Add(n)
		if err := b.writePart(part.Name(), part); err != nil {
			return err
		}

		// return nil or context.Canceled or UnexpectedEOF (resumable)
		return err
	})

	g.Go(func() error {
		ticker := time.NewTicker(time.Second)
		for {
			select {
			case <-ticker.C:
				if part.Completed.Load() >= part.Size {
					return nil
				}

				part.lastUpdatedMu.Lock()
				lastUpdated := part.lastUpdated
				part.lastUpdatedMu.Unlock()

				if !lastUpdated.IsZero() && time.Since(lastUpdated) > 30*time.Second {
					const msg = "%s part %d stalled; retrying. If this persists, press ctrl-c to exit, then 'ollama pull' to find a faster connection."
					slog.Info(fmt.Sprintf(msg, b.Digest[7:19], part.N))
					// reset last updated
					part.lastUpdatedMu.Lock()
					part.lastUpdated = time.Time{}
					part.lastUpdatedMu.Unlock()
					return errPartStalled
				}
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})

	return g.Wait()
}

func (b *blobDownload) newPart(offset, size int64) error {
	part := blobDownloadPart{blobDownload: b, Offset: offset, Size: size, N: len(b.Parts)}
	if err := b.writePart(part.Name(), &part); err != nil {
		return err
	}

	b.Parts = append(b.Parts, &part)
	return nil
}

func (b *blobDownload) readPart(partName string) (*blobDownloadPart, error) {
	var part blobDownloadPart
	partFile, err := os.Open(partName)
	if err != nil {
		return nil, err
	}
	defer partFile.Close()

	if err := json.NewDecoder(partFile).Decode(&part); err != nil {
		return nil, err
	}

	part.blobDownload = b
	return &part, nil
}

func (b *blobDownload) writePart(partName string, part *blobDownloadPart) error {
	partFile, err := os.OpenFile(partName, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer partFile.Close()

	return json.NewEncoder(partFile).Encode(part)
}

func (b *blobDownload) acquire() {
	b.references.Add(1)
}

func (b *blobDownload) release() {
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

func (b *blobDownload) Wait(ctx context.Context, fn func(api.ProgressResponse)) error {
	b.acquire()
	defer b.release()

	ticker := time.NewTicker(60 * time.Millisecond)
	for {
		select {
		case <-b.done:
			return b.err
		case <-ticker.C:
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("pulling %s", b.Digest[7:19]),
				Digest:    b.Digest,
				Total:     b.Total,
				Completed: b.Completed.Load(),
			})
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

type downloadOpts struct {
	n       model.Name
	digest  string
	regOpts *registryOptions
	fn      func(api.ProgressResponse)
}

// downloadBlob downloads a blob from the registry and stores it in the
// blobs directory. It returns cacheHit=true if the blob was already
// present on disk, and inlineVerified=true if the blob's sha256 was
// verified by an inline hasher during download (so the caller can skip
// the separate verifyBlob() pass).
func downloadBlob(ctx context.Context, opts downloadOpts) (cacheHit bool, inlineVerified bool, _ error) {
	if opts.digest == "" {
		return false, false, fmt.Errorf(("%s: %s"), opts.n.DisplayNamespaceModel(), "digest is empty")
	}

	fp, err := manifest.BlobsPath(opts.digest)
	if err != nil {
		return false, false, err
	}

	fi, err := os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return false, false, err
	default:
		opts.fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pulling %s", opts.digest[7:19]),
			Digest:    opts.digest,
			Total:     fi.Size(),
			Completed: fi.Size(),
		})

		return true, false, nil
	}

	data, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest})
	download := data.(*blobDownload)
	if !ok {
		requestURL := opts.n.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.n.DisplayNamespaceModel(), "blobs", opts.digest)
		if err := download.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			blobDownloadManager.Delete(opts.digest)
			return false, false, err
		}

		//nolint:contextcheck
		go download.Run(context.Background(), requestURL, opts.regOpts)
	}

	if err := download.Wait(ctx, opts.fn); err != nil {
		return false, false, err
	}
	return false, download.inlineHash != nil, nil
}
