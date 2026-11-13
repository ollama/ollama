package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	"github.com/ollama/ollama/transfer"
	"github.com/ollama/ollama/types/model"
)

const maxRetries = 6 // used by upload.go; pulls use envconfig.MaxPullRetries

// rateLimiter is a token bucket shared by every stream of one download, so the
// configured rate is the total on the wire rather than a per-stream figure. A
// bucket per stream multiplied the limit by OLLAMA_MAX_PULL_STREAMS, which is
// not what anyone asking for "500k" means.
//
// The bucket starts empty on purpose. Starting it full handed out one free
// second per stream, which made the measured average (t+1)/t times the limit —
// still 9% high after ten seconds.
type rateLimiter struct {
	mu          sync.Mutex
	bytesPerSec int64
	budget      int64
	lastRefill  time.Time
}

func newRateLimiter(bytesPerSec int64) *rateLimiter {
	return &rateLimiter{bytesPerSec: bytesPerSec, lastRefill: time.Now()}
}

// refillLocked credits the time since the last call. The remainder below one
// byte stays on the clock instead of being dropped, so frequent small reads do
// not lose budget to truncation.
func (l *rateLimiter) refillLocked() {
	now := time.Now()
	elapsed := now.Sub(l.lastRefill).Seconds()
	add := int64(elapsed * float64(l.bytesPerSec))
	if add <= 0 {
		return
	}
	l.lastRefill = now
	l.budget += add
	if l.budget > l.bytesPerSec {
		l.budget = l.bytesPerSec
	}
}

// reserve blocks until the bucket holds at least one byte and returns how many
// bytes the caller may read, never more than want.
func (l *rateLimiter) reserve(want int64) int64 {
	if want < 1 {
		return want
	}
	for {
		l.mu.Lock()
		l.refillLocked()
		if l.budget > 0 {
			take := min(l.budget, want)
			l.budget -= take
			l.mu.Unlock()
			return take
		}
		l.mu.Unlock()
		// Warten, bis wieder etwas im Eimer ist. Eine Zehntelsekunde ist grob
		// genug, um nicht zu pollen, und fein genug, dass die Rate stimmt.
		time.Sleep(100 * time.Millisecond)
	}
}

// refund returns budget that reserve granted but the reader did not use.
func (l *rateLimiter) refund(n int64) {
	if n <= 0 {
		return
	}
	l.mu.Lock()
	l.budget += n
	if l.budget > l.bytesPerSec {
		l.budget = l.bytesPerSec
	}
	l.mu.Unlock()
}

// throttledTransport applies a rateLimiter to every response body it returns.
//
// This is how the tensor-layer pull path gets the same total rate as the legacy
// one. transfer.DownloadOptions carries no rate hook, but it does accept an
// *http.Client, so the limit can be installed underneath without touching the
// transfer package at all. One limiter per pull means the rate is shared across
// the blobs transferring in parallel, exactly as on the legacy path.
type throttledTransport struct {
	base http.RoundTripper
	l    *rateLimiter
}

func (t *throttledTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		return resp, err
	}
	resp.Body = struct {
		io.Reader
		io.Closer
	}{Reader: newThrottledReader(resp.Body, t.l), Closer: resp.Body}
	return resp, nil
}

// throttledClient returns a client that reads at no more than the limiter
// allows. A nil limiter returns base unchanged, so the no-limit path keeps
// using whatever client it had.
func throttledClient(base *http.Client, l *rateLimiter) *http.Client {
	if l == nil {
		return base
	}
	c := http.Client{}
	if base != nil {
		c = *base
	}
	rt := c.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	c.Transport = &throttledTransport{base: rt, l: l}
	return &c
}

// throttledReader draws from a shared rateLimiter before each read.
type throttledReader struct {
	r io.Reader
	l *rateLimiter
}

func newThrottledReader(r io.Reader, l *rateLimiter) *throttledReader {
	return &throttledReader{r: r, l: l}
}

func (t *throttledReader) Read(p []byte) (int, error) {
	allowed := t.l.reserve(int64(len(p)))
	n, err := t.r.Read(p[:allowed])
	t.l.refund(allowed - int64(n))
	return n, err
}

var (
	errMaxRetriesExceeded   = errors.New("max retries exceeded")
	errPartStalled          = errors.New("part stalled")
	errMaxRedirectsExceeded = errors.New("maximum redirects exceeded (10) for directURL")
)

var blobDownloadManager sync.Map

type blobDownload struct {
	Name   string
	Digest string

	// Limits are the per-request limits; a zero field falls back to the
	// environment. Read in Run, before the parts start.
	Limits api.PullLimits

	// limiter is shared by all parts of this blob, so Limits.Rate is the total
	// rate rather than a per-stream one. Built once in Run.
	limiter *rateLimiter

	Total     int64
	Completed atomic.Int64

	Parts []*blobDownloadPart

	context.CancelFunc

	done       chan struct{}
	err        error
	references atomic.Int32
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
	minDownloadPartSize int64 = 100 * format.MegaByte
	maxDownloadPartSize int64 = 1000 * format.MegaByte
)

var downloadStallTimeout = 30 * time.Second

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

// effectiveLimits resolves the per-request limits against the environment.
// It lives here alone so the rule cannot drift between its call sites: the
// request wins, a zero field falls back to the environment, and streams never
// drop below one. Taking a maximum of the two instead would let the
// environment override a deliberately smaller request.
func (b *blobDownload) effectiveLimits() (rate int64, streams int) {
	rate = b.Limits.Rate
	if rate == 0 {
		rate = int64(envconfig.MaxPullRate())
	}
	streams = b.Limits.Streams
	if streams == 0 {
		streams = int(envconfig.MaxPullStreams())
	}
	return rate, max(streams, 1)
}

func (b *blobDownload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
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

		_, streams := b.effectiveLimits()
		size := b.Total / int64(streams)
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

			newOpts.CheckRedirect = downloadDiscoveryRedirect(requestURL)

			resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, nil, nil, newOpts)
			if err != nil {
				slog.Warn("failed to get direct URL; backing off and retrying", "err", err)
				if err := backoff(ctx); err != nil {
					return nil, err
				}
				continue
			}
			defer resp.Body.Close()
			switch resp.StatusCode {
			case http.StatusOK, http.StatusMovedPermanently, http.StatusFound,
				http.StatusSeeOther, http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
			default:
				return nil, fmt.Errorf("unexpected status code %d", resp.StatusCode)
			}
			loc, err := resp.Location()
			if err != nil {
				return nil, err
			}
			if err := transfer.ValidateRedirectTarget(ctx, loc, requestURL.String(), opts.Insecure); err != nil {
				return nil, err
			}
			return loc, nil
		}
	}()
	if err != nil {
		return err
	}

	client := transfer.NewRedirectClient(requestURL.String(), opts.Insecure)
	defer client.CloseIdleConnections()

	// One limiter for the whole blob, not one per part: a bucket per stream
	// would multiply the configured rate by the number of streams.
	bw, streams := b.effectiveLimits()
	if bw > 0 {
		b.limiter = newRateLimiter(bw)
		slog.Debug("pull rate limit active", "bytes_per_sec", bw,
			"streams", streams, "digest", b.Digest[:min(16, len(b.Digest))])
	}

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(streams)
	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed.Load() == part.Size {
			continue
		}

		g.Go(func() error {
			var err error
			for try := 0; try < int(envconfig.MaxPullRetries()); try++ {
				w := io.NewOffsetWriter(file, part.StartsAt())
				err = b.downloadChunk(inner, client, directURL, w, part)
				switch {
				case errors.Is(err, context.Canceled), errors.Is(err, syscall.ENOSPC):
					// return immediately if the context is canceled or the device is out of space
					return err
				case errors.Is(err, errPartStalled):
					try--
					continue
				case err != nil:
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

	if err := g.Wait(); err != nil {
		return err
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

// downloadDiscoveryRedirect follows only redirects within the original
// registry origin. A new scheme, host, or port must first pass the redirect
// target policy before the blob downloader makes another request.
func downloadDiscoveryRedirect(base *url.URL) func(*http.Request, []*http.Request) error {
	return func(req *http.Request, via []*http.Request) error {
		if len(via) > 10 {
			return errMaxRedirectsExceeded
		}
		if err := transfer.ValidateRedirectScheme(req.URL, base.String()); err != nil {
			return err
		}
		if sameRegistryOrigin(req.URL, base) {
			return nil
		}
		return http.ErrUseLastResponse
	}
}

func sameRegistryOrigin(a, b *url.URL) bool {
	if a == nil || b == nil || a.Scheme != b.Scheme || !strings.EqualFold(a.Hostname(), b.Hostname()) {
		return false
	}
	port := func(u *url.URL) string {
		if p := u.Port(); p != "" {
			return p
		}
		if u.Scheme == "https" {
			return "443"
		}
		return "80"
	}
	return port(a) == port(b)
}

func (b *blobDownload) downloadChunk(ctx context.Context, client *http.Client, requestURL *url.URL, w io.Writer, part *blobDownloadPart) error {
	g, ctx := errgroup.WithContext(ctx)
	attemptStarted := time.Now()
	transferDone := make(chan struct{})
	g.Go(func() error {
		defer close(transferDone)

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", part.StartsAt(), part.StopsAt()-1))
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		body := io.Reader(resp.Body)
		if b.limiter != nil {
			body = newThrottledReader(body, b.limiter)
		}

		n, err := io.CopyN(w, io.TeeReader(body, part), part.Size-part.Completed.Load())
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
		ticker := time.NewTicker(min(time.Second, downloadStallTimeout/2))
		defer ticker.Stop()
		for {
			select {
			case <-transferDone:
				return nil
			case <-ticker.C:
				part.lastUpdatedMu.Lock()
				lastUpdated := part.lastUpdated
				part.lastUpdatedMu.Unlock()
				if lastUpdated.Before(attemptStarted) {
					lastUpdated = attemptStarted
				}

				if time.Since(lastUpdated) > downloadStallTimeout {
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
	if b.references.Add(-1) == 0 && b.CancelFunc != nil {
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
	limits  api.PullLimits
}

// downloadBlob downloads a blob from the registry and stores it in the blobs directory
func downloadBlob(ctx context.Context, opts downloadOpts) (cacheHit bool, _ error) {
	if opts.digest == "" {
		return false, fmt.Errorf("%s: %s", opts.n.DisplayNamespaceModel(), "digest is empty")
	}

	fp, err := manifest.BlobsPath(opts.digest)
	if err != nil {
		return false, err
	}

	fi, err := os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return false, err
	default:
		opts.fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pulling %s", opts.digest[7:19]),
			Digest:    opts.digest,
			Total:     fi.Size(),
			Completed: fi.Size(),
		})

		return true, nil
	}

	data, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest, Limits: opts.limits})
	download := data.(*blobDownload)
	if !ok {
		requestURL := opts.n.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.n.DisplayNamespaceModel(), "blobs", opts.digest)
		if err := download.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			blobDownloadManager.Delete(opts.digest)
			return false, err
		}

		//nolint:contextcheck
		go download.Run(context.Background(), requestURL, opts.regOpts)
	}

	return false, download.Wait(ctx, opts.fn)
}
