package transfer

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

var (
	errStalled = errors.New("download stalled")
	errSlow    = errors.New("download too slow")
)

type downloader struct {
	client       *http.Client
	baseURL      string
	destDir      string
	repository   string // Repository path for blob URLs (e.g., "library/model")
	token        *string
	getToken     func(context.Context, AuthChallenge) (string, error)
	userAgent    string
	stallTimeout time.Duration
	progress     *progressTracker
	speeds       *speedTracker
	logger       *slog.Logger
}

func download(ctx context.Context, opts DownloadOptions) error {
	if len(opts.Blobs) == 0 {
		return nil
	}

	// Calculate total from all blobs (for accurate progress reporting on resume)
	var total int64
	for _, b := range opts.Blobs {
		total += b.Size
	}

	// Filter out already-downloaded blobs and track completed bytes
	var blobs []Blob
	var alreadyCompleted int64
	for _, b := range opts.Blobs {
		if fi, _ := os.Stat(filepath.Join(opts.DestDir, digestToPath(b.Digest))); fi != nil && fi.Size() == b.Size {
			if opts.Logger != nil {
				opts.Logger.Debug("blob already exists", "digest", b.Digest, "size", b.Size)
			}
			alreadyCompleted += b.Size
			continue
		}
		blobs = append(blobs, b)
	}
	if len(blobs) == 0 {
		return nil
	}

	token := opts.Token
	progress := newProgressTracker(total, opts.Progress)
	progress.add(alreadyCompleted) // Report already-downloaded bytes upfront

	d := &downloader{
		client:       cmp.Or(opts.Client, defaultClient),
		baseURL:      opts.BaseURL,
		destDir:      opts.DestDir,
		repository:   cmp.Or(opts.Repository, "library/_"),
		token:        &token,
		getToken:     opts.GetToken,
		userAgent:    cmp.Or(opts.UserAgent, defaultUserAgent),
		stallTimeout: cmp.Or(opts.StallTimeout, defaultStallTimeout),
		progress:     progress,
		speeds:       &speedTracker{},
		logger:       opts.Logger,
	}

	concurrency := cmp.Or(opts.Concurrency, DefaultDownloadConcurrency)
	sem := semaphore.NewWeighted(int64(concurrency))

	g, ctx := errgroup.WithContext(ctx)
	for _, blob := range blobs {
		g.Go(func() error {
			if err := sem.Acquire(ctx, 1); err != nil {
				return err
			}
			defer sem.Release(1)
			return d.download(ctx, blob)
		})
	}
	return g.Wait()
}

func (d *downloader) download(ctx context.Context, blob Blob) error {
	var lastErr error
	var slowRetries int
	attempt := 0

	for attempt < maxRetries {
		if attempt > 0 {
			if err := backoff(ctx, attempt, time.Second<<uint(attempt-1)); err != nil {
				return err
			}
		}

		start := time.Now()
		n, err := d.downloadOnce(ctx, blob)
		if err == nil {
			if s := time.Since(start).Seconds(); s > 0 {
				d.speeds.record(float64(blob.Size) / s)
			}
			return nil
		}

		d.progress.add(-n) // rollback

		switch {
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			return err
		case errors.Is(err, errStalled):
			// Don't count stall retries against limit
		case errors.Is(err, errSlow):
			if slowRetries++; slowRetries >= 3 {
				attempt++ // Only count after 3 slow retries
			}
		default:
			attempt++
		}
		lastErr = err
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
}

func (d *downloader) downloadOnce(ctx context.Context, blob Blob) (int64, error) {
	if d.logger != nil {
		d.logger.Debug("downloading blob", "digest", blob.Digest, "size", blob.Size)
	}

	baseURL, _ := url.Parse(d.baseURL)
	u, err := d.resolve(ctx, fmt.Sprintf("%s/v2/%s/blobs/%s", d.baseURL, d.repository, blob.Digest))
	if err != nil {
		return 0, err
	}

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	req.Header.Set("User-Agent", d.userAgent)
	// Add auth only for same-host (not CDN)
	if u.Host == baseURL.Host && *d.token != "" {
		req.Header.Set("Authorization", "Bearer "+*d.token)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("status %d", resp.StatusCode)
	}

	return d.save(ctx, blob, resp.Body)
}

func (d *downloader) save(ctx context.Context, blob Blob, r io.Reader) (int64, error) {
	dest := filepath.Join(d.destDir, digestToPath(blob.Digest))
	tmp := dest + ".tmp"
	os.MkdirAll(filepath.Dir(dest), 0o755)

	f, err := os.Create(tmp)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	setSparse(f)

	h := sha256.New()
	n, err := d.copy(ctx, f, r, h)
	if err != nil {
		os.Remove(tmp)
		return n, err
	}
	f.Close()

	if got := fmt.Sprintf("sha256:%x", h.Sum(nil)); got != blob.Digest {
		os.Remove(tmp)
		return n, fmt.Errorf("digest mismatch")
	}
	if n != blob.Size {
		os.Remove(tmp)
		return n, fmt.Errorf("size mismatch")
	}
	return n, os.Rename(tmp, dest)
}

func (d *downloader) copy(ctx context.Context, dst io.Writer, src io.Reader, h io.Writer) (int64, error) {
	var n int64
	var lastRead atomic.Int64
	lastRead.Store(time.Now().UnixNano())
	start := time.Now()

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	go func() {
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				if time.Since(time.Unix(0, lastRead.Load())) > d.stallTimeout {
					cancel(errStalled)
					return
				}
				if e := time.Since(start); e > 5*time.Second {
					if m := d.speeds.median(); m > 0 && float64(n)/e.Seconds() < m*0.1 {
						cancel(errSlow)
						return
					}
				}
			}
		}
	}()

	buf := make([]byte, 32*1024)
	for {
		if err := ctx.Err(); err != nil {
			if c := context.Cause(ctx); c != nil {
				return n, c
			}
			return n, err
		}

		nr, err := src.Read(buf)
		if nr > 0 {
			lastRead.Store(time.Now().UnixNano())
			dst.Write(buf[:nr])
			h.Write(buf[:nr])
			d.progress.add(int64(nr))
			n += int64(nr)
		}
		if err == io.EOF {
			return n, nil
		}
		if err != nil {
			return n, err
		}
	}
}

func (d *downloader) resolve(ctx context.Context, rawURL string) (*url.URL, error) {
	u, _ := url.Parse(rawURL)
	for range 10 {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		req.Header.Set("User-Agent", d.userAgent)
		if *d.token != "" {
			req.Header.Set("Authorization", "Bearer "+*d.token)
		}

		resp, err := d.client.Do(req)
		if err != nil {
			return nil, err
		}
		resp.Body.Close()

		switch resp.StatusCode {
		case http.StatusOK:
			return u, nil
		case http.StatusUnauthorized:
			if d.getToken == nil {
				return nil, fmt.Errorf("unauthorized")
			}
			ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
			if *d.token, err = d.getToken(ctx, ch); err != nil {
				return nil, err
			}
		case http.StatusTemporaryRedirect, http.StatusFound, http.StatusMovedPermanently:
			loc, _ := resp.Location()
			if loc.Host != u.Host {
				return loc, nil
			}
			u = loc
		default:
			return nil, fmt.Errorf("status %d", resp.StatusCode)
		}
	}
	return nil, fmt.Errorf("too many redirects")
}

type speedTracker struct {
	mu     sync.Mutex
	speeds []float64
}

func (s *speedTracker) record(v float64) {
	s.mu.Lock()
	s.speeds = append(s.speeds, v)
	if len(s.speeds) > 30 {
		s.speeds = s.speeds[1:]
	}
	s.mu.Unlock()
}

func (s *speedTracker) median() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.speeds) < 5 {
		return 0
	}
	sorted := make([]float64, len(s.speeds))
	copy(sorted, s.speeds)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}

const defaultStallTimeout = 10 * time.Second
