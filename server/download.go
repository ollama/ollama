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
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

const maxRetries = 6

var (
	errMaxRetriesExceeded   = errors.New("max retries exceeded")
	errPartStalled          = errors.New("part stalled")
	errPartSlow             = errors.New("part slow, racing")
	errMaxRedirectsExceeded = errors.New("maximum redirects exceeded (10) for directURL")
)

// speedTracker tracks download speeds and computes rolling median.
type speedTracker struct {
	mu     sync.Mutex
	speeds []float64 // bytes per second
}

func (s *speedTracker) Record(bytesPerSec float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.speeds = append(s.speeds, bytesPerSec)
	// Keep last 100 samples
	if len(s.speeds) > 100 {
		s.speeds = s.speeds[1:]
	}
}

func (s *speedTracker) Median() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.speeds) < 3 {
		return 0 // not enough data
	}

	sorted := slices.Clone(s.speeds)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}

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

var (
	downloadPartSize    = int64(envInt("OLLAMA_DOWNLOAD_PART_SIZE", 64)) * format.MegaByte
	downloadConcurrency = envInt("OLLAMA_DOWNLOAD_CONCURRENCY", 48)
)

func envInt(key string, defaultVal int) int {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			return v
		}
	}
	return defaultVal
}

// streamHasher reads a file sequentially and hashes it as chunks complete.
// Memory usage: ~64KB (just the read buffer), regardless of file size or concurrency.
// Works by reading from OS page cache - data just written is still in RAM.
type streamHasher struct {
	file   *os.File
	hasher hash.Hash
	parts  []*blobDownloadPart
	total  int64 // total bytes to hash
	hashed atomic.Int64

	mu        sync.Mutex
	cond      *sync.Cond
	completed []bool
	done      bool
	err       error
}

func newStreamHasher(file *os.File, parts []*blobDownloadPart, total int64) *streamHasher {
	h := &streamHasher{
		file:      file,
		hasher:    sha256.New(),
		parts:     parts,
		total:     total,
		completed: make([]bool, len(parts)),
	}
	h.cond = sync.NewCond(&h.mu)
	return h
}

// MarkComplete signals that a part has been written to disk.
func (h *streamHasher) MarkComplete(partIndex int) {
	h.mu.Lock()
	h.completed[partIndex] = true
	h.cond.Broadcast()
	h.mu.Unlock()
}

// Run reads and hashes the file sequentially
func (h *streamHasher) Run() {
	buf := make([]byte, 64*1024) // 64KB read buffer
	var offset int64

	for i, part := range h.parts {
		// Wait for this part to be written
		h.mu.Lock()
		for !h.completed[i] && !h.done {
			h.cond.Wait()
		}
		if h.done {
			h.mu.Unlock()
			return
		}
		h.mu.Unlock()

		// Read and hash this part (from page cache)
		remaining := part.Size
		for remaining > 0 {
			n := int64(len(buf))
			if n > remaining {
				n = remaining
			}
			nr, err := h.file.ReadAt(buf[:n], offset)
			if err != nil && err != io.EOF {
				h.mu.Lock()
				h.err = err
				h.mu.Unlock()
				return
			}
			h.hasher.Write(buf[:nr])
			offset += int64(nr)
			remaining -= int64(nr)
			h.hashed.Store(offset)
		}
	}
}

// Stop signals the hasher to exit early.
func (h *streamHasher) Stop() {
	h.mu.Lock()
	h.done = true
	h.cond.Broadcast()
	h.mu.Unlock()
}

// Hashed returns bytes hashed so far.
func (h *streamHasher) Hashed() int64 {
	return h.hashed.Load()
}

// Digest returns the computed hash.
func (h *streamHasher) Digest() string {
	return fmt.Sprintf("sha256:%x", h.hasher.Sum(nil))
}

// Err returns any error from hashing.
func (h *streamHasher) Err() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

func (p *blobDownloadPart) Name() string {
	return strings.Join([]string{
		p.blobDownload.Name, "partial", strconv.Itoa(p.N),
	}, "-")
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

		size := downloadPartSize
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

	// Download chunks to disk, hash sequentially
	// The hasher follows behind the downloaders, reading recently-written
	// data from OS page cache (RAM) rather than disk.
	sh := newStreamHasher(file, b.Parts, b.Total)
	tracker := &speedTracker{}

	hashDone := make(chan struct{})
	go func() {
		sh.Run()
		close(hashDone)
	}()

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(downloadConcurrency)
	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed.Load() == part.Size {
			sh.MarkComplete(part.N)
			continue
		}

		g.Go(func() error {
			var err error
			var slowRetries int
			for try := 0; try < maxRetries; try++ {
				// After 3 slow retries, stop checking slowness and let it complete
				skipSlowCheck := slowRetries >= 3
				err = b.downloadChunkToDisk(inner, directURL, file, part, tracker, skipSlowCheck)
				switch {
				case errors.Is(err, context.Canceled), errors.Is(err, syscall.ENOSPC):
					return err
				case errors.Is(err, errPartStalled):
					try--
					continue
				case errors.Is(err, errPartSlow):
					// Kill slow request, retry immediately (stays within concurrency limit)
					slowRetries++
					try--
					continue
				case err != nil:
					sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
					slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
					time.Sleep(sleep)
					continue
				default:
					sh.MarkComplete(part.N)
					return nil
				}
			}
			return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
		})
	}

	if err := g.Wait(); err != nil {
		sh.Stop()
		return err
	}

	// Wait for hasher to finish
	<-hashDone
	if err := sh.Err(); err != nil {
		return err
	}

	// Verify hash
	if computed := sh.Digest(); computed != b.Digest {
		return fmt.Errorf("digest mismatch: got %s, want %s", computed, b.Digest)
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

// downloadChunkToDisk streams a part directly to disk at its offset.
// If skipSlowCheck is true, don't flag slow parts (used after repeated slow retries).
func (b *blobDownload) downloadChunkToDisk(ctx context.Context, requestURL *url.URL, file *os.File, part *blobDownloadPart, tracker *speedTracker, skipSlowCheck bool) error {
	g, ctx := errgroup.WithContext(ctx)
	startTime := time.Now()
	var bytesAtLastCheck atomic.Int64

	g.Go(func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", part.Offset, part.Offset+part.Size-1))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		w := io.NewOffsetWriter(file, part.Offset)
		buf := make([]byte, 32*1024)

		var written int64
		for written < part.Size {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				if _, werr := w.Write(buf[:n]); werr != nil {
					return werr
				}
				written += int64(n)
				b.Completed.Add(int64(n))
				bytesAtLastCheck.Store(written)

				part.lastUpdatedMu.Lock()
				part.lastUpdated = time.Now()
				part.lastUpdatedMu.Unlock()
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				b.Completed.Add(-written)
				return err
			}
		}

		// Record speed for this part
		elapsed := time.Since(startTime).Seconds()
		if elapsed > 0 {
			tracker.Record(float64(part.Size) / elapsed)
		}

		part.Completed.Store(part.Size)
		return b.writePart(part.Name(), part)
	})

	g.Go(func() error {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		var lastBytes int64
		checksWithoutProgress := 0

		for {
			select {
			case <-ticker.C:
				if part.Completed.Load() >= part.Size {
					return nil
				}

				currentBytes := bytesAtLastCheck.Load()

				// Check for stall (no progress for 10 seconds)
				if currentBytes == lastBytes {
					checksWithoutProgress++
					if checksWithoutProgress >= 10 {
						slog.Info(fmt.Sprintf("%s part %d stalled; retrying", b.Digest[7:19], part.N))
						return errPartStalled
					}
				} else {
					checksWithoutProgress = 0
				}
				lastBytes = currentBytes

				// Check for slow speed after 5+ seconds (only for multi-part downloads)
				// Skip if we've already retried for slowness too many times
				elapsed := time.Since(startTime).Seconds()
				if !skipSlowCheck && elapsed >= 5 && currentBytes > 0 && len(b.Parts) > 1 {
					currentSpeed := float64(currentBytes) / elapsed
					median := tracker.Median()

					// If we're below 10% of median speed, flag as slow
					if median > 0 && currentSpeed < median*0.1 {
						slog.Info(fmt.Sprintf("%s part %d slow (%.0f KB/s vs median %.0f KB/s); retrying",
							b.Digest[7:19], part.N, currentSpeed/1024, median/1024))
						return errPartSlow
					}
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
	mp      ModelPath
	digest  string
	regOpts *registryOptions
	fn      func(api.ProgressResponse)
}

// downloadBlob downloads a blob from the registry and stores it in the blobs directory
func downloadBlob(ctx context.Context, opts downloadOpts) (cacheHit bool, _ error) {
	if opts.digest == "" {
		return false, fmt.Errorf(("%s: %s"), opts.mp.GetNamespaceRepository(), "digest is empty")
	}

	fp, err := GetBlobsPath(opts.digest)
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

	data, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest})
	download := data.(*blobDownload)
	if !ok {
		requestURL := opts.mp.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.mp.GetNamespaceRepository(), "blobs", opts.digest)
		if err := download.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			blobDownloadManager.Delete(opts.digest)
			return false, err
		}

		//nolint:contextcheck
		go download.Run(context.Background(), requestURL, opts.regOpts)
	}

	return false, download.Wait(ctx, opts.fn)
}
