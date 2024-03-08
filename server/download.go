package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
)

const maxRetries = 6

var errMaxRetriesExceeded = errors.New("max retries exceeded")
var errPartStalled = errors.New("part stalled")

var blobDownloadManager sync.Map

type blobDownload struct {
	Name   string
	Digest string

	Total     int64
	Completed atomic.Int64

	Parts []*blobDownloadPart

	context.CancelFunc

	done       bool
	err        error
	references atomic.Int32
}

type blobDownloadPart struct {
	N           int
	Offset      int64
	Size        int64
	Completed   int64
	lastUpdated time.Time

	*blobDownload `json:"-"`
}

const (
	numDownloadParts          = 64
	minDownloadPartSize int64 = 100 * format.MegaByte
	maxDownloadPartSize int64 = 1000 * format.MegaByte
)

func (p *blobDownloadPart) Name() string {
	return strings.Join([]string{
		p.blobDownload.Name, "partial", strconv.Itoa(p.N),
	}, "-")
}

func (p *blobDownloadPart) StartsAt() int64 {
	return p.Offset + p.Completed
}

func (p *blobDownloadPart) StopsAt() int64 {
	return p.Offset + p.Size
}

func (p *blobDownloadPart) Write(b []byte) (n int, err error) {
	n = len(b)
	p.blobDownload.Completed.Add(int64(n))
	p.lastUpdated = time.Now()
	return n, nil
}

func (b *blobDownload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	partFilePaths, err := filepath.Glob(b.Name + "-partial-*")
	if err != nil {
		return err
	}

	for _, partFilePath := range partFilePaths {
		part, err := b.readPart(partFilePath)
		if err != nil {
			return err
		}

		b.Total += part.Size
		b.Completed.Add(part.Completed)
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

	slog.Info(fmt.Sprintf("downloading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size)))
	return nil
}

func (b *blobDownload) Run(ctx context.Context, requestURL *url.URL, opts *registryOptions) {
	defer blobDownloadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	file, err := os.OpenFile(b.Name+"-partial", os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		b.err = err
		return
	}
	defer file.Close()

	_ = file.Truncate(b.Total)

	limit := int64(runtime.NumCPU())
	g, inner := NewLimitGroup(ctx, numDownloadParts, limit)
	go watchDelta(inner, g, &b.Completed, limit)

	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed == part.Size {
			continue
		}

		g.Go(inner, func() error {
			var err error
			for try := 0; try < maxRetries; try++ {
				w := io.NewOffsetWriter(file, part.StartsAt())
				err = b.downloadChunk(inner, requestURL, w, part, opts)
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
		b.err = err
		return
	}

	// explicitly close the file so we can rename it
	if err := file.Close(); err != nil {
		b.err = err
		return
	}

	for i := range b.Parts {
		if err := os.Remove(file.Name() + "-" + strconv.Itoa(i)); err != nil {
			b.err = err
			return
		}
	}

	if err := os.Rename(file.Name(), b.Name); err != nil {
		b.err = err
		return
	}

	b.done = true
}

func (b *blobDownload) downloadChunk(ctx context.Context, requestURL *url.URL, w io.Writer, part *blobDownloadPart, opts *registryOptions) error {
	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		headers := make(http.Header)
		headers.Set("Range", fmt.Sprintf("bytes=%d-%d", part.StartsAt(), part.StopsAt()-1))
		resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, opts)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		n, err := io.Copy(w, io.TeeReader(resp.Body, part))
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, io.ErrUnexpectedEOF) {
			// rollback progress
			b.Completed.Add(-n)
			return err
		}

		part.Completed += n
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
				if part.Completed >= part.Size {
					return nil
				}

				if !part.lastUpdated.IsZero() && time.Since(part.lastUpdated) > 5*time.Second {
					slog.Info(fmt.Sprintf("%s part %d stalled; retrying", b.Digest[7:19], part.N))
					// reset last updated
					part.lastUpdated = time.Time{}
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
		case <-ticker.C:
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("pulling %s", b.Digest[7:19]),
				Digest:    b.Digest,
				Total:     b.Total,
				Completed: b.Completed.Load(),
			})

			if b.done || b.err != nil {
				return b.err
			}
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
func downloadBlob(ctx context.Context, opts downloadOpts) error {
	fp, err := GetBlobsPath(opts.digest)
	if err != nil {
		return err
	}

	fi, err := os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return err
	default:
		opts.fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pulling %s", opts.digest[7:19]),
			Digest:    opts.digest,
			Total:     fi.Size(),
			Completed: fi.Size(),
		})

		return nil
	}

	data, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest})
	download := data.(*blobDownload)
	if !ok {
		requestURL := opts.mp.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.mp.GetNamespaceRepository(), "blobs", opts.digest)
		if err := download.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			blobDownloadManager.Delete(opts.digest)
			return err
		}

		// nolint: contextcheck
		go download.Run(context.Background(), requestURL, opts.regOpts)
	}

	return download.Wait(ctx, opts.fn)
}

type LimitGroup struct {
	*errgroup.Group
	*semaphore.Weighted
	size, limit int64
}

func NewLimitGroup(ctx context.Context, size, limit int64) (*LimitGroup, context.Context) {
	g, ctx := errgroup.WithContext(ctx)
	return &LimitGroup{
		Group:    g,
		Weighted: semaphore.NewWeighted(size),
		size:     size,
		limit:    limit,
	}, ctx
}

func (g *LimitGroup) Go(ctx context.Context, fn func() error) {
	var weight int64 = 1
	if g.limit > 0 {
		weight = g.size / g.limit
	}

	_ = g.Acquire(ctx, weight)
	if ctx.Err() != nil {
		return
	}

	g.Group.Go(func() error {
		defer g.Release(weight)
		return fn()
	})
}

func (g *LimitGroup) SetLimit(limit int64) {
	if limit > g.limit {
		g.limit = limit
	}
}

func watchDelta(ctx context.Context, g *LimitGroup, c *atomic.Int64, limit int64) {
	var maxDelta float64
	var buckets []int64

	// 3s ramp up period
	nextUpdate := time.Now().Add(3 * time.Second)

	ticker := time.NewTicker(time.Second)
	for {
		select {
		case <-ticker.C:
			buckets = append(buckets, c.Load())
			if len(buckets) < 2 {
				continue
			} else if len(buckets) > 10 {
				buckets = buckets[1:]
			}

			delta := float64((buckets[len(buckets)-1] - buckets[0])) / float64(len(buckets))
			slog.Debug("", "limit", limit, "delta", format.HumanBytes(int64(delta)), "max_delta", format.HumanBytes(int64(maxDelta)))

			if time.Now().Before(nextUpdate) {
				// quiet period; do not update ccy if recently updated
				continue
			} else if maxDelta > 0 {
				x := delta / maxDelta
				if x < 1 {
					continue
				}

				limit += min(int64(x), int64(runtime.NumCPU()))
				slog.Debug("setting", "limit", limit)
				g.SetLimit(limit)
			}

			// 3s cooldown period
			nextUpdate = time.Now().Add(2 * time.Second)
			maxDelta = delta

		case <-ctx.Done():
			return
		}
	}
}
