package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/jmorganca/ollama/api"
)

var blobDownloadManager sync.Map

type blobDownload struct {
	Name   string
	Digest string

	Total     int64
	Completed atomic.Int64

	*os.File
	Parts []*blobDownloadPart

	done chan struct{}
	context.CancelFunc
	references atomic.Int32
}

type blobDownloadPart struct {
	N         int
	Offset    int64
	Size      int64
	Completed int64

	*blobDownload `json:"-"`
}

func (p *blobDownloadPart) Name() string {
	return strings.Join([]string{
		p.blobDownload.Name, "partial", strconv.Itoa(p.N),
	}, "-")
}

func (b *blobDownload) Prepare(ctx context.Context, requestURL *url.URL, opts *RegistryOptions) error {
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
		resp, err := makeRequest(ctx, "HEAD", requestURL, nil, nil, opts)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		if resp.StatusCode >= http.StatusBadRequest {
			body, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
		}

		b.Total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

		var offset int64
		var size int64 = 64 * 1024 * 1024

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

	log.Printf("downloading %s in %d part(s)", b.Digest[7:19], len(b.Parts))
	return nil
}

func (b *blobDownload) Run(ctx context.Context, requestURL *url.URL, opts *RegistryOptions) (err error) {
	defer blobDownloadManager.Delete(b.Digest)

	ctx, b.CancelFunc = context.WithCancel(ctx)

	b.File, err = os.OpenFile(b.Name+"-partial", os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer b.Close()

	b.Truncate(b.Total)

	b.done = make(chan struct{}, 1)
	defer close(b.done)

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(64)
	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed == part.Size {
			continue
		}

		i := i
		g.Go(func() error {
			for try := 0; try < maxRetries; try++ {
				err := b.downloadChunk(ctx, requestURL, i, opts)
				switch {
				case errors.Is(err, context.Canceled):
					return err
				case err != nil:
					log.Printf("%s part %d attempt %d failed: %v, retrying", b.Digest[7:19], i, try, err)
					continue
				default:
					return nil
				}
			}

			return errors.New("max retries exceeded")
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	if err := b.Close(); err != nil {
		return err
	}

	for i := range b.Parts {
		if err := os.Remove(b.File.Name() + "-" + strconv.Itoa(i)); err != nil {
			return err
		}
	}

	if err := os.Rename(b.File.Name(), b.Name); err != nil {
		return err
	}

	return nil
}

func (b *blobDownload) downloadChunk(ctx context.Context, requestURL *url.URL, i int, opts *RegistryOptions) error {
	part := b.Parts[i]

	offset := part.Offset + part.Completed
	w := io.NewOffsetWriter(b.File, offset)

	headers := make(http.Header)
	headers.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, part.Offset+part.Size-1))
	resp, err := makeRequest(ctx, "GET", requestURL, headers, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	n, err := io.Copy(w, io.TeeReader(resp.Body, b))
	if err != nil && !errors.Is(err, context.Canceled) {
		// rollback progress
		b.Completed.Add(-n)
		return err
	}

	part.Completed += n
	if err := b.writePart(part.Name(), part); err != nil {
		return err
	}

	// return nil or context.Canceled
	return err
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
	partFile, err := os.OpenFile(partName, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer partFile.Close()

	return json.NewEncoder(partFile).Encode(part)
}

func (b *blobDownload) Write(p []byte) (n int, err error) {
	n = len(p)
	b.Completed.Add(int64(n))
	return n, nil
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
			if b.Completed.Load() != b.Total {
				return io.ErrUnexpectedEOF
			}
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}

		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("downloading %s", b.Digest),
			Digest:    b.Digest,
			Total:     b.Total,
			Completed: b.Completed.Load(),
		})

		if b.Completed.Load() >= b.Total {
			<-b.done
			return nil
		}
	}
}

type downloadOpts struct {
	mp      ModelPath
	digest  string
	regOpts *RegistryOptions
	fn      func(api.ProgressResponse)
}

const maxRetries = 3

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
			Status:    fmt.Sprintf("downloading %s", opts.digest),
			Digest:    opts.digest,
			Total:     fi.Size(),
			Completed: fi.Size(),
		})

		return nil
	}

	value, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest})
	blobDownload := value.(*blobDownload)
	if !ok {
		requestURL := opts.mp.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.mp.GetNamespaceRepository(), "blobs", opts.digest)
		if err := blobDownload.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			return err
		}

		go blobDownload.Run(context.Background(), requestURL, opts.regOpts)
	}

	return blobDownload.Wait(ctx, opts.fn)
}
