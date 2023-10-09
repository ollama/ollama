package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"golang.org/x/sync/errgroup"
)

var blobUploadManager sync.Map

type blobUpload struct {
	*Layer

	Total     int64
	Completed atomic.Int64

	Parts []blobUploadPart

	nextURL chan *url.URL

	context.CancelFunc

	done       bool
	err        error
	references atomic.Int32
}

type blobUploadPart struct {
	// N is the part number
	N      int
	Offset int64
	Size   int64
}

const (
	numUploadParts          = 64
	minUploadPartSize int64 = 95 * 1000 * 1000
	maxUploadPartSize int64 = 1000 * 1000 * 1000
)

func (b *blobUpload) Prepare(ctx context.Context, requestURL *url.URL, opts *RegistryOptions) error {
	p, err := GetBlobsPath(b.Digest)
	if err != nil {
		return err
	}

	if b.From != "" {
		values := requestURL.Query()
		values.Add("mount", b.Digest)
		values.Add("from", b.From)
		requestURL.RawQuery = values.Encode()
	}

	resp, err := makeRequestWithRetry(ctx, "POST", requestURL, nil, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	fi, err := os.Stat(p)
	if err != nil {
		return err
	}

	b.Total = fi.Size()

	var size = b.Total / numUploadParts
	switch {
	case size < minUploadPartSize:
		size = minUploadPartSize
	case size > maxUploadPartSize:
		size = maxUploadPartSize
	}

	var offset int64
	for offset < fi.Size() {
		if offset+size > fi.Size() {
			size = fi.Size() - offset
		}

		b.Parts = append(b.Parts, blobUploadPart{N: len(b.Parts), Offset: offset, Size: size})
		offset += size
	}

	log.Printf("uploading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(size))

	requestURL, err = url.Parse(location)
	if err != nil {
		return err
	}

	b.nextURL = make(chan *url.URL, 1)
	b.nextURL <- requestURL
	return nil
}

func (b *blobUpload) Run(ctx context.Context, opts *RegistryOptions) {
	b.err = b.run(ctx, opts)
}

func (b *blobUpload) run(ctx context.Context, opts *RegistryOptions) error {
	defer blobUploadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	p, err := GetBlobsPath(b.Digest)
	if err != nil {
		return err
	}

	f, err := os.Open(p)
	if err != nil {
		return err
	}
	defer f.Close()

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numUploadParts)
	for i := range b.Parts {
		part := &b.Parts[i]
		requestURL := <-b.nextURL
		g.Go(func() error {
			for try := 0; try < maxRetries; try++ {
				r := io.NewSectionReader(f, part.Offset, part.Size)
				err := b.uploadChunk(inner, http.MethodPatch, requestURL, r, part, opts)
				switch {
				case errors.Is(err, context.Canceled):
					return err
				case errors.Is(err, errMaxRetriesExceeded):
					return err
				case err != nil:
					log.Printf("%s part %d attempt %d failed: %v, retrying", b.Digest[7:19], part.N, try, err)
					continue
				}

				return nil
			}

			return errMaxRetriesExceeded
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	requestURL := <-b.nextURL

	values := requestURL.Query()
	values.Add("digest", b.Digest)
	requestURL.RawQuery = values.Encode()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", "0")

	resp, err := makeRequest(ctx, "PUT", requestURL, headers, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	b.done = true
	return nil
}

func (b *blobUpload) uploadChunk(ctx context.Context, method string, requestURL *url.URL, rs io.ReadSeeker, part *blobUploadPart, opts *RegistryOptions) error {
	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", fmt.Sprintf("%d", part.Size))
	headers.Set("X-Redirect-Uploads", "1")

	if method == http.MethodPatch {
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", part.Offset, part.Offset+part.Size-1))
	}

	buw := blobUploadWriter{blobUpload: b}
	resp, err := makeRequest(ctx, method, requestURL, headers, io.TeeReader(rs, &buw), opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	nextURL, err := url.Parse(location)
	if err != nil {
		return err
	}

	switch {
	case resp.StatusCode == http.StatusTemporaryRedirect:
		b.nextURL <- nextURL

		redirectURL, err := resp.Location()
		if err != nil {
			return err
		}

		for try := 0; try < maxRetries; try++ {
			rs.Seek(0, io.SeekStart)
			b.Completed.Add(-buw.written)
			err := b.uploadChunk(ctx, http.MethodPut, redirectURL, rs, part, nil)
			switch {
			case errors.Is(err, context.Canceled):
				return err
			case errors.Is(err, errMaxRetriesExceeded):
				return err
			case err != nil:
				log.Printf("%s part %d attempt %d failed: %v, retrying", b.Digest[7:19], part.N, try, err)
				continue
			}

			return nil
		}

		return errMaxRetriesExceeded

	case resp.StatusCode == http.StatusUnauthorized:
		auth := resp.Header.Get("www-authenticate")
		authRedir := ParseAuthRedirectString(auth)
		token, err := getAuthToken(ctx, authRedir)
		if err != nil {
			return err
		}

		opts.Token = token
		fallthrough
	case resp.StatusCode >= http.StatusBadRequest:
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		rs.Seek(0, io.SeekStart)
		b.Completed.Add(-buw.written)
		return fmt.Errorf("http status %d %s: %s", resp.StatusCode, resp.Status, body)
	}

	if method == http.MethodPatch {
		b.nextURL <- nextURL
	}

	return nil
}

func (b *blobUpload) acquire() {
	b.references.Add(1)
}

func (b *blobUpload) release() {
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

func (b *blobUpload) Wait(ctx context.Context, fn func(api.ProgressResponse)) error {
	b.acquire()
	defer b.release()

	ticker := time.NewTicker(60 * time.Millisecond)
	for {
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}

		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("uploading %s", b.Digest),
			Digest:    b.Digest,
			Total:     b.Total,
			Completed: b.Completed.Load(),
		})

		if b.done || b.err != nil {
			return b.err
		}
	}
}

type blobUploadWriter struct {
	written int64
	*blobUpload
}

func (b *blobUploadWriter) Write(p []byte) (n int, err error) {
	n = len(p)
	b.written += int64(n)
	b.Completed.Add(int64(n))
	return n, nil
}

func uploadBlob(ctx context.Context, mp ModelPath, layer *Layer, opts *RegistryOptions, fn func(api.ProgressResponse)) error {
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "blobs", layer.Digest)

	resp, err := makeRequest(ctx, "HEAD", requestURL, nil, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusNotFound:
	case http.StatusOK:
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("uploading %s", layer.Digest),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: layer.Size,
		})

		return nil
	default:
		return fmt.Errorf("unexpected status code %d", resp.StatusCode)
	}

	data, ok := blobUploadManager.LoadOrStore(layer.Digest, &blobUpload{Layer: layer})
	upload := data.(*blobUpload)
	if !ok {
		requestURL := mp.BaseURL()
		requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "blobs/uploads/")
		if err := upload.Prepare(ctx, requestURL, opts); err != nil {
			blobUploadManager.Delete(layer.Digest)
			return err
		}

		go upload.Run(context.Background(), opts)
	}

	return upload.Wait(ctx, fn)
}
