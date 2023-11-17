package server

import (
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"hash"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
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
		values.Add("from", ParseModelPath(b.From).GetNamespaceRepository())
		requestURL.RawQuery = values.Encode()
	}

	resp, err := makeRequestWithRetry(ctx, http.MethodPost, requestURL, nil, nil, opts)
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

	// http.StatusCreated indicates a blob has been mounted
	// ref: https://distribution.github.io/distribution/spec/api/#cross-repository-blob-mount
	if resp.StatusCode == http.StatusCreated {
		b.Completed.Store(b.Total)
		b.done = true
		return nil
	}

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

		// set part.N to the current number of parts
		b.Parts = append(b.Parts, blobUploadPart{blobUpload: b, N: len(b.Parts), Offset: offset, Size: size})
		offset += size
	}

	log.Printf("uploading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size))

	requestURL, err = url.Parse(location)
	if err != nil {
		return err
	}

	b.nextURL = make(chan *url.URL, 1)
	b.nextURL <- requestURL
	return nil
}

// Run uploads blob parts to the upstream. If the upstream supports redirection, parts will be uploaded
// in parallel as defined by Prepare. Otherwise, parts will be uploaded serially. Run sets b.err on error.
func (b *blobUpload) Run(ctx context.Context, opts *RegistryOptions) {
	defer blobUploadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	p, err := GetBlobsPath(b.Digest)
	if err != nil {
		b.err = err
		return
	}

	f, err := os.Open(p)
	if err != nil {
		b.err = err
		return
	}
	defer f.Close()

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numUploadParts)
	for i := range b.Parts {
		part := &b.Parts[i]
		select {
		case <-inner.Done():
		case requestURL := <-b.nextURL:
			g.Go(func() error {
				var err error
				for try := 0; try < maxRetries; try++ {
					part.ReadSeeker = io.NewSectionReader(f, part.Offset, part.Size)
					err = b.uploadChunk(inner, http.MethodPatch, requestURL, part, opts)
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

				return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
			})
		}
	}

	if err := g.Wait(); err != nil {
		b.err = err
		return
	}

	requestURL := <-b.nextURL

	var sb strings.Builder
	for _, part := range b.Parts {
		sb.Write(part.Sum(nil))
	}

	md5sum := md5.Sum([]byte(sb.String()))

	values := requestURL.Query()
	values.Add("digest", b.Digest)
	values.Add("etag", fmt.Sprintf("%x-%d", md5sum, len(b.Parts)))
	requestURL.RawQuery = values.Encode()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", "0")

	resp, err := makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, nil, opts)
	if err != nil {
		b.err = err
		return
	}
	defer resp.Body.Close()

	b.done = true
}

func (b *blobUpload) uploadChunk(ctx context.Context, method string, requestURL *url.URL, part *blobUploadPart, opts *RegistryOptions) error {
	part.Reset()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", fmt.Sprintf("%d", part.Size))
	headers.Set("X-Redirect-Uploads", "1")

	if method == http.MethodPatch {
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", part.Offset, part.Offset+part.Size-1))
	}

	resp, err := makeRequest(ctx, method, requestURL, headers, io.TeeReader(part.ReadSeeker, io.MultiWriter(part, part.Hash)), opts)
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
			err = b.uploadChunk(ctx, http.MethodPut, redirectURL, part, nil)
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

		return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)

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

		return fmt.Errorf("http status %s: %s", resp.Status, body)
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
			Status:    fmt.Sprintf("uploading %s", b.Digest[7:19]),
			Digest:    b.Digest,
			Total:     b.Total,
			Completed: b.Completed.Load(),
		})

		if b.done || b.err != nil {
			return b.err
		}
	}
}

type blobUploadPart struct {
	// N is the part number
	N      int
	Offset int64
	Size   int64
	hash.Hash

	written int64

	io.ReadSeeker
	*blobUpload
}

func (p *blobUploadPart) Write(b []byte) (n int, err error) {
	n = len(b)
	p.written += int64(n)
	p.Completed.Add(int64(n))
	return n, nil
}

func (p *blobUploadPart) Reset() {
	p.Seek(0, io.SeekStart)
	p.Completed.Add(-int64(p.written))
	p.written = 0
	p.Hash = md5.New()
}

func uploadBlob(ctx context.Context, mp ModelPath, layer *Layer, opts *RegistryOptions, fn func(api.ProgressResponse)) error {
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "blobs", layer.Digest)

	resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return err
	default:
		defer resp.Body.Close()
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("uploading %s", layer.Digest[7:19]),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: layer.Size,
		})

		return nil
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
