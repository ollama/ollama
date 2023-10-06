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
	"strconv"
	"sync"

	"github.com/jmorganca/ollama/api"
)

const (
	redirectChunkSize int64 = 1024 * 1024 * 1024
	regularChunkSize  int64 = 95 * 1024 * 1024
)

func startUpload(ctx context.Context, mp ModelPath, layer *Layer, regOpts *RegistryOptions) (*url.URL, int64, error) {
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "blobs/uploads/")
	if layer.From != "" {
		values := requestURL.Query()
		values.Add("mount", layer.Digest)
		values.Add("from", layer.From)
		requestURL.RawQuery = values.Encode()
	}

	resp, err := makeRequestWithRetry(ctx, "POST", requestURL, nil, nil, regOpts)
	if err != nil {
		log.Printf("couldn't start upload: %v", err)
		return nil, 0, err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	chunkSize := redirectChunkSize
	if location == "" {
		location = resp.Header.Get("Location")
		chunkSize = regularChunkSize
	}

	locationURL, err := url.Parse(location)
	if err != nil {
		return nil, 0, err
	}

	return locationURL, chunkSize, nil
}

func uploadBlob(ctx context.Context, requestURL *url.URL, layer *Layer, chunkSize int64, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	// TODO allow resumability
	// TODO allow canceling uploads via DELETE

	fp, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	pw := ProgressWriter{
		status: fmt.Sprintf("uploading %s", layer.Digest),
		digest: layer.Digest,
		total:  layer.Size,
		fn:     fn,
	}

	for offset := int64(0); offset < layer.Size; {
		chunk := layer.Size - offset
		if chunk > chunkSize {
			chunk = chunkSize
		}

		resp, err := uploadBlobChunk(ctx, http.MethodPatch, requestURL, f, offset, chunk, regOpts, &pw)
		if err != nil {
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("error uploading chunk: %v", err),
				Digest:    layer.Digest,
				Total:     layer.Size,
				Completed: offset,
			})

			return err
		}

		offset += chunk
		location := resp.Header.Get("Docker-Upload-Location")
		if location == "" {
			location = resp.Header.Get("Location")
		}

		requestURL, err = url.Parse(location)
		if err != nil {
			return err
		}
	}

	values := requestURL.Query()
	values.Add("digest", layer.Digest)
	requestURL.RawQuery = values.Encode()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", "0")

	// finish the upload
	resp, err := makeRequest(ctx, "PUT", requestURL, headers, nil, regOpts)
	if err != nil {
		log.Printf("couldn't finish upload: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("on finish upload registry responded with code %d: %v", resp.StatusCode, string(body))
	}
	return nil
}

func uploadBlobChunk(ctx context.Context, method string, requestURL *url.URL, r io.ReaderAt, offset, limit int64, opts *RegistryOptions, pw *ProgressWriter) (*http.Response, error) {
	sectionReader := io.NewSectionReader(r, offset, limit)

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", strconv.Itoa(int(limit)))
	headers.Set("X-Redirect-Uploads", "1")

	if method == http.MethodPatch {
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", offset, offset+sectionReader.Size()-1))
	}

	for try := 0; try < maxRetries; try++ {
		resp, err := makeRequest(ctx, method, requestURL, headers, io.TeeReader(sectionReader, pw), opts)
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, err
		}
		defer resp.Body.Close()

		switch {
		case resp.StatusCode == http.StatusTemporaryRedirect:
			location, err := resp.Location()
			if err != nil {
				return nil, err
			}

			pw.completed = offset
			if _, err := uploadBlobChunk(ctx, http.MethodPut, location, r, offset, limit, nil, pw); err != nil {
				// retry
				log.Printf("retrying redirected upload: %v", err)
				continue
			}

			return resp, nil
		case resp.StatusCode == http.StatusUnauthorized:
			auth := resp.Header.Get("www-authenticate")
			authRedir := ParseAuthRedirectString(auth)
			token, err := getAuthToken(ctx, authRedir)
			if err != nil {
				return nil, err
			}

			opts.Token = token

			pw.completed = offset
			sectionReader = io.NewSectionReader(r, offset, limit)
			continue
		case resp.StatusCode >= http.StatusBadRequest:
			body, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("on upload registry responded with code %d: %s", resp.StatusCode, body)
		}

		return resp, nil
	}

	return nil, fmt.Errorf("max retries exceeded")
}

type ProgressWriter struct {
	status    string
	digest    string
	bucket    int64
	completed int64
	total     int64
	fn        func(api.ProgressResponse)
	mu        sync.Mutex
}

func (pw *ProgressWriter) Write(b []byte) (int, error) {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	n := len(b)
	pw.bucket += int64(n)

	// throttle status updates to not spam the client
	if pw.bucket >= 1024*1024 || pw.completed+pw.bucket >= pw.total {
		pw.completed += pw.bucket
		pw.fn(api.ProgressResponse{
			Status:    pw.status,
			Digest:    pw.digest,
			Total:     pw.total,
			Completed: pw.completed,
		})

		pw.bucket = 0
	}

	return n, nil
}
