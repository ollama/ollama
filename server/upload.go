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

	"github.com/jmorganca/ollama/api"
)

func startUpload(ctx context.Context, mp ModelPath, layer *Layer, regOpts *RegistryOptions) (*url.URL, error) {
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
		return nil, err
	}
	defer resp.Body.Close()

	// Extract UUID location from header
	location := resp.Header.Get("Location")
	if location == "" {
		return nil, fmt.Errorf("location header is missing in response")
	}

	return url.Parse(location)
}

func uploadBlobChunked(ctx context.Context, requestURL *url.URL, layer *Layer, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
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

	// 95MiB chunk size
	chunkSize := 95 * 1024 * 1024
	pw := ProgressWriter{
		status: fmt.Sprintf("uploading %s", layer.Digest),
		digest: layer.Digest,
		total:  layer.Size,
		fn:     fn,
	}

	for offset := int64(0); offset < int64(layer.Size); {
		chunk := int64(layer.Size) - offset
		if chunk > int64(chunkSize) {
			chunk = int64(chunkSize)
		}

		resp, err := uploadBlobChunk(ctx, http.MethodPatch, requestURL, f, offset, chunk, regOpts, &pw)
		if err != nil {
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("error uploading chunk: %v", err),
				Digest:    layer.Digest,
				Total:     layer.Size,
				Completed: int(offset),
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
	sectionReader := io.NewSectionReader(r, int64(offset), limit)

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", strconv.Itoa(int(limit)))
	headers.Set("X-Redirect-Uploads", "1")

	if method == http.MethodPatch {
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", offset, offset+sectionReader.Size()-1))
	}

	for try := 0; try < MaxRetries; try++ {
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

			pw.completed = int(offset)
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

			pw.completed = int(offset)
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
	bucket    int
	completed int
	total     int
	fn        func(api.ProgressResponse)
}

func (pw *ProgressWriter) Write(b []byte) (int, error) {
	n := len(b)
	pw.bucket += n
	pw.completed += n

	// throttle status updates to not spam the client
	if pw.bucket >= 1024*1024 || pw.completed >= pw.total {
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
