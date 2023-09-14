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

	// 95MB chunk size
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

		sectionReader := io.NewSectionReader(f, int64(offset), chunk)

		var errStatus error
		for try := 0; try < MaxRetries; try++ {
			errStatus = nil

			headers := make(http.Header)
			headers.Set("Content-Type", "application/octet-stream")
			headers.Set("Content-Length", strconv.Itoa(int(chunk)))
			headers.Set("Content-Range", fmt.Sprintf("%d-%d", offset, offset+sectionReader.Size()-1))
			resp, err := makeRequest(ctx, "PATCH", requestURL, headers, io.TeeReader(sectionReader, &pw), regOpts)
			if err != nil && !errors.Is(err, io.EOF) {
				fn(api.ProgressResponse{
					Status:    fmt.Sprintf("error uploading chunk: %v", err),
					Digest:    layer.Digest,
					Total:     layer.Size,
					Completed: int(offset),
				})

				return err
			}
			defer resp.Body.Close()

			switch {
			case resp.StatusCode == http.StatusUnauthorized:
				errStatus = errors.New("unauthorized")

				auth := resp.Header.Get("www-authenticate")
				authRedir := ParseAuthRedirectString(auth)
				token, err := getAuthToken(ctx, authRedir)
				if err != nil {
					return err
				}

				regOpts.Token = token

				pw.completed = int(offset)
				sectionReader = io.NewSectionReader(f, offset, chunk)
				continue
			case resp.StatusCode >= http.StatusBadRequest:
				body, _ := io.ReadAll(resp.Body)
				return fmt.Errorf("on upload registry responded with code %d: %s", resp.StatusCode, body)
			}

			offset += sectionReader.Size()
			requestURL, err = url.Parse(resp.Header.Get("Location"))
			if err != nil {
				return err
			}

			break
		}

		if errStatus != nil {
			return fmt.Errorf("max retries exceeded: %w", errStatus)
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
