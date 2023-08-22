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

func uploadBlobChunked(ctx context.Context, mp ModelPath, requestURL *url.URL, layer *Layer, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
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

	var completed int64
	chunkSize := 10 * 1024 * 1024

	for {
		chunk := int64(layer.Size) - completed
		if chunk > int64(chunkSize) {
			chunk = int64(chunkSize)
		}

		sectionReader := io.NewSectionReader(f, int64(completed), chunk)

		headers := make(http.Header)
		headers.Set("Content-Type", "application/octet-stream")
		headers.Set("Content-Length", strconv.Itoa(int(chunk)))
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", completed, completed+sectionReader.Size()-1))
		resp, err := makeRequestWithRetry(ctx, "PATCH", requestURL, headers, sectionReader, regOpts)
		if err != nil && !errors.Is(err, io.EOF) {
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("error uploading chunk: %v", err),
				Digest:    layer.Digest,
				Total:     layer.Size,
				Completed: int(completed),
			})

			return err
		}
		defer resp.Body.Close()

		completed += sectionReader.Size()
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("uploading %s", layer.Digest),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: int(completed),
		})

		requestURL, err = url.Parse(resp.Header.Get("Location"))
		if err != nil {
			return err
		}

		if completed >= int64(layer.Size) {
			break
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

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("on finish upload registry responded with code %d: %v", resp.StatusCode, string(body))
	}
	return nil
}
