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

	"github.com/jmorganca/ollama/api"
	"golang.org/x/sync/errgroup"
)

type BlobDownloadPart struct {
	Offset    int64
	Size      int64
	Completed int64
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

	f, err := os.OpenFile(fp+"-partial", os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	partFilePaths, err := filepath.Glob(fp + "-partial-*")
	if err != nil {
		return err
	}

	var total, completed int64
	var parts []BlobDownloadPart
	for _, partFilePath := range partFilePaths {
		var part BlobDownloadPart
		partFile, err := os.Open(partFilePath)
		if err != nil {
			return err
		}
		defer partFile.Close()

		if err := json.NewDecoder(partFile).Decode(&part); err != nil {
			return err
		}

		total += part.Size
		completed += part.Completed

		parts = append(parts, part)
	}

	requestURL := opts.mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", opts.mp.GetNamespaceRepository(), "blobs", opts.digest)

	if len(parts) == 0 {
		resp, err := makeRequest(ctx, "HEAD", requestURL, nil, nil, opts.regOpts)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

		// reserve the file
		f.Truncate(total)

		var offset int64
		var size int64 = 64 * 1024 * 1024

		for offset < total {
			if offset+size > total {
				size = total - offset
			}

			parts = append(parts, BlobDownloadPart{
				Offset: offset,
				Size:   size,
			})

			offset += size
		}
	}

	pw := &ProgressWriter{
		status:    fmt.Sprintf("downloading %s", opts.digest),
		digest:    opts.digest,
		total:     total,
		completed: completed,
		fn:        opts.fn,
	}

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(64)
	for i := range parts {
		part := parts[i]
		if part.Completed == part.Size {
			continue
		}

		i := i
		g.Go(func() error {
			for try := 0; try < maxRetries; try++ {
				if err := downloadBlobChunk(ctx, f, requestURL, parts, i, pw, opts); err != nil {
					log.Printf("%s part %d attempt %d failed: %v, retrying", opts.digest[7:19], i, try, err)
					continue
				}

				return nil
			}

			return errors.New("max retries exceeded")
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	if err := f.Close(); err != nil {
		return err
	}

	for i := range parts {
		if err := os.Remove(f.Name() + "-" + strconv.Itoa(i)); err != nil {
			return err
		}
	}

	return os.Rename(f.Name(), fp)
}

func downloadBlobChunk(ctx context.Context, f *os.File, requestURL *url.URL, parts []BlobDownloadPart, i int, pw *ProgressWriter, opts downloadOpts) error {
	part := &parts[i]

	partName := f.Name() + "-" + strconv.Itoa(i)
	if err := flushPart(partName, part); err != nil {
		return err
	}

	offset := part.Offset + part.Completed
	w := io.NewOffsetWriter(f, offset)

	headers := make(http.Header)
	headers.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, part.Offset+part.Size-1))
	resp, err := makeRequest(ctx, "GET", requestURL, headers, nil, opts.regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	n, err := io.Copy(w, io.TeeReader(resp.Body, pw))
	if err != nil && !errors.Is(err, io.EOF) {
		// rollback progress bar
		pw.completed -= n
		return err
	}

	part.Completed += n

	return flushPart(partName, part)
}

func flushPart(name string, part *BlobDownloadPart) error {
	partFile, err := os.OpenFile(name, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer partFile.Close()

	return json.NewEncoder(partFile).Encode(part)
}
