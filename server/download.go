package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"sync"
	"time"

	"github.com/jmorganca/ollama/api"
)

type FileDownload struct {
	Digest    string
	FilePath  string
	Total     int64
	Completed int64
}

var inProgress sync.Map // map of digests currently being downloaded to their current download progress

// downloadBlob downloads a blob from the registry and stores it in the blobs directory
func downloadBlob(ctx context.Context, mp ModelPath, digest string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	if fi, _ := os.Stat(fp); fi != nil {
		// we already have the file, so return
		fn(api.ProgressResponse{
			Digest:    digest,
			Total:     int(fi.Size()),
			Completed: int(fi.Size()),
		})

		return nil
	}

	fileDownload := &FileDownload{
		Digest:    digest,
		FilePath:  fp,
		Total:     1, // dummy value to indicate that we don't know the total size yet
		Completed: 0,
	}

	_, downloading := inProgress.LoadOrStore(digest, fileDownload)
	if downloading {
		// this is another client requesting the server to download the same blob concurrently
		return monitorDownload(ctx, mp, regOpts, fileDownload, fn)
	}
	resp, err := requestDownload(ctx, mp, regOpts, fileDownload)
	if err != nil {
		return err
	}
	return doDownload(ctx, fileDownload, resp, fn)
}

var downloadMu sync.Mutex // mutex to check to resume a download while monitoring

// monitorDownload monitors the download progress of a blob and resumes it if it is interrupted
func monitorDownload(ctx context.Context, mp ModelPath, regOpts *RegistryOptions, f *FileDownload, fn func(api.ProgressResponse)) error {
	tick := time.NewTicker(time.Second)
	for range tick.C {
		downloadMu.Lock()
		val, downloading := inProgress.Load(f.Digest)
		if !downloading {
			// check once again if the download is complete
			if fi, _ := os.Stat(f.FilePath); fi != nil {
				downloadMu.Unlock()
				// successfull download while monitoring
				fn(api.ProgressResponse{
					Digest:    f.Digest,
					Total:     int(fi.Size()),
					Completed: int(fi.Size()),
				})
				return nil
			}
			// resume the download
			resp, err := requestDownload(ctx, mp, regOpts, f)
			if err != nil {
				return fmt.Errorf("resume: %w", err)
			}
			inProgress.Store(f.Digest, f)
			downloadMu.Unlock()
			return doDownload(ctx, f, resp, fn)
		}
		downloadMu.Unlock()
		f, ok := val.(*FileDownload)
		if !ok {
			return fmt.Errorf("invalid type for in progress download: %T", val)
		}
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("downloading %s", f.Digest),
			Digest:    f.Digest,
			Total:     int(f.Total),
			Completed: int(f.Completed),
		})
	}
	return nil
}

var chunkSize = 1024 * 1024 // 1 MiB in bytes

// requestDownload requests a blob from the registry and returns the response
func requestDownload(ctx context.Context, mp ModelPath, regOpts *RegistryOptions, f *FileDownload) (*http.Response, error) {
	var size int64

	fi, err := os.Stat(f.FilePath + "-partial")
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop, file doesn't exist so create it
	case err != nil:
		return nil, fmt.Errorf("stat: %w", err)
	default:
		size = fi.Size()
		// Ensure the size is divisible by the chunk size by removing excess bytes
		size -= size % int64(chunkSize)

		err := os.Truncate(f.FilePath+"-partial", size)
		if err != nil {
			return nil, fmt.Errorf("truncate: %w", err)
		}
	}

	url := fmt.Sprintf("%s/v2/%s/blobs/%s", mp.Registry, mp.GetNamespaceRepository(), f.Digest)
	headers := map[string]string{
		"Range": fmt.Sprintf("bytes=%d-", size),
	}

	resp, err := makeRequest("GET", url, headers, nil, regOpts)
	if err != nil {
		log.Printf("couldn't download blob: %v", err)
		return nil, err
	}
	// resp MUST be closed by doDownload, which should follow this function

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("on download registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	err = os.MkdirAll(path.Dir(f.FilePath), 0o700)
	if err != nil {
		return nil, fmt.Errorf("make blobs directory: %w", err)
	}

	remaining, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	f.Completed = size
	f.Total = remaining + f.Completed

	inProgress.Store(f.Digest, f)
	return resp, nil
}

// doDownload downloads a blob from the registry and stores it in the blobs directory
func doDownload(ctx context.Context, f *FileDownload, resp *http.Response, fn func(api.ProgressResponse)) error {
	defer resp.Body.Close()
	out, err := os.OpenFile(f.FilePath+"-partial", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer out.Close()
outerLoop:
	for {
		select {
		case <-ctx.Done():
			// handle client request cancellation
			inProgress.Delete(f.Digest)
			return nil
		default:
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("downloading %s", f.Digest),
				Digest:    f.Digest,
				Total:     int(f.Total),
				Completed: int(f.Completed),
			})

			if f.Completed >= f.Total {
				if err := out.Close(); err != nil {
					return err
				}

				if err := os.Rename(f.FilePath+"-partial", f.FilePath); err != nil {
					fn(api.ProgressResponse{
						Status:    fmt.Sprintf("error renaming file: %v", err),
						Digest:    f.Digest,
						Total:     int(f.Total),
						Completed: int(f.Completed),
					})
					return err
				}

				break outerLoop
			}
		}

		n, err := io.CopyN(out, resp.Body, int64(chunkSize))
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		f.Completed += n

		inProgress.Store(f.Digest, f)
	}

	inProgress.Delete(f.Digest)

	log.Printf("success getting %s\n", f.Digest)
	return nil
}
