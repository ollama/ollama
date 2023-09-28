package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
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

type downloadOpts struct {
	mp      ModelPath
	digest  string
	regOpts *RegistryOptions
	fn      func(api.ProgressResponse)
	retry   int // track the number of retries on this download
}

const maxRetry = 3

// downloadBlob downloads a blob from the registry and stores it in the blobs directory
func downloadBlob(ctx context.Context, opts downloadOpts) error {
	fp, err := GetBlobsPath(opts.digest)
	if err != nil {
		return err
	}

	if fi, _ := os.Stat(fp); fi != nil {
		// we already have the file, so return
		opts.fn(api.ProgressResponse{
			Digest:    opts.digest,
			Total:     fi.Size(),
			Completed: fi.Size(),
		})

		return nil
	}

	fileDownload := &FileDownload{
		Digest:    opts.digest,
		FilePath:  fp,
		Total:     1, // dummy value to indicate that we don't know the total size yet
		Completed: 0,
	}

	_, downloading := inProgress.LoadOrStore(opts.digest, fileDownload)
	if downloading {
		// this is another client requesting the server to download the same blob concurrently
		return monitorDownload(ctx, opts, fileDownload)
	}
	if err := doDownload(ctx, opts, fileDownload); err != nil {
		if errors.Is(err, errDownload) && opts.retry < maxRetry {
			opts.retry++
			log.Print(err)
			log.Printf("retrying download of %s", opts.digest)
			return downloadBlob(ctx, opts)
		}
		return err
	}
	return nil
}

var downloadMu sync.Mutex // mutex to check to resume a download while monitoring

// monitorDownload monitors the download progress of a blob and resumes it if it is interrupted
func monitorDownload(ctx context.Context, opts downloadOpts, f *FileDownload) error {
	tick := time.NewTicker(time.Second)
	for range tick.C {
		done, resume, err := func() (bool, bool, error) {
			downloadMu.Lock()
			defer downloadMu.Unlock()
			val, downloading := inProgress.Load(f.Digest)
			if !downloading {
				// check once again if the download is complete
				if fi, _ := os.Stat(f.FilePath); fi != nil {
					// successful download while monitoring
					opts.fn(api.ProgressResponse{
						Digest:    f.Digest,
						Total:     fi.Size(),
						Completed: fi.Size(),
					})
					return true, false, nil
				}
				// resume the download
				inProgress.Store(f.Digest, f) // store the file download again to claim the resume
				return false, true, nil
			}
			f, ok := val.(*FileDownload)
			if !ok {
				return false, false, fmt.Errorf("invalid type for in progress download: %T", val)
			}
			opts.fn(api.ProgressResponse{
				Status:    fmt.Sprintf("downloading %s", f.Digest),
				Digest:    f.Digest,
				Total:     f.Total,
				Completed: f.Completed,
			})
			return false, false, nil
		}()
		if err != nil {
			return err
		}
		if done {
			// done downloading
			return nil
		}
		if resume {
			return doDownload(ctx, opts, f)
		}
	}
	return nil
}

var (
	chunkSize   int64 = 1024 * 1024 // 1 MiB in bytes
	errDownload       = fmt.Errorf("download failed")
)

// doDownload downloads a blob from the registry and stores it in the blobs directory
func doDownload(ctx context.Context, opts downloadOpts, f *FileDownload) error {
	defer inProgress.Delete(f.Digest)
	var size int64

	fi, err := os.Stat(f.FilePath + "-partial")
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop, file doesn't exist so create it
	case err != nil:
		return fmt.Errorf("stat: %w", err)
	default:
		size = fi.Size()
		// Ensure the size is divisible by the chunk size by removing excess bytes
		size -= size % chunkSize

		err := os.Truncate(f.FilePath+"-partial", size)
		if err != nil {
			return fmt.Errorf("truncate: %w", err)
		}
	}

	requestURL := opts.mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", opts.mp.GetNamespaceRepository(), "blobs", f.Digest)

	headers := make(http.Header)
	headers.Set("Range", fmt.Sprintf("bytes=%d-", size))

	resp, err := makeRequest(ctx, "GET", requestURL, headers, nil, opts.regOpts)
	if err != nil {
		log.Printf("couldn't download blob: %v", err)
		return fmt.Errorf("%w: %w", errDownload, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%w: on download registry responded with code %d: %v", errDownload, resp.StatusCode, string(body))
	}

	err = os.MkdirAll(filepath.Dir(f.FilePath), 0o700)
	if err != nil {
		return fmt.Errorf("make blobs directory: %w", err)
	}

	remaining, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	f.Completed = size
	f.Total = remaining + f.Completed

	inProgress.Store(f.Digest, f)

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
			opts.fn(api.ProgressResponse{
				Status:    fmt.Sprintf("downloading %s", f.Digest),
				Digest:    f.Digest,
				Total:     f.Total,
				Completed: f.Completed,
			})

			if f.Completed >= f.Total {
				if err := out.Close(); err != nil {
					return err
				}

				if err := os.Rename(f.FilePath+"-partial", f.FilePath); err != nil {
					opts.fn(api.ProgressResponse{
						Status:    fmt.Sprintf("error renaming file: %v", err),
						Digest:    f.Digest,
						Total:     f.Total,
						Completed: f.Completed,
					})
					return err
				}

				break outerLoop
			}
		}

		n, err := io.CopyN(out, resp.Body, chunkSize)
		if err != nil && !errors.Is(err, io.EOF) {
			return fmt.Errorf("%w: %w", errDownload, err)
		}
		f.Completed += n

		inProgress.Store(f.Digest, f)
	}

	log.Printf("success getting %s\n", f.Digest)
	return nil
}
