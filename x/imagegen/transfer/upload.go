package transfer

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

type uploader struct {
	client     *http.Client
	baseURL    string
	srcDir     string
	repository string // Repository path for blob URLs (e.g., "library/model")
	token      *string
	getToken   func(context.Context, AuthChallenge) (string, error)
	userAgent  string
	progress   *progressTracker
	logger     *slog.Logger
}

func upload(ctx context.Context, opts UploadOptions) error {
	if len(opts.Blobs) == 0 && len(opts.Manifest) == 0 {
		return nil
	}

	token := opts.Token
	u := &uploader{
		client:     cmp.Or(opts.Client, defaultClient),
		baseURL:    opts.BaseURL,
		srcDir:     opts.SrcDir,
		repository: cmp.Or(opts.Repository, "library/_"),
		token:      &token,
		getToken:   opts.GetToken,
		userAgent:  cmp.Or(opts.UserAgent, defaultUserAgent),
		logger:     opts.Logger,
	}

	if len(opts.Blobs) > 0 {
		// Phase 1: Fast parallel HEAD checks to find which blobs need uploading
		needsUpload := make([]bool, len(opts.Blobs))
		{
			sem := semaphore.NewWeighted(128) // High concurrency for HEAD checks
			g, gctx := errgroup.WithContext(ctx)
			for i, blob := range opts.Blobs {
				g.Go(func() error {
					if err := sem.Acquire(gctx, 1); err != nil {
						return err
					}
					defer sem.Release(1)
					exists, err := u.exists(gctx, blob)
					if err != nil {
						return err
					}
					if !exists {
						needsUpload[i] = true
					} else if u.logger != nil {
						u.logger.Debug("blob exists", "digest", blob.Digest)
					}
					return nil
				})
			}
			if err := g.Wait(); err != nil {
				return err
			}
		}

		// Filter to only blobs that need uploading
		var toUpload []Blob
		var total int64
		for i, blob := range opts.Blobs {
			if needsUpload[i] {
				toUpload = append(toUpload, blob)
				total += blob.Size
			}
		}

		if len(toUpload) == 0 {
			if u.logger != nil {
				u.logger.Debug("all blobs exist, nothing to upload")
			}
		} else {
			// Phase 2: Upload blobs that don't exist
			u.progress = newProgressTracker(total, opts.Progress)
			concurrency := cmp.Or(opts.Concurrency, DefaultUploadConcurrency)
			sem := semaphore.NewWeighted(int64(concurrency))

			g, gctx := errgroup.WithContext(ctx)
			for _, blob := range toUpload {
				g.Go(func() error {
					if err := sem.Acquire(gctx, 1); err != nil {
						return err
					}
					defer sem.Release(1)
					return u.upload(gctx, blob)
				})
			}
			if err := g.Wait(); err != nil {
				return err
			}
		}
	}

	if len(opts.Manifest) > 0 && opts.ManifestRef != "" && opts.Repository != "" {
		return u.pushManifest(ctx, opts.Repository, opts.ManifestRef, opts.Manifest)
	}
	return nil
}

func (u *uploader) upload(ctx context.Context, blob Blob) error {
	var lastErr error
	var n int64

	for attempt := range maxRetries {
		if attempt > 0 {
			if err := backoff(ctx, attempt, time.Second<<uint(attempt-1)); err != nil {
				return err
			}
		}

		var err error
		n, err = u.uploadOnce(ctx, blob)
		if err == nil {
			return nil
		}

		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}

		u.progress.add(-n)
		lastErr = err
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
}

func (u *uploader) uploadOnce(ctx context.Context, blob Blob) (int64, error) {
	if u.logger != nil {
		u.logger.Debug("uploading blob", "digest", blob.Digest, "size", blob.Size)
	}

	// Init upload
	uploadURL, err := u.initUpload(ctx, blob)
	if err != nil {
		return 0, err
	}

	// Open file
	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		return 0, err
	}
	defer f.Close()

	// PUT blob
	return u.put(ctx, uploadURL, f, blob.Size)
}

func (u *uploader) exists(ctx context.Context, blob Blob) (bool, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodHead, fmt.Sprintf("%s/v2/%s/blobs/%s", u.baseURL, u.repository, blob.Digest), nil)
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return false, err
	}
	resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return false, err
		}
		return u.exists(ctx, blob)
	}

	return resp.StatusCode == http.StatusOK, nil
}

func (u *uploader) initUpload(ctx context.Context, blob Blob) (string, error) {
	endpoint, _ := url.Parse(fmt.Sprintf("%s/v2/%s/blobs/uploads/", u.baseURL, u.repository))
	q := endpoint.Query()
	q.Set("digest", blob.Digest)
	endpoint.RawQuery = q.Encode()

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), nil)
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return "", err
	}
	resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return "", err
		}
		return u.initUpload(ctx, blob)
	}

	if resp.StatusCode != http.StatusAccepted {
		return "", fmt.Errorf("init: status %d", resp.StatusCode)
	}

	loc := resp.Header.Get("Docker-Upload-Location")
	if loc == "" {
		loc = resp.Header.Get("Location")
	}
	if loc == "" {
		return "", fmt.Errorf("no upload location")
	}

	locURL, _ := url.Parse(loc)
	if !locURL.IsAbs() {
		base, _ := url.Parse(u.baseURL)
		locURL = base.ResolveReference(locURL)
	}
	q = locURL.Query()
	q.Set("digest", blob.Digest)
	locURL.RawQuery = q.Encode()

	return locURL.String(), nil
}

func (u *uploader) put(ctx context.Context, uploadURL string, f *os.File, size int64) (int64, error) {
	pr := &progressReader{reader: f, tracker: u.progress}

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, pr)
	req.ContentLength = size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return pr.n, err
	}
	defer resp.Body.Close()

	// Handle auth retry
	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return pr.n, err
		}
		f.Seek(0, 0)
		u.progress.add(-pr.n)
		return u.put(ctx, uploadURL, f, size)
	}

	// Handle redirect to CDN
	if resp.StatusCode == http.StatusTemporaryRedirect {
		loc, _ := resp.Location()
		f.Seek(0, 0)
		u.progress.add(-pr.n)
		pr2 := &progressReader{reader: f, tracker: u.progress}

		req2, _ := http.NewRequestWithContext(ctx, http.MethodPut, loc.String(), pr2)
		req2.ContentLength = size
		req2.Header.Set("Content-Type", "application/octet-stream")
		req2.Header.Set("User-Agent", u.userAgent)

		resp2, err := u.client.Do(req2)
		if err != nil {
			return pr2.n, err
		}
		defer resp2.Body.Close()

		if resp2.StatusCode != http.StatusCreated && resp2.StatusCode != http.StatusAccepted {
			body, _ := io.ReadAll(resp2.Body)
			return pr2.n, fmt.Errorf("status %d: %s", resp2.StatusCode, body)
		}
		return pr2.n, nil
	}

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return pr.n, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	return pr.n, nil
}

func (u *uploader) pushManifest(ctx context.Context, repo, ref string, manifest []byte) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", u.baseURL, repo, ref), bytes.NewReader(manifest))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return err
		}
		return u.pushManifest(ctx, repo, ref, manifest)
	}

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	return nil
}

type progressReader struct {
	reader  io.Reader
	tracker *progressTracker
	n       int64
}

func (r *progressReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if n > 0 {
		r.n += int64(n)
		r.tracker.add(int64(n))
	}
	return n, err
}
