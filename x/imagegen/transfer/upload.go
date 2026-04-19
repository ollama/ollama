package transfer

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/logutil"

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

	// computeParts controls how blobs are split for chunked upload.
	// Defaults to the standard computeParts function.
	// Tests may override this to use smaller part sizes.
	makeParts func(int64) []uploadPart
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

		// Filter to only blobs that need uploading, but track total across all blobs
		var toUpload []Blob
		var totalSize, alreadyExists int64
		for i, blob := range opts.Blobs {
			totalSize += blob.Size
			if needsUpload[i] {
				toUpload = append(toUpload, blob)
			} else {
				alreadyExists += blob.Size
			}
		}

		// Progress includes all blobs — already-existing ones start as completed
		u.progress = newProgressTracker(totalSize, opts.Progress)
		u.progress.add(alreadyExists)

		logutil.Trace("upload plan", "total_blobs", len(opts.Blobs), "need_upload", len(toUpload), "already_exist", len(opts.Blobs)-len(toUpload), "total_bytes", totalSize, "existing_bytes", alreadyExists)

		if len(toUpload) == 0 {
			if u.logger != nil {
				u.logger.Debug("all blobs exist, nothing to upload")
			}
		} else {
			// Phase 2: Upload blobs that don't exist
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
		logutil.Trace("pushing manifest", "repo", opts.Repository, "ref", opts.ManifestRef, "size", len(opts.Manifest))
		if err := u.pushManifest(ctx, opts.Repository, opts.ManifestRef, opts.Manifest); err != nil {
			logutil.Trace("manifest push failed", "error", err)
			return err
		}
		logutil.Trace("manifest push succeeded", "repo", opts.Repository, "ref", opts.ManifestRef)
	}
	return nil
}

func (u *uploader) upload(ctx context.Context, blob Blob) error {
	var lastErr error
	var n int64

	for attempt := range maxRetries {
		if attempt > 0 {
			// Use longer backoff for uploads — server-side rate limiting
			// and S3 upload session creation need real breathing room.
			// attempt 1: up to 2s, attempt 2: up to 4s, attempt 3: up to 8s, etc.
			if err := backoff(ctx, attempt, 2*time.Second<<uint(attempt-1)); err != nil {
				return err
			}
		}

		var err error
		n, err = u.uploadOnce(ctx, blob)
		if err == nil {
			logutil.Trace("blob upload complete", "digest", blob.Digest, "bytes", n, "attempt", attempt+1)
			return nil
		}

		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			logutil.Trace("blob upload cancelled", "digest", blob.Digest, "error", err)
			return err
		}

		logutil.Trace("blob upload failed, retrying", "digest", blob.Digest, "attempt", attempt+1, "bytes", n, "error", err)
		u.progress.add(-n)
		lastErr = err
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
}

func (u *uploader) uploadOnce(ctx context.Context, blob Blob) (int64, error) {
	if u.logger != nil {
		u.logger.Debug("uploading blob", "digest", blob.Digest, "size", blob.Size)
	}

	uploadURL, err := u.initUpload(ctx, blob)
	if err != nil {
		return 0, err
	}

	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if blob.Size >= resumeThreshold {
		return u.putChunked(ctx, uploadURL, f, blob)
	}
	return u.put(ctx, uploadURL, f, blob)
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
	io.Copy(io.Discard, resp.Body)
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

const maxInitRetries = 12

// initUpload initiates an upload session and returns the session URL.
// The returned URL does NOT include the digest query parameter — callers
// (put, putChunked/finalize) append it as needed.
func (u *uploader) initUpload(ctx context.Context, blob Blob) (string, error) {
	endpoint, _ := url.Parse(fmt.Sprintf("%s/v2/%s/blobs/uploads/", u.baseURL, u.repository))

	var lastErr error
	for attempt := range maxInitRetries {
		if attempt > 0 {
			if err := backoff(ctx, attempt, min(5*time.Second<<uint(attempt-1), 30*time.Second)); err != nil {
				return "", err
			}
			logutil.Trace("retrying init upload", "digest", blob.Digest, "attempt", attempt+1, "error", lastErr)
		}

		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), nil)
		req.Header.Set("User-Agent", u.userAgent)
		if *u.token != "" {
			req.Header.Set("Authorization", "Bearer "+*u.token)
		}

		resp, err := u.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
			ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
			if *u.token, err = u.getToken(ctx, ch); err != nil {
				return "", err
			}
			continue
		}

		if resp.StatusCode == http.StatusCreated {
			return "", nil
		}

		if resp.StatusCode != http.StatusAccepted {
			lastErr = fmt.Errorf("init upload: status %d", resp.StatusCode)
			continue
		}

		loc := resp.Header.Get("Docker-Upload-Location")
		if loc == "" {
			loc = resp.Header.Get("Location")
		}
		if loc == "" {
			lastErr = fmt.Errorf("no upload location (server returned 202 without Location header)")
			continue
		}

		locURL, _ := url.Parse(loc)
		if !locURL.IsAbs() {
			base, _ := url.Parse(u.baseURL)
			locURL = base.ResolveReference(locURL)
		}

		return locURL.String(), nil
	}
	return "", lastErr
}

// put uploads a blob as a single PUT request (for blobs below resumeThreshold).
func (u *uploader) put(ctx context.Context, uploadURL string, f *os.File, blob Blob) (int64, error) {
	if uploadURL == "" {
		return 0, nil
	}

	// Append digest to the upload URL
	uu, _ := url.Parse(uploadURL)
	q := uu.Query()
	q.Set("digest", blob.Digest)
	uu.RawQuery = q.Encode()

	br := bufio.NewReaderSize(f, 256*1024)
	pr := &progressReader{reader: br, tracker: u.progress}

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, uu.String(), pr)
	req.ContentLength = blob.Size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return pr.n, fmt.Errorf("put request: %w", err)
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	// Handle auth retry
	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return pr.n, err
		}
		f.Seek(0, 0)
		u.progress.add(-pr.n)
		return u.put(ctx, uploadURL, f, blob)
	}

	// Handle redirect to CDN
	if resp.StatusCode == http.StatusTemporaryRedirect {
		loc, _ := resp.Location()
		f.Seek(0, 0)
		u.progress.add(-pr.n)

		br2 := bufio.NewReaderSize(f, 256*1024)
		pr2 := &progressReader{reader: br2, tracker: u.progress}

		req2, _ := http.NewRequestWithContext(ctx, http.MethodPut, loc.String(), pr2)
		req2.ContentLength = blob.Size
		req2.Header.Set("Content-Type", "application/octet-stream")
		req2.Header.Set("User-Agent", u.userAgent)

		resp2, err := u.client.Do(req2)
		if err != nil {
			return pr2.n, fmt.Errorf("cdn put request: %w", err)
		}
		defer func() { io.Copy(io.Discard, resp2.Body); resp2.Body.Close() }()

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

// uploadPart represents a chunk of a blob for chunked upload.
type uploadPart struct {
	n      int
	offset int64
	size   int64
	md5sum hash.Hash
}

// computeParts divides a blob into upload parts using default limits.
func computeParts(totalSize int64) []uploadPart {
	return computePartsWithLimits(totalSize, numUploadParts, minUploadPartSize, maxUploadPartSize)
}

// computePartsWithLimits divides a blob into upload parts with configurable limits.
func computePartsWithLimits(totalSize int64, nParts int, minPart, maxPart int64) []uploadPart {
	partSize := totalSize / int64(nParts)
	partSize = max(partSize, minPart)
	partSize = min(partSize, maxPart)

	var parts []uploadPart
	var offset int64
	for offset < totalSize {
		size := partSize
		if offset+size > totalSize {
			size = totalSize - offset
		}
		parts = append(parts, uploadPart{n: len(parts), offset: offset, size: size})
		offset += size
	}
	return parts
}

// putChunked uploads a large blob using parallel PATCH requests with Content-Range
// headers, followed by a PUT to finalize. This matches the OCI chunked upload
// protocol used by server/upload.go.
func (u *uploader) putChunked(ctx context.Context, uploadURL string, f *os.File, blob Blob) (int64, error) {
	if uploadURL == "" {
		return 0, nil
	}

	makeParts := computeParts
	if u.makeParts != nil {
		makeParts = u.makeParts
	}
	parts := makeParts(blob.Size)

	sessionURL, err := url.Parse(uploadURL)
	if err != nil {
		return 0, fmt.Errorf("parse upload URL: %w", err)
	}

	nextURL := make(chan *url.URL, 1)
	nextURL <- sessionURL

	var totalWritten atomic.Int64

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(numUploadParts)
	for i := range parts {
		part := &parts[i]
		select {
		case <-gctx.Done():
		case patchURL := <-nextURL:
			g.Go(func() error {
				var lastErr error
				for try := range 3 {
					if try > 0 {
						if err := backoff(gctx, try, 2*time.Second<<uint(try-1)); err != nil {
							return err
						}
					}

					next, n, err := u.patchPart(gctx, patchURL, part, f, nextURL)
					totalWritten.Add(n)
					if err == nil {
						if next != nil {
							nextURL <- next
						}
						return nil
					}

					// Rollback progress for this attempt
					u.progress.add(-n)
					totalWritten.Add(-n)

					if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
						return err
					}
					lastErr = err
				}
				return fmt.Errorf("part %d: %w", part.n, lastErr)
			})
		}
	}

	if err := g.Wait(); err != nil {
		return totalWritten.Load(), err
	}

	finalURL := <-nextURL
	if err := u.finalize(ctx, finalURL, parts, blob.Digest); err != nil {
		return totalWritten.Load(), err
	}
	return totalWritten.Load(), nil
}

// patchPart uploads a single part via PATCH with Content-Range.
// On 307 redirect, it pushes the next session URL to the channel and PUTs the
// data to the CDN. On success, it returns the next session URL for the caller
// to push to the channel.
func (u *uploader) patchPart(ctx context.Context, patchURL *url.URL, part *uploadPart, f *os.File, nextURL chan<- *url.URL) (*url.URL, int64, error) {
	sr := io.NewSectionReader(f, part.offset, part.size)
	br := bufio.NewReaderSize(sr, 256*1024)
	md5sum := md5.New()
	pr := &progressReader{reader: br, tracker: u.progress}
	body := io.TeeReader(pr, md5sum)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPatch, patchURL.String(), body)
	req.ContentLength = part.size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Content-Range", fmt.Sprintf("%d-%d", part.offset, part.offset+part.size-1))
	req.Header.Set("X-Redirect-Uploads", "1")
	req.Header.Set("User-Agent", u.userAgent)
	if *u.token != "" {
		req.Header.Set("Authorization", "Bearer "+*u.token)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return nil, pr.n, fmt.Errorf("patch part %d: %w", part.n, err)
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	// Parse next session URL from response
	loc := resp.Header.Get("Docker-Upload-Location")
	if loc == "" {
		loc = resp.Header.Get("Location")
	}
	var next *url.URL
	if loc != "" {
		next, _ = url.Parse(loc)
	}

	switch {
	case resp.StatusCode == http.StatusTemporaryRedirect:
		// Push session URL to channel immediately so the next part can start
		// its PATCH while we re-upload data to the CDN.
		if next != nil {
			nextURL <- next
		}

		redirectURL, _ := resp.Location()
		if redirectURL == nil {
			return nil, pr.n, fmt.Errorf("patch part %d: 307 without Location", part.n)
		}

		// Rollback progress from the PATCH attempt, re-upload to CDN
		u.progress.add(-pr.n)
		cdnWritten := pr.n // remember for total accounting

		cdnN, cdnErr := u.putPartToCDN(ctx, redirectURL, part, f)
		if cdnErr != nil {
			return nil, cdnWritten, cdnErr
		}

		part.md5sum = md5sum
		return nil, cdnN, nil

	case resp.StatusCode == http.StatusUnauthorized && u.getToken != nil:
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if *u.token, err = u.getToken(ctx, ch); err != nil {
			return nil, pr.n, err
		}
		return nil, pr.n, fmt.Errorf("patch part %d: auth retry", part.n)

	case resp.StatusCode >= http.StatusBadRequest:
		body, _ := io.ReadAll(resp.Body)
		return nil, pr.n, fmt.Errorf("patch part %d: status %d: %s", part.n, resp.StatusCode, body)
	}

	if next == nil {
		return nil, pr.n, fmt.Errorf("patch part %d: no next URL in response", part.n)
	}

	part.md5sum = md5sum
	return next, pr.n, nil
}

// putPartToCDN re-uploads a part's data to a CDN redirect URL via PUT.
func (u *uploader) putPartToCDN(ctx context.Context, cdnURL *url.URL, part *uploadPart, f *os.File) (int64, error) {
	sr := io.NewSectionReader(f, part.offset, part.size)
	br := bufio.NewReaderSize(sr, 256*1024)
	pr := &progressReader{reader: br, tracker: u.progress}

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, cdnURL.String(), pr)
	req.ContentLength = part.size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", u.userAgent)
	// No auth header for CDN

	resp, err := u.client.Do(req)
	if err != nil {
		return pr.n, fmt.Errorf("cdn put part %d: %w", part.n, err)
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return pr.n, fmt.Errorf("cdn put part %d: status %d: %s", part.n, resp.StatusCode, body)
	}
	return pr.n, nil
}

// finalize completes a chunked upload by sending a PUT with the composite etag.
func (u *uploader) finalize(ctx context.Context, finalURL *url.URL, parts []uploadPart, digest string) error {
	// Compute composite MD5 etag: md5(part1_hash || part2_hash || ... || partN_hash)
	composite := md5.New()
	for i := range parts {
		if parts[i].md5sum == nil {
			return fmt.Errorf("finalize: part %d has no hash", i)
		}
		composite.Write(parts[i].md5sum.Sum(nil))
	}
	etag := fmt.Sprintf("%x-%d", composite.Sum(nil), len(parts))

	q := finalURL.Query()
	q.Set("digest", digest)
	q.Set("etag", etag)
	finalURL.RawQuery = q.Encode()

	var lastErr error
	for try := range maxRetries {
		if try > 0 {
			if err := backoff(ctx, try, 2*time.Second<<uint(try-1)); err != nil {
				return err
			}
		}

		req, _ := http.NewRequestWithContext(ctx, http.MethodPut, finalURL.String(), nil)
		req.ContentLength = 0
		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set("User-Agent", u.userAgent)
		if *u.token != "" {
			req.Header.Set("Authorization", "Bearer "+*u.token)
		}

		resp, err := u.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
			ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
			if *u.token, err = u.getToken(ctx, ch); err != nil {
				return err
			}
			continue
		}

		if resp.StatusCode == http.StatusCreated || resp.StatusCode == http.StatusOK {
			return nil
		}

		body := make([]byte, 512)
		n, _ := resp.Body.Read(body)
		lastErr = fmt.Errorf("finalize: status %d: %s", resp.StatusCode, body[:n])
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
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
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

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
