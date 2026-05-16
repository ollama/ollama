package transfer

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
	tokenMu    sync.RWMutex
	token      string
	getToken   func(context.Context, AuthChallenge) (string, error)
	userAgent  string
	progress   *progressTracker
	logger     *slog.Logger
	// bodySem caps the number of simultaneous body-bearing transfers so a
	// modest home uplink isn't saturated. Always set by upload(); nil only
	// when tests build uploader directly (in which case holdBody is a no-op).
	bodySem   *semaphore.Weighted
	makeParts func(int64) []uploadPart // controls how blobs are split for chunked upload
}

// authToken returns the current bearer token. Safe to call concurrently with
// refreshToken.
func (u *uploader) authToken() string {
	u.tokenMu.RLock()
	defer u.tokenMu.RUnlock()
	return u.token
}

// refreshToken coalesces token fetches so concurrent 401s don't all hit the
// auth server. prev is the token the caller used in the request that got
// rejected: if the stored token has already moved past prev, another
// goroutine has refreshed and we just observe its result; otherwise the
// caller holds the lock and performs the fetch.
func (u *uploader) refreshToken(ctx context.Context, ch AuthChallenge, prev string) error {
	u.tokenMu.Lock()
	defer u.tokenMu.Unlock()
	if u.token != prev {
		return nil
	}
	if u.getToken == nil {
		return errors.New("no token refresh callback")
	}
	t, err := u.getToken(ctx, ch)
	if err != nil {
		return err
	}
	u.token = t
	return nil
}

// holdBody acquires a body-transfer slot. The returned release must be called
// exactly once after the body-bearing request completes (defer is fine).
func (u *uploader) holdBody(ctx context.Context) (func(), error) {
	if u.bodySem == nil {
		return func() {}, nil
	}
	if err := u.bodySem.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	return func() { u.bodySem.Release(1) }, nil
}

func upload(ctx context.Context, opts UploadOptions) error {
	if len(opts.Blobs) == 0 && len(opts.Manifest) == 0 {
		return nil
	}

	u := &uploader{
		client:     cmp.Or(opts.Client, defaultClient),
		baseURL:    opts.BaseURL,
		srcDir:     opts.SrcDir,
		repository: cmp.Or(opts.Repository, "library/_"),
		token:      opts.Token,
		getToken:   opts.GetToken,
		userAgent:  cmp.Or(opts.UserAgent, defaultUserAgent),
		logger:     opts.Logger,
	}
	// 0 or negative serializes; never unbounded.
	u.bodySem = semaphore.NewWeighted(int64(max(1, opts.BodyConcurrency)))

	if len(opts.Blobs) > 0 {
		// Discover which blobs the server already has so we can skip uploading
		needsUpload := make([]bool, len(opts.Blobs))
		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(128)
		for i, blob := range opts.Blobs {
			g.Go(func() error {
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
			// Upload the blobs the server doesn't already have. Concurrency
			// caps blob-level parallelism.
			concurrency := cmp.Or(opts.Concurrency, DefaultUploadConcurrency)
			sem := semaphore.NewWeighted(int64(concurrency))

			start := time.Now()
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
			err := g.Wait()
			elapsed := time.Since(start)
			done := u.progress.completed.Load() - alreadyExists
			mbps := float64(done) / 1e6 / max(0.001, elapsed.Seconds())
			slog.Debug("upload summary",
				"blobs", len(toUpload),
				"bytes", done,
				"duration", elapsed.Round(time.Millisecond),
				"mb_per_sec", fmt.Sprintf("%.1f", mbps),
				"max_transfers", max(1, opts.BodyConcurrency),
			)
			if err != nil {
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
			// Longer backoff for uploads — server-side rate limiting and
			// upload-session bookkeeping need real breathing room.
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

	ep, err := u.initUpload(ctx, blob)
	if err != nil {
		return 0, err
	}
	if ep.sessionURL == "" {
		// Server matched ?digest= against existing storage; nothing to
		// upload. Credit the full size to progress so a retry-after-failure
		// (where the prior attempt streamed bytes that were rolled back)
		// still finishes at 100%.
		u.progress.add(blob.Size)
		return blob.Size, nil
	}

	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if ep.directUploadURL != "" {
		// Body goes straight to the URL the server returned; the server
		// only sees a tiny commit roundtrip.
		return u.putDirect(ctx, ep, f, blob)
	}

	// Body goes to the server in parts via PATCH, followed by a finalize PUT.
	return u.putChunked(ctx, ep.sessionURL, f, blob)
}

func (u *uploader) exists(ctx context.Context, blob Blob) (bool, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodHead, fmt.Sprintf("%s/v2/%s/blobs/%s", u.baseURL, u.repository, blob.Digest), nil)
	req.Header.Set("User-Agent", u.userAgent)
	prev := u.authToken()
	if prev != "" {
		req.Header.Set("Authorization", "Bearer "+prev)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return false, err
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if err := u.refreshToken(ctx, ch, prev); err != nil {
			return false, err
		}
		return u.exists(ctx, blob)
	}

	return resp.StatusCode == http.StatusOK, nil
}

const maxInitRetries = 12

// uploadEndpoint describes where a blob's body should be uploaded after init.
//
// A zero-valued endpoint (sessionURL == "") means the server already has the
// blob and the caller should skip upload.
//
// When sessionURL is set but directUploadURL is empty, the body goes to the
// server in parts via PATCH, then commits with a finalize PUT.
//
// When directUploadURL is set, the body is PUT directly to that URL with any
// signedHeaders the server provided echoed back as request headers. A
// bodyless commit PUT to sessionURL?digest=... then records the blob.
type uploadEndpoint struct {
	sessionURL      string
	directUploadURL string
	signedHeaders   http.Header // headers the server provided that the client must echo on the direct PUT
}

// initUpload announces the upload to the server and discovers which flow to
// use. The server may return a direct-upload URL alongside the session URL;
// the caller branches on whether one came back.
func (u *uploader) initUpload(ctx context.Context, blob Blob) (uploadEndpoint, error) {
	endpoint, _ := url.Parse(fmt.Sprintf("%s/v2/%s/blobs/uploads/", u.baseURL, u.repository))
	q := endpoint.Query()
	q.Set("digest", blob.Digest)
	endpoint.RawQuery = q.Encode()

	var lastErr error
	for attempt := range maxInitRetries {
		if attempt > 0 {
			if err := backoff(ctx, attempt, min(5*time.Second<<uint(attempt-1), 30*time.Second)); err != nil {
				return uploadEndpoint{}, err
			}
			logutil.Trace("retrying init upload", "digest", blob.Digest, "attempt", attempt+1, "error", lastErr)
		}

		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), nil)
		req.Header.Set("User-Agent", u.userAgent)
		req.Header.Set("X-Redirect-Uploads", "2")
		req.Header.Set("X-Content-Length", fmt.Sprintf("%d", blob.Size))
		prev := u.authToken()
		if prev != "" {
			req.Header.Set("Authorization", "Bearer "+prev)
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
			if err := u.refreshToken(ctx, ch, prev); err != nil {
				return uploadEndpoint{}, err
			}
			continue
		}

		if resp.StatusCode == http.StatusCreated {
			// Server matched our ?digest= against existing storage —
			// nothing to upload.
			return uploadEndpoint{}, nil
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

		sessionURL, _ := url.Parse(loc)
		if !sessionURL.IsAbs() {
			base, _ := url.Parse(u.baseURL)
			sessionURL = base.ResolveReference(sessionURL)
		}

		ep := uploadEndpoint{sessionURL: sessionURL.String()}

		// Opt-in direct-upload path: enabled only when the server returns an
		// upload URL. Any X-Signed-Header-<name> response headers must be
		// echoed back on the direct PUT under <name> — the client doesn't
		// need to know which headers, just to forward whatever was signed.
		if directURL := resp.Header.Get("X-Direct-Upload-URL"); directURL != "" {
			// Validate it parses and is absolute, but keep the original
			// string. url.Parse + String() round-trips with normalization
			// (percent-encoding case, query ordering) which can change the
			// canonical form a signed URL was computed over.
			if d, err := url.Parse(directURL); err == nil && d.IsAbs() {
				ep.directUploadURL = directURL
				ep.signedHeaders = make(http.Header)
				const signedPrefix = "X-Signed-Header-"
				for k, vs := range resp.Header {
					name, ok := strings.CutPrefix(k, signedPrefix)
					if !ok {
						continue
					}
					for _, v := range vs {
						ep.signedHeaders.Add(name, v)
					}
				}
			}
		}
		return ep, nil
	}
	return uploadEndpoint{}, lastErr
}

// putDirect PUTs the blob body to the URL the server returned, echoing any
// signed headers it provided. The follow-up commit PUT records the blob on
// the server side with no body.
func (u *uploader) putDirect(ctx context.Context, ep uploadEndpoint, f *os.File, blob Blob) (int64, error) {
	pr, err := u.streamPutBody(ctx, ep, f, blob)
	if err != nil {
		return pr.bytes(), err
	}
	// Body slot is released; commit is bookkeeping (no body) and shouldn't
	// hold the cap from other body uploads.
	if err := u.commit(ctx, ep.sessionURL, blob.Digest); err != nil {
		return pr.bytes(), err
	}
	return pr.bytes(), nil
}

// streamPutBody PUTs the blob body to the server-supplied URL, holding a
// body-transfer slot only for the duration of the body PUT (not the
// follow-up commit). Returns the progressReader so the caller can report
// pr.n on commit failure.
func (u *uploader) streamPutBody(ctx context.Context, ep uploadEndpoint, f *os.File, blob Blob) (*progressReader, error) {
	release, err := u.holdBody(ctx)
	if err != nil {
		return &progressReader{}, err
	}
	defer release()

	br := bufio.NewReaderSize(f, 256*1024)
	pr := &progressReader{reader: br, tracker: u.progress}

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, ep.directUploadURL, pr)
	req.ContentLength = blob.Size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", u.userAgent)
	// Echo signed headers — overwrite any defaults we set above so the
	// signed value wins. Appending would leave duplicates that change the
	// signature canonical form and the upload would be rejected.
	maps.Copy(req.Header, ep.signedHeaders)
	// No Authorization — the direct-upload URL carries its own credential.

	resp, err := u.client.Do(req)
	if err != nil {
		return pr, fmt.Errorf("direct put: %w", err)
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return pr, fmt.Errorf("direct put: status %d: %s", resp.StatusCode, body)
	}
	return pr, nil
}

// commit sends a bodyless PUT to the session URL with ?digest= so the server
// records a blob whose body was uploaded out-of-band.
func (u *uploader) commit(ctx context.Context, sessionURL, digest string) error {
	finalURL, err := url.Parse(sessionURL)
	if err != nil {
		return fmt.Errorf("parse session URL: %w", err)
	}
	q := finalURL.Query()
	q.Set("digest", digest)
	finalURL.RawQuery = q.Encode()

	return u.bodylessRegistryPUT(ctx, finalURL.String(), "commit")
}

// bodylessRegistryPUT sends a zero-body PUT to a registry URL, retrying with
// backoff on transport/server errors and once on auth challenge. op is used
// as the error prefix.
func (u *uploader) bodylessRegistryPUT(ctx context.Context, url string, op string) error {
	var lastErr error
	for try := range maxRetries {
		if try > 0 {
			if err := backoff(ctx, try, 2*time.Second<<uint(try-1)); err != nil {
				return err
			}
		}

		req, _ := http.NewRequestWithContext(ctx, http.MethodPut, url, nil)
		req.ContentLength = 0
		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set("User-Agent", u.userAgent)
		prev := u.authToken()
		if prev != "" {
			req.Header.Set("Authorization", "Bearer "+prev)
		}

		resp, err := u.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized && u.getToken != nil:
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
			if err := u.refreshToken(ctx, ch, prev); err != nil {
				return err
			}
		case resp.StatusCode == http.StatusCreated || resp.StatusCode == http.StatusOK:
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			return nil
		default:
			// Capture body for the error message before closing.
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			resp.Body.Close()
			lastErr = fmt.Errorf("%s: status %d: %s", op, resp.StatusCode, body)
		}
	}
	return fmt.Errorf("%w: %v", errMaxRetriesExceeded, lastErr)
}

// putChunked is the fallback used when the server doesn't return a
// direct-upload URL. It splits the blob into parts and sends each via
// PATCH with a Content-Range, following any redirect on the response,
// then finalizes with a composite-MD5 etag PUT.
//
// On failure the function rolls back any progress it accumulated for this
// blob and returns 0 bytes written, so the outer per-blob retry can start
// from a clean state.
func (u *uploader) putChunked(ctx context.Context, uploadURL string, f *os.File, blob Blob) (int64, error) {
	if uploadURL == "" {
		return 0, nil
	}

	splitParts := computeParts
	if u.makeParts != nil {
		splitParts = u.makeParts
	}
	parts := splitParts(blob.Size)

	current, err := url.Parse(uploadURL)
	if err != nil {
		return 0, fmt.Errorf("parse upload URL: %w", err)
	}

	composite := md5.New()
	var written int64

	for i := range parts {
		part := &parts[i]
		next, partHash, err := u.uploadOnePartWithRetry(ctx, current, part, f)
		if err != nil {
			u.progress.add(-written)
			return 0, err
		}
		composite.Write(partHash)
		written += part.size
		if next != nil {
			current = next
		}
	}

	q := current.Query()
	q.Set("digest", blob.Digest)
	q.Set("etag", fmt.Sprintf("%x-%d", composite.Sum(nil), len(parts)))
	current.RawQuery = q.Encode()
	if err := u.bodylessRegistryPUT(ctx, current.String(), "finalize"); err != nil {
		u.progress.add(-written)
		return 0, err
	}
	return written, nil
}

// uploadOnePartWithRetry sends a single part with up to maxPartRetries
// attempts; rolls back per-attempt progress on transient failures so the
// progress tracker stays consistent.
func (u *uploader) uploadOnePartWithRetry(ctx context.Context, sessionURL *url.URL, part *uploadPart, f *os.File) (*url.URL, []byte, error) {
	const maxPartRetries = 3
	var lastErr error
	for try := range maxPartRetries {
		if try > 0 {
			if err := backoff(ctx, try, 2*time.Second<<uint(try-1)); err != nil {
				return nil, nil, err
			}
		}
		next, partHash, n, err := u.uploadOnePart(ctx, sessionURL, part, f)
		if err == nil {
			return next, partHash, nil
		}
		// Roll back this attempt's progress so retries don't double-count.
		u.progress.add(-n)
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, nil, err
		}
		lastErr = err
	}
	return nil, nil, fmt.Errorf("part %d: %w", part.n, lastErr)
}

// uploadOnePart sends one PATCH for a single part and returns the next
// session URL, the part's MD5 sum, the bytes written, and any error. If the
// server replies 307, the body is re-uploaded to the redirect URL via a
// follow-up PUT; the next session URL still comes from the 307 response.
func (u *uploader) uploadOnePart(ctx context.Context, sessionURL *url.URL, part *uploadPart, f *os.File) (*url.URL, []byte, int64, error) {
	// Hold the body slot across both the PATCH and any subsequent CDN PUT —
	// both transfer body bytes and shouldn't double-count against the cap.
	release, err := u.holdBody(ctx)
	if err != nil {
		return nil, nil, 0, err
	}
	defer release()

	sr := io.NewSectionReader(f, part.offset, part.size)
	br := bufio.NewReaderSize(sr, 256*1024)
	partHash := md5.New()
	pr := &progressReader{reader: br, tracker: u.progress}
	body := io.TeeReader(pr, partHash)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPatch, sessionURL.String(), body)
	req.ContentLength = part.size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Content-Range", fmt.Sprintf("%d-%d", part.offset, part.offset+part.size-1))
	req.Header.Set("X-Redirect-Uploads", "1")
	req.Header.Set("User-Agent", u.userAgent)
	prev := u.authToken()
	if prev != "" {
		req.Header.Set("Authorization", "Bearer "+prev)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return nil, nil, pr.bytes(), fmt.Errorf("patch part %d: %w", part.n, err)
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	// The server may return either an absolute or a relative URL in
	// Location / Docker-Upload-Location; resolve relative ones against the
	// request URL.
	loc := resp.Header.Get("Docker-Upload-Location")
	if loc == "" {
		loc = resp.Header.Get("Location")
	}
	var next *url.URL
	if loc != "" {
		next, _ = url.Parse(loc)
		if next != nil && !next.IsAbs() {
			next = sessionURL.ResolveReference(next)
		}
	}

	switch {
	case resp.StatusCode == http.StatusTemporaryRedirect:
		redirectURL, _ := resp.Location()
		if redirectURL == nil {
			return nil, nil, pr.bytes(), fmt.Errorf("patch part %d: 307 without Location", part.n)
		}
		// The PATCH attempt's progress is wasted — we re-upload to CDN.
		// We can't safely Reset partHash here: the http transport's
		// writeLoop may still be feeding TeeReader bytes into it, so
		// abandon it and let putPartToCDN compute a fresh hash from the
		// bytes that actually land on the storage backend.
		u.progress.add(-pr.bytes())
		cdnSum, cdnN, err := u.putPartToCDN(ctx, redirectURL, part, f)
		if err != nil {
			return nil, nil, cdnN, err
		}
		return next, cdnSum, cdnN, nil

	case resp.StatusCode == http.StatusUnauthorized && u.getToken != nil:
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if err := u.refreshToken(ctx, ch, prev); err != nil {
			return nil, nil, pr.bytes(), err
		}
		return nil, nil, pr.bytes(), fmt.Errorf("patch part %d: auth retry", part.n)

	case resp.StatusCode >= http.StatusBadRequest:
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, pr.bytes(), fmt.Errorf("patch part %d: status %d: %s", part.n, resp.StatusCode, body)
	}

	if next == nil {
		return nil, nil, pr.bytes(), fmt.Errorf("patch part %d: no next URL in response", part.n)
	}
	return next, partHash.Sum(nil), pr.bytes(), nil
}

// putPartToCDN re-uploads a part's data to a CDN redirect URL via PUT.
// Returns the md5 sum of bytes actually streamed to the CDN, the byte count,
// and any error. The hash is fed inline so the composite etag we eventually
// send to the registry reflects what the storage backend stored, not what the
// client tried to PATCH.
func (u *uploader) putPartToCDN(ctx context.Context, cdnURL *url.URL, part *uploadPart, f *os.File) ([]byte, int64, error) {
	sr := io.NewSectionReader(f, part.offset, part.size)
	br := bufio.NewReaderSize(sr, 256*1024)
	pr := &progressReader{reader: br, tracker: u.progress}
	partHash := md5.New()
	body := io.TeeReader(pr, partHash)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, cdnURL.String(), body)
	req.ContentLength = part.size
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", u.userAgent)
	// No Authorization — the redirect URL carries its own credential.

	resp, err := u.client.Do(req)
	if err != nil {
		return nil, pr.bytes(), fmt.Errorf("cdn put part %d: %w", part.n, err)
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, pr.bytes(), fmt.Errorf("cdn put part %d: status %d: %s", part.n, resp.StatusCode, body)
	}
	return partHash.Sum(nil), pr.bytes(), nil
}

// Chunked-upload sizing — when computeParts splits a blob into parts for the
// multipart fallback, parts are sized in [minUploadPartSize, maxUploadPartSize]
// with a target count of numUploadParts. Smaller blobs end up as a single
// sub-minimum part.
const (
	numUploadParts          = 16
	minUploadPartSize int64 = 100 << 20  // 100 MB
	maxUploadPartSize int64 = 1000 << 20 // ~1 GB
)

// uploadPart represents a chunk of a blob for the multipart fallback.
type uploadPart struct {
	n      int
	offset int64
	size   int64
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

func (u *uploader) pushManifest(ctx context.Context, repo, ref string, manifest []byte) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, fmt.Sprintf("%s/v2/%s/manifests/%s", u.baseURL, repo, ref), bytes.NewReader(manifest))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	req.Header.Set("User-Agent", u.userAgent)
	prev := u.authToken()
	if prev != "" {
		req.Header.Set("Authorization", "Bearer "+prev)
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()

	if resp.StatusCode == http.StatusUnauthorized && u.getToken != nil {
		ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
		if err := u.refreshToken(ctx, ch, prev); err != nil {
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

// progressReader counts bytes streamed through Read. The byte counter is
// atomic because the HTTP transport's writeLoop runs concurrently with the
// goroutine that returns the count after a non-2xx response — the transport
// may still be calling Read while we're already returning from uploadOnePart.
type progressReader struct {
	reader  io.Reader
	tracker *progressTracker
	n       atomic.Int64
}

func (r *progressReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if n > 0 {
		r.n.Add(int64(n))
		r.tracker.add(int64(n))
	}
	return n, err
}

func (r *progressReader) bytes() int64 {
	return r.n.Load()
}
