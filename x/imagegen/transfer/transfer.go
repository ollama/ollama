// Package transfer provides minimal, fast blob transfer for tensor-based models.
//
// This package is in x/ because the tensor model storage format is under development.
// It provides optimized transfer for models with many small blobs (tensor models)
// rather than few large blobs (typical LLMs).
//
// TODO (jmorganca): Integrate into server/download.go and server/upload.go when stable.
//
// Design Philosophy:
// This package is intentionally simpler than the main server's download/upload code.
// Key simplifications for many-small-blob workloads:
//
//   - Whole-blob transfers: No part-based chunking. Each blob downloads/uploads as one unit.
//   - No resume: If a transfer fails, it restarts from scratch (fine for small blobs).
//   - Inline hashing: SHA256 computed during streaming, not asynchronously after parts complete.
//   - Stall and speed detection: Cancels on no data (stall) or speed below 10% of median.
//
// For large models (multi-GB), use the server's download/upload code which has:
//   - Part-based transfers with 64MB chunks
//   - Resumable downloads with JSON state files
//   - Async streamHasher that hashes from OS page cache as parts complete
//   - Speed tracking with rolling median to detect and restart slow parts
package transfer

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Blob represents a content-addressed blob to transfer.
type Blob struct {
	Digest string // sha256:...
	Size   int64

	// From enables cross-repository blob mounting (upload only).
	// When set, the upload will first attempt to mount the blob from this source
	// repository instead of uploading the data. This is a Docker Registry v2 API
	// feature that avoids re-uploading blobs that already exist elsewhere.
	//
	// Example: From="library/source-model" will add ?mount=<digest>&from=library/source-model
	// to the POST /blobs/uploads/ request. If the registry returns 201 Created,
	// the blob was mounted successfully and no upload is needed.
	//
	// See: https://distribution.github.io/distribution/spec/api/#cross-repository-blob-mount
	From string
}

// DownloadOptions configures a parallel download operation.
type DownloadOptions struct {
	Blobs        []Blob                                                             // Blobs to download
	BaseURL      string                                                             // Registry base URL
	DestDir      string                                                             // Destination directory for blobs
	Repository   string                                                             // Repository path for blob URLs (e.g., "library/model")
	Concurrency  int                                                                // Max parallel downloads (default 64)
	Progress     func(completed, total int64)                                       // Progress callback (optional)
	Client       *http.Client                                                       // HTTP client (optional, uses default)
	Token        string                                                             // Auth token (optional)
	GetToken     func(ctx context.Context, challenge AuthChallenge) (string, error) // Token refresh callback
	Logger       *slog.Logger                                                       // Optional structured logger
	UserAgent    string                                                             // User-Agent header (optional, has default)
	StallTimeout time.Duration                                                      // Timeout for stall detection (default 10s)
}

// UploadOptions configures a parallel upload operation.
type UploadOptions struct {
	Blobs       []Blob                                                             // Blobs to upload
	BaseURL     string                                                             // Registry base URL
	SrcDir      string                                                             // Source directory containing blobs
	Concurrency int                                                                // Max parallel uploads (default 32)
	Progress    func(completed, total int64)                                       // Progress callback (optional)
	Client      *http.Client                                                       // HTTP client (optional, uses default)
	Token       string                                                             // Auth token (optional)
	GetToken    func(ctx context.Context, challenge AuthChallenge) (string, error) // Token refresh callback
	Logger      *slog.Logger                                                       // Optional structured logger
	UserAgent   string                                                             // User-Agent header (optional, has default)

	// Manifest fields (optional) - if set, manifest is pushed after all blobs complete
	Manifest    []byte // Raw manifest JSON to push
	ManifestRef string // Tag or digest for the manifest (e.g., "latest", "sha256:...")
	Repository  string // Repository path for manifest URL (e.g., "library/model")
}

// AuthChallenge represents a parsed WWW-Authenticate challenge.
type AuthChallenge struct {
	Realm   string
	Service string
	Scope   string
}

// Default concurrency limits and settings
const (
	DefaultDownloadConcurrency = 64
	DefaultUploadConcurrency   = 32
	maxRetries                 = 6
	defaultUserAgent           = "ollama-transfer/1.0"
)

var errMaxRetriesExceeded = errors.New("max retries exceeded")

// defaultClient is a shared HTTP client with connection pooling.
var defaultClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
	},
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	},
}

// progressTracker aggregates progress across concurrent operations.
type progressTracker struct {
	completed atomic.Int64
	total     int64
	callback  func(completed, total int64)
}

func newProgressTracker(total int64, callback func(completed, total int64)) *progressTracker {
	return &progressTracker{
		total:    total,
		callback: callback,
	}
}

func (p *progressTracker) add(n int64) {
	if p == nil || p.callback == nil {
		return
	}
	completed := p.completed.Add(n)
	p.callback(completed, p.total)
}

// Download downloads blobs in parallel with streaming hash verification.
func Download(ctx context.Context, opts DownloadOptions) error {
	return download(ctx, opts)
}

// Upload uploads blobs in parallel.
func Upload(ctx context.Context, opts UploadOptions) error {
	return upload(ctx, opts)
}

// digestToPath converts sha256:abc123 to sha256-abc123
func digestToPath(digest string) string {
	if len(digest) > 7 && digest[6] == ':' {
		return digest[:6] + "-" + digest[7:]
	}
	return digest
}

// parseAuthChallenge parses a WWW-Authenticate header value.
// Example: Bearer realm="https://auth.example.com",service="registry",scope="repository:foo:pull"
func parseAuthChallenge(header string) AuthChallenge {
	header = strings.TrimPrefix(header, "Bearer ")

	getValue := func(key string) string {
		startIdx := strings.Index(header, key+"=")
		if startIdx == -1 {
			return ""
		}
		startIdx += len(key) + 1
		if startIdx >= len(header) {
			return ""
		}

		// Handle quoted values
		if header[startIdx] == '"' {
			startIdx++
			endIdx := strings.Index(header[startIdx:], "\"")
			if endIdx == -1 {
				return header[startIdx:]
			}
			return header[startIdx : startIdx+endIdx]
		}

		// Unquoted value - ends at comma or end of string
		endIdx := strings.Index(header[startIdx:], ",")
		if endIdx == -1 {
			return header[startIdx:]
		}
		return header[startIdx : startIdx+endIdx]
	}

	return AuthChallenge{
		Realm:   getValue("realm"),
		Service: getValue("service"),
		Scope:   getValue("scope"),
	}
}

// backoff returns a function that sleeps with exponential backoff.
func backoff(ctx context.Context, attempt int, maxBackoff time.Duration) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// n^2 backoff with jitter
	d := min(time.Duration(attempt*attempt)*10*time.Millisecond, maxBackoff)
	d = time.Duration(float64(d) * (rand.Float64() + 0.5))

	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
