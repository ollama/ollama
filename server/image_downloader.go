package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// ImageDownloader handles downloading and caching of external images.
type ImageDownloader struct {
	client  *http.Client
	maxSize int64
}

// NewImageDownloader creates a new image downloader with default settings.
func NewImageDownloader() *ImageDownloader {
	return &ImageDownloader{
		client: &http.Client{
			Timeout:   time.Duration(envconfig.ImageURLTimeout()) * time.Second,
			Transport: &http.Transport{Proxy: http.ProxyFromEnvironment},
		},
		maxSize: envconfig.ImageURLMaxSize(),
	}
}

// DownloadImage downloads an image from a URL and returns the image data.
// It validates the URL, checks size limits, and caches the result.
func (d *ImageDownloader) DownloadImage(ctx context.Context, imageURL api.ImageURL) (api.ImageData, error) {
	// Parse and validate URL
	u, err := url.Parse(imageURL.URL)
	if err != nil {
		return nil, fmt.Errorf("invalid URL %q: %w", imageURL.URL, err)
	}

	// Validate scheme
	if u.Scheme != "https" && (u.Scheme != "http" || !imageURL.AllowHTTP) {
		return nil, fmt.Errorf("unsupported URL scheme %q: use https or set allow_http=true", u.Scheme)
	}

	// Check host whitelist if configured
	if allowed := envconfig.ImageURLAllowedHosts(); len(allowed) > 0 && !containsHost(allowed, u.Host) {
		return nil, fmt.Errorf("host %q not in allowed list", u.Host)
	}

	// Check cache first
	cacheKey := generateCacheKey(imageURL.URL)
	if data := getCachedImage(cacheKey); data != nil {
		return data, nil
	}

	// Download with context timeout
	ctx, cancel := context.WithTimeout(ctx, d.client.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download failed: status %d %s", resp.StatusCode, resp.Status)
	}

	// Validate content type
	contentType := resp.Header.Get("Content-Type")
	if !isValidImageType(contentType) {
		return nil, fmt.Errorf("invalid content type %q", contentType)
	}

	// Read with size limit
	limitedReader := io.LimitReader(resp.Body, d.maxSize+1)
	data, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if int64(len(data)) > d.maxSize {
		return nil, fmt.Errorf("image size %d exceeds max %d", len(data), d.maxSize)
	}

	// Save to cache
	saveCachedImage(cacheKey, data)

	return data, nil
}

// DownloadImages downloads multiple images and returns them as ImageData slice.
func (d *ImageDownloader) DownloadImages(ctx context.Context, urls []api.ImageURL) ([]api.ImageData, error) {
	if !envconfig.ImageURLEnabled() {
		return nil, fmt.Errorf("image URL feature is disabled")
	}

	result := make([]api.ImageData, 0, len(urls))
	for _, u := range urls {
		data, err := d.DownloadImage(ctx, u)
		if err != nil {
			return nil, err
		}
		result = append(result, data)
	}
	return result, nil
}

// isValidImageType checks if the content type is a valid image type.
func isValidImageType(contentType string) bool {
	validTypes := []string{
		"image/jpeg",
		"image/jpg",
		"image/png",
		"image/gif",
		"image/webp",
		"image/bmp",
		"image/tiff",
	}
	for _, t := range validTypes {
		if strings.HasPrefix(contentType, t) {
			return true
		}
	}
	// Allow empty content type (some servers don't set it correctly)
	return contentType == ""
}

// generateCacheKey creates a cache key from URL.
func generateCacheKey(url string) string {
	h := sha256.Sum256([]byte(url))
	return hex.EncodeToString(h[:16])
}

// getCachedImage retrieves cached image data.
func getCachedImage(key string) api.ImageData {
	cacheDir := envconfig.ImageURLCacheDir()
	if cacheDir == "" {
		return nil
	}

	path := filepath.Join(cacheDir, key)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return data
}

// saveCachedImage saves image data to cache.
func saveCachedImage(key string, data api.ImageData) {
	cacheDir := envconfig.ImageURLCacheDir()
	if cacheDir == "" {
		return
	}

	// Ensure cache directory exists
	if err := os.MkdirAll(cacheDir, 0750); err != nil {
		return
	}

	path := filepath.Join(cacheDir, key)
	_ = os.WriteFile(path, data, 0640)
}

// containsHost checks if host is in the allowed list.
func containsHost(allowed []string, host string) bool {
	for _, h := range allowed {
		// Support wildcards like *.example.com
		if strings.HasPrefix(h, "*.") {
			suffix := h[1:] // Remove the leading *
			if strings.HasSuffix(host, suffix) {
				return true
			}
		} else if h == host {
			return true
		}
	}
	return false
}
