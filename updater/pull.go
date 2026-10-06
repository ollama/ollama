package updater

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

type progressTrackingReader struct {
	reader     io.Reader
	total      int64
	current    int64
	progressFn func(downloaded, total int64)
}

func (r *progressTrackingReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if n > 0 {
		r.current += int64(n)
		if r.progressFn != nil {
			r.progressFn(r.current, r.total)
		}
	}
	return n, err
}

// Pull downloads the update asset associated with the given release.
func Pull(ctx context.Context, release *ReleaseInfo, opts PullOptions) (*PullResult, error) {
	if release == nil {
		return nil, fmt.Errorf("release information is required")
	}
	asset := release.Asset
	if asset == nil {
		return nil, fmt.Errorf("no suitable download asset found for this platform")
	}

	if !opts.Force && !release.UpdateAvailable {
		return nil, fmt.Errorf("ollama is already up to date (%s)", release.CurrentVersion)
	}

	targetDir := opts.Dir
	if targetDir == "" {
		cacheDir, err := os.UserCacheDir()
		if err != nil {
			cacheDir = os.TempDir()
		}
		targetDir = filepath.Join(cacheDir, "ollama", "updates")
	}

	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create update directory %s: %w", targetDir, err)
	}

	targetPath := filepath.Join(targetDir, asset.Name)

	// Check if already downloaded
	if !opts.Force {
		if fi, err := os.Stat(targetPath); err == nil && fi.Size() > 0 {
			if asset.Size <= 0 || fi.Size() == asset.Size {
				return &PullResult{
					FilePath:       targetPath,
					Filename:       asset.Name,
					Version:        release.LatestVersion,
					Size:           fi.Size(),
					AlreadyExisted: true,
				}, nil
			}
		}
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, asset.DownloadURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create download request: %w", err)
	}
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s", release.CurrentVersion))

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download update: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusNotFound && release.IsRC {
			return nil, fmt.Errorf("download for pre-release %s is not available yet (binaries may still be building or pending release; see %s)", release.LatestVersion, release.ReleaseURL)
		}
		return nil, fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	totalSize := resp.ContentLength
	if totalSize <= 0 && asset.Size > 0 {
		totalSize = asset.Size
	}

	pr := &progressTrackingReader{
		reader:     resp.Body,
		total:      totalSize,
		progressFn: opts.ProgressFn,
	}

	// If a custom writer is provided, stream into it directly
	if opts.Writer != nil {
		n, err := io.Copy(opts.Writer, pr)
		if err != nil {
			return nil, fmt.Errorf("failed to write update: %w", err)
		}
		return &PullResult{
			FilePath: targetPath,
			Filename: asset.Name,
			Version:  release.LatestVersion,
			Size:     n,
		}, nil
	}

	tmpPath := targetPath + ".tmp"
	outFile, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return nil, fmt.Errorf("failed to create temporary file %s: %w", tmpPath, err)
	}

	var copyErr error
	var written int64
	defer func() {
		outFile.Close()
		if copyErr != nil {
			_ = os.Remove(tmpPath)
		}
	}()

	written, copyErr = io.Copy(outFile, pr)
	if copyErr != nil {
		return nil, fmt.Errorf("failed to save update: %w", copyErr)
	}

	if err := outFile.Sync(); err != nil {
		copyErr = err
		return nil, fmt.Errorf("failed to sync update file: %w", err)
	}

	if err := outFile.Close(); err != nil {
		copyErr = err
		return nil, fmt.Errorf("failed to close update file: %w", err)
	}

	// Atomic rename to final path
	if err := os.Rename(tmpPath, targetPath); err != nil {
		copyErr = err
		return nil, fmt.Errorf("failed to finalize update file: %w", err)
	}

	return &PullResult{
		FilePath: targetPath,
		Filename: asset.Name,
		Version:  release.LatestVersion,
		Size:     written,
	}, nil
}
