package lifecycle

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/version"
)

var (
	UpdateCheckURLBase  = "https://ollama.com/api/update"
	UpdateDownloaded    = false
	UpdateCheckInterval = 60 * 60 * time.Second
)

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	UpdateURL     string `json:"url"`
	UpdateVersion string `json:"version"`
}

func IsNewReleaseAvailable(ctx context.Context) (bool, UpdateResponse) {
	var updateResp UpdateResponse

	requestURL, err := url.Parse(UpdateCheckURLBase)
	if err != nil {
		return false, updateResp
	}

	query := requestURL.Query()
	query.Add("os", runtime.GOOS)
	query.Add("arch", runtime.GOARCH)
	query.Add("version", version.Version)
	query.Add("ts", fmt.Sprintf("%d", time.Now().Unix()))

	nonce, err := auth.NewNonce(rand.Reader, 16)
	if err != nil {
		return false, updateResp
	}

	query.Add("nonce", nonce)
	requestURL.RawQuery = query.Encode()

	data := []byte(fmt.Sprintf("%s,%s", http.MethodGet, requestURL.RequestURI()))
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return false, updateResp
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	req.Header.Set("Authorization", signature)
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	slog.Info("checking for available update", "requestURL", requestURL)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	defer resp.Body.Close()

	if resp.StatusCode == 204 {
		slog.Debug("check update response 204 (current version is up to date)")
		return false, updateResp
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to read body response: %s", err))
	}

	if resp.StatusCode != 200 {
		slog.Info(fmt.Sprintf("check update error %d - %.96s", resp.StatusCode, string(body)))
		return false, updateResp
	}
	err = json.Unmarshal(body, &updateResp)
	if err != nil {
		slog.Warn(fmt.Sprintf("malformed response checking for update: %s", err))
		return false, updateResp
	}
	// Extract the version string from the URL in the github release artifact path
	updateResp.UpdateVersion = path.Base(path.Dir(updateResp.UpdateURL))

	slog.Info("New update available at " + updateResp.UpdateURL)
	return true, updateResp
}

func DownloadNewRelease(ctx context.Context, updateResp UpdateResponse) error {
	etagfilename := filepath.Join(UpdateStageDir, updateResp.UpdateVersion, "etag")

	// parse the base of the url for the filename
	u, err := url.Parse(updateResp.UpdateURL)
	if err != nil {
		return fmt.Errorf("could not parse update url: %w", err)
	}

	var etag string
	filename := filepath.Join(UpdateStageDir, updateResp.UpdateVersion, filepath.Base(u.Path))
	_, err = os.Stat(filename)
	if err == nil {
		file, err := os.Open(etagfilename)
		if err == nil {
			content, err := io.ReadAll(file)
			if err != nil {
				return fmt.Errorf("could not read etag file: %w", err)
			}

			etag = string(content)
			file.Close()
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, updateResp.UpdateURL, nil)
	if err != nil {
		return err
	}

	if etag != "" {
		req.Header.Set("If-None-Match", "\""+etag+"\"")
	}

	slog.Info("sending update request", "url", req.URL.String())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("error checking update: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 304 {
		slog.Info("update already downloaded")
		return nil
	}

	slog.Info("got status", "status", resp.StatusCode)
	if resp.StatusCode != 200 {
		return fmt.Errorf("unexpected status attempting to download update %d", resp.StatusCode)
	}

	cleanupOldDownloads()

	// Create the directory for the update
	_, err = os.Stat(filepath.Dir(filename))
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(filename), 0o755); err != nil {
			return fmt.Errorf("create update dir %s: %v", filepath.Dir(filename), err)
		}
	}

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read body response: %w", err)
	}
	fp, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return fmt.Errorf("write payload %s: %w", filename, err)
	}
	defer fp.Close()
	if n, err := fp.Write(payload); err != nil || n != len(payload) {
		return fmt.Errorf("write payload %s: %d vs %d -- %w", filename, n, len(payload), err)
	}

	etag = strings.Trim(resp.Header.Get("etag"), "\"")
	if etag != "" {
		file, err := os.OpenFile(etagfilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("could not open etag file for writing: %w", err)
		}

		if _, err := file.Write([]byte(etag)); err != nil {
			return fmt.Errorf("could not write etag file: %w", err)
		}

		file.Close()
	}

	slog.Info("new update downloaded " + filename)

	UpdateDownloaded = true

	return nil
}

func cleanupOldDownloads() {
	files, err := os.ReadDir(UpdateStageDir)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		// Expected behavior on first run
		return
	} else if err != nil {
		slog.Warn(fmt.Sprintf("failed to list stage dir: %s", err))
		return
	}
	for _, file := range files {
		fullname := filepath.Join(UpdateStageDir, file.Name())
		slog.Debug("cleaning up old download: " + fullname)
		err = os.RemoveAll(fullname)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to cleanup stale update download %s", err))
		}
	}
}

func StartBackgroundUpdaterChecker(ctx context.Context, cb func(string) error) {
	go func() {
		// Don't blast an update message immediately after startup
		// time.Sleep(30 * time.Second)
		time.Sleep(3 * time.Second)

		for {
			available, resp := IsNewReleaseAvailable(ctx)
			if available {
				err := DownloadNewRelease(ctx, resp)
				if err != nil {
					slog.Error(fmt.Sprintf("failed to download new release: %s", err))
				}
				err = cb(resp.UpdateVersion)
				if err != nil {
					slog.Warn(fmt.Sprintf("failed to register update available with tray: %s", err))
				}
			}
			select {
			case <-ctx.Done():
				slog.Debug("stopping background update checker")
				return
			default:
				time.Sleep(UpdateCheckInterval)
			}
		}
	}()
}
