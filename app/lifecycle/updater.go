package lifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/jmorganca/ollama/auth"
	"github.com/jmorganca/ollama/version"
)

var (
	UpdateCheckURLBase = "https://ollama.ai/api/update"
	UpdateDownloaded   = false
)

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	UpdateURL     string `json:"url"`
	UpdateVersion string `json:"version"`
}

func getClient(req *http.Request) http.Client {
	proxyURL, err := http.ProxyFromEnvironment(req)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to handle proxy: %s", err))
		return http.Client{}
	}

	return http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}
}

func IsNewReleaseAvailable(ctx context.Context) (bool, UpdateResponse) {
	var updateResp UpdateResponse
	updateCheckURL := UpdateCheckURLBase + "?os=" + runtime.GOOS + "&arch=" + runtime.GOARCH + "&version=" + version.Version
	headers := make(http.Header)
	err := auth.SignRequest(http.MethodGet, updateCheckURL, nil, headers)
	if err != nil {
		slog.Info(fmt.Sprintf("failed to sign update request %s", err))
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, updateCheckURL, nil)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	req.Header = headers
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))
	client := getClient(req)

	slog.Debug(fmt.Sprintf("checking for available update at %s with headers %v", updateCheckURL, headers))
	resp, err := client.Do(req)
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

// Returns true if we downloaded a new update, false if we already had it
func DownloadNewRelease(ctx context.Context, updateResp UpdateResponse) error {
	// Do a head first to check etag info
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, updateResp.UpdateURL, nil)
	if err != nil {
		return err
	}
	client := getClient(req)
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("error checking update: %w", err)
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("unexpected status attempting to download update %d", resp.StatusCode)
	}
	resp.Body.Close()
	etag := strings.Trim(resp.Header.Get("etag"), "\"")
	if etag == "" {
		slog.Debug("no etag detected, falling back to filename based dedup")
		etag = "_"
	}
	filename := Installer
	_, params, err := mime.ParseMediaType(resp.Header.Get("content-disposition"))
	if err == nil {
		filename = params["filename"]
	}

	stageFilename := filepath.Join(UpdateStageDir, etag, filename)

	// Check to see if we already have it downloaded
	_, err = os.Stat(stageFilename)
	if err == nil {
		slog.Debug("update already downloaded")
		return nil
	}

	cleanupOldDownloads()

	req.Method = http.MethodGet
	resp, err = client.Do(req)
	if err != nil {
		return fmt.Errorf("error checking update: %w", err)
	}
	defer resp.Body.Close()
	etag = strings.Trim(resp.Header.Get("etag"), "\"")
	if etag == "" {
		slog.Debug("no etag detected, falling back to filename based dedup") // TODO probably can get rid of this redundant log
		etag = "_"
	}

	stageFilename = filepath.Join(UpdateStageDir, etag, filename)

	_, err = os.Stat(filepath.Dir(stageFilename))
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(stageFilename), 0o755); err != nil {
			return fmt.Errorf("create ollama dir %s: %v", filepath.Dir(stageFilename), err)
		}
	}

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read body response: %w", err)
	}
	fp, err := os.OpenFile(stageFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return fmt.Errorf("write payload %s: %w", stageFilename, err)
	}
	defer fp.Close()
	if n, err := fp.Write(payload); err != nil || n != len(payload) {
		return fmt.Errorf("write payload %s: %d vs %d -- %w", stageFilename, n, len(payload), err)
	}
	slog.Info("new update downloaded " + stageFilename)

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
				time.Sleep(60 * 60 * time.Second)
			}
		}
	}()
}
