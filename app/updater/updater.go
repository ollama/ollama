//go:build windows || darwin

package updater

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/auth"
)

var (
	UpdateCheckURLBase      = "https://ollama.com/api/update"
	UpdateDownloaded        = false
	UpdateCheckInterval     = 60 * 60 * time.Second
	UpdateCheckInitialDelay = 3 * time.Second // 30 * time.Second

	UpdateStageDir    string
	UpgradeLogFile    string
	UpgradeMarkerFile string
	Installer         string
	UserAgentOS       string

	VerifyDownload func() error
)

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	UpdateURL     string `json:"url"`
	UpdateVersion string `json:"version"`
}

func (u *Updater) checkForUpdate(ctx context.Context) (bool, UpdateResponse) {
	var updateResp UpdateResponse

	requestURL, err := url.Parse(UpdateCheckURLBase)
	if err != nil {
		return false, updateResp
	}

	query := requestURL.Query()
	query.Add("os", runtime.GOOS)
	query.Add("arch", runtime.GOARCH)
	currentVersion := version.GetVersion()
	query.Add("version", currentVersion)
	query.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))
	
	slog.Debug("checking for update", "current_version", currentVersion, "os", runtime.GOOS, "arch", runtime.GOARCH)

	// The original macOS app used to use the device ID
	// to check for updates so include it if present
	if runtime.GOOS == "darwin" {
		if id, err := u.Store.ID(); err == nil && id != "" {
			query.Add("id", id)
		}
	}

	var signature string

	nonce, err := auth.NewNonce(rand.Reader, 16)
	if err != nil {
		// Don't sign if we haven't yet generated a key pair for the server
		slog.Debug("unable to generate nonce for update check request", "error", err)
	} else {
		query.Add("nonce", nonce)
		requestURL.RawQuery = query.Encode()

		data := []byte(fmt.Sprintf("%s,%s", http.MethodGet, requestURL.RequestURI()))
		signature, err = auth.Sign(ctx, data)
		if err != nil {
			slog.Debug("unable to generate signature for update check request", "error", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	if signature != "" {
		req.Header.Set("Authorization", signature)
	}
	ua := fmt.Sprintf("ollama/%s %s Go/%s %s", version.GetVersion(), runtime.GOARCH, runtime.Version(), UserAgentOS)
	req.Header.Set("User-Agent", ua)

	slog.Debug("checking for available update", "requestURL", requestURL, "User-Agent", ua)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		slog.Debug("check update response 204 (current version is up to date)")
		return false, updateResp
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to read body response: %s", err))
	}

	if resp.StatusCode != http.StatusOK {
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

func (u *Updater) DownloadNewRelease(ctx context.Context, updateResp UpdateResponse) error {
	// Do a head first to check etag info
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, updateResp.UpdateURL, nil)
	if err != nil {
		return err
	}

	// In case of slow downloads, continue the update check in the background
	bgctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		for {
			select {
			case <-bgctx.Done():
				return
			case <-time.After(UpdateCheckInterval):
				u.checkForUpdate(bgctx)
			}
		}
	}()

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("error checking update: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
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
		slog.Info("update already downloaded", "bundle", stageFilename)
		UpdateDownloaded = true
		return nil
	}

	cleanupOldDownloads(UpdateStageDir)

	req.Method = http.MethodGet
	resp, err = http.DefaultClient.Do(req)
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

	if err := VerifyDownload(); err != nil {
		_ = os.Remove(stageFilename)
		return fmt.Errorf("%s - %s", resp.Request.URL.String(), err)
	}
	UpdateDownloaded = true
	return nil
}

func cleanupOldDownloads(stageDir string) {
	files, err := os.ReadDir(stageDir)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		// Expected behavior on first run
		return
	} else if err != nil {
		slog.Warn(fmt.Sprintf("failed to list stage dir: %s", err))
		return
	}
	for _, file := range files {
		fullname := filepath.Join(stageDir, file.Name())
		slog.Debug("cleaning up old download: " + fullname)
		err = os.RemoveAll(fullname)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to cleanup stale update download %s", err))
		}
	}
}

type Updater struct {
	Store *store.Store
}

func (u *Updater) StartBackgroundUpdaterChecker(ctx context.Context, cb func(string) error) {
	go func() {
		// Don't blast an update message immediately after startup
		time.Sleep(UpdateCheckInitialDelay)
		slog.Info("beginning update checker", "interval", UpdateCheckInterval)
		for {
			// Check if auto-update is enabled
			settings, err := u.Store.Settings()
			if err != nil {
				slog.Error("failed to load settings", "error", err)
				time.Sleep(UpdateCheckInterval)
				continue
			}
			
			if !settings.AutoUpdateEnabled {
				// When auto-update is disabled, don't check or download anything
				slog.Debug("auto-update disabled, skipping check")
				time.Sleep(UpdateCheckInterval)
				continue
			}

			// Auto-update is enabled - proceed with normal flow
			available, resp := u.checkForUpdate(ctx)
			if available {
				err := u.DownloadNewRelease(ctx, resp)
				if err != nil {
					slog.Error(fmt.Sprintf("failed to download new release: %s", err))
				} else {
					err = cb(resp.UpdateVersion)
					if err != nil {
						slog.Warn(fmt.Sprintf("failed to register update available with tray: %s", err))
					}
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

func (u *Updater) CheckForUpdate(ctx context.Context) (bool, string, error) {
	available, resp := u.checkForUpdate(ctx)
	return available, resp.UpdateVersion, nil
}

func (u *Updater) DownloadUpdate(ctx context.Context, updateVersion string) error {
	// First check for update to get the actual download URL
	available, updateResp := u.checkForUpdate(ctx)
	
	if !available {
		return fmt.Errorf("no update available")
	}
	if updateResp.UpdateVersion != updateVersion {
		slog.Error("version mismatch", "requested", updateVersion, "available", updateResp.UpdateVersion)
		return fmt.Errorf("version mismatch: requested %s, available %s", updateVersion, updateResp.UpdateVersion)
	}
	
	slog.Info("downloading update", "version", updateVersion)
	err := u.DownloadNewRelease(ctx, updateResp)
	if err != nil {
		return err
	}
	
	slog.Info("update downloaded successfully", "version", updateVersion)
	return nil
}

func (u *Updater) InstallAndRestart() error {
	if !UpdateDownloaded {
		return fmt.Errorf("no update downloaded")
	}
	
	slog.Info("installing update and restarting")
	return DoUpgrade(true)
}
