package lifecycle

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"path"
	"runtime"
	"strconv"
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
	query.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))

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

	slog.Debug("checking for available update", "requestURL", requestURL)
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
