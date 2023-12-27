package lifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/jmorganca/ollama/desktop/store"
	"github.com/jmorganca/ollama/version"
)

var (
	UpdateCheckURLBase = "https://ollama.ai/api/update"
	UpdateDownloaded   = false
)

func GetUpdateCheckURL(id string) string {
	return UpdateCheckURLBase + "?os=" + runtime.GOOS + "&arch=" + runtime.GOARCH + "&version=" + version.Version + "&id=" + id
}

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	UpdateURL     string `json:"url"`
	UpdateVersion string `json:"version"`
}

func IsNewReleaseAvailable() (bool, UpdateResponse) {
	var updateResp UpdateResponse
	updateCheckURL := GetUpdateCheckURL(store.GetID())
	log.Printf("XXX checking for update via %s", updateCheckURL)
	resp, err := http.Get(updateCheckURL)
	if err != nil {
		log.Printf("XXX error checking for update: %s", err)
		return false, updateResp
	}
	defer resp.Body.Close()
	if resp.StatusCode == 204 {
		log.Printf("XXX got 204 when checking for update")
		return false, updateResp
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("XXX failed to read body response: %s", err)
	}
	json.Unmarshal(body, &updateResp)
	log.Printf("XXX New update available at %s", updateResp.UpdateURL)
	return true, updateResp
}

func DownloadNewRelease(updateResp UpdateResponse) error {
	updateURL, err := url.Parse(updateResp.UpdateURL)
	if err != nil {
		return fmt.Errorf("failed to parse update URL %s: %w", updateResp.UpdateURL, err)
	}
	escapedFilename := filepath.Join(UpdateStageDir, url.PathEscape(updateURL.Path))
	_, err = os.Stat(UpdateStageDir)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(UpdateStageDir, 0o755); err != nil {
			return fmt.Errorf("create ollama dir %s: %v", UpdateStageDir, err)
		}
	}
	_, err = os.Stat(escapedFilename)
	if errors.Is(err, os.ErrNotExist) {
		log.Printf("XXX downloading %s", updateResp.UpdateURL)
		resp, err := http.Get(updateResp.UpdateURL)
		if err != nil {
			return fmt.Errorf("error downloading update: %w", err)
		}
		defer resp.Body.Close()
		payload, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("failed to read body response: %w", err)
		}
		fp, err := os.OpenFile(escapedFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
		if err != nil {
			return fmt.Errorf("write payload %s: %w", escapedFilename, err)
		}
		defer fp.Close()
		if n, err := fp.Write(payload); err != nil || n != len(payload) {
			return fmt.Errorf("write payload %s: %d vs %d -- %w", escapedFilename, n, len(payload), err)
		}
		log.Printf("XXX completed writing out update payload to %s", escapedFilename)
	} else if err != nil {
		return fmt.Errorf("XXX unexpected stat error %w", err)
	} else {
		log.Printf("XXX update already downloaded")
	}
	UpdateDownloaded = true
	return nil
}

func StartBackgroundUpdaterChecker(ctx context.Context, cb func(bool)) {
	// TODO - shutdown mechanism
	go func() {
		for {
			available, resp := IsNewReleaseAvailable()
			if available {
				DownloadNewRelease(resp)
				cb(UpdateDownloaded)
			}
			select {
			case <-ctx.Done():
				log.Printf("XXX stopping background update checker")
				return
			default:
				time.Sleep(60 * 60 * time.Second)
			}
		}
	}()
}

func DoUpgrade() {
	log.Printf("XXX Would be performing upgrade magic here...")
	// Start some sort of helper that does the heavy lifting while the main app is allowed to exit

}
