package usage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ollama/ollama/version"
)

const (
	reportTimeout = 10 * time.Second
	usageURL      = "https://ollama.com/api/usage"
)

// HeartbeatResponse is the response from the heartbeat endpoint.
type HeartbeatResponse struct {
	UpdateVersion string `json:"update_version,omitempty"`
}

// UpdateAvailable returns the available update version, if any.
func (t *Stats) UpdateAvailable() string {
	if v := t.updateAvailable.Load(); v != nil {
		return v.(string)
	}
	return ""
}

// sendHeartbeat sends usage stats and checks for updates.
func (t *Stats) sendHeartbeat(payload *Payload) {
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), reportTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, usageURL, bytes.NewReader(data))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s", version.Version))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return
	}

	var heartbeat HeartbeatResponse
	if err := json.NewDecoder(resp.Body).Decode(&heartbeat); err != nil {
		return
	}

	t.updateAvailable.Store(heartbeat.UpdateVersion)
}
