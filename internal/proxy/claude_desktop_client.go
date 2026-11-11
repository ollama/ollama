package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"
	"time"
)

func ClaudeDesktopBaseURL(ollamaURL string) string {
	return strings.TrimRight(ollamaURL, "/") + ClaudeDesktopPathPrefix
}

type claudeRuntimeModel struct {
	Model            ClaudeDesktopModel `json:"model"`
	Route            string             `json:"route"`
	EntitlementKnown bool               `json:"entitlement_known"`
}

type ClaudeDesktopRuntimeStatus struct {
	Models []claudeRuntimeModel `json:"models"`
	Counts ClaudeDesktopCounts  `json:"counts"`
}

func (p *ClaudeDesktop) serveRuntimeStatus(w http.ResponseWriter, models []ClaudeDesktopModel) {
	status := ClaudeDesktopRuntimeStatus{Counts: p.Counts()}
	for _, model := range models {
		status.Models = append(status.Models, claudeRuntimeModel{model, model.GatewayID(), model.entitlementKnown})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(status)
}

// ClaudeDesktopClient observes the server-owned gateway. SetModels only
// verifies the saved selection: it never creates a second configuration store.
type ClaudeDesktopClient struct {
	BaseURL string
	HTTP    *http.Client
}

func NewClaudeDesktopClient(baseURL string) *ClaudeDesktopClient {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	// Cold discovery can take two 3s cloud fallbacks plus local/account checks.
	// A caller's context can still impose a shorter deadline.
	return &ClaudeDesktopClient{strings.TrimRight(baseURL, "/"), &http.Client{Transport: transport, Timeout: 10 * time.Second}}
}

func (c *ClaudeDesktopClient) Status(ctx context.Context) (*ClaudeDesktopRuntimeStatus, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/_ollama/status", nil)
	if err != nil {
		return nil, err
	}
	response, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Claude gateway status returned %s", response.Status)
	}
	var status ClaudeDesktopRuntimeStatus
	if err := json.NewDecoder(response.Body).Decode(&status); err != nil {
		return nil, err
	}
	for i := range status.Models {
		row := &status.Models[i]
		for _, slot := range claudeDesktopModelSlots {
			if row.Route == slot.id {
				row.Model.assignClaudeDesktopSlot(slot)
			}
		}
		row.Model.entitlementKnown = row.EntitlementKnown
	}
	return &status, nil
}

func (c *ClaudeDesktopClient) Models() []ClaudeDesktopModel {
	status, err := c.Status(context.Background())
	if err != nil {
		return nil
	}
	models := make([]ClaudeDesktopModel, len(status.Models))
	for i, row := range status.Models {
		models[i] = row.Model
	}
	return models
}

func (c *ClaudeDesktopClient) SetModels(models []ClaudeDesktopModel) error {
	return c.SetModelsContext(context.Background(), models)
}

func (c *ClaudeDesktopClient) SetModelsContext(ctx context.Context, models []ClaudeDesktopModel) error {
	status, err := c.Status(ctx)
	if err != nil {
		return err
	}
	actual := make(map[string]string)
	for _, row := range status.Models {
		actual[row.Route] = row.Model.OllamaModel
	}
	if !maps.Equal(actual, ClaudeDesktopMappings(models)) {
		return errors.New("Claude gateway has not loaded the saved model mappings")
	}
	return nil
}

func (c *ClaudeDesktopClient) Counts() ClaudeDesktopCounts {
	status, err := c.Status(context.Background())
	if err != nil {
		return ClaudeDesktopCounts{}
	}
	return status.Counts
}

func (c *ClaudeDesktopClient) Close(context.Context) error {
	c.HTTP.CloseIdleConnections()
	return nil
}
