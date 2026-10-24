package cmd

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/version"
)

func listCloudModels(ctx context.Context) (*api.ListResponse, error) {
	if internalcloud.Disabled() {
		return nil, errors.New(internalcloud.DisabledMessagePrefix)
	}

	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://ollama.com/api/tags", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "ollama/"+version.Version)

	// The catalog is public. api.Client would require a signing key for ollama.com.
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list cloud models: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		apiErr := api.StatusError{StatusCode: resp.StatusCode, Status: resp.Status}
		if err := json.Unmarshal(body, &apiErr); err != nil {
			apiErr.ErrorMessage = strings.TrimSpace(string(body))
		}
		return nil, fmt.Errorf("list cloud models: %w", apiErr)
	}

	var models api.ListResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&models); err != nil {
		return nil, fmt.Errorf("decode cloud models: %w", err)
	}
	for i := range models.Models {
		m := &models.Models[i]
		ref, err := modelref.ParseRef(cmp.Or(m.Model, m.Name))
		if err != nil {
			return nil, fmt.Errorf("invalid cloud model: %w", err)
		}
		m.Name = ref.Base + ":cloud"
		m.Model = m.Name
		m.RemoteModel = ref.Base
		m.RemoteHost = "https://ollama.com"
	}
	slices.SortFunc(models.Models, func(a, b api.ListModelResponse) int {
		return strings.Compare(a.Name, b.Name)
	})
	return &models, nil
}
