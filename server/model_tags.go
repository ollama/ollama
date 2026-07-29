package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/types/model"
)

const (
	maxModelTagSuggestions    = 8
	modelTagSuggestionTimeout = 3 * time.Second
)

type modelTagsResponse struct {
	Tags []string `json:"tags"`
}

func modelTagsURL(n model.Name) *url.URL {
	base := n.BaseURL()
	if strings.EqualFold(n.Host, model.DefaultName().Host) {
		// The default registry's tag index is served by ollama.com.
		base.Scheme = "https"
		base.Host = "ollama.com"
	}

	return base.JoinPath(n.Namespace, n.Model, "tags")
}

func getModelTagSuggestions(ctx context.Context, n model.Name, regOpts *registryOptions) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, modelTagSuggestionTimeout)
	defer cancel()

	headers := make(http.Header)
	headers.Set("Accept", "application/json")

	requestOpts := &registryOptions{}
	if regOpts != nil && !strings.EqualFold(n.Host, model.DefaultName().Host) {
		requestOpts.Insecure = regOpts.Insecure
		requestOpts.CheckRedirect = regOpts.CheckRedirect
	}

	resp, err := makeRequest(ctx, http.MethodGet, modelTagsURL(n), headers, nil, requestOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code %d", resp.StatusCode)
	}

	var payload modelTagsResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return nil, err
	}

	suggestions := make([]string, 0, min(len(payload.Tags), maxModelTagSuggestions))
	seen := make(map[string]struct{}, len(payload.Tags))
	for _, tag := range payload.Tags {
		candidate := n
		candidate.Tag = tag
		if strings.EqualFold(tag, n.Tag) || !candidate.IsValid() {
			continue
		}

		suggestion := candidate.DisplayShortest()
		key := strings.ToLower(suggestion)
		if _, ok := seen[key]; ok {
			continue
		}

		seen[key] = struct{}{}
		suggestions = append(suggestions, suggestion)
		if len(suggestions) == maxModelTagSuggestions {
			break
		}
	}

	return suggestions, nil
}
