//go:build windows || darwin

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"sync"
)

const maxFeatureFlagResponseSize = 4096

var errFeatureFlagUnavailable = errors.New("feature flag unavailable")

type featureFlagResult struct {
	ready chan struct{}
	value any
}

type featureFlagService struct {
	mu            sync.Mutex
	results       map[string]*featureFlagResult
	fetch         func(context.Context, string) (any, error)
	cloudDisabled func() (bool, error)
}

func newFeatureFlagService(
	fetch func(context.Context, string) (any, error),
	cloudDisabled func() (bool, error),
) *featureFlagService {
	return &featureFlagService{
		results:       make(map[string]*featureFlagResult),
		fetch:         fetch,
		cloudDisabled: cloudDisabled,
	}
}

func (s *featureFlagService) resolve(ctx context.Context, key string, defaultValue any) any {
	if !validFeatureFlagKey(key) {
		return defaultValue
	}

	s.mu.Lock()
	result, ok := s.results[key]
	if ok {
		s.mu.Unlock()
		select {
		case <-result.ready:
			return result.value
		case <-ctx.Done():
			return defaultValue
		}
	}
	result = &featureFlagResult{ready: make(chan struct{})}
	s.results[key] = result
	s.mu.Unlock()

	result.value = defaultValue
	disabled, err := s.cloudDisabled()
	if err == nil && !disabled {
		value, err := s.fetch(ctx, key)
		if err == nil {
			switch defaultValue.(type) {
			case bool:
				if _, ok := value.(bool); ok {
					result.value = value
				}
			case string:
				if _, ok := value.(string); ok {
					result.value = value
				}
			}
		}
	}
	close(result.ready)
	return result.value
}

func validFeatureFlagKey(key string) bool {
	if key == "" || len(key) > 128 {
		return false
	}
	for _, r := range key {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '-', r == '_', r == '.', r == ':':
		default:
			return false
		}
	}
	return true
}

func (s *Server) featureFlagResolver() *featureFlagService {
	s.featureFlagsMu.Lock()
	defer s.featureFlagsMu.Unlock()
	if s.featureFlags == nil {
		s.featureFlags = newFeatureFlagService(
			s.fetchFeatureFlag,
			func() (bool, error) {
				if s.Store == nil {
					return false, errFeatureFlagUnavailable
				}
				return s.Store.CloudDisabled()
			},
		)
	}
	return s.featureFlags
}

// FeatureFlagBool returns one session-stable boolean value or defaultValue.
func (s *Server) FeatureFlagBool(ctx context.Context, key string, defaultValue bool) bool {
	valueBool, ok := s.featureFlagResolver().resolve(ctx, key, defaultValue).(bool)
	if !ok {
		return defaultValue
	}
	return valueBool
}

// FeatureFlagString returns one session-stable string value or defaultValue.
func (s *Server) FeatureFlagString(ctx context.Context, key, defaultValue string) string {
	valueString, ok := s.featureFlagResolver().resolve(ctx, key, defaultValue).(string)
	if !ok {
		return defaultValue
	}
	return valueString
}

func (s *Server) fetchFeatureFlag(ctx context.Context, key string) (any, error) {
	if !validFeatureFlagKey(key) {
		return nil, errFeatureFlagUnavailable
	}
	resp, err := s.doSelfSigned(ctx, http.MethodGet, "/api/app/feature-flags/"+url.PathEscape(key))
	if err != nil {
		return nil, errFeatureFlagUnavailable
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errFeatureFlagUnavailable
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFeatureFlagResponseSize+1))
	if err != nil || len(body) > maxFeatureFlagResponseSize {
		return nil, errFeatureFlagUnavailable
	}
	var response struct {
		Value any `json:"value"`
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&response); err != nil {
		return nil, errFeatureFlagUnavailable
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errFeatureFlagUnavailable
	}

	switch value := response.Value.(type) {
	case bool, string:
		return value, nil
	}
	return nil, errFeatureFlagUnavailable
}

func (s *Server) getFeatureFlag(w http.ResponseWriter, r *http.Request) error {
	key := r.PathValue("key")
	defaultValue := r.URL.Query().Get("default")
	var value any
	switch r.URL.Query().Get("type") {
	case "boolean":
		if defaultValue != "true" && defaultValue != "false" {
			w.WriteHeader(http.StatusBadRequest)
			return nil
		}
		fallback, _ := strconv.ParseBool(defaultValue)
		value = s.FeatureFlagBool(r.Context(), key, fallback)
	case "string":
		value = s.FeatureFlagString(r.Context(), key, defaultValue)
	default:
		w.WriteHeader(http.StatusBadRequest)
		return nil
	}

	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(struct {
		Value any `json:"value"`
	}{Value: value})
}
