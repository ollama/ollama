// Package launchconfig reads the launcher configuration without migrating or
// writing it. The server and launcher share the same persisted model choices.
package launchconfig

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Integration struct {
	Models  []string          `json:"models"`
	Aliases map[string]string `json:"aliases,omitempty"`
}

func Path() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config.json"), nil
}

func LoadIntegration(name string) (*Integration, error) {
	path, err := Path()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		data, err = os.ReadFile(filepath.Join(filepath.Dir(path), "config", "config.json"))
	}
	if err != nil {
		return nil, err
	}
	var document *struct {
		Integrations map[string]json.RawMessage `json:"integrations"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("parse launcher config: %w", err)
	}
	if document == nil {
		return nil, fmt.Errorf("launcher config must be an object")
	}
	raw, ok := document.Integrations[strings.ToLower(name)]
	if !ok {
		return nil, os.ErrNotExist
	}
	var integration *Integration
	if err := json.Unmarshal(raw, &integration); err != nil {
		return nil, fmt.Errorf("parse %s config: %w", name, err)
	}
	if integration == nil {
		return nil, fmt.Errorf("%s config must be an object", name)
	}
	return integration, nil
}
