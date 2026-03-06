package model

import (
	"encoding/json"
	"fmt"
)

// ConfigV2 represents the configuration metadata for a model.
type ConfigV2 struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"` // shown as Parameter Size
	FileType      string   `json:"file_type"`  // shown as Quantization Level
	Renderer      string   `json:"renderer,omitempty"`
	Parser        string   `json:"parser,omitempty"`
	Think         *Think   `json:"think,omitempty"`
	Requires      string   `json:"requires,omitempty"`

	RemoteHost  string `json:"remote_host,omitempty"`
	RemoteModel string `json:"remote_model,omitempty"`

	// used for remotes
	Capabilities []string `json:"capabilities,omitempty"`
	ContextLen   int      `json:"context_length,omitempty"`
	EmbedLen     int      `json:"embedding_length,omitempty"`
	BaseName     string   `json:"base_name,omitempty"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`
}

// Think stores the model's default thinking mode.
// Valid values: true, false, "high", "medium", "low".
type Think struct {
	Value any
}

func (t *Think) UnmarshalJSON(data []byte) error {
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		t.Value = b
		return nil
	}

	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		switch s {
		case "high", "medium", "low":
			t.Value = s
		default:
			return fmt.Errorf("invalid think value: %q", s)
		}
		return nil
	}

	return fmt.Errorf("think must be a boolean or string")
}

func (t *Think) MarshalJSON() ([]byte, error) {
	if t == nil || t.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}

// RootFS represents the root filesystem configuration for a model.
type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}
