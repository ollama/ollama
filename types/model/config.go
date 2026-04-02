package model

// ConfigV2 represents the configuration metadata for a model.
type ConfigV2 struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"` // shown as Parameter Size
	FileType      string   `json:"file_type"`  // shown as Quantization Level
	Renderer      string   `json:"renderer,omitempty"`
	Parser        string   `json:"parser,omitempty"`
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

// RootFS represents the root filesystem configuration for a model.
type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}
