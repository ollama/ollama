package model

// SkillRef represents a reference to a skill, either by local path or by registry digest.
type SkillRef struct {
	// Name is the local path (for development) or registry name (e.g., "skill/calculator:1.0.0")
	Name string `json:"name,omitempty"`
	// Digest is the content-addressable digest of the skill blob (e.g., "sha256:abc123...")
	Digest string `json:"digest,omitempty"`
}

// MCPRef represents a reference to an MCP (Model Context Protocol) server.
type MCPRef struct {
	// Name is the identifier for the MCP server (used for tool namespacing)
	Name string `json:"name,omitempty"`
	// Digest is the content-addressable digest of the bundled MCP server blob
	Digest string `json:"digest,omitempty"`
	// Command is the executable to run (e.g., "uv", "node", "python3")
	Command string `json:"command,omitempty"`
	// Args are the arguments to pass to the command
	Args []string `json:"args,omitempty"`
	// Env is optional environment variables for the MCP server
	Env map[string]string `json:"env,omitempty"`
	// Type is the transport type (currently only "stdio" is supported)
	Type string `json:"type,omitempty"`
}

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

	// agent-specific fields
	Skills     []SkillRef `json:"skills,omitempty"`
	MCPs       []MCPRef   `json:"mcps,omitempty"`
	AgentType  string     `json:"agent_type,omitempty"`
	Entrypoint string     `json:"entrypoint,omitempty"`

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
