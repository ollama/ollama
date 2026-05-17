// Package imagegen provides a unified MLX runner for both LLM and image generation models.
//
// This package handles safetensors models created with `ollama create --experimental`,
// supporting both text generation (LLM) and image generation (diffusion) models
// through a single unified interface.
package imagegen

// Request is the request format for completion requests.
type Request struct {
	Prompt string `json:"prompt"`

	// LLM-specific fields
	Options *RequestOptions `json:"options,omitempty"`

	// Image generation fields
	Width  int32    `json:"width,omitempty"`
	Height int32    `json:"height,omitempty"`
	Steps  int      `json:"steps,omitempty"`
	Seed   int64    `json:"seed,omitempty"`
	Images [][]byte `json:"images,omitempty"` // Input images for image editing/conditioning
}

// RequestOptions contains LLM-specific generation options.
type RequestOptions struct {
	NumPredict  int      `json:"num_predict,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	TopK        int      `json:"top_k,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// Response is streamed back for each progress update.
type Response struct {
	// Text generation response
	Content string `json:"content,omitempty"`

	// Image generation response
	Image string `json:"image,omitempty"` // Base64-encoded PNG

	// Common fields
	Done       bool   `json:"done"`
	DoneReason int    `json:"done_reason,omitempty"`
	StopReason string `json:"stop_reason,omitempty"` // Debug: why generation stopped

	// Progress fields
	Step  int `json:"step,omitempty"`
	Total int `json:"total,omitempty"`

	// Statistics
	PromptEvalCount    int `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int `json:"prompt_eval_duration,omitempty"`
	EvalCount          int `json:"eval_count,omitempty"`
	EvalDuration       int `json:"eval_duration,omitempty"`
}

// HealthResponse is returned by the health endpoint.
type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress,omitempty"`
}

// ModelMode represents the type of model being run.
type ModelMode int

const (
	// ModeLLM indicates a text generation model.
	ModeLLM ModelMode = iota
	// ModeImageGen indicates an image generation model.
	ModeImageGen
)

func (m ModelMode) String() string {
	switch m {
	case ModeLLM:
		return "llm"
	case ModeImageGen:
		return "imagegen"
	default:
		return "unknown"
	}
}
