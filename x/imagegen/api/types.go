// Package api provides OpenAI-compatible image generation API types.
package api

// ImageGenerationRequest is an OpenAI-compatible image generation request.
type ImageGenerationRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	Stream         bool   `json:"stream,omitempty"`
}

// ImageGenerationResponse is an OpenAI-compatible image generation response.
type ImageGenerationResponse struct {
	Created int64       `json:"created"`
	Data    []ImageData `json:"data"`
}

// ImageData contains the generated image data.
type ImageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// ImageProgressEvent is sent during streaming to indicate generation progress.
type ImageProgressEvent struct {
	Step  int `json:"step"`
	Total int `json:"total"`
}
