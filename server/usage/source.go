package usage

import (
	"strings"
)

// API type constants
const (
	APITypeOllama    = "ollama"
	APITypeOpenAI    = "openai"
	APITypeAnthropic = "anthropic"
)

// ClassifyAPIType determines the API type from the request path.
func ClassifyAPIType(path string) string {
	if strings.HasPrefix(path, "/v1/messages") {
		return APITypeAnthropic
	}
	if strings.HasPrefix(path, "/v1/") {
		return APITypeOpenAI
	}
	return APITypeOllama
}
