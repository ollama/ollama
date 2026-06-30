//go:build integration && (fast || release || library)

package integration

import (
	"strings"
	"testing"
)

func runIntegrationGroup(t *testing.T, cases ...string) {
	t.Helper()
	selected := map[string]struct{}{}
	for _, c := range cases {
		selected[c] = struct{}{}
	}
	var ran bool
	for _, c := range integrationCases {
		if _, ok := selected[c.Case]; !ok {
			continue
		}
		ran = true
		c := c
		name := c.Case
		if c.Model != "" {
			name += "/" + testName(c.Model)
		}
		t.Run(name, c.Run)
	}
	if !ran {
		t.Skip("no integration cases selected")
	}
}

func TestAPI(t *testing.T) {
	runIntegrationGroup(t,
		"api-generate",
		"api-chat",
		"api-list-models",
		"api-show-model",
		"generate-logprobs",
		"chat-logprobs",
	)
}

func TestBasic(t *testing.T) {
	runIntegrationGroup(t,
		"blue-sky",
		"unicode-input",
		"unicode-output",
		"unicode-model-dir",
		"num-predict",
		"thinking-enabled",
		"thinking-suppressed",
	)
}

func TestChat(t *testing.T) {
	runIntegrationGroup(t,
		"chat",
		"chat-history",
	)
}

func TestEmbedding(t *testing.T) {
	runIntegrationGroup(t,
		"embed",
		"embed-correlation",
		"embedding-api",
		"embed-api",
		"embed-api-batch",
		"embed-api-truncate",
		"embed-truncation",
		"embed-large-input",
		"embed-status-code",
	)
}

func TestVision(t *testing.T) {
	runIntegrationGroup(t,
		"vision-multiturn",
		"vision-count",
		"vision-scene",
		"vision-spatial",
		"vision-detail",
		"vision-multi-image",
		"vision-description",
		"vision-split-batch",
		"vision-text",
	)
}

func TestAudio(t *testing.T) {
	runIntegrationGroup(t,
		"audio-transcription",
		"audio-response",
		"openai-audio-transcription",
		"openai-chat-audio",
	)
}

func TestContext(t *testing.T) {
	runIntegrationGroup(t,
		"context-long-input",
		"context-exhaustion",
		"generate-history",
		"parallel-generate-history",
		"parallel-chat-history",
	)
}

func TestConcurrency(t *testing.T) {
	runIntegrationGroup(t,
		"concurrent-chat",
		"scheduler-multimodel",
		"scheduler-max-queue",
	)
}

func TestTools(t *testing.T) {
	runIntegrationGroup(t,
		"tools",
		"tools-stress",
	)
}

func TestCreate(t *testing.T) {
	runIntegrationGroup(t,
		"create-safetensors",
		"create-gguf",
	)
}

func TestQuantization(t *testing.T) {
	runIntegrationGroup(t, "quantization")
}

func TestImageGeneration(t *testing.T) {
	runIntegrationGroup(t, "image-generation")
}

func testName(s string) string {
	return strings.NewReplacer("/", "~", " ", "_").Replace(s)
}
