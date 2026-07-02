//go:build integration && migration

package integration

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestPublishedCompatibilityModels(t *testing.T) {
	if testModel != "" {
		t.Skip("uses a fixed published compatibility matrix, not applicable with model override")
	}
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("published compatibility validation requires a harness-managed server")
	}
	skipIfRemote(t)

	softTimeout, hardTimeout := getTimeouts(t)
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()

	// TODO - temporary prefix prior to publishing to library
	prefix := "dhiltgen/"

	for _, base := range compatibilityPublishedModelNames() {
		name := prefixedCompatibilityModel(prefix, base)
		t.Run(name, func(t *testing.T) {
			if time.Since(started) > softTimeout {
				t.Skip("skipping remaining tests to avoid excessive runtime")
			}

			runPublishedCompatibilityModelCase(ctx, t, name)
		})
	}
}

func runPublishedCompatibilityModelCase(ctx context.Context, t *testing.T, name string) {
	t.Helper()

	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)
	t.Setenv("OLLAMA_DEBUG", "2")
	t.Logf("%s: using published compatibility model store %s", name, modelsDir)

	t.Logf("%s: starting server with %s=0", name, migrationCompatEnv)
	t.Setenv(migrationCompatEnv, "0")
	patchDisabledClient, _, patchDisabledCleanup := InitServerConnection(ctx, t)
	defer func() {
		if patchDisabledCleanup != nil {
			patchDisabledCleanup()
		}
	}()

	t.Logf("%s: pulling published model if needed", name)
	if err := PullIfMissing(ctx, patchDisabledClient, name); err != nil {
		t.Fatalf("model %s not available: %v", name, err)
	}

	patchDisabledShow := showOrFatal(ctx, t, patchDisabledClient, name)
	if len(patchDisabledShow.Capabilities) == 0 {
		t.Fatalf("show %s returned no capabilities", name)
	}
	slog.Info("validating published compatibility model", "model", name, "capabilities", normalizedCapabilities(patchDisabledShow.Capabilities))

	skipIfModelTooLargeForVRAM(ctx, t, patchDisabledClient, name)
	patchDisabledValidationStart := len(serverLog.String())
	validatePublishedCapabilities(ctx, t, patchDisabledClient, name, patchDisabledShow, 10*time.Second)
	waitForNoRunningModel(ctx, t, patchDisabledClient, name)

	patchDisabledLogs := serverLog.String()[patchDisabledValidationStart:]
	if hasCompatPatchEvidence(patchDisabledLogs) {
		t.Fatalf("%s triggered compatibility patch with %s=0", name, migrationCompatEnv)
	}

	t.Logf("%s: stopping patch-disabled server", name)
	patchDisabledCleanup()
	patchDisabledCleanup = nil

	t.Logf("%s: restarting server with default compatibility patch settings", name)
	os.Unsetenv(migrationCompatEnv)
	defaultClient, _, defaultCleanup := InitServerConnection(ctx, t)
	defer defaultCleanup()

	defaultShow := showOrFatal(ctx, t, defaultClient, name)
	defaultValidationStart := len(serverLog.String())
	validateCompatibilityPrimaryCapability(ctx, t, defaultClient, name, defaultShow, 10*time.Second)
	waitForNoRunningModel(ctx, t, defaultClient, name)

	defaultLogs := serverLog.String()[defaultValidationStart:]
	if hasCompatPatchEvidence(defaultLogs) {
		t.Fatalf("%s triggered compatibility patch from published llama.cpp-compatible artifact", name)
	}
}

func validatePublishedCapabilities(ctx context.Context, t *testing.T, client *api.Client, name string, resp *api.ShowResponse, keepAlive time.Duration) {
	t.Helper()
	capabilities := resp.Capabilities
	validated := false
	if hasCapability(capabilities, model.CapabilityEmbedding) {
		t.Logf("%s: validating embedding capability", name)
		testEmbedCosineDistanceCorrelationForModel(t, ctx, client, name, keepAlive)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityAudio) {
		t.Logf("%s: validating audio capability", name)
		testAudioResponseForModel(t, ctx, client, name, keepAlive, 120*time.Second, 30*time.Second)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityVision) {
		t.Logf("%s: validating vision capability", name)
		validateCompatibilityVision(ctx, t, client, name, keepAlive)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityCompletion) {
		t.Logf("%s: validating completion capability", name)
		if isPlainCompletionOnlyModel(resp) {
			validateGenerateCompletion(ctx, t, client, name, keepAlive)
		} else {
			validateCompatibilityChat(ctx, t, client, name, keepAlive)
		}
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityThinking) {
		t.Logf("%s: validating thinking capability", name)
		validateCompatibilityThinking(ctx, t, client, name, keepAlive)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityTools) {
		t.Logf("%s: validating tools capability", name)
		testBasicToolCallWithNumPredict(t, ctx, client, name, 60*time.Second, 60*time.Second, migrationToolNumPredict(name))
		validated = true
	}
	if !validated {
		t.Fatalf("%s advertised unsupported capability set %v", name, capabilities)
	}
}

func validateCompatibilityPrimaryCapability(ctx context.Context, t *testing.T, client *api.Client, name string, resp *api.ShowResponse, keepAlive time.Duration) {
	t.Helper()
	capabilities := resp.Capabilities
	if hasCapability(capabilities, model.CapabilityEmbedding) {
		testEmbedCosineDistanceCorrelationForModel(t, ctx, client, name, keepAlive)
		return
	}
	if hasCapability(capabilities, model.CapabilityVision) {
		validateCompatibilityVision(ctx, t, client, name, keepAlive)
		if hasCapability(capabilities, model.CapabilityTools) && shouldSmokeTools(name) {
			testBasicToolCallWithNumPredict(t, ctx, client, name, 60*time.Second, 60*time.Second, migrationToolNumPredict(name))
		}
		return
	}
	if hasCapability(capabilities, model.CapabilityTools) && shouldSmokeTools(name) {
		testBasicToolCallWithNumPredict(t, ctx, client, name, 60*time.Second, 60*time.Second, migrationToolNumPredict(name))
		return
	}
	if hasCapability(capabilities, model.CapabilityCompletion) {
		if isPlainCompletionOnlyModel(resp) {
			validateGenerateCompletion(ctx, t, client, name, keepAlive)
		} else {
			validateCompatibilityChat(ctx, t, client, name, keepAlive)
		}
	}
}

func validateCompatibilityCapabilities(ctx context.Context, t *testing.T, client *api.Client, name string, resp *api.ShowResponse, keepAlive time.Duration) {
	t.Helper()
	capabilities := resp.Capabilities
	if hasCapability(capabilities, model.CapabilityEmbedding) {
		testEmbedCosineDistanceCorrelationForModel(t, ctx, client, name, keepAlive)
		return
	}
	if hasCapability(capabilities, model.CapabilityAudio) {
		testAudioResponseForModel(t, ctx, client, name, keepAlive, 120*time.Second, 30*time.Second)
	}
	if hasCapability(capabilities, model.CapabilityVision) {
		validateCompatibilityVision(ctx, t, client, name, keepAlive)
		if hasCapability(capabilities, model.CapabilityTools) && shouldSmokeTools(name) {
			testBasicToolCallWithNumPredict(t, ctx, client, name, 60*time.Second, 60*time.Second, migrationToolNumPredict(name))
		}
		return
	}
	if hasCapability(capabilities, model.CapabilityCompletion) {
		if isPlainCompletionOnlyModel(resp) {
			validateGenerateCompletion(ctx, t, client, name, keepAlive)
		} else {
			validateCompatibilityChat(ctx, t, client, name, keepAlive)
		}
	}
	if hasCapability(capabilities, model.CapabilityTools) {
		testBasicToolCallWithNumPredict(t, ctx, client, name, 60*time.Second, 60*time.Second, migrationToolNumPredict(name))
	}
}

func isPlainCompletionOnlyModel(resp *api.ShowResponse) bool {
	if !hasCapability(resp.Capabilities, model.CapabilityCompletion) ||
		hasCapability(resp.Capabilities, model.CapabilityTools) ||
		hasCapability(resp.Capabilities, model.CapabilityVision) ||
		hasCapability(resp.Capabilities, model.CapabilityAudio) ||
		hasCapability(resp.Capabilities, model.CapabilityThinking) {
		return false
	}
	return strings.TrimSpace(resp.Template) == "" || strings.TrimSpace(resp.Template) == "{{ .Prompt }}"
}

func validateGenerateCompletion(ctx context.Context, t *testing.T, client *api.Client, name string, keepAlive time.Duration) {
	t.Helper()
	req := api.GenerateRequest{
		Model:     name,
		Prompt:    "Q: What is the capital of France?\nA:",
		KeepAlive: &api.Duration{Duration: keepAlive},
		Options: map[string]any{
			"temperature": 0.0,
			"seed":        123,
			"num_predict": 32,
		},
	}
	DoGenerate(ctx, t, client, req, []string{"paris"}, 120*time.Second, 30*time.Second)
}

func validateCompatibilityChat(ctx context.Context, t *testing.T, client *api.Client, model string, keepAlive time.Duration) {
	t.Helper()
	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: blueSkyPrompt,
			},
		},
		KeepAlive: &api.Duration{Duration: keepAlive},
		Options: map[string]any{
			"temperature": 0.1,
			"seed":        123,
			"num_predict": migrationChatNumPredict,
		},
	}
	msg := DoChat(ctx, t, client, req, blueSkyExpected, 120*time.Second, 30*time.Second)
	validateNoTokenizerArtifacts(t, model, msg)
}

func validateCompatibilityVision(ctx context.Context, t *testing.T, client *api.Client, model string, keepAlive time.Duration) {
	t.Helper()
	image, _, defaultImage := decodeTestImages(t)
	prompt := "Describe what you see in this image briefly."
	expected := []string{"llama", "pig", "animal", "drawing", "sketch", "build", "model", "open", "cartoon", "character"}
	if isModelFamily(model, "llama4") {
		// Avoid llama.cpp's current llama4 multi-tile image path; migration only
		// needs to prove the vision encoder survives conversion.
		prompt = "What text is shown in this image?"
		expected = []string{"ollam", "text"}
	} else {
		image = defaultImage
	}
	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: prompt,
				Images:  []api.ImageData{image},
			},
		},
		KeepAlive: &api.Duration{Duration: keepAlive},
		Options: map[string]any{
			"temperature": 0.0,
			"seed":        42,
			"num_predict": migrationChatNumPredict,
		},
	}
	msg := DoChat(ctx, t, client, req, expected, 120*time.Second, 30*time.Second)
	validateNoTokenizerArtifacts(t, model, msg)
}

func validateCompatibilityThinking(ctx context.Context, t *testing.T, client *api.Client, model string, keepAlive time.Duration) {
	t.Helper()
	think := api.ThinkValue{Value: true}
	stream := false
	req := api.ChatRequest{
		Model:  model,
		Stream: &stream,
		Think:  &think,
		Messages: []api.Message{
			{Role: "user", Content: "What is 12 * 15? Think briefly."},
		},
		KeepAlive: &api.Duration{Duration: keepAlive},
		Options: map[string]any{
			"temperature": 0,
			"seed":        42,
			"num_predict": 512,
		},
	}

	var response api.ChatResponse
	if err := client.Chat(ctx, &req, func(cr api.ChatResponse) error {
		response = cr
		return nil
	}); err != nil {
		t.Fatalf("thinking chat failed: %v", err)
	}

	combined := response.Message.Thinking + " " + response.Message.Content
	if !strings.Contains(combined, "180") {
		t.Fatalf("expected thinking response for %s to contain 180, got thinking=%q content=%q", model, response.Message.Thinking, response.Message.Content)
	}
	validateNoTokenizerArtifacts(t, model, &response.Message)
}

func validateNoTokenizerArtifacts(t *testing.T, model string, msg *api.Message) {
	t.Helper()
	if msg == nil {
		t.Fatalf("%s did not return a chat response", model)
	}
	if strings.Contains(msg.Content, "[UNK_BYTE_") {
		t.Fatalf("%s returned visible tokenizer byte fallback artifacts: %s", model, msg.Content)
	}
}

func prefixedCompatibilityModel(prefix, name string) string {
	if prefix == "" || strings.Contains(strings.Split(name, ":")[0], "/") {
		return name
	}
	return strings.TrimRight(prefix, "/") + "/" + name
}
