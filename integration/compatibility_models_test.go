//go:build integration && migration

package integration

import (
	"context"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestPublishedCompatibilityModels(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("published compatibility validation requires a harness-managed server")
	}
	skipIfRemote(t)

	softTimeout, hardTimeout := getTimeouts(t)
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()

	// OLLAMA_TEST_COMPAT_NAMESPACE points the published-artifact matrix at a
	// staging namespace until the compatible artifacts land in the library.
	prefix := os.Getenv("OLLAMA_TEST_COMPAT_NAMESPACE")

	if testModel != "" {
		runPublishedCompatibilityModelCase(ctx, t, testModel)
		return
	}

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

	if strings.HasSuffix(name, "-ggml") {
		t.Fatalf("%s is a legacy GGML source tag; run migration validation on a temporary copy instead", name)
	}

	if testModel == "" || os.Getenv("OLLAMA_MODELS") == "" {
		modelsDir := t.TempDir()
		t.Setenv("OLLAMA_MODELS", modelsDir)
		t.Logf("%s: using published compatibility model store %s", name, modelsDir)
	} else {
		t.Logf("%s: using configured model store %s", name, os.Getenv("OLLAMA_MODELS"))
	}
	t.Setenv("OLLAMA_DEBUG", "2")

	t.Logf("%s: starting server for published-model validation", name)
	firstClient, _, firstCleanup := InitServerConnection(ctx, t)
	defer func() {
		if firstCleanup != nil {
			firstCleanup()
		}
	}()

	t.Logf("%s: pulling published model if needed", name)
	if err := PullIfMissing(ctx, firstClient, name); err != nil {
		t.Fatalf("model %s not available: %v", name, err)
	}

	firstShow := showOrFatal(ctx, t, firstClient, name)
	originalDigests := compatibilityModelDigests(ctx, t, firstClient, name)
	if len(firstShow.Capabilities) == 0 {
		t.Fatalf("show %s returned no capabilities", name)
	}
	slog.Info("validating published compatibility model", "model", name, "capabilities", normalizedCapabilities(firstShow.Capabilities))

	skipIfModelTooLargeForVRAM(ctx, t, firstClient, name)
	firstValidationStart := len(serverLog.String())
	validateCompatibilityCapabilities(ctx, t, firstClient, name, firstShow, compatKeepAlive)
	checkCompatibilityRunner(ctx, t, firstClient, name, firstShow)
	waitForNoRunningModel(ctx, t, firstClient, name)

	firstLogs := serverLog.String()[firstValidationStart:]
	if hasCompatPatchEvidence(firstLogs) || strings.Contains(firstLogs, "starting local compat GGUF migration") {
		t.Fatalf("%s required migration or patching; published model is not clean", name)
	}

	t.Logf("%s: stopping first server", name)
	firstCleanup()
	firstCleanup = nil

	t.Logf("%s: restarting server to verify unchanged published model", name)
	defaultClient, _, defaultCleanup := InitServerConnection(ctx, t)
	defer defaultCleanup()

	defaultShow := showOrFatal(ctx, t, defaultClient, name)
	if got := compatibilityModelDigests(ctx, t, defaultClient, name); !slices.Equal(got, originalDigests) {
		t.Fatalf("published manifest digests changed across inference/restart: before=%v after=%v", originalDigests, got)
	}
	t.Logf("%s: unchanged model format=%s manifests=%+v", name, defaultShow.Details.Format, defaultShow.Manifests)
	defaultValidationStart := len(serverLog.String())
	validateCompatibilityPrimaryCapability(ctx, t, defaultClient, name, defaultShow, compatKeepAlive)
	checkCompatibilityRunner(ctx, t, defaultClient, name, defaultShow)
	waitForNoRunningModel(ctx, t, defaultClient, name)

	defaultLogs := serverLog.String()[defaultValidationStart:]
	if hasCompatPatchEvidence(defaultLogs) || strings.Contains(defaultLogs, "starting local compat GGUF migration") {
		t.Fatalf("%s required migration or patching after restart", name)
	}
}

func compatibilityModelDigests(ctx context.Context, t *testing.T, client *api.Client, name string) []string {
	t.Helper()
	list, err := client.List(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var digests []string
	for _, entry := range list.Models {
		if sameModelName(entry.Name, name) {
			digests = append(digests, entry.Digest)
		}
	}
	if len(digests) == 0 {
		t.Fatalf("no manifest identities returned for %s", name)
	}
	slices.Sort(digests)
	return digests
}

func checkCompatibilityRunner(ctx context.Context, t *testing.T, client *api.Client, name string, show *api.ShowResponse) {
	t.Helper()
	want := manifest.RunnerLlamaCPP
	if show.Details.Format == "safetensors" {
		want = manifest.RunnerMLX
	}
	// When both children are local, verify platform preference rather than
	// trusting whichever child show happened to select.
	for _, child := range show.Manifests {
		if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" && child.Runner == manifest.RunnerMLX {
			want = manifest.RunnerMLX
			break
		}
		if runtime.GOOS != "darwin" && child.Runner == manifest.RunnerLlamaCPP {
			want = manifest.RunnerLlamaCPP
			break
		}
	}
	running, err := client.ListRunning(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, loaded := range running.Models {
		if sameModelName(loaded.Name, name) {
			if loaded.Runner != want {
				t.Fatalf("%s: running via %s, want %s", name, loaded.Runner, want)
			}
			t.Logf("%s: runner=%s digest=%s format=%s", name, loaded.Runner, loaded.Digest, loaded.Details.Format)
			return
		}
	}
	t.Fatalf("%s was not resident after capability validation", name)
}

func validateCompatibilityCapabilities(ctx context.Context, t *testing.T, client *api.Client, name string, resp *api.ShowResponse, keepAlive time.Duration) {
	t.Helper()
	capabilities := resp.Capabilities
	if isModelFamily(name, "glm-ocr") || isModelFamily(name, "deepseek-ocr") {
		// OCR specialists fail generic chat probes on released builds too
		// (DeepSeek OCR confirmed on v0.40.0); validate image content instead.
		if !hasCapability(capabilities, model.CapabilityVision) {
			t.Fatalf("%s did not advertise expected vision capability: %v", name, capabilities)
		}
		t.Logf("%s: validating vision capability", name)
		validateCompatibilityVision(ctx, t, client, name, keepAlive)
		return
	}

	validated := false
	if hasCapability(capabilities, model.CapabilityDecision) {
		t.Logf("%s: validating System One decisions, concurrency and error recovery", name)
		validateSystemOne(ctx, t, client, name, resp)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityEmbedding) {
		t.Logf("%s: validating embedding capability", name)
		testEmbedCosineDistanceCorrelationForModel(t, ctx, client, name, keepAlive)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityAudio) {
		t.Logf("%s: validating audio capability", name)
		testAudioResponseForModel(t, ctx, client, name, keepAlive, compatValidationInitialTimeout, compatValidationStreamTimeout)
		validated = true
	}
	if hasCapability(capabilities, model.CapabilityVision) && !hasCapability(capabilities, model.CapabilityDecision) {
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
		testBasicToolCallWithNumPredict(t, ctx, client, name, compatToolInitialTimeout, compatToolStreamTimeout, migrationToolNumPredict(name))
		validated = true
	}
	if !validated {
		t.Fatalf("%s advertised unsupported capability set %v", name, capabilities)
	}
}

func validateCompatibilityPrimaryCapability(ctx context.Context, t *testing.T, client *api.Client, name string, resp *api.ShowResponse, keepAlive time.Duration) {
	t.Helper()
	capabilities := resp.Capabilities
	if hasCapability(capabilities, model.CapabilityDecision) {
		validateSystemOne(ctx, t, client, name, resp)
		return
	}
	if hasCapability(capabilities, model.CapabilityEmbedding) {
		testEmbedCosineDistanceCorrelationForModel(t, ctx, client, name, keepAlive)
		return
	}
	if hasCapability(capabilities, model.CapabilityVision) {
		validateCompatibilityVision(ctx, t, client, name, keepAlive)
		return
	}
	if hasCapability(capabilities, model.CapabilityCompletion) {
		if isPlainCompletionOnlyModel(resp) {
			validateGenerateCompletion(ctx, t, client, name, keepAlive)
		} else {
			validateCompatibilityChat(ctx, t, client, name, keepAlive)
		}
		return
	}
	t.Fatalf("%s has no supported primary capability: %v", name, capabilities)
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
	DoGenerate(ctx, t, client, req, []string{"paris"}, compatValidationInitialTimeout, compatValidationStreamTimeout)
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
	msg := DoChat(ctx, t, client, req, blueSkyExpected, compatValidationInitialTimeout, compatValidationStreamTimeout)
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
	msg := DoChat(ctx, t, client, req, expected, compatValidationInitialTimeout, compatValidationStreamTimeout)
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
			"num_predict": migrationToolNumPredict(model),
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
