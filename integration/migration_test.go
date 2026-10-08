//go:build integration && migration

package integration

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/model"
)

const (
	migrationChatNumPredict    = 2048
	migrationDefaultGoTemplate = "{{ .Prompt }}"
	migrationGibiByte          = 1 << 30

	// Timeouts for vision/chat/audio/completion capability validations.
	compatValidationInitialTimeout = 120 * time.Second
	compatValidationStreamTimeout  = 30 * time.Second
	// Timeouts for the tool-call smoke test.
	compatToolInitialTimeout = 60 * time.Second
	compatToolStreamTimeout  = 60 * time.Second
	// keepAlive passed to validation loads so models unload promptly.
	compatKeepAlive = 10 * time.Second

	// Conversion wait when the source model size is unknown.
	migrationConversionFallbackTimeout = 5 * time.Minute
	// Conversion wait scales with source size: base + per-GiB, capped at max.
	migrationConversionBaseTimeout = 2 * time.Minute
	migrationConversionPerGiB      = 10 * time.Second
	migrationConversionMaxTimeout  = 12 * time.Minute
	// Wait and poll interval for models to unload after validation.
	migrationUnloadTimeout      = 2 * time.Minute
	migrationUnloadPollInterval = 500 * time.Millisecond
)

func TestLocalCompatibilityMigration(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("local compatibility migration requires a harness-managed server")
	}
	skipIfRemote(t)

	names := compatibilityMigrationModelNames()
	if testModel != "" {
		names = []string{testModel}
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			runLocalCompatibilityMigrationCase(t, name)
		})
	}
}

func runLocalCompatibilityMigrationCase(t *testing.T, name string) {
	t.Helper()

	modelsDir := os.Getenv("OLLAMA_MODELS")
	isolated := testModel == "" || modelsDir == ""
	if isolated {
		modelsDir = t.TempDir()
		t.Setenv("OLLAMA_MODELS", modelsDir)
	}
	t.Setenv("OLLAMA_DEBUG", "2")
	t.Logf("%s: using migration model store %s", name, modelsDir)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	t.Logf("%s: starting harness-managed server", name)
	client, _, cleanup := InitServerConnection(ctx, t)
	defer func() {
		if cleanup != nil {
			cleanup()
		}
	}()

	t.Logf("%s: pulling source model if needed", name)
	if err := PullIfMissing(ctx, client, name); err != nil {
		t.Fatalf("source model unavailable: %v", err)
	}

	t.Logf("%s: reading source manifest", name)
	firstShow := showOrFatal(ctx, t, client, name)
	expectedCapabilities := normalizedCapabilities(firstShow.Capabilities)
	expectedPromptConfig := promptConfigFromShow(t, firstShow)
	if expectedPromptConfig.renderer == "clef" && hasCapability(firstShow.Capabilities, model.CapabilityDecision) {
		// Upstream System One replaces Clef's private Go renderer.
		expectedPromptConfig.renderer = ""
	}

	t.Logf("%s: loading legacy source, waiting for conversion before inference", name)
	firstValidationStart := len(serverLog.String())
	loadCtx, loadCancel := context.WithTimeout(ctx, migrationConversionTimeout(ctx, t, client, name))
	validateCompatibilityPrimaryCapability(loadCtx, t, client, name, firstShow, compatKeepAlive)
	loadCancel()
	t.Logf("%s: waiting for first load to unload", name)
	waitForNoRunningModel(ctx, t, client, name)

	firstValidationLogs := serverLog.String()[firstValidationStart:]
	completed := strings.Index(firstValidationLogs, "completed local compat GGUF migration")
	loaded := strings.Index(firstValidationLogs, "loading model via llama-server")
	if completed < 0 || loaded < completed || strings.Count(firstValidationLogs, "completed local compat GGUF migration") != 1 {
		t.Fatal("first load must complete conversion before starting llama-server; use a verified legacy source")
	}
	if hasCompatPatchEvidence(firstValidationLogs) {
		t.Fatal("first load used a patched native binary")
	}

	t.Logf("%s: reading converted manifest selection", name)
	secondShow := showOrFatal(ctx, t, client, name)
	children, err := client.ShowManifests(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		t.Fatal(err)
	}
	if len(children.Manifests) != 1 || children.Manifests[0].Runner != manifest.RunnerLlamaCPP {
		t.Fatalf("converted model must have one llamacpp child, got %d children", len(children.Manifests))
	}
	convertedDigests := ggufDigestsFromShow(t, secondShow)
	if !slices.Equal(ggufDigestsFromShow(t, &children.Manifests[0].ShowResponse), convertedDigests) {
		t.Fatal("default selection differs from the converted child")
	}
	if isolated {
		for _, digest := range ggufDigestsFromShow(t, firstShow) {
			if slices.Contains(convertedDigests, digest) {
				continue
			}
			if exists, err := client.HeadBlob(ctx, digest); err != nil || exists {
				t.Fatalf("obsolete GGUF %s was not reclaimed: exists=%v err=%v", digest, exists, err)
			}
		}
	}
	if got := normalizedCapabilities(secondShow.Capabilities); !slices.Equal(got, expectedCapabilities) {
		t.Fatalf("converted %s child capabilities changed: before=%v after=%v", name, expectedCapabilities, got)
	}
	if got := promptConfigFromShow(t, secondShow); got != expectedPromptConfig {
		t.Fatalf("converted %s child prompt config changed: before=%+v after=%+v", name, expectedPromptConfig, got)
	}
	t.Logf("%s: validating converted child", name)
	convertedValidationStart := len(serverLog.String())
	validateCompatibilityCapabilities(ctx, t, client, name, firstShow, compatKeepAlive)
	t.Logf("%s: waiting for converted child to unload", name)
	waitForNoRunningModel(ctx, t, client, name)

	logs := serverLog.String()

	t.Logf("%s: stopping first server before restart validation", name)
	cleanup()
	cleanup = nil

	t.Logf("%s: checking migration and converted-load log evidence", name)
	convertedValidationLogs := logs[convertedValidationStart:]
	if !strings.Contains(convertedValidationLogs, "loading model via llama-server") {
		t.Fatalf("server log does not show converted model load")
	}
	if hasCompatPatchEvidence(convertedValidationLogs) {
		t.Fatalf("converted %s child still triggered compatibility patch after migration", name)
	}
	if strings.Contains(convertedValidationLogs, "starting local compat GGUF migration") {
		t.Fatal("second load converted the model again")
	}

	t.Logf("%s: restarting server with the converted store", name)
	restartedClient, _, restartedCleanup := InitServerConnection(ctx, t)
	defer restartedCleanup()

	restartedValidationStart := len(serverLog.String())
	restartedShow := showOrFatal(ctx, t, restartedClient, name)
	if !slices.Equal(ggufDigestsFromShow(t, restartedShow), convertedDigests) {
		t.Fatal("restart selected a different model")
	}
	validateCompatibilityPrimaryCapability(ctx, t, restartedClient, name, restartedShow, compatKeepAlive)
	waitForNoRunningModel(ctx, t, restartedClient, name)

	restartedLogs := serverLog.String()[restartedValidationStart:]
	if hasCompatPatchEvidence(restartedLogs) || strings.Contains(restartedLogs, "starting local compat GGUF migration") {
		t.Fatalf("converted %s child needed conversion or patching after restart", name)
	}

	t.Logf("%s: migration validation complete", name)
}

func migrationConversionTimeout(ctx context.Context, t *testing.T, client *api.Client, model string) time.Duration {
	t.Helper()

	list, err := client.List(ctx)
	if err != nil {
		t.Logf("%s: could not list local model size for conversion timeout: %v", model, err)
		return migrationConversionFallbackTimeout
	}
	for _, candidate := range list.Models {
		if sameModelName(candidate.Name, model) || sameModelName(candidate.Model, model) {
			timeout := migrationConversionTimeoutForSize(candidate.Size)
			t.Logf("%s: waiting up to %s for conversion of %d byte source", model, timeout, candidate.Size)
			return timeout
		}
	}
	t.Logf("%s: local model size not found for conversion timeout", model)
	return migrationConversionFallbackTimeout
}

func migrationConversionTimeoutForSize(size int64) time.Duration {
	timeout := migrationConversionBaseTimeout
	if size > 0 {
		timeout += time.Duration((size+migrationGibiByte-1)/migrationGibiByte) * migrationConversionPerGiB
	}
	if timeout > migrationConversionMaxTimeout {
		return migrationConversionMaxTimeout
	}
	return timeout
}

// Detect accidentally testing an older, still-patched native binary.
func hasCompatPatchEvidence(logs string) bool {
	return strings.Contains(logs, "detected Ollama-format")
}

func normalizedCapabilities(capabilities []model.Capability) []string {
	out := make([]string, 0, len(capabilities))
	for _, capability := range capabilities {
		out = append(out, string(capability))
	}
	slices.Sort(out)
	return out
}

func hasCapability(capabilities []model.Capability, capability model.Capability) bool {
	return slices.Contains(capabilities, capability)
}

type migrationPromptConfig struct {
	selectedTemplate string
	goTemplate       string
	renderer         string
	parser           string
}

func promptConfigFromShow(t *testing.T, resp *api.ShowResponse) migrationPromptConfig {
	t.Helper()

	out := migrationPromptConfig{selectedTemplate: resp.Template}
	mf, err := parser.ParseFile(strings.NewReader(resp.Modelfile))
	if err != nil {
		t.Fatalf("parse show modelfile: %v", err)
	}
	for _, cmd := range mf.Commands {
		switch cmd.Name {
		case "template":
			if strings.TrimSpace(cmd.Args) != migrationDefaultGoTemplate {
				out.goTemplate = cmd.Args
			}
		case "renderer":
			out.renderer = cmd.Args
		case "parser":
			out.parser = cmd.Args
		}
	}
	return out
}

func showOrFatal(ctx context.Context, t *testing.T, client *api.Client, model string) *api.ShowResponse {
	t.Helper()
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		t.Fatalf("show %s failed: %v", model, err)
	}
	return resp
}

func ggufDigestsFromShow(t *testing.T, resp *api.ShowResponse) []string {
	t.Helper()
	mf, err := parser.ParseFile(strings.NewReader(resp.Modelfile))
	if err != nil {
		t.Fatal(err)
	}
	var digests []string
	for _, cmd := range mf.Commands {
		if cmd.Name == "model" || cmd.Name == "draft" {
			digest, ok := manifest.DigestReference(filepath.Base(cmd.Args))
			if !ok {
				t.Fatalf("expected a local GGUF in show output, got %q", cmd.Args)
			}
			digests = append(digests, digest)
		}
	}
	if len(digests) == 0 {
		t.Fatal("show output has no GGUF model")
	}
	return digests
}

func migrationToolNumPredict(name string) int {
	if isModelFamily(name, "qwen3-vl") || isModelFamily(name, "qwen3-next") || isModelFamily(name, "laguna-xs.2") || isModelFamily(name, "nemotron-3-nano") {
		return 2048
	}
	return 512
}

func isModelFamily(model, family string) bool {
	name := strings.SplitN(model, ":", 2)[0]
	if slash := strings.LastIndexByte(name, '/'); slash >= 0 {
		name = name[slash+1:]
	}
	return name == family
}

func waitForNoRunningModel(ctx context.Context, t *testing.T, client *api.Client, model string) {
	t.Helper()
	deadline := time.Now().Add(migrationUnloadTimeout)
	var last []string
	for time.Now().Before(deadline) {
		resp, err := client.ListRunning(ctx)
		if err != nil {
			t.Fatalf("ps failed while waiting for unload: %v", err)
		}
		last = last[:0]
		running := false
		for _, m := range resp.Models {
			last = append(last, m.Name)
			if sameModelName(m.Name, model) {
				running = true
			}
		}
		if !running {
			return
		}
		time.Sleep(migrationUnloadPollInterval)
	}
	slices.Sort(last)
	t.Fatalf("timed out waiting for %s to unload; running models: %s", model, strings.Join(last, ", "))
}
