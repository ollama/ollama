//go:build integration && migration

package integration

import (
	"context"
	"os"
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
	migrationCompatEnv         = "OLLAMA_LLAMA_CPP_COMPAT"
	migrationGibiByte          = 1 << 30
)

func TestLocalCompatibilityMigration(t *testing.T) {
	if testModel != "" {
		t.Skip("uses a fixed migration matrix, not applicable with model override")
	}
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("local compatibility migration requires a harness-managed server")
	}
	skipIfRemote(t)

	for _, name := range compatibilityMigrationModelNames() {
		t.Run(name, func(t *testing.T) {
			runLocalCompatibilityMigrationCase(t, name)
		})
	}
}

func runLocalCompatibilityMigrationCase(t *testing.T, name string) {
	t.Helper()

	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)
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
	pullOrSkip(ctx, t, client, name)

	t.Logf("%s: reading source manifest", name)
	firstShow := showOrFatal(ctx, t, client, name)
	if childSummaryForRunner(firstShow, manifest.RunnerLlamaCPP) != nil {
		t.Fatalf("model %s still has llamacpp child before first load", name)
	}
	expectedCapabilities := normalizedCapabilities(firstShow.Capabilities)
	expectedPromptConfig := promptConfigFromShow(t, firstShow)

	t.Logf("%s: validating first load through legacy source", name)
	firstValidationStart := len(serverLog.String())
	validateCompatibilityPrimaryCapability(ctx, t, client, name, firstShow, time.Second)
	t.Logf("%s: waiting for first load to unload", name)
	waitForNoRunningModel(ctx, t, client, name)

	conversionTimeout := migrationConversionTimeout(ctx, t, client, name)
	t.Logf("%s: waiting for background %s conversion", name, manifest.RunnerLlamaCPP)
	converted := waitForRunnerManifestChild(ctx, t, client, name, manifest.RunnerLlamaCPP, conversionTimeout)
	t.Logf("%s: converted manifest digest %s", name, converted.Digest)
	t.Logf("%s: reading converted manifest selection", name)
	secondShow := showOrFatal(ctx, t, client, name)
	if child := childSummaryForRunner(secondShow, manifest.RunnerLlamaCPP); child == nil || !child.Selected {
		t.Fatalf("converted %s child is not selected for %s after migration", manifest.RunnerLlamaCPP, name)
	}
	if got := normalizedCapabilities(secondShow.Capabilities); !slices.Equal(got, expectedCapabilities) {
		t.Fatalf("converted %s child capabilities changed: before=%v after=%v", name, expectedCapabilities, got)
	}
	if got := promptConfigFromShow(t, secondShow); got != expectedPromptConfig {
		t.Fatalf("converted %s child prompt config changed: before=%+v after=%+v", name, expectedPromptConfig, got)
	}
	t.Logf("%s: validating converted child", name)
	convertedValidationStart := len(serverLog.String())
	validateCompatibilityCapabilities(ctx, t, client, name, firstShow, 10*time.Second)
	t.Logf("%s: waiting for converted child to unload", name)
	waitForNoRunningModel(ctx, t, client, name)

	logs := serverLog.String()

	t.Logf("%s: stopping first server before patch-disabled validation", name)
	cleanup()
	cleanup = nil

	t.Logf("%s: checking first-load and converted-load log evidence", name)
	firstValidationLogs := logs[firstValidationStart:]
	if !hasCompatPatchEvidence(firstValidationLogs) {
		t.Fatalf("server log does not prove first load used compatibility patch")
	}
	migrationComplete := strings.Index(logs, "local compatibility migration completed")
	if migrationComplete < 0 {
		t.Fatalf("server log does not contain background migration completion")
	}
	convertedValidationLogs := logs[convertedValidationStart:]
	if !strings.Contains(convertedValidationLogs, "loading model via llama-server") {
		t.Fatalf("server log does not show converted model load")
	}
	if hasCompatPatchEvidence(convertedValidationLogs) {
		t.Fatalf("converted %s child still triggered compatibility patch after migration", name)
	}

	t.Logf("%s: restarting server with %s=0", name, migrationCompatEnv)
	t.Setenv(migrationCompatEnv, "0")
	patchDisabledClient, _, patchDisabledCleanup := InitServerConnection(ctx, t)
	defer patchDisabledCleanup()

	t.Logf("%s: validating converted child with compatibility patch disabled", name)
	patchDisabledValidationStart := len(serverLog.String())
	patchDisabledShow := showOrFatal(ctx, t, patchDisabledClient, name)
	if child := childSummaryForRunner(patchDisabledShow, manifest.RunnerLlamaCPP); child == nil || !child.Selected {
		t.Fatalf("converted %s child is not selected for %s with compatibility patch disabled", manifest.RunnerLlamaCPP, name)
	}
	validateCompatibilityCapabilities(ctx, t, patchDisabledClient, name, firstShow, 10*time.Second)
	waitForNoRunningModel(ctx, t, patchDisabledClient, name)

	patchDisabledLogs := serverLog.String()[patchDisabledValidationStart:]
	if hasCompatPatchEvidence(patchDisabledLogs) {
		t.Fatalf("converted %s child triggered compatibility patch with %s=0", name, migrationCompatEnv)
	}

	t.Logf("%s: migration validation complete", name)
}

func migrationConversionTimeout(ctx context.Context, t *testing.T, client *api.Client, model string) time.Duration {
	t.Helper()

	list, err := client.List(ctx)
	if err != nil {
		t.Logf("%s: could not list local model size for conversion timeout: %v", model, err)
		return 5 * time.Minute
	}
	for _, candidate := range list.Models {
		if sameModelName(candidate.Name, model) || sameModelName(candidate.Model, model) {
			timeout := migrationConversionTimeoutForSize(candidate.Size)
			t.Logf("%s: waiting up to %s for conversion of %d byte source", model, timeout, candidate.Size)
			return timeout
		}
	}
	t.Logf("%s: local model size not found for conversion timeout", model)
	return 5 * time.Minute
}

func migrationConversionTimeoutForSize(size int64) time.Duration {
	timeout := 2 * time.Minute
	if size > 0 {
		timeout += time.Duration((size+migrationGibiByte-1)/migrationGibiByte) * 10 * time.Second
	}
	if timeout > 12*time.Minute {
		return 12 * time.Minute
	}
	return timeout
}

func hasCompatPatchEvidence(logs string) bool {
	return strings.Contains(logs, "detected Ollama-format") ||
		strings.Contains(logs, "detected Llama 3 tokenizer metadata gap") ||
		strings.Contains(logs, "detected qwen3next GGUF")
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

func childSummaryForRunner(resp *api.ShowResponse, runner string) *api.ManifestSummary {
	for i := range resp.Manifests {
		if resp.Manifests[i].Runner == runner {
			return &resp.Manifests[i]
		}
	}
	return nil
}

func waitForRunnerManifestChild(ctx context.Context, t *testing.T, client *api.Client, model, runner string, timeout time.Duration) api.ManifestSummary {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var lastErr error
	attempt := 0
	for time.Now().Before(deadline) {
		attempt++
		resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
		if err == nil {
			if child := childSummaryForRunner(resp, runner); child != nil {
				t.Logf("%s: found %s manifest child after %d poll(s)", model, runner, attempt)
				return *child
			}
		} else {
			lastErr = err
		}
		if attempt == 1 || attempt%10 == 0 {
			t.Logf("%s: waiting for %s manifest child, poll=%d", model, runner, attempt)
		}
		time.Sleep(2 * time.Second)
	}
	if lastErr != nil {
		t.Fatalf("timed out waiting for %s manifest child for %s; last show error: %v", runner, model, lastErr)
	}
	t.Fatalf("timed out waiting for %s manifest child for %s", runner, model)
	return api.ManifestSummary{}
}

func migrationToolNumPredict(name string) int {
	if isModelFamily(name, "qwen3-vl") || isModelFamily(name, "qwen3-next") {
		return 2048
	}
	return 512
}

func shouldSmokeTools(name string) bool {
	for _, candidate := range libraryToolsModels {
		if name == candidate || strings.HasPrefix(name, candidate+":") {
			return true
		}
	}
	return false
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
	deadline := time.Now().Add(2 * time.Minute)
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
		time.Sleep(500 * time.Millisecond)
	}
	slices.Sort(last)
	t.Fatalf("timed out waiting for %s to unload; running models: %s", model, strings.Join(last, ", "))
}
