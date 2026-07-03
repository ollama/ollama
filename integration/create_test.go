//go:build integration

package integration

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	testdataModelsDir = "testdata/models"

	tinyLlamaRepo     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	tinyLlamaRevision = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
	tinyLlamaModelDir = "TinyLlama-1.1B"

	llama32GGUFRepo     = "bartowski/Llama-3.2-1B-Instruct-GGUF"
	llama32GGUFRevision = "067b946cf014b7c697f3654f621d577a3e3afd1c"
	llama32GGUFFile     = "Llama-3.2-1B-Instruct-IQ3_M.gguf"
)

// findHFCLI returns the path to the HuggingFace CLI, or "" if not found.
func findHFCLI() string {
	if p, err := exec.LookPath("hf"); err == nil {
		return p
	}
	return ""
}

// downloadHFModel idempotently downloads a HuggingFace model to destDir.
// Skips the test if CLI is missing and model isn't already present.
func downloadHFModel(t *testing.T, repo, revision, destDir string, extraArgs ...string) {
	t.Helper()

	// Check if model already exists
	if _, err := os.Stat(destDir); err == nil {
		entries, err := os.ReadDir(destDir)
		if err == nil && len(entries) > 0 {
			t.Logf("Model %s already present at %s", repo, destDir)
			return
		}
	}

	cli := findHFCLI()
	if cli == "" {
		t.Skipf("HuggingFace CLI not found and model %s not present at %s", repo, destDir)
	}

	t.Logf("Downloading %s to %s", repo, destDir)
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		t.Fatalf("Failed to create model directory %s: %v", destDir, err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Minute)
	defer cancel()

	args := []string{"download", repo, "--revision", revision, "--local-dir", destDir}
	args = append(args, extraArgs...)
	cmd := exec.CommandContext(ctx, cli, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to download %s: %v", repo, err)
	}
}

// ollamaBin returns the path to the ollama binary to use for tests.
// Prefers OLLAMA_BIN env, then falls back to the built binary at ../ollama
// (same binary the integration test server uses).
func ollamaBin() string {
	if bin := os.Getenv("OLLAMA_BIN"); bin != "" {
		return bin
	}
	if abs, err := filepath.Abs("../ollama"); err == nil {
		if _, err := os.Stat(abs); err == nil {
			return abs
		}
	}
	return "ollama"
}

// ensureMLXLibraryPath sets OLLAMA_LIBRARY_PATH so the MLX dynamic library
// is discoverable. Integration tests run from integration/ dir, so the
// default CWD-based search won't find the library at the repo root.
func ensureMLXLibraryPath(t *testing.T) {
	t.Helper()
	if libPath, err := filepath.Abs("../build/lib/ollama"); err == nil {
		if _, err := os.Stat(libPath); err == nil {
			if existing := os.Getenv("OLLAMA_LIBRARY_PATH"); existing != "" {
				t.Setenv("OLLAMA_LIBRARY_PATH", existing+string(filepath.ListSeparator)+libPath)
			} else {
				t.Setenv("OLLAMA_LIBRARY_PATH", libPath)
			}
		}
	}
}

// runOllamaCreate runs "ollama create" as a subprocess.
func runOllamaCreate(ctx context.Context, t *testing.T, args ...string) {
	t.Helper()
	createCmd := exec.CommandContext(ctx, ollamaBin(), append([]string{"create"}, args...)...)
	createCmd.Stdout = os.Stdout
	createCmd.Stderr = os.Stderr
	if err := createCmd.Run(); err != nil {
		t.Fatalf("ollama create failed: %v", err)
	}
}

func isolateCreateModelStore(t *testing.T) {
	t.Helper()
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
	}
}

func createIntegrationModelName(prefix string) string {
	return fmt.Sprintf("%s-%x", prefix, time.Now().UnixNano())
}

func tinyLlamaModelfile(t *testing.T, modelDir string) string {
	t.Helper()
	absModelDir, err := filepath.Abs(modelDir)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}

	// Include an Ollama template because the importer does not convert a Hugging
	// Face chat template into a Modelfile template.
	contents := "FROM " + absModelDir + "\n" +
		"TEMPLATE \"{{ if .System }}<|system|>\n{{ .System }}</s>\n{{ end }}" +
		"{{ if .Prompt }}<|user|>\n{{ .Prompt }}</s>\n{{ end }}" +
		"<|assistant|>\n{{ .Response }}</s>\n\"\n"
	name := filepath.Join(t.TempDir(), "Modelfile")
	if err := os.WriteFile(name, []byte(contents), 0o644); err != nil {
		t.Fatalf("Failed to write Modelfile: %v", err)
	}
	return name
}

func cleanupCreatedModel(t *testing.T, client *api.Client, modelName string) {
	t.Helper()
	t.Cleanup(func() {
		// The test context is canceled before cleanup functions run.
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if err := client.Delete(ctx, &api.DeleteRequest{Model: modelName}); err != nil {
			t.Logf("failed to delete test model %s: %v", modelName, err)
		}
	})
}

func verifyTinyLlamaChat(ctx context.Context, t *testing.T, client *api.Client, modelName string) {
	t.Helper()
	chatReq := &api.ChatRequest{
		Model: modelName,
		Messages: []api.Message{
			{Role: "user", Content: "Write a short sentence about the weather."},
		},
		Options: map[string]interface{}{
			"num_predict": 20,
			"temperature": 0.0,
		},
	}

	var output strings.Builder
	err := client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
		output.WriteString(resp.Message.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}

	text := output.String()
	t.Logf("Generated output: %q", text)
	assertCoherentOutput(t, text)
}

func runCreateSafetensorsLLM(t *testing.T) {
	if testModel != "" {
		t.Skip("exercises create pipeline with a fixed source model, not applicable with model override")
	}
	isolateCreateModelStore(t)
	modelDir := filepath.Join(testdataModelsDir, tinyLlamaModelDir)
	downloadHFModel(t, tinyLlamaRepo, tinyLlamaRevision, modelDir)

	ensureMLXLibraryPath(t)
	t.Setenv("OLLAMA_CREATE_REMOTE", "1")

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	t.Cleanup(cleanup)

	modelName := createIntegrationModelName("test-tinyllama-safetensors")
	cleanupCreatedModel(t, client, modelName)

	runOllamaCreate(ctx, t, modelName, "-f", tinyLlamaModelfile(t, modelDir))

	// Verify model exists via show
	showReq := &api.ShowRequest{Name: modelName}
	showResp, err := client.Show(ctx, showReq)
	if err != nil {
		t.Fatalf("Model show failed after create: %v", err)
	}
	t.Logf("Created model details: %+v", showResp.Details)

	verifyTinyLlamaChat(ctx, t, client, modelName)
}

func runCreateGGUF(t *testing.T) {
	if testModel != "" {
		t.Skip("exercises create pipeline with a fixed source model, not applicable with model override")
	}
	isolateCreateModelStore(t)
	modelDir := filepath.Join(testdataModelsDir, "Llama-3.2-1B-GGUF")
	downloadHFModel(t, llama32GGUFRepo, llama32GGUFRevision, modelDir,
		"--include", llama32GGUFFile)

	ggufPath := filepath.Join(modelDir, llama32GGUFFile)
	absGGUF, err := filepath.Abs(ggufPath)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	t.Cleanup(cleanup)

	modelName := createIntegrationModelName("test-llama32-gguf")
	cleanupCreatedModel(t, client, modelName)

	// Create a Modelfile and use the CLI
	tmpModelfile := filepath.Join(t.TempDir(), "Modelfile")
	if err := os.WriteFile(tmpModelfile, []byte("FROM "+absGGUF+"\n"), 0o644); err != nil {
		t.Fatalf("Failed to write Modelfile: %v", err)
	}

	runOllamaCreate(ctx, t, modelName, "-f", tmpModelfile)

	// Verify model exists
	showReq := &api.ShowRequest{Name: modelName}
	_, err = client.Show(ctx, showReq)
	if err != nil {
		t.Fatalf("Model show failed after create: %v", err)
	}

	// Generate and verify output is coherent
	genReq := &api.GenerateRequest{
		Model:  modelName,
		Prompt: "Write a short sentence about the weather.",
		Options: map[string]interface{}{
			"num_predict": 20,
			"temperature": 0.0,
		},
	}

	var output strings.Builder
	err = client.Generate(ctx, genReq, func(resp api.GenerateResponse) error {
		output.WriteString(resp.Response)
		return nil
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	text := output.String()
	t.Logf("Generated output: %q", text)
	assertCoherentOutput(t, text)
}

// assertCoherentOutput checks that model output looks like real language, not
// garbled binary or repeated garbage. This catches corrupted model creation
// where inference "works" but produces nonsense.
func assertCoherentOutput(t *testing.T, text string) {
	t.Helper()

	if len(text) == 0 {
		t.Fatal("model produced empty output")
	}

	// Check minimum length — 20 tokens should produce at least a few words
	if len(text) < 5 {
		t.Fatalf("model output suspiciously short (%d bytes): %q", len(text), text)
	}

	// Check for mostly-printable ASCII/Unicode — garbled models often emit
	// high ratios of control characters or replacement characters
	unprintable := 0
	for _, r := range text {
		if r < 0x20 && r != '\n' && r != '\r' && r != '\t' {
			unprintable++
		}
		if r == '\ufffd' { // Unicode replacement character
			unprintable++
		}
	}
	ratio := float64(unprintable) / float64(len([]rune(text)))
	if ratio > 0.3 {
		t.Fatalf("model output is %.0f%% unprintable characters (likely garbled): %q", ratio*100, text)
	}

	// Check it contains at least one space — real language has word boundaries
	if !strings.Contains(text, " ") {
		t.Fatalf("model output contains no spaces (likely garbled): %q", text)
	}

	// Check for excessive repetition — a broken model might repeat one token
	words := strings.Fields(text)
	if len(words) >= 4 {
		counts := map[string]int{}
		for _, w := range words {
			counts[strings.ToLower(w)]++
		}
		for w, c := range counts {
			if c > len(words)*3/4 {
				t.Fatalf("model output is excessively repetitive (%q appears %d/%d times): %q", w, c, len(words), text)
			}
		}
	}
}
