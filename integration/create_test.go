//go:build integration

package integration

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

const testdataModelsDir = "testdata/models"

const (
	defaultSafetensorsCreateRepo = "Qwen/Qwen3-0.6B"
	defaultSafetensorsCreateDir  = "Qwen3-0.6B"
	createIdleTimeout            = 3 * time.Minute
	createMaxTimeout             = 30 * time.Minute
	hfDownloadIdleTimeout        = 5 * time.Minute
	hfDownloadMaxTimeout         = 45 * time.Minute
)

type activityWriter struct {
	dst        io.Writer
	lastOutput *atomic.Int64
}

func (w activityWriter) Write(p []byte) (int, error) {
	w.lastOutput.Store(time.Now().UnixNano())
	return w.dst.Write(p)
}

func runCommandWithIdleTimeout(parent context.Context, t *testing.T, cmd *exec.Cmd, stdoutDst, stderrDst io.Writer, idleTimeout, maxTimeout time.Duration) error {
	t.Helper()

	origPath := cmd.Path
	origArgs := append([]string(nil), cmd.Args...)
	origEnv := append([]string(nil), cmd.Env...)
	origDir := cmd.Dir

	ctx, cancel := context.WithTimeout(parent, maxTimeout)
	defer cancel()
	cmd = exec.CommandContext(ctx, origPath, origArgs[1:]...)
	cmd.Env = origEnv
	cmd.Dir = origDir

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	var lastOutput atomic.Int64
	lastOutput.Store(time.Now().UnixNano())

	var stderrCopy bytes.Buffer
	stdoutWriter := activityWriter{dst: stdoutDst, lastOutput: &lastOutput}
	stderrWriter := activityWriter{dst: io.MultiWriter(stderrDst, &stderrCopy), lastOutput: &lastOutput}

	if err := cmd.Start(); err != nil {
		return err
	}

	var copyWG sync.WaitGroup
	copyWG.Add(2)
	go func() {
		defer copyWG.Done()
		_, _ = io.Copy(stdoutWriter, stdoutPipe)
	}()
	go func() {
		defer copyWG.Done()
		_, _ = io.Copy(stderrWriter, stderrPipe)
	}()

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case err := <-done:
			copyWG.Wait()
			if err == nil {
				return nil
			}
			if ctx.Err() == context.DeadlineExceeded {
				return fmt.Errorf("command exceeded max timeout %s: %w\nstderr: %s", maxTimeout, err, stderrCopy.String())
			}
			return fmt.Errorf("%w\nstderr: %s", err, stderrCopy.String())
		case <-ticker.C:
			last := time.Unix(0, lastOutput.Load())
			if time.Since(last) > idleTimeout {
				cancel()
				err := <-done
				copyWG.Wait()
				return fmt.Errorf("command stalled for %s: %w\nstderr: %s", idleTimeout, err, stderrCopy.String())
			}
		case <-parent.Done():
			cancel()
			err := <-done
			copyWG.Wait()
			if errors.Is(parent.Err(), context.DeadlineExceeded) {
				return fmt.Errorf("parent context expired: %w\nstderr: %s", err, stderrCopy.String())
			}
			return fmt.Errorf("command canceled by parent context: %w\nstderr: %s", err, stderrCopy.String())
		}
	}
}

// skipIfRemote skips the test if OLLAMA_HOST points to a non-local server.
// Imagegen creation requires localhost since it writes blobs directly to disk.
// Safetensors LLM creation works against any server via the API pipeline.
func skipIfRemote(t *testing.T) {
	t.Helper()
	if !ollamaHostIsRemote(os.Getenv("OLLAMA_HOST")) {
		return
	}
	t.Skipf("safetensors/imagegen creation requires a local server (OLLAMA_HOST=%s)", os.Getenv("OLLAMA_HOST"))
}

func ollamaHostIsRemote(host string) bool {
	if host == "" {
		return false // default is localhost
	}
	// Strip scheme if present
	_, hostport, ok := strings.Cut(host, "://")
	if !ok {
		hostport = host
	}
	h, _, err := net.SplitHostPort(hostport)
	if err != nil {
		h = hostport
	}
	if h == "" || h == "localhost" {
		return false
	}
	ip := net.ParseIP(h)
	if ip != nil && (ip.IsLoopback() || ip.IsUnspecified()) {
		return false
	}
	return true
}

// findHFCLI returns the path to the HuggingFace CLI, or "" if not found.
func findHFCLI() string {
	for _, name := range []string{"huggingface-cli", "hf"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return ""
}

// downloadHFModel idempotently downloads a HuggingFace model to destDir.
// Skips the test if CLI is missing and model isn't already present.
func downloadHFModel(t *testing.T, repo, destDir string, extraArgs ...string) {
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
	os.MkdirAll(destDir, 0o755)

	args := []string{"download", repo, "--local-dir", destDir}
	args = append(args, extraArgs...)
	cmd := exec.Command(cli, args...)
	if err := runCommandWithIdleTimeout(context.Background(), t, cmd, os.Stdout, os.Stderr, hfDownloadIdleTimeout, hfDownloadMaxTimeout); err != nil {
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

// runOllamaCreate runs "ollama create" as a subprocess. Skips the test if
// the error indicates the server is remote.
func runOllamaCreate(ctx context.Context, t *testing.T, args ...string) {
	t.Helper()
	createCmd := exec.Command(ollamaBin(), append([]string{"create"}, args...)...)
	var createStderr bytes.Buffer
	if err := runCommandWithIdleTimeout(ctx, t, createCmd, os.Stdout, io.MultiWriter(os.Stderr, &createStderr), createIdleTimeout, createMaxTimeout); err != nil {
		stderr := createStderr.String()
		if strings.Contains(stderr, "requires a local server") || strings.Contains(stderr, "remote server") {
			t.Skip("safetensors creation requires a local server")
		}
		t.Fatalf("ollama create failed: %v\nstderr: %s", err, stderr)
	}
}

func TestCreateSafetensorsLLM(t *testing.T) {
	if testModel != "" {
		t.Skip("exercises create pipeline with a fixed source model, not applicable with model override")
	}
	// Allow overriding the model directory via env var for testing with
	// larger models (e.g., OLLAMA_TEST_SAFETENSORS_MODEL_DIR=/path/to/Qwen3-32B).
	modelDir := os.Getenv("OLLAMA_TEST_SAFETENSORS_MODEL_DIR")
	if modelDir == "" {
		// Keep the default safetensors integration fixture separate from the GGUF
		// llama fixture for now. Qwen3 is smaller than TinyLlama and exercises a
		// real MLX-backed safetensors create flow without pulling a full repo snapshot.
		modelDir = filepath.Join(testdataModelsDir, defaultSafetensorsCreateDir)
		downloadHFModel(t, defaultSafetensorsCreateRepo, modelDir,
			"--include", "*.safetensors",
			"--include", "*.json",
			"--include", "tokenizer*",
			"--include", "*.model")
	} else {
		t.Logf("Using existing safetensors model at %s", modelDir)
	}

	// Verify it looks like a valid safetensors model directory
	if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
		t.Fatalf("config.json not found in %s — not a valid safetensors model directory", modelDir)
	}

	ensureMLXLibraryPath(t)

	absModelDir, err := filepath.Abs(modelDir)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}
	customModel := os.Getenv("OLLAMA_TEST_SAFETENSORS_MODEL_DIR") != ""

	testCases := []struct {
		name        string
		forceRemote string
		skipLocal   bool
	}{
		{name: "local", forceRemote: "0", skipLocal: true},
		{name: "remote", forceRemote: "1"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.skipLocal && ollamaHostIsRemote(os.Getenv("OLLAMA_HOST")) {
				t.Skipf("local experimental create requires a local host (OLLAMA_HOST=%s)", os.Getenv("OLLAMA_HOST"))
			}

			t.Setenv("OLLAMA_CREATE_REMOTE", tc.forceRemote)

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			modelName := "test-safetensors-llm-" + tc.name
			defer func() {
				cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), time.Minute)
				defer cleanupCancel()
				if err := client.Delete(cleanupCtx, &api.DeleteRequest{Model: modelName}); err != nil {
					t.Logf("Warning: failed to delete test model %s: %v", modelName, err)
				}
			}()

			// Create a Modelfile pointing to the model directory and rely on the
			// imported model metadata/parser behavior rather than a hard-coded template.
			modelfileContent := "FROM " + absModelDir + "\n"
			tmpModelfile := filepath.Join(t.TempDir(), "Modelfile")
			if err := os.WriteFile(tmpModelfile, []byte(modelfileContent), 0o644); err != nil {
				t.Fatalf("Failed to write Modelfile: %v", err)
			}

			createArgs := []string{modelName, "--experimental", "-f", tmpModelfile}
			if q := os.Getenv("OLLAMA_TEST_QUANTIZE"); q != "" {
				createArgs = append(createArgs, "--quantize", q)
			}

			createStart := time.Now()
			runOllamaCreate(ctx, t, createArgs...)
			t.Logf("Create took %s", time.Since(createStart))

			showReq := &api.ShowRequest{Name: modelName}
			showResp, err := client.Show(ctx, showReq)
			if err != nil {
				t.Fatalf("Model show failed after create: %v", err)
			}
			t.Logf("Created model details: %+v", showResp.Details)

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
			err = client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
				output.WriteString(resp.Message.Content)
				return nil
			})
			if err != nil {
				if customModel {
					t.Logf("Chat failed (may be unsupported architecture): %v", err)
					return
				}
				t.Fatalf("Chat failed: %v", err)
			}

			text := output.String()
			t.Logf("Generated output: %q", text)
			if customModel && text == "" {
				t.Logf("Empty output from custom model (may need specific chat template) — skipping coherence check")
			} else {
				assertCoherentOutput(t, text)
			}

		})
	}
}

func TestCreateGGUF(t *testing.T) {
	if testModel != "" {
		t.Skip("exercises create pipeline with a fixed source model, not applicable with model override")
	}
	modelDir := filepath.Join(testdataModelsDir, "Llama-3.2-1B-GGUF")
	downloadHFModel(t, "bartowski/Llama-3.2-1B-Instruct-GGUF", modelDir,
		"--include", "Llama-3.2-1B-Instruct-IQ3_M.gguf")

	// Find the GGUF file
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		t.Fatalf("Failed to read model dir: %v", err)
	}

	var ggufPath string
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".gguf" {
			ggufPath = filepath.Join(modelDir, e.Name())
			break
		}
	}
	if ggufPath == "" {
		t.Skip("No GGUF file found in model directory")
	}

	absGGUF, err := filepath.Abs(ggufPath)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	modelName := "test-llama32-gguf"
	defer func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), time.Minute)
		defer cleanupCancel()
		if err := client.Delete(cleanupCtx, &api.DeleteRequest{Model: modelName}); err != nil {
			t.Logf("Warning: failed to delete test model %s: %v", modelName, err)
		}
	}()

	// Create a Modelfile and use the CLI
	tmpModelfile := filepath.Join(t.TempDir(), "Modelfile")
	if err := os.WriteFile(tmpModelfile, []byte("FROM "+absGGUF+"\n"), 0o644); err != nil {
		t.Fatalf("Failed to write Modelfile: %v", err)
	}

	createCmd := exec.CommandContext(ctx, ollamaBin(), "create", modelName, "-f", tmpModelfile)
	createCmd.Stdout = os.Stdout
	createCmd.Stderr = os.Stderr
	if err := createCmd.Run(); err != nil {
		t.Fatalf("ollama create failed: %v", err)
	}

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
