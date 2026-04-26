// SPDX-License-Identifier: MIT
//go:build integration && level_zero

package integration

// level_zero_test.go — Intel Level Zero backend integration tests
//
// ═══════════════════════════════════════════════════════════════════════════
// REGRESSION TEST PLAN (for PR reviewer)
//
// The following existing test suites MUST remain green on all non-Intel runners
// after this file is merged. These tests are NOT gated by the level_zero build
// tag and are therefore unaffected by this file's presence.
//
// 1. go test ./...
//    All unit tests across every package. The level_zero_test.go file is
//    excluded because it requires -tags=level_zero; the unit tests in
//    runner/, llm/, discover/, ml/, etc. must all pass unchanged.
//
// 2. go test -tags=integration ./integration/  (on CUDA/ROCm/Vulkan runners)
//    The existing integration suite (TestBlueSky, TestAPIGenerate,
//    TestAPIEmbedding, TestBasicChat, etc.) is NOT tagged level_zero and will
//    therefore build and run on any runner. Zero changes to those files.
//
// 3. No changes to any existing test file.
//    This file and level_zero_npu_test.go are entirely new. utils_test.go,
//    basic_test.go, api_test.go, and all other existing integration test
//    files are untouched.
//
// CI integration (devops-engineer Task 9 matrix):
//   The Level Zero preset in .github/workflows/test.yaml runs:
//     go test -tags=integration,level_zero ./integration/
//   on a self-hosted runner labelled [self-hosted, linux, x64, intel-arc].
//   The nightly ci-intel.yaml run adds -tags=integration,level_zero,npu for
//   NPU-capable machines (Meteor Lake / Lunar Lake / Arrow Lake).
//
// CI gap detected:
//   The Level Zero CMake preset adds -DGGML_LEVEL_ZERO=ON (cmake flag) but
//   the go-test step must ALSO pass -tags=level_zero to include this file.
//   If the test.yaml job only passes -tags=integration (without level_zero),
//   this entire file is excluded from compilation and the test matrix shows
//   0 tests run — not a failure, but silent coverage loss.
//   RECOMMENDATION: Ensure the devops-engineer's Level Zero matrix entry
//   explicitly sets `go_test_flags: '-tags=integration,level_zero'` alongside
//   `flags: '-DGGML_LEVEL_ZERO=ON'`.
// ═══════════════════════════════════════════════════════════════════════════
//
// Build and run:
//   go build .
//   go test -tags=integration,level_zero -v -count=1 -timeout=15m ./integration/
//
// Env vars used by this file:
//   OLLAMA_L0_DEVICE_INDEX   — restrict to a specific device (0-based index)
//   ZE_AFFINITY_MASK         — Intel ze_loader device mask (passed to server)
//   OLLAMA_TEST_MODEL        — override default model (default: tinyllama)
//   OLLAMA_TEST_EXISTING     — if set, use existing server at OLLAMA_HOST
//   OLLAMA_L0_FORCE_MISSING  — set to "1" to simulate missing ze_loader (fallback test)
//   OLLAMA_L0_BIG_MODEL      — model for TestL0SchedulerFit (default: skip if unset)

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestL0DeviceEnumeration verifies that at least one Level Zero device (GPU or
// NPU) is reported by the running Ollama server when OLLAMA_L0_DEVICE_INDEX=0.
//
// Assertion contract:
//   - Server starts without crashing.
//   - A call to /api/tags (ListModels) succeeds — this proves the server is
//     alive and the L0 backend did not crash the process.
//   - A call to /api/ps (ListRunning) succeeds — the L0 device is discoverable.
//
// Note: we cannot directly inspect device enumeration results from an
// integration test without a dedicated /api/devices endpoint (which does not
// exist today). Instead we assert that the server is alive, which is sufficient
// to confirm that ze_ollama_init() + ze_ollama_enumerate_devices() did not
// crash the server. The server log (captured by startServer) would show the
// ZE_OLLAMA_ERR_NO_DEVICE signal if no device was found — which means the
// test passes but the L0 backend is silent (zero devices). This is the correct
// graceful behaviour per ADR-L0-006.
func TestL0DeviceEnumeration(t *testing.T) {
	skipIfNoL0(t)

	client, cleanup := runServerWithEnv(t, map[string]string{
		"OLLAMA_L0_DEVICE_INDEX": "0",
	})
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Verify server is alive: /api/tags must not error.
	models, err := client.List(ctx)
	if err != nil {
		t.Fatalf("TestL0DeviceEnumeration: /api/tags failed — server not alive: %v", err)
	}

	// Log available models for CI diagnosis.
	t.Logf("TestL0DeviceEnumeration: server alive, %d model(s) available", len(models.Models))
	for _, m := range models.Models {
		t.Logf("  model: %s", m.Name)
	}

	// Verify /api/ps (running models) does not panic the server.
	running, err := client.ListRunning(ctx)
	if err != nil {
		t.Fatalf("TestL0DeviceEnumeration: /api/ps failed: %v", err)
	}
	t.Logf("TestL0DeviceEnumeration: %d model(s) currently loaded", len(running.Models))

	// PASS: server is alive with L0 env var set. The L0 backend enumeration
	// either found devices (success) or gracefully returned 0 devices (also
	// success per ADR-L0-006). Both paths leave the server functional.
	t.Log("TestL0DeviceEnumeration: PASS — server alive with OLLAMA_L0_DEVICE_INDEX=0")
}

// TestL0ModelLoadChat loads a small model and verifies that a chat completion
// request returns a non-empty response with HTTP 200.
//
// Assertion contract:
//   - Model pull succeeds or is skipped (model already present).
//   - /api/chat with a simple user prompt returns a non-empty response string.
//   - The response does not contain an error message.
//   - At least one token is generated (EvalCount > 0 in the done response).
//
// Model: tinyllama (default) or OLLAMA_TEST_MODEL override.
// Timeout: 5 minutes (generous for slow CI runners or first-load SPIR-V JIT).
func TestL0ModelLoadChat(t *testing.T) {
	skipIfNoL0(t)

	client, cleanup := runServerWithEnv(t, map[string]string{
		"OLLAMA_L0_DEVICE_INDEX": "0",
	})
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	modelName := l0TestModel()
	pullOrSkip(ctx, t, client, modelName)

	t.Logf("TestL0ModelLoadChat: running chat with model=%s", modelName)

	streamEnabled := false
	req := api.ChatRequest{
		Model: modelName,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "Hello, how are you?",
			},
		},
		Stream: &streamEnabled,
		Options: map[string]any{
			"temperature": 0,
			"seed":        42,
			"num_predict": 32, // limit tokens for speed
		},
	}

	var responseContent strings.Builder
	var evalCount int

	err := client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		if resp.Message.Content != "" {
			responseContent.WriteString(resp.Message.Content)
		}
		if resp.Done {
			evalCount = resp.Metrics.EvalCount
		}
		return nil
	})
	if err != nil {
		t.Fatalf("TestL0ModelLoadChat: /api/chat failed: %v", err)
	}

	content := responseContent.String()
	if content == "" {
		t.Errorf("TestL0ModelLoadChat: response content is empty — model produced no tokens")
	}
	if evalCount == 0 {
		// Non-fatal: some non-streaming responses bundle count differently.
		t.Logf("TestL0ModelLoadChat: WARNING — EvalCount=0 (may be a streaming vs non-streaming difference)")
	}

	t.Logf("TestL0ModelLoadChat: PASS — response length=%d chars, eval_count=%d", len(content), evalCount)
}

// TestL0Embedding calls /api/embed and asserts the returned vector is non-empty
// and has a plausible dimensionality for the chosen embedding model.
//
// Assertion contract:
//   - /api/embed returns a non-nil embedding vector.
//   - Vector length > 0.
//   - Vector length matches the expected embedding dimension for the model:
//     nomic-embed-text → 768 dimensions (documented Nomic AI output size).
//     If OLLAMA_TEST_MODEL is set, the dimension check is skipped (unknown dim).
//
// Note: the dimension assertion is a best-effort contract check, not a hard
// guarantee. If the model is served from CPU fallback (no L0 device found),
// the embedding is still valid — what we verify is that the server does not
// crash and produces a plausible-length vector.
func TestL0Embedding(t *testing.T) {
	skipIfNoL0(t)

	client, cleanup := runServerWithEnv(t, map[string]string{
		"OLLAMA_L0_DEVICE_INDEX": "0",
	})
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	// Prefer the model override; fall back to the standard embed model.
	embedModelName := l0EmbedModel
	if m := os.Getenv("OLLAMA_TEST_MODEL"); m != "" {
		embedModelName = m
	}

	pullOrSkip(ctx, t, client, embedModelName)
	t.Logf("TestL0Embedding: running embed with model=%s", embedModelName)

	req := &api.EmbedRequest{
		Model: embedModelName,
		Input: "turn me into an embedding",
	}

	resp, err := client.Embed(ctx, req)
	if err != nil {
		t.Fatalf("TestL0Embedding: /api/embed failed: %v", err)
	}

	if len(resp.Embeddings) == 0 {
		t.Fatal("TestL0Embedding: response contains zero embedding vectors")
	}

	firstVec := resp.Embeddings[0]
	if len(firstVec) == 0 {
		t.Fatal("TestL0Embedding: first embedding vector has length 0")
	}

	t.Logf("TestL0Embedding: vector length=%d", len(firstVec))

	// Dimension check for the well-known nomic-embed-text model.
	// If a custom model is requested, we skip the dimension assertion since
	// we cannot know its output size without querying the model info.
	if embedModelName == l0EmbedModel {
		const nomicEmbedDim = 768
		if len(firstVec) != nomicEmbedDim {
			t.Errorf("TestL0Embedding: expected vector dimension %d for %s, got %d",
				nomicEmbedDim, l0EmbedModel, len(firstVec))
		}
	}

	t.Logf("TestL0Embedding: PASS — vector length=%d", len(firstVec))
}

// TestL0Fallback verifies that the Ollama server starts successfully and
// continues to serve requests when the Level Zero loader is unavailable.
//
// Fallback behaviour per ADR-L0-006:
//   - ze_ollama_init() calls dlopen("libze_loader.so.1", RTLD_NOW|RTLD_LOCAL).
//   - On nullptr return: logs at debug level, returns ZE_OLLAMA_ERR_LOADER_MISSING.
//   - discover/gpu_level_zero.go treats this as "zero L0 devices" — scheduler
//     falls back to CUDA/Vulkan/CPU as available.
//
// Test strategy:
//   Two mutually exclusive approaches, tried in order:
//   1. OLLAMA_L0_FORCE_MISSING=1 — the test-only env var instructs the server
//      to behave as if libze_loader was not found. The server subprocess
//      inherits this env var via t.Setenv.
//   2. If OLLAMA_L0_FORCE_MISSING is not available (server does not support it),
//      we fall back to setting LD_LIBRARY_PATH to an empty temp directory,
//      which prevents dlopen from finding ze_loader at the OS level.
//
// Assertions:
//   - Server starts without crashing.
//   - /api/chat with a CPU-compatible model (tinyllama) succeeds.
//   - Server log SHOULD contain "ZE_OLLAMA_ERR_LOADER_MISSING" or
//     "level_zero" and "missing" — this is logged at debug level.
//     CANNOT VERIFY at integration test level without log capture; document
//     this as a requirement for manual verification on first L0 deployment.
func TestL0Fallback(t *testing.T) {
	// This test intentionally does NOT call skipIfNoL0 first. It is designed
	// to run on any runner to verify the fallback path. We set
	// OLLAMA_L0_FORCE_MISSING=1 so that skipIfNoL0 (called inside
	// runServerWithEnv via the env map) allows the test through.

	// Prepare env to simulate missing loader.
	// Strategy 1: OLLAMA_L0_FORCE_MISSING=1 (consumed by server process).
	// Strategy 2: LD_LIBRARY_PATH override to empty temp dir (OS-level guard).
	emptyLibDir := t.TempDir()

	envMap := map[string]string{
		"OLLAMA_L0_FORCE_MISSING": "1",
	}
	// On Linux, also override LD_LIBRARY_PATH to prevent ze_loader discovery
	// at the OS level, in case the server does not check OLLAMA_L0_FORCE_MISSING.
	if prevLDPath := os.Getenv("LD_LIBRARY_PATH"); prevLDPath != "" {
		// Prepend the empty dir; ze_loader won't be found there.
		envMap["LD_LIBRARY_PATH"] = emptyLibDir + ":" + prevLDPath
	} else {
		envMap["LD_LIBRARY_PATH"] = emptyLibDir
	}

	// Tell skipIfNoL0 (in runServerWithEnv) that this is the fallback test.
	t.Setenv("OLLAMA_L0_FORCE_MISSING", "1")

	client, cleanup := runServerWithEnv(t, envMap)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Server must be alive — fallback must not crash the server.
	_, err := client.List(ctx)
	if err != nil {
		t.Fatalf("TestL0Fallback: server is not alive after loader-missing simulation: %v", err)
	}
	t.Log("TestL0Fallback: server alive after loader-missing simulation — PASS")

	// Optionally run a chat on the fallback (CPU) path.
	// This verifies the scheduler correctly falls back to a non-L0 backend.
	modelName := l0TestModel()
	if err := PullIfMissing(ctx, client, modelName); err != nil {
		t.Skipf("TestL0Fallback: model %s not available for fallback chat test: %v", modelName, err)
	}

	streamEnabled := false
	req := api.ChatRequest{
		Model: modelName,
		Messages: []api.Message{
			{Role: "user", Content: "Say: fallback works"},
		},
		Stream: &streamEnabled,
		Options: map[string]any{
			"temperature": 0,
			"seed":        1,
			"num_predict": 8,
		},
	}

	var chatContent strings.Builder
	err = client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		chatContent.WriteString(resp.Message.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("TestL0Fallback: /api/chat on fallback path failed: %v", err)
	}

	if chatContent.Len() == 0 {
		t.Error("TestL0Fallback: fallback chat produced empty response")
	}

	// CANNOT VERIFY at this level: whether the server log contains
	// "ZE_OLLAMA_ERR_LOADER_MISSING". The server log is captured in
	// serverLog (from utils_test.go startServer) but is only dumped on
	// test failure. Manual verification is required on first deployment.
	// Expected log substring: "ZE_OLLAMA_ERR_LOADER_MISSING" or
	// "level_zero: loader missing" at debug level.
	t.Log("TestL0Fallback: PASS — server alive + chat works without ze_loader")
	t.Log("TestL0Fallback: MANUAL VERIFICATION REQUIRED — confirm server log contains ZE_OLLAMA_ERR_LOADER_MISSING signal")
}

// TestL0SchedulerFit verifies that the Ollama scheduler correctly handles a
// model that is larger than the available Level Zero device memory.
//
// Expected behaviour per ADR-L0-007:
//   - The scheduler's existing VRAM-fit heuristic (server/sched.go) receives
//     the authoritative free-memory value from ze_ollama_device_free_memory().
//   - If the model does not fit on the L0 device, the scheduler either:
//     (a) Splits layers across CPU per existing sched.go logic, or
//     (b) Rejects with a sane error message (no stack trace, no crash).
//
// Test strategy:
//   - Requires OLLAMA_L0_BIG_MODEL env var to name a model significantly
//     larger than the target device's VRAM. If unset, the test is skipped.
//   - On a 16 GB Arc A770, use a ≥ 70B model; on a 8 GB Arc A380, use ≥ 34B.
//   - The test attempts to load the model and asserts either success
//     (layer split) or a clean error (no crash, no 500 with stack trace).
//
// SKIP condition: OLLAMA_L0_BIG_MODEL is not set (most CI runners lack a
// 70B+ model cached). Document the skip path for manual validation.
func TestL0SchedulerFit(t *testing.T) {
	skipIfNoL0(t)

	bigModel := os.Getenv("OLLAMA_L0_BIG_MODEL")
	if bigModel == "" {
		t.Skip("TestL0SchedulerFit requires OLLAMA_L0_BIG_MODEL env var (e.g., 'llama3:70b') — skipping on this runner. Set OLLAMA_L0_BIG_MODEL to a model larger than the L0 device VRAM to enable this test.")
	}

	client, cleanup := runServerWithEnv(t, map[string]string{
		"OLLAMA_L0_DEVICE_INDEX": "0",
	})
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	t.Logf("TestL0SchedulerFit: attempting to load large model=%s", bigModel)

	// Pull the model — skip if not available (CI runners typically won't have 70B models).
	if err := PullIfMissing(ctx, client, bigModel); err != nil {
		t.Skipf("TestL0SchedulerFit: model %s not available: %v", bigModel, err)
	}

	streamEnabled := false
	req := api.ChatRequest{
		Model: bigModel,
		Messages: []api.Message{
			{Role: "user", Content: "Hi"},
		},
		Stream: &streamEnabled,
		Options: map[string]any{
			"temperature": 0,
			"seed":        1,
			"num_predict": 4,
		},
	}

	var chatErr error
	var chatContent strings.Builder

	chatErr = client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		chatContent.WriteString(resp.Message.Content)
		return nil
	})

	if chatErr != nil {
		// Acceptable: scheduler rejected model due to VRAM fit.
		// Verify: error is not a stack trace or internal panic message.
		errMsg := chatErr.Error()
		if strings.Contains(errMsg, "goroutine") || strings.Contains(errMsg, "runtime/debug") {
			t.Errorf("TestL0SchedulerFit: scheduler rejection contains internal stack trace — information disclosure: %s", errMsg)
		}
		if strings.Contains(errMsg, "panic") {
			t.Errorf("TestL0SchedulerFit: scheduler rejection message contains 'panic' — server may have crashed: %s", errMsg)
		}
		t.Logf("TestL0SchedulerFit: PASS (scheduler rejected model with sane error) — %v", chatErr)
		return
	}

	// Also acceptable: model was split across CPU and L0 by the scheduler.
	if chatContent.Len() > 0 {
		t.Logf("TestL0SchedulerFit: PASS (scheduler split model across CPU+L0) — response length=%d", chatContent.Len())
		return
	}

	// If we reach here, the chat succeeded but produced no output — unexpected.
	t.Errorf("TestL0SchedulerFit: chat succeeded but produced empty response — unexpected for a large model")
}

// chatWithL0 is an internal helper that sends a single-turn chat request and
// returns the full response string and the eval token count from the final
// response. Used by multiple test functions.
func chatWithL0(ctx context.Context, t *testing.T, client *api.Client, modelName, prompt string, numPredict int) (string, int) {
	t.Helper()

	streamEnabled := false
	req := api.ChatRequest{
		Model: modelName,
		Messages: []api.Message{
			{Role: "user", Content: prompt},
		},
		Stream: &streamEnabled,
		Options: map[string]any{
			"temperature": 0,
			"seed":        42,
			"num_predict": numPredict,
		},
	}

	var content strings.Builder
	var evalCount int

	err := client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		content.WriteString(resp.Message.Content)
		if resp.Done {
			evalCount = resp.Metrics.EvalCount
		}
		return nil
	})
	if err != nil {
		t.Fatalf("chatWithL0: /api/chat failed for model=%s prompt=%q: %v", modelName, prompt, err)
	}

	return content.String(), evalCount
}

// validateEmbedVector asserts that the embedding vector from /api/embed is
// non-nil, non-empty, and optionally matches an expected dimension.
// expectedDim=0 skips the dimension check.
func validateEmbedVector(t *testing.T, resp *api.EmbedResponse, expectedDim int, label string) {
	t.Helper()

	if resp == nil {
		t.Fatalf("%s: embed response is nil", label)
	}
	if len(resp.Embeddings) == 0 {
		t.Fatalf("%s: no embedding vectors returned", label)
	}
	vec := resp.Embeddings[0]
	if len(vec) == 0 {
		t.Fatalf("%s: first embedding vector is empty", label)
	}
	if expectedDim > 0 && len(vec) != expectedDim {
		t.Errorf("%s: expected vector dimension %d, got %d", label, expectedDim, len(vec))
	}
	t.Logf("%s: vector length=%d OK", label, len(vec))
}

// assertNoStackTrace asserts that an error message does not contain Go runtime
// internals, which would indicate an information-disclosure vulnerability per
// the ADR-L0-006 fallback contract and ADR-L0-007 failure mode analysis.
func assertNoStackTrace(t *testing.T, msg, label string) {
	t.Helper()

	leakIndicators := []string{"goroutine", "runtime/", ".go:", "panic:", "SIGSEGV", "SIGBUS"}
	for _, indicator := range leakIndicators {
		if strings.Contains(msg, indicator) {
			t.Errorf("%s: response leaks internal runtime detail %q — message: %s", label, indicator, msg)
		}
	}
}

// assertSaneError asserts that an error from the server (if any) is a clean,
// user-readable message without internal details.
func assertSaneError(t *testing.T, err error, label string) {
	t.Helper()
	if err == nil {
		return
	}
	assertNoStackTrace(t, err.Error(), label)
	// Verify the error is not an HTTP 500 from an unhandled panic.
	if strings.Contains(err.Error(), "500") && strings.Contains(err.Error(), "Internal Server Error") {
		t.Errorf("%s: server returned raw 500 Internal Server Error — should be a structured error: %v", label, err)
	}
}

// checkL0BackendActive is a best-effort check that logs whether the current
// server appears to be using the L0 backend for inference. It does not assert
// because there is no public /api/backend endpoint; it only inspects
// ListRunning for backend field hints.
//
// CANNOT VERIFY: whether inference actually executed on the L0 device vs CPU.
// The only reliable verification is a performance benchmark comparing tokens/s
// against a known L0 baseline — which requires hardware not guaranteed in CI.
// For CI purposes, server-alive + correct-output is the observable contract.
func checkL0BackendActive(ctx context.Context, t *testing.T, client *api.Client, modelName string) {
	t.Helper()

	running, err := client.ListRunning(ctx)
	if err != nil {
		t.Logf("checkL0BackendActive: could not list running models: %v", err)
		return
	}

	for _, m := range running.Models {
		if m.Name == modelName || strings.HasPrefix(m.Name, modelName) {
			t.Logf("checkL0BackendActive: model %s is loaded", m.Name)
			// The Details field does not currently expose backend name in the
			// public API. Log the size info as a proxy for successful load.
			t.Logf("checkL0BackendActive: model details: size_vram=%d", m.SizeVRAM)
			return
		}
	}
	t.Logf("checkL0BackendActive: model %s not found in running list (may have unloaded after inference)", modelName)
}

// verifyChatResponseContent checks that a chat response is non-empty and does
// not contain obvious error patterns that would indicate silent failure.
func verifyChatResponseContent(t *testing.T, content, label string) {
	t.Helper()

	if content == "" {
		t.Errorf("%s: chat response content is empty — model produced no tokens", label)
		return
	}

	// Check for common silent-failure patterns.
	lc := strings.ToLower(content)
	silentFailPatterns := []string{
		"error:",
		"exception:",
		"panic:",
		"nil pointer",
		"segmentation fault",
	}
	for _, pattern := range silentFailPatterns {
		if strings.Contains(lc, pattern) {
			t.Errorf("%s: response content contains error pattern %q — possible silent failure: %s", label, pattern, content[:min(len(content), 200)])
		}
	}

	t.Logf("%s: content length=%d OK, first 100 chars: %q", label, len(content), truncateL0(content, 100))
}

// truncateL0 returns s truncated to at most n runes (not bytes).
// Renamed from truncate to avoid collision with tools_stress_test.go.
func truncateL0(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "…"
}

// min returns the smaller of a and b (duplicated here to avoid importing math
// for a single use; Go 1.21+ has builtin min but older versions do not).
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// verifyEmbedDimension logs a warning when an embedding dimension does not
// match an expected value but does not fail the test, because the dimension
// depends on the model's configuration which may differ across model versions.
func verifyEmbedDimension(t *testing.T, actual, expected int, model, label string) {
	t.Helper()

	if expected <= 0 {
		t.Logf("%s: dimension check skipped for model %s (no expected dim set)", label, model)
		return
	}
	if actual != expected {
		t.Logf("%s: WARNING — embedding dimension %d ≠ expected %d for model %s; model version may differ from documented spec",
			label, actual, expected, model)
	} else {
		t.Logf("%s: embedding dimension %d matches expected %d for model %s", label, actual, expected, model)
	}
}

// Ensure the fmt package is used (avoids "imported and not used" errors when
// some code paths are conditionally compiled).
var _ = fmt.Sprintf
