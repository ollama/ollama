// SPDX-License-Identifier: MIT
//go:build integration && level_zero

package integration

// utils_level_zero_test.go — Level Zero test helpers
//
// Provides shared utilities consumed by level_zero_test.go and
// level_zero_npu_test.go. These helpers follow the same conventions as the
// existing utils_test.go helpers (InitServerConnection, pullOrSkip, etc.) and
// are designed so that every L0 test skips cleanly on runners that lack Intel
// hardware or a ze_loader installation.
//
// Build requirements:
//   go test -tags=integration,level_zero ./integration/
//
// Key env vars (consumed by helpers):
//   OLLAMA_L0_DEVICE_INDEX  — restrict enumeration to specific device index
//   ZE_AFFINITY_MASK        — Intel loader mask (pass-through to subprocess)
//   OLLAMA_L0_NPU_ENABLE    — set to "1" to include NPU devices
//   OLLAMA_TEST_MODEL       — override default model name for chat/embed tests
//   OLLAMA_TEST_EXISTING    — if set, skip local server launch (use OLLAMA_HOST)
//   OLLAMA_L0_BIG_MODEL     — model name used for TestL0SchedulerFit (optional)
//   OLLAMA_L0_FORCE_MISSING — if "1", simulate missing ze_loader for fallback test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// l0SmallModel is the default small model used in Level Zero chat tests.
// Override with OLLAMA_TEST_MODEL to test a different model.
// tinyllama is chosen because it is the smallest commonly cached model on CI
// runners and fits comfortably in any Intel Arc GPU or NPU VRAM budget.
const l0SmallModel = "tinyllama"

// l0EmbedModel is the model used for the embedding test.
// nomic-embed-text is the standard embedding model in Ollama and is always
// present on runners that have pulled at least one embedding-capable model.
const l0EmbedModel = "nomic-embed-text"

// hasLevelZeroDevice returns true when at least one Intel Level Zero device
// (GPU or NPU) is discoverable on this machine.
//
// Detection strategy (in order of reliability):
//  1. Check whether the ze_loader shared library exists in standard paths.
//     On Linux: /usr/lib/x86_64-linux-gnu/libze_loader.so.1 or /usr/lib64/libze_loader.so.1
//     On Windows: check WINDIR\System32\ze_loader.dll existence.
//  2. If the library is present, attempt to run "ollama ps" against a running
//     server and check whether any entry contains "level_zero" in the backend
//     field. Because we cannot CGO into ze_loader from the test binary itself
//     (the test binary is built without CGO by go test), we rely on the
//     filesystem probe as the primary gate.
//
// CANNOT VERIFY: actual device enumeration without ze_loader. If the library
// exists but no compatible GPU/NPU is present, the test will start and then
// report ZE_OLLAMA_ERR_NO_DEVICE via the server log — which is acceptable.
// The skip guard here only prevents running on machines that lack the loader
// entirely (e.g., CI Ubuntu runners without Intel packages installed).
func hasLevelZeroDevice() bool {
	// Windows path check
	if runtime.GOOS == "windows" {
		windir := os.Getenv("WINDIR")
		if windir == "" {
			windir = `C:\Windows`
		}
		candidates := []string{
			filepath.Join(windir, "System32", "ze_loader.dll"),
			filepath.Join(windir, "SysWOW64", "ze_loader.dll"),
		}
		for _, p := range candidates {
			if _, err := os.Stat(p); err == nil {
				return true
			}
		}
		// Also check common Intel oneAPI install paths on Windows
		for _, base := range []string{
			`C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler`,
			`C:\Windows\System32`,
		} {
			if _, err := os.Stat(filepath.Join(base, "ze_loader.dll")); err == nil {
				return true
			}
		}
		return false
	}

	// Linux / other Unix — check standard library paths for ze_loader
	candidates := []string{
		"/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
		"/usr/lib64/libze_loader.so.1",
		"/usr/lib/libze_loader.so.1",
		"/opt/intel/oneapi/compiler/latest/linux/lib/libze_loader.so.1",
		"/usr/local/lib/libze_loader.so.1",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}

	// Also check via ldconfig cache on Linux — a best-effort probe.
	// We call `ldconfig -p` and grep for ze_loader. If ldconfig is unavailable
	// (which is fine), we fall through.
	if out, err := exec.Command("ldconfig", "-p").Output(); err == nil {
		if strings.Contains(string(out), "libze_loader") {
			return true
		}
	}

	return false
}

// buildWithL0 verifies that the ollama binary at the repo root was compiled
// with the level_zero build tag (i.e., -tags=level_zero or CMake -DGGML_LEVEL_ZERO=ON).
//
// Detection approach: check for the presence of OLLAMA_BUILD_TAGS environment
// variable (set by CI during the Level Zero preset build), or fall back to
// probing whether the libggml-level-zero shared library exists alongside the
// ollama binary. If neither is available the test is allowed to proceed anyway
// — the runtime behaviour (server failing to discover L0 devices) is what we
// ultimately gate on, not the binary metadata.
func buildWithL0(t *testing.T) string {
	t.Helper()

	// Check CI-provided env var first.
	if tags := os.Getenv("OLLAMA_BUILD_TAGS"); tags != "" {
		if !strings.Contains(tags, "level_zero") {
			t.Skip("OLLAMA_BUILD_TAGS does not include level_zero — binary not built with L0 support")
		}
		return tags
	}

	// Fall back: locate the ollama binary and check for co-located L0 library.
	cliBin, err := filepath.Abs("../ollama")
	if err != nil {
		t.Logf("buildWithL0: could not resolve binary path: %v", err)
		return ""
	}
	if runtime.GOOS == "windows" {
		cliBin += ".exe"
	}
	if _, err := os.Stat(cliBin); err != nil {
		t.Skipf("buildWithL0: ollama binary missing at %s — run 'go build .' first", cliBin)
	}

	// Check for libggml-level-zero alongside the binary.
	binDir := filepath.Dir(cliBin)
	libName := "libggml-level-zero.so"
	if runtime.GOOS == "windows" {
		libName = "ggml-level-zero.dll"
	}
	// The shared library is typically placed in build/lib/ollama/ relative to repo root.
	libCandidates := []string{
		filepath.Join(binDir, libName),
		filepath.Join(binDir, "lib", libName),
		filepath.Join(binDir, "..", "lib", "ollama", libName),
		filepath.Join(binDir, "build", "lib", "ollama", libName),
	}
	for _, p := range libCandidates {
		if _, err := os.Stat(p); err == nil {
			t.Logf("buildWithL0: found %s at %s", libName, p)
			return "level_zero"
		}
	}

	// Could not confirm; proceed without skipping — the test itself will
	// discover absence via hasLevelZeroDevice() skip guard.
	t.Logf("buildWithL0: could not confirm level_zero library presence; proceeding")
	return ""
}

// skipIfNoL0 is the idiomatic t.Skip wrapper that every L0 test must call
// as its very first statement. It combines the loader presence check and
// allows the OLLAMA_L0_FORCE_MISSING override used by TestL0Fallback.
func skipIfNoL0(t *testing.T) {
	t.Helper()
	// If the caller set OLLAMA_L0_FORCE_MISSING=1 they want to simulate a
	// missing loader; allow the test to proceed (the server will see the
	// missing loader and fall back gracefully — that is what is being tested).
	if os.Getenv("OLLAMA_L0_FORCE_MISSING") == "1" {
		t.Log("skipIfNoL0: OLLAMA_L0_FORCE_MISSING=1 — skipping device check for fallback test")
		return
	}
	if !hasLevelZeroDevice() {
		t.Skip("no Intel Level Zero device available (libze_loader not found in standard paths)")
	}
}

// skipIfNoNPU skips the test when the NPU opt-in env var is not set. NPU tests
// are doubly-gated: the L0 device must exist (skipIfNoL0) AND the NPU must be
// explicitly enabled.
func skipIfNoNPU(t *testing.T) {
	t.Helper()
	skipIfNoL0(t)
	if os.Getenv("OLLAMA_L0_NPU_ENABLE") != "1" {
		t.Skip("OLLAMA_L0_NPU_ENABLE is not set to '1' — skipping NPU test (set OLLAMA_L0_NPU_ENABLE=1 on a Meteor Lake / Lunar Lake / Arrow Lake system)")
	}
}

// runServerWithEnv starts (or connects to) an Ollama server with additional
// environment variables injected. It reuses the existing InitServerConnection
// harness from utils_test.go, injecting env vars via t.Setenv so they are
// automatically cleaned up.
//
// Returns a connected api.Client and a cleanup function that shuts down the
// server (if we started it).
func runServerWithEnv(t *testing.T, envMap map[string]string) (*api.Client, func()) {
	t.Helper()

	// Apply env overrides before InitServerConnection starts the subprocess.
	for k, v := range envMap {
		t.Setenv(k, v)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)

	client, _, cleanup := InitServerConnection(ctx, t)

	combinedCleanup := func() {
		cancel()
		cleanup()
	}

	return client, combinedCleanup
}

// l0TestModel returns the model name to use for Level Zero integration tests.
// Preference order:
//  1. OLLAMA_TEST_MODEL env var (allows CI to pin a specific model)
//  2. l0SmallModel constant ("tinyllama")
func l0TestModel() string {
	if m := os.Getenv("OLLAMA_TEST_MODEL"); m != "" {
		return m
	}
	return l0SmallModel
}
