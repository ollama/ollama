//go:build mlx

package mlxrunner

import "testing"

func TestPrefillChunkSize(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "")
	if got := prefillChunkSize(false); got != 2<<10 {
		t.Fatalf("prefillChunkSize(false) = %d, want %d", got, 2<<10)
	}
	if got := prefillChunkSize(true); got != 32 {
		t.Fatalf("prefillChunkSize(true) = %d, want %d", got, 32)
	}
}

func TestPrefillChunkSizeEnvOverride(t *testing.T) {
	t.Setenv("OLLAMA_MLX_PREFILL_CHUNK", "96")
	if got := prefillChunkSize(false); got != 96 {
		t.Fatalf("prefillChunkSize(false) with env = %d, want %d", got, 96)
	}
	if got := prefillChunkSize(true); got != 96 {
		t.Fatalf("prefillChunkSize(true) with env = %d, want %d", got, 96)
	}
}

func TestMLXDebugMemoryEnabled(t *testing.T) {
	t.Setenv("OLLAMA_MLX_DEBUG_MEMORY", "")
	if mlxDebugMemoryEnabled() {
		t.Fatal("mlxDebugMemoryEnabled() = true, want false")
	}

	t.Setenv("OLLAMA_MLX_DEBUG_MEMORY", "1")
	if !mlxDebugMemoryEnabled() {
		t.Fatal("mlxDebugMemoryEnabled() = false, want true")
	}
}

func TestFinalizeRequestCachesUsesPromptCachePath(t *testing.T) {
	insertCalls := 0
	freeCalls := 0
	logPhase := ""

	finalizeRequestCaches(
		true,
		func() { insertCalls++ },
		func() { freeCalls++ },
		func(phase string, _ int) { logPhase = phase },
	)

	if insertCalls != 1 {
		t.Fatalf("insert calls = %d, want 1", insertCalls)
	}
	if freeCalls != 0 {
		t.Fatalf("free calls = %d, want 0", freeCalls)
	}
	if logPhase != "request_done_cached" {
		t.Fatalf("log phase = %q, want %q", logPhase, "request_done_cached")
	}
}

func TestFinalizeRequestCachesUsesFreePath(t *testing.T) {
	insertCalls := 0
	freeCalls := 0
	logPhase := ""

	finalizeRequestCaches(
		false,
		func() { insertCalls++ },
		func() { freeCalls++ },
		func(phase string, _ int) { logPhase = phase },
	)

	if insertCalls != 0 {
		t.Fatalf("insert calls = %d, want 0", insertCalls)
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
	if logPhase != "request_done_freed" {
		t.Fatalf("log phase = %q, want %q", logPhase, "request_done_freed")
	}
}
