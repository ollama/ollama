package ml

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestMergeEnvWithRunnerEnvOverrides(t *testing.T) {
	devices := []DeviceInfo{
		{
			DeviceID:           DeviceID{Library: "Metal", ID: "0"},
			RunnerEnvOverrides: map[string]string{"GGML_METAL_TENSOR_DISABLE": "1"},
		},
		{
			DeviceID: DeviceID{Library: "CUDA", ID: "3"},
		},
	}

	env := GetDevicesEnv(devices, true)

	if got, want := env["GGML_METAL_TENSOR_DISABLE"], "1"; got != want {
		t.Fatalf("GGML_METAL_TENSOR_DISABLE = %q, want %q", got, want)
	}

	if got, want := env["CUDA_VISIBLE_DEVICES"], "3"; got != want {
		t.Fatalf("CUDA_VISIBLE_DEVICES = %q, want %q", got, want)
	}
}

func TestGetDevicesEnvWarnsOnConflictingOverrides(t *testing.T) {
	var logs bytes.Buffer
	oldLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() {
		slog.SetDefault(oldLogger)
	})

	devices := []DeviceInfo{
		{
			DeviceID:           DeviceID{Library: "Metal", ID: "0"},
			RunnerEnvOverrides: map[string]string{"TEST_OVERRIDE": "one"},
		},
		{
			DeviceID:           DeviceID{Library: "Metal", ID: "1"},
			RunnerEnvOverrides: map[string]string{"TEST_OVERRIDE": "two"},
		},
	}

	env := GetDevicesEnv(devices, false)

	if got, want := env["TEST_OVERRIDE"], "two"; got != want {
		t.Fatalf("TEST_OVERRIDE = %q, want %q", got, want)
	}

	if !strings.Contains(logs.String(), "conflicting device environment override") {
		t.Fatalf("expected warning log, got %q", logs.String())
	}
}
