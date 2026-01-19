package imagegen

import (
	"runtime"
	"testing"
)

func TestCheckPlatformSupport(t *testing.T) {
	err := CheckPlatformSupport()

	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			if err != nil {
				t.Errorf("Expected nil error on darwin/arm64, got: %v", err)
			}
		} else {
			if err == nil {
				t.Error("Expected error on darwin/non-arm64")
			}
		}
	case "linux", "windows":
		if err != nil {
			t.Errorf("Expected nil error on %s, got: %v", runtime.GOOS, err)
		}
	default:
		if err == nil {
			t.Errorf("Expected error on unsupported platform %s", runtime.GOOS)
		}
	}
}

func TestCheckMemoryRequirements(t *testing.T) {
	tests := []struct {
		name            string
		availableMemory uint64
		wantErr         bool
	}{
		{
			name:            "sufficient memory",
			availableMemory: 32 * GB,
			wantErr:         false,
		},
		{
			name:            "exactly enough memory",
			availableMemory: 21 * GB,
			wantErr:         false,
		},
		{
			name:            "insufficient memory",
			availableMemory: 16 * GB,
			wantErr:         true,
		},
		{
			name:            "zero memory",
			availableMemory: 0,
			wantErr:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use a non-existent model name which will default to 21GB estimate
			err := CheckMemoryRequirements("nonexistent-model", tt.availableMemory)
			if (err != nil) != tt.wantErr {
				t.Errorf("CheckMemoryRequirements() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestModelVRAMEstimates(t *testing.T) {
	// Verify the VRAM estimates map has expected entries
	expected := map[string]uint64{
		"ZImagePipeline": 21 * GB,
		"FluxPipeline":   20 * GB,
	}

	for name, expectedVRAM := range expected {
		if actual, ok := modelVRAMEstimates[name]; !ok {
			t.Errorf("Missing VRAM estimate for %s", name)
		} else if actual != expectedVRAM {
			t.Errorf("VRAM estimate for %s = %d GB, want %d GB", name, actual/GB, expectedVRAM/GB)
		}
	}
}

func TestEstimateVRAMDefault(t *testing.T) {
	// Non-existent model should return default 21GB
	vram := EstimateVRAM("nonexistent-model-that-does-not-exist")
	if vram != 21*GB {
		t.Errorf("EstimateVRAM() = %d GB, want 21 GB", vram/GB)
	}
}

func TestResolveModelName(t *testing.T) {
	// Non-existent model should return empty string
	result := ResolveModelName("nonexistent-model")
	if result != "" {
		t.Errorf("ResolveModelName() = %q, want empty string", result)
	}
}
