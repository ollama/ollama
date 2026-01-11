package imagegen

import (
	"runtime"
	"testing"
)

// TestPlatformSupport verifies platform validation works correctly.
func TestPlatformSupport(t *testing.T) {
	err := CheckPlatformSupport()

	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			// Apple Silicon should be supported
			if err != nil {
				t.Errorf("Expected nil error on darwin/arm64, got: %v", err)
			}
		} else {
			// Intel Mac should fail
			if err == nil {
				t.Error("Expected error on darwin/amd64 (Intel), got nil")
			}
			if err != nil && err.Error() == "" {
				t.Error("Expected meaningful error message for unsupported platform")
			}
		}
	case "linux", "windows":
		// Linux/Windows are allowed (CUDA support checked at runtime)
		if err != nil {
			t.Errorf("Expected nil error on %s, got: %v", runtime.GOOS, err)
		}
	default:
		// Other platforms should fail
		if err == nil {
			t.Errorf("Expected error on unsupported platform %s, got nil", runtime.GOOS)
		}
	}
}

// TestMemoryRequirementsError verifies memory check returns clear error.
func TestMemoryRequirementsError(t *testing.T) {
	// Test with insufficient memory
	err := CheckMemoryRequirements("test-model", 8*GB)
	if err == nil {
		t.Error("Expected error for insufficient memory (8GB < 21GB default)")
	}

	// Test with sufficient memory
	err = CheckMemoryRequirements("test-model", 32*GB)
	if err != nil {
		t.Errorf("Expected no error for sufficient memory (32GB), got: %v", err)
	}
}

// TestEstimateVRAMReturnsReasonableDefaults verifies VRAM estimates are sensible.
func TestEstimateVRAMReturnsReasonableDefaults(t *testing.T) {
	// Unknown model should return default (21GB)
	vram := EstimateVRAM("unknown-model")
	if vram < 10*GB || vram > 100*GB {
		t.Errorf("VRAM estimate %d GB is outside reasonable range (10-100 GB)", vram/GB)
	}

	// Verify known pipeline estimates exist and are reasonable
	for name, estimate := range modelVRAMEstimates {
		if estimate < 10*GB {
			t.Errorf("VRAM estimate for %s (%d GB) is suspiciously low", name, estimate/GB)
		}
		if estimate > 200*GB {
			t.Errorf("VRAM estimate for %s (%d GB) is suspiciously high", name, estimate/GB)
		}
	}
}

// TestServerInterfaceCompliance verifies Server implements llm.LlamaServer.
// This is a compile-time check but we document it as a test.
func TestServerInterfaceCompliance(t *testing.T) {
	// The var _ llm.LlamaServer = (*Server)(nil) line in server.go
	// ensures compile-time interface compliance.
	// This test documents that requirement.
	t.Log("Server implements llm.LlamaServer interface (compile-time checked)")
}
