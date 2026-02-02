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

// TestServerInterfaceCompliance verifies Server implements llm.LlamaServer.
// This is a compile-time check but we document it as a test.
func TestServerInterfaceCompliance(t *testing.T) {
	// The var _ llm.LlamaServer = (*Server)(nil) line in server.go
	// ensures compile-time interface compliance.
	// This test documents that requirement.
	t.Log("Server implements llm.LlamaServer interface (compile-time checked)")
}
