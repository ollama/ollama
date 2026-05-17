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

func TestResolveModelName(t *testing.T) {
	// Non-existent model should return empty string
	result := ResolveModelName("nonexistent-model")
	if result != "" {
		t.Errorf("ResolveModelName() = %q, want empty string", result)
	}
}
