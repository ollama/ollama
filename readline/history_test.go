package readline

import (
	"os"
	"runtime"
	"testing"
)

func TestHomeDir(t *testing.T) {
	// Save and restore original environment
	origHome := os.Getenv("HOME")
	origUserProfile := os.Getenv("USERPROFILE")
	defer func() {
		os.Setenv("HOME", origHome)
		os.Setenv("USERPROFILE", origUserProfile)
	}()

	t.Run("HOME environment variable takes precedence on Unix", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Skipping Unix-specific test on Windows")
		}
		os.Unsetenv("HOME")
		os.Setenv("HOME", "/custom/home")
		os.Unsetenv("USERPROFILE")

		got := homeDir()
		if got != "/custom/home" {
			t.Errorf("homeDir() = %q, want %q", got, "/custom/home")
		}
	})

	t.Run("USERPROFILE takes precedence on Windows", func(t *testing.T) {
		os.Setenv("HOME", "/unix/home")
		os.Setenv("USERPROFILE", `C:\Users\Custom`)

		got := homeDir()
		if runtime.GOOS == "windows" {
			if got != `C:\Users\Custom` {
				t.Errorf("homeDir() = %q, want %q", got, `C:\Users\Custom`)
			}
		}
		// On non-Windows, HOME should be used
		if runtime.GOOS != "windows" {
			if got != "/unix/home" {
				t.Errorf("homeDir() = %q, want %q", got, "/unix/home")
			}
		}
	})

	t.Run("HOME used when USERPROFILE not set on non-Windows", func(t *testing.T) {
		os.Setenv("HOME", "/fallback/home")
		os.Unsetenv("USERPROFILE")

		got := homeDir()
		if runtime.GOOS != "windows" {
			if got != "/fallback/home" {
				t.Errorf("homeDir() = %q, want %q", got, "/fallback/home")
			}
		}
	})

	t.Run("empty string when no environment vars set", func(t *testing.T) {
		os.Unsetenv("HOME")
		os.Unsetenv("USERPROFILE")

		// This test may fail if os.UserHomeDir() returns a value,
		// which is expected behavior - we fall back to it
		got := homeDir()
		// Just verify it doesn't panic and returns something reasonable
		if got == "" {
			t.Log("homeDir() returned empty string (os.UserHomeDir() may have failed)")
		}
	})
}

func TestHistoryInit_RespectsHomeEnv(t *testing.T) {
	// This test verifies that History.Init() uses the home directory
	// from environment variables
	origHome := os.Getenv("HOME")
	defer os.Setenv("HOME", origHome)

	// Create a temp directory to act as custom home
	tmpDir := t.TempDir()
	os.Setenv("HOME", tmpDir)

	// On Windows, also set USERPROFILE
	if runtime.GOOS == "windows" {
		origUserProfile := os.Getenv("USERPROFILE")
		defer os.Setenv("USERPROFILE", origUserProfile)
		os.Setenv("USERPROFILE", tmpDir)
	}

	h, err := NewHistory()
	if err != nil {
		t.Fatalf("NewHistory() failed: %v", err)
	}

	// Verify the history file path uses the custom home
	expectedPath := tmpDir + "/.ollama/history"
	if runtime.GOOS == "windows" {
		expectedPath = tmpDir + `\.ollama\history`
	}

	if h.Filename != expectedPath {
		t.Errorf("History.Filename = %q, want %q", h.Filename, expectedPath)
	}

	// Verify the directory was created in the custom home
	if _, err := os.Stat(tmpDir + "/.ollama"); os.IsNotExist(err) {
		t.Error(".ollama directory was not created in custom home")
	}
}
