//go:build windows

package startup

import (
	"os"
	"path/filepath"
	"testing"
)

func newMockRegistrar(t *testing.T) (Registrar, pathConfig) {
	t.Helper()

	// TempDir is unique per test, so it should be safe to run in parallel
	tmp := t.TempDir()

	tempShortcutOrigin := filepath.Join(tmp, "app", "lib", "Ollama.lnk")
	tempStartupShortcut := filepath.Join(tmp, "startup", "Ollama.lnk")

	return NewRegistrar(func(pathConfig *pathConfig) {
		pathConfig.shortcutOrigin = tempShortcutOrigin
		pathConfig.startupShortcut = tempStartupShortcut
	}), pathConfig{shortcutOrigin: tempShortcutOrigin, startupShortcut: tempStartupShortcut}
}

func TestGetState(t *testing.T) {
	tests := []struct {
		name                  string
		createOriginShortcut  bool
		createStartupShortcut bool
		want                  RegistrationState
	}{
		{
			name: "unsupported when origin shortcut is missing",
			want: RegistrationState{Supported: false, Registered: false},
		},
		{
			name:                 "unregistered when startup shortcut is missing",
			createOriginShortcut: true,
			want:                 RegistrationState{Supported: true, Registered: false},
		},
		{
			name:                  "registered when startup shortcut exists",
			createOriginShortcut:  true,
			createStartupShortcut: true,
			want:                  RegistrationState{Supported: true, Registered: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registrar, paths := newMockRegistrar(t)

			if tt.createOriginShortcut {
				if err := os.MkdirAll(filepath.Dir(paths.shortcutOrigin), 0o755); err != nil {
					t.Fatalf("Failed to set up temp install dir: %v", err)
				}
				if err := os.WriteFile(paths.shortcutOrigin, []byte("shortcut"), 0o644); err != nil {
					t.Fatalf("Failed to create origin shortcut: %v", err)
				}
			}

			if tt.createStartupShortcut {
				if err := os.MkdirAll(filepath.Dir(paths.startupShortcut), 0o755); err != nil {
					t.Fatalf("Failed to set up temp startup dir: %v", err)
				}
				if err := os.WriteFile(paths.startupShortcut, []byte("shortcut"), 0o644); err != nil {
					t.Fatalf("Failed to create startup shortcut: %v", err)
				}
			}

			state, err := registrar.GetState()
			if err != nil {
				t.Fatalf("GetState returned error: %v", err)
			}
			if state != tt.want {
				t.Fatalf("GetState = %+v, want %+v", state, tt.want)
			}
		})
	}
}

func TestRegisterCopiesShortcut(t *testing.T) {
	registrar, paths := newMockRegistrar(t)

	shortcutOrigin := filepath.Join(filepath.Dir(paths.shortcutOrigin), filepath.Base(paths.startupShortcut))
	if err := os.MkdirAll(filepath.Dir(shortcutOrigin), 0o755); err != nil {
		t.Fatalf("Failed to create origin dir: %v", err)
	}
	originContent := []byte("origin-shortcut")
	if err := os.WriteFile(shortcutOrigin, originContent, 0o644); err != nil {
		t.Fatalf("Failed to write origin shortcut: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.startupShortcut), 0o755); err != nil {
		t.Fatalf("Failed to set up temp startup dir: %v", err)
	}

	if err := registrar.Register(); err != nil {
		t.Fatalf("Register returned error: %v", err)
	}

	data, err := os.ReadFile(paths.startupShortcut)
	if err != nil {
		t.Fatalf("Failed to read created shortcut: %v", err)
	}
	if string(data) != string(originContent) {
		t.Fatalf("shortcut content mismatch: got %q want %q", string(data), string(originContent))
	}

	// Ensure re-registering when the file already exists succeeds and leaves content intact.
	if err := registrar.Register(); err != nil {
		t.Fatalf("Register second call returned error: %v", err)
	}
	data, err = os.ReadFile(paths.startupShortcut)
	if err != nil {
		t.Fatalf("Failed to read shortcut after second register: %v", err)
	}
	if string(data) != string(originContent) {
		t.Fatalf("shortcut content changed after second register: got %q want %q", string(data), string(originContent))
	}
}

func TestDeregisterRemovesShortcut(t *testing.T) {
	registrar, paths := newMockRegistrar(t)

	if err := os.MkdirAll(filepath.Dir(paths.shortcutOrigin), 0o755); err != nil {
		t.Fatalf("Failed to set up temp install dir: %v", err)
	}
	if err := os.WriteFile(paths.shortcutOrigin, []byte("shortcut"), 0o644); err != nil {
		t.Fatalf("Failed to create origin shortcut: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.startupShortcut), 0o755); err != nil {
		t.Fatalf("Failed to create startup dir: %v", err)
	}
	if err := os.WriteFile(paths.startupShortcut, []byte("shortcut"), 0o644); err != nil {
		t.Fatalf("Failed to write shortcut: %v", err)
	}

	if err := registrar.Deregister(); err != nil {
		t.Fatalf("Deregister returned error: %v", err)
	}

	if _, err := os.Stat(paths.startupShortcut); !os.IsNotExist(err) {
		t.Fatalf("expected shortcut to be removed, err=%v", err)
	}
}

func TestDeregisterNoopWhenMissing(t *testing.T) {
	registrar, paths := newMockRegistrar(t)

	if err := os.MkdirAll(filepath.Dir(paths.shortcutOrigin), 0o755); err != nil {
		t.Fatalf("Failed to set up temp install dir: %v", err)
	}
	if err := os.WriteFile(paths.shortcutOrigin, []byte("shortcut"), 0o644); err != nil {
		t.Fatalf("Failed to create origin shortcut: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.startupShortcut), 0o755); err != nil {
		t.Fatalf("Failed to create startup dir: %v", err)
	}

	if err := registrar.Deregister(); err != nil {
		t.Fatalf("Deregister returned error: %v", err)
	}
}
