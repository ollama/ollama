//go:build windows || darwin

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/app/store"
)

type recordingUpdater struct {
	cancels, checks int
}

func (u *recordingUpdater) CancelOngoingDownload() { u.cancels++ }
func (u *recordingUpdater) TriggerImmediateCheck() { u.checks++ }

type testSettings struct {
	*settingsController
	restarts      int
	notifications int
	lookups       int
	updater       *recordingUpdater
}

func newTestSettings(t *testing.T) *testSettings {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("OLLAMA_MODELS", "")
	t.Setenv("OLLAMA_NO_CLOUD", "")
	st := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	t.Cleanup(func() { st.Close() })

	ts := &testSettings{updater: &recordingUpdater{}}
	ts.settingsController = &settingsController{
		store:         st,
		restartServer: func() { ts.restarts++ },
		updater:       ts.updater,
		notifyUpdate:  func() { ts.notifications++ },
		updateReady:   func() bool { return false },
		defaultContextLength: func(context.Context) int {
			ts.lookups++
			return 32768
		},
	}
	return ts
}

func (ts *testSettings) state(t *testing.T) settingsState {
	t.Helper()
	state, err := ts.State(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return state
}

func TestSettingsRestartServerOnlyForServerSettings(t *testing.T) {
	ts := newTestSettings(t)

	steps := []struct {
		name         string
		change       func() error
		wantRestarts int
		wantErr      bool
	}{
		{"expose", func() error { return ts.SetExpose(true) }, 1, false},
		{"expose unchanged", func() error { return ts.SetExpose(true) }, 1, false},
		{"context length", func() error { return ts.SetContextLength(8192) }, 2, false},
		{"context length unchanged", func() error { return ts.SetContextLength(8192) }, 2, false},
		{"unsupported context length", func() error { return ts.SetContextLength(12345) }, 2, true},
		{"automatic context length", func() error { return ts.SetContextLength(0) }, 3, false},
		{"auto update", func() error { return ts.SetAutoUpdate(false) }, 3, false},
	}
	for _, step := range steps {
		err := step.change()
		if (err != nil) != step.wantErr {
			t.Fatalf("%s: error = %v, want error %v", step.name, err, step.wantErr)
		}
		if ts.restarts != step.wantRestarts {
			t.Fatalf("%s: restarts = %d, want %d", step.name, ts.restarts, step.wantRestarts)
		}
	}

	state := ts.state(t)
	if !state.Expose || state.ContextLength != 0 || state.AutoUpdate {
		t.Fatalf("state = %+v", state)
	}
}

func TestSettingsModelsPath(t *testing.T) {
	ts := newTestSettings(t)
	if state := ts.state(t); !state.ModelsPathIsDefault || state.ModelsPath != state.DefaultModelsPath {
		t.Fatalf("new settings should use the default models folder: %+v", state)
	}

	custom := t.TempDir()
	if err := ts.SetModelsPath(custom); err != nil {
		t.Fatal(err)
	}
	if state := ts.state(t); state.ModelsPath != custom || state.ModelsPathIsDefault || ts.restarts != 1 {
		t.Fatalf("custom folder: state = %+v, restarts = %d", state, ts.restarts)
	}

	if err := ts.SetModelsPath(filepath.Join(custom, "missing")); err == nil {
		t.Fatal("expected an error for a folder that doesn't exist")
	}
	file := filepath.Join(custom, "file")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ts.SetModelsPath(file); err == nil {
		t.Fatal("expected an error for a file")
	}
	if ts.restarts != 1 {
		t.Fatalf("rejected folders restarted the server: restarts = %d", ts.restarts)
	}

	// Choosing the default folder stores no path, like Reset does.
	defaultPath := ts.state(t).DefaultModelsPath
	if err := os.MkdirAll(defaultPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := ts.SetModelsPath(defaultPath); err != nil {
		t.Fatal(err)
	}
	stored, err := ts.store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if state := ts.state(t); !state.ModelsPathIsDefault || stored.Models != defaultPath || ts.restarts != 2 {
		t.Fatalf("default folder: state = %+v, restarts = %d", state, ts.restarts)
	}
	if err := ts.SetModelsPath(""); err != nil || ts.restarts != 2 {
		t.Fatalf("reset to the folder in use: error = %v, restarts = %d", err, ts.restarts)
	}
}

func TestIsDefaultModelsPath(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("OLLAMA_MODELS", "")
	defaultPath := filepath.Join(home, ".ollama", "models")
	if err := os.MkdirAll(defaultPath, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "models")
	if err := os.Symlink(defaultPath, link); err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		path string
		want bool
	}{
		{"", true},
		{defaultPath, true},
		{defaultPath + "/", true},
		{link, true},
		{t.TempDir(), false},
	} {
		if got := isDefaultModelsPath(tt.path); got != tt.want {
			t.Errorf("isDefaultModelsPath(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestSettingsAutoUpdate(t *testing.T) {
	ts := newTestSettings(t)

	if err := ts.SetAutoUpdate(false); err != nil {
		t.Fatal(err)
	}
	if ts.updater.cancels != 1 || ts.updater.checks != 0 {
		t.Fatalf("turning off updates: cancels = %d, checks = %d", ts.updater.cancels, ts.updater.checks)
	}
	if err := ts.SetAutoUpdate(true); err != nil {
		t.Fatal(err)
	}
	if ts.updater.checks != 1 || ts.notifications != 0 {
		t.Fatalf("turning on updates: checks = %d, notifications = %d", ts.updater.checks, ts.notifications)
	}

	// A downloaded update is announced instead of checked for again.
	ts.updateReady = func() bool { return true }
	if err := ts.SetAutoUpdate(false); err != nil {
		t.Fatal(err)
	}
	if err := ts.SetAutoUpdate(true); err != nil {
		t.Fatal(err)
	}
	if ts.updater.checks != 1 || ts.notifications != 1 {
		t.Fatalf("update ready: checks = %d, notifications = %d", ts.updater.checks, ts.notifications)
	}
	if err := ts.SetAutoUpdate(true); err != nil || ts.notifications != 1 {
		t.Fatalf("unchanged setting: error = %v, notifications = %d", err, ts.notifications)
	}
}

func TestSettingsCloud(t *testing.T) {
	ts := newTestSettings(t)
	if state := ts.state(t); !state.CloudEnabled || state.CloudDisabledByEnv {
		t.Fatalf("cloud should start on: %+v", state)
	}

	if err := ts.SetCloudEnabled(false); err != nil {
		t.Fatal(err)
	}
	if state := ts.state(t); state.CloudEnabled || ts.restarts != 1 {
		t.Fatalf("turning off cloud: state = %+v, restarts = %d", state, ts.restarts)
	}
	if err := ts.SetCloudEnabled(false); err != nil || ts.restarts != 1 {
		t.Fatalf("unchanged cloud setting: error = %v, restarts = %d", err, ts.restarts)
	}

	t.Setenv("OLLAMA_NO_CLOUD", "1")
	if err := ts.SetCloudEnabled(true); err == nil {
		t.Fatal("OLLAMA_NO_CLOUD should keep cloud off")
	}
	if state := ts.state(t); state.CloudEnabled || !state.CloudDisabledByEnv || ts.restarts != 1 {
		t.Fatalf("cloud disabled by env: state = %+v, restarts = %d", state, ts.restarts)
	}
}

func TestSettingsDefaultContextLengthLastsUntilRestart(t *testing.T) {
	ts := newTestSettings(t)
	if got := ts.state(t).DefaultContextLength; got != 32768 {
		t.Fatalf("default context length = %d", got)
	}
	ts.state(t)
	if ts.lookups != 1 {
		t.Fatalf("lookups = %d, want 1 before a restart", ts.lookups)
	}

	if err := ts.SetExpose(true); err != nil {
		t.Fatal(err)
	}
	ts.state(t)
	if ts.lookups != 2 {
		t.Fatalf("lookups = %d, want 2 after a restart", ts.lookups)
	}

	// An unknown length is looked up again rather than kept.
	ts.defaultContextLength = func(context.Context) int { ts.lookups++; return 0 }
	if err := ts.SetExpose(false); err != nil {
		t.Fatal(err)
	}
	ts.state(t)
	ts.state(t)
	if ts.lookups != 4 {
		t.Fatalf("lookups = %d, want 4 while the length is unknown", ts.lookups)
	}
}

func TestSettingsChatCount(t *testing.T) {
	ts := newTestSettings(t)
	if got := ts.state(t).ChatCount; got != 0 {
		t.Fatalf("chat count = %d, want 0", got)
	}
	if err := ts.store.SetChat(*store.NewChat("chat")); err != nil {
		t.Fatal(err)
	}
	if got := ts.state(t).ChatCount; got != 1 {
		t.Fatalf("chat count = %d, want 1", got)
	}
}
