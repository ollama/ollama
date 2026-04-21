package launch

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
)

func captureStderr(t *testing.T, fn func()) string {
	t.Helper()

	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create stderr pipe: %v", err)
	}
	os.Stderr = w
	defer func() {
		os.Stderr = oldStderr
	}()

	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()

	fn()

	_ = w.Close()
	return <-done
}

func TestLaunchCmd(t *testing.T) {
	mockCheck := func(cmd *cobra.Command, args []string) error {
		return nil
	}
	mockTUI := func(cmd *cobra.Command) {}
	cmd := LaunchCmd(mockCheck, mockTUI)

	t.Run("command structure", func(t *testing.T) {
		if cmd.Use != "launch [INTEGRATION] [-- [EXTRA_ARGS...]]" {
			t.Errorf("Use = %q, want %q", cmd.Use, "launch [INTEGRATION] [-- [EXTRA_ARGS...]]")
		}
		if cmd.Short == "" {
			t.Error("Short description should not be empty")
		}
		if cmd.Long == "" {
			t.Error("Long description should not be empty")
		}
		if !strings.Contains(cmd.Long, "hermes") {
			t.Error("Long description should mention hermes")
		}
		if !strings.Contains(cmd.Long, "kimi") {
			t.Error("Long description should mention kimi")
		}
	})

	t.Run("flags exist", func(t *testing.T) {
		if cmd.Flags().Lookup("model") == nil {
			t.Error("--model flag should exist")
		}
		if cmd.Flags().Lookup("config") == nil {
			t.Error("--config flag should exist")
		}
		if cmd.Flags().Lookup("yes") == nil {
			t.Error("--yes flag should exist")
		}
	})

	t.Run("PreRunE is set", func(t *testing.T) {
		if cmd.PreRunE == nil {
			t.Error("PreRunE should be set to checkServerHeartbeat")
		}
	})
}

func TestLaunchCmdTUICallback(t *testing.T) {
	mockCheck := func(cmd *cobra.Command, args []string) error {
		return nil
	}

	t.Run("no args calls TUI", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{})
		_ = cmd.Execute()

		if !tuiCalled {
			t.Error("TUI callback should be called when no args provided")
		}
	})

	t.Run("integration arg bypasses TUI", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		defer srv.Close()
		t.Setenv("OLLAMA_HOST", srv.URL)

		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"claude"})
		_ = cmd.Execute()

		if tuiCalled {
			t.Error("TUI callback should NOT be called when integration arg provided")
		}
	})

	t.Run("--model flag without integration returns error", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--model", "test-model"})
		err := cmd.Execute()

		if err == nil {
			t.Fatal("expected --model without an integration to fail")
		}
		if !strings.Contains(err.Error(), "require an integration name") {
			t.Fatalf("expected integration-name guidance, got %v", err)
		}
		if tuiCalled {
			t.Error("TUI callback should NOT be called when --model is provided without an integration")
		}
	})

	t.Run("--config flag without integration returns error", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--config"})
		err := cmd.Execute()

		if err == nil {
			t.Fatal("expected --config without an integration to fail")
		}
		if !strings.Contains(err.Error(), "require an integration name") {
			t.Fatalf("expected integration-name guidance, got %v", err)
		}
		if tuiCalled {
			t.Error("TUI callback should NOT be called when --config is provided without an integration")
		}
	})

	t.Run("--yes flag without integration returns error", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--yes"})
		err := cmd.Execute()

		if err == nil {
			t.Fatal("expected --yes without an integration to fail")
		}
		if !strings.Contains(err.Error(), "require an integration name") {
			t.Fatalf("expected integration-name guidance, got %v", err)
		}
		if tuiCalled {
			t.Error("TUI callback should NOT be called when --yes is provided without an integration")
		}
	})

	t.Run("extra args without integration return error", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--model", "test-model", "--", "--sandbox", "workspace-write"})
		err := cmd.Execute()

		if err == nil {
			t.Fatal("expected flags and extra args without an integration to fail")
		}
		if !strings.Contains(err.Error(), "require an integration name") {
			t.Fatalf("expected integration-name guidance, got %v", err)
		}
		if tuiCalled {
			t.Error("TUI callback should NOT be called when flags or extra args are provided without an integration")
		}
	})
}

func TestLaunchCmdNilHeartbeat(t *testing.T) {
	cmd := LaunchCmd(nil, nil)
	if cmd == nil {
		t.Fatal("LaunchCmd returned nil")
	}
	if cmd.PreRunE != nil {
		t.Log("Note: PreRunE is set even when nil is passed (acceptable)")
	}
}

func TestLaunchCmdModelFlagFiltersDisabledCloudFromSavedConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SaveIntegration("stubeditor", []string{"glm-5:cloud"}); err != nil {
		t.Fatalf("failed to seed saved config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprintf(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		case "/api/show":
			fmt.Fprintf(w, `{"model":"llama3.2"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherEditorRunner{}
	restore := OverrideIntegration("stubeditor", stub)
	defer restore()

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubeditor", "--model", "llama3.2"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("launch command failed: %v", err)
	}

	saved, err := config.LoadIntegration("stubeditor")
	if err != nil {
		t.Fatalf("failed to reload integration config: %v", err)
	}
	if diff := cmp.Diff([]string{"llama3.2"}, saved.Models); diff != "" {
		t.Fatalf("saved models mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([][]string{{"llama3.2"}}, stub.edited); diff != "" {
		t.Fatalf("editor models mismatch (-want +got):\n%s", diff)
	}
	if stub.ranModel != "llama3.2" {
		t.Fatalf("expected launch to run with llama3.2, got %q", stub.ranModel)
	}
}

func TestLaunchCmdModelFlagClearsDisabledCloudOverride(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprintf(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherSingleRunner{}
	restore := OverrideIntegration("stubapp", stub)
	defer restore()

	oldSelector := DefaultSingleSelector
	defer func() { DefaultSingleSelector = oldSelector }()

	var selectorCalls int
	var gotCurrent string
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		selectorCalls++
		gotCurrent = current
		return "llama3.2", nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubapp", "--model", "glm-5:cloud"})
	stderr := captureStderr(t, func() {
		if err := cmd.Execute(); err != nil {
			t.Fatalf("launch command failed: %v", err)
		}
	})

	if selectorCalls != 1 {
		t.Fatalf("expected disabled cloud override to fall back to selector, got %d calls", selectorCalls)
	}
	if gotCurrent != "" {
		t.Fatalf("expected disabled override to be cleared before selection, got current %q", gotCurrent)
	}
	if stub.ranModel != "llama3.2" {
		t.Fatalf("expected launch to run with replacement local model, got %q", stub.ranModel)
	}
	if !strings.Contains(stderr, "Warning: ignoring --model glm-5:cloud because cloud is disabled") {
		t.Fatalf("expected disabled-cloud warning, got stderr: %q", stderr)
	}

	saved, err := config.LoadIntegration("stubapp")
	if err != nil {
		t.Fatalf("failed to reload integration config: %v", err)
	}
	if diff := cmp.Diff([]string{"llama3.2"}, saved.Models); diff != "" {
		t.Fatalf("saved models mismatch (-want +got):\n%s", diff)
	}
}

func TestLaunchCmdYes_AutoConfirmsLaunchPromptPath(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherEditorRunner{paths: []string{"/tmp/stubeditor.json"}}
	restore := OverrideIntegration("stubeditor", stub)
	defer restore()

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt with --yes: %q", prompt)
		return false, nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubeditor", "--model", "llama3.2", "--yes"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("launch command with --yes failed: %v", err)
	}

	if diff := cmp.Diff([][]string{{"llama3.2"}}, stub.edited); diff != "" {
		t.Fatalf("editor models mismatch (-want +got):\n%s", diff)
	}
	if stub.ranModel != "llama3.2" {
		t.Fatalf("expected launch to run with llama3.2, got %q", stub.ranModel)
	}
}

func TestLaunchCmdHeadlessWithYes_AutoPullsMissingLocalModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherSingleRunner{}
	restore := OverrideIntegration("stubapp", stub)
	defer restore()

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt with --yes in headless autopull path: %q", prompt)
		return false, nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubapp", "--model", "missing-model", "--yes"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("launch command with --yes failed: %v", err)
	}

	if !pullCalled {
		t.Fatal("expected missing local model to be auto-pulled with --yes in headless mode")
	}
	if stub.ranModel != "missing-model" {
		t.Fatalf("expected launch to run with pulled model, got %q", stub.ranModel)
	}
}

func TestLaunchCmdHeadlessWithoutYes_ReturnsActionableConfirmError(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherEditorRunner{paths: []string{"/tmp/stubeditor.json"}}
	restore := OverrideIntegration("stubeditor", stub)
	defer restore()

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt in headless non-yes mode: %q", prompt)
		return false, nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubeditor", "--model", "llama3.2"})
	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected launch command to fail without --yes in headless mode")
	}
	if !strings.Contains(err.Error(), "re-run with --yes") {
		t.Fatalf("expected actionable --yes guidance, got %v", err)
	}
	if len(stub.edited) != 0 {
		t.Fatalf("expected no editor writes when confirmation is blocked, got %v", stub.edited)
	}
	if stub.ranModel != "" {
		t.Fatalf("expected launch to abort before run, got %q", stub.ranModel)
	}
}

func TestLaunchCmdIntegrationArgPromptsForModelWithSavedSelection(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SaveIntegration("stubapp", []string{"llama3.2"}); err != nil {
		t.Fatalf("failed to seed saved config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"},{"name":"qwen3:8b"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"qwen3:8b"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherSingleRunner{}
	restore := OverrideIntegration("stubapp", stub)
	defer restore()

	oldSelector := DefaultSingleSelector
	defer func() { DefaultSingleSelector = oldSelector }()

	var gotCurrent string
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		gotCurrent = current
		return "qwen3:8b", nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubapp"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("launch command failed: %v", err)
	}

	if gotCurrent != "llama3.2" {
		t.Fatalf("expected selector current model to be saved model llama3.2, got %q", gotCurrent)
	}
	if stub.ranModel != "qwen3:8b" {
		t.Fatalf("expected launch to run selected model qwen3:8b, got %q", stub.ranModel)
	}

	saved, err := config.LoadIntegration("stubapp")
	if err != nil {
		t.Fatalf("failed to reload integration config: %v", err)
	}
	if diff := cmp.Diff([]string{"qwen3:8b"}, saved.Models); diff != "" {
		t.Fatalf("saved models mismatch (-want +got):\n%s", diff)
	}
}

func TestLaunchCmdHeadlessYes_IntegrationRequiresModelEvenWhenSaved(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	if err := config.SaveIntegration("stubapp", []string{"llama3.2"}); err != nil {
		t.Fatalf("failed to seed saved config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherSingleRunner{}
	restore := OverrideIntegration("stubapp", stub)
	defer restore()

	oldSelector := DefaultSingleSelector
	defer func() { DefaultSingleSelector = oldSelector }()
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		t.Fatal("selector should not be called for headless --yes saved-model launch")
		return "", nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubapp", "--yes"})
	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected launch command to fail when --yes is used headlessly without --model")
	}
	if !strings.Contains(err.Error(), "requires --model <model>") {
		t.Fatalf("expected actionable --model guidance, got %v", err)
	}
	if stub.ranModel != "" {
		t.Fatalf("expected launch to abort before run, got %q", stub.ranModel)
	}
}

func TestLaunchCmdHeadlessYes_IntegrationWithoutSavedModelReturnsError(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	stub := &launcherSingleRunner{}
	restore := OverrideIntegration("stubapp", stub)
	defer restore()

	oldSelector := DefaultSingleSelector
	defer func() { DefaultSingleSelector = oldSelector }()
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		t.Fatal("selector should not be called for headless --yes without saved model")
		return "", nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command) {})
	cmd.SetArgs([]string{"stubapp", "--yes"})
	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected launch command to fail when --yes is used headlessly without --model")
	}
	if !strings.Contains(err.Error(), "requires --model <model>") {
		t.Fatalf("expected actionable --model guidance, got %v", err)
	}
	if stub.ranModel != "" {
		t.Fatalf("expected launch to abort before run, got %q", stub.ranModel)
	}
}
