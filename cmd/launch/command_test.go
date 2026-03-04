package launch

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
)

func TestLaunchCmd(t *testing.T) {
	mockCheck := func(cmd *cobra.Command, args []string) error {
		return nil
	}
	mockTUI := func(cmd *cobra.Command, inv LauncherInvocation) {}
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
	})

	t.Run("flags exist", func(t *testing.T) {
		if cmd.Flags().Lookup("model") == nil {
			t.Error("--model flag should exist")
		}
		if cmd.Flags().Lookup("config") == nil {
			t.Error("--config flag should exist")
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
		var gotInv LauncherInvocation
		mockTUI := func(cmd *cobra.Command, inv LauncherInvocation) {
			tuiCalled = true
			gotInv = inv
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{})
		_ = cmd.Execute()

		if !tuiCalled {
			t.Error("TUI callback should be called when no args provided")
		}
		if diff := cmp.Diff(LauncherInvocation{}, gotInv); diff != "" {
			t.Fatalf("launcher invocation mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("integration arg bypasses TUI", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		defer srv.Close()
		t.Setenv("OLLAMA_HOST", srv.URL)

		tuiCalled := false
		mockTUI := func(cmd *cobra.Command, inv LauncherInvocation) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"claude"})
		_ = cmd.Execute()

		if tuiCalled {
			t.Error("TUI callback should NOT be called when integration arg provided")
		}
	})

	t.Run("--model flag opens TUI with invocation", func(t *testing.T) {
		tuiCalled := false
		var gotInv LauncherInvocation
		mockTUI := func(cmd *cobra.Command, inv LauncherInvocation) {
			tuiCalled = true
			gotInv = inv
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--model", "test-model"})
		_ = cmd.Execute()

		if !tuiCalled {
			t.Error("TUI callback should be called when --model flag provided without an integration")
		}
		want := LauncherInvocation{ModelOverride: "test-model"}
		if diff := cmp.Diff(want, gotInv); diff != "" {
			t.Fatalf("launcher invocation mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("--config flag bypasses TUI", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command, inv LauncherInvocation) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--config"})
		_ = cmd.Execute()

		if tuiCalled {
			t.Error("TUI callback should NOT be called when --config flag provided")
		}
	})

	t.Run("--model flag forwards extra args through TUI invocation", func(t *testing.T) {
		tuiCalled := false
		var gotInv LauncherInvocation
		mockTUI := func(cmd *cobra.Command, inv LauncherInvocation) {
			tuiCalled = true
			gotInv = inv
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--model", "test-model", "--", "--sandbox", "workspace-write"})
		_ = cmd.Execute()

		if !tuiCalled {
			t.Error("TUI callback should be called when --model flag is provided without an integration")
		}
		want := LauncherInvocation{
			ModelOverride: "test-model",
			ExtraArgs:     []string{"--sandbox", "workspace-write"},
		}
		if diff := cmp.Diff(want, gotInv); diff != "" {
			t.Fatalf("launcher invocation mismatch (-want +got):\n%s", diff)
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

func TestLaunchIntegrationByNameUnknownIntegration(t *testing.T) {
	err := LaunchIntegrationByName("nonexistent-integration")
	if err == nil {
		t.Fatal("expected error for unknown integration")
	}
	if !strings.Contains(err.Error(), "unknown integration") {
		t.Errorf("error should mention 'unknown integration', got: %v", err)
	}
}

func TestLaunchIntegrationByNameNotConfigured(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	err := LaunchIntegrationByName("claude")
	if err == nil {
		t.Fatal("expected error when integration is not configured")
	}
	if !strings.Contains(err.Error(), "no selector configured") {
		t.Errorf("error should mention missing selector, got: %v", err)
	}
}

func TestSaveAndEditIntegrationUnknownIntegration(t *testing.T) {
	err := SaveAndEditIntegration("nonexistent", []string{"model"})
	if err == nil {
		t.Fatal("expected error for unknown integration")
	}
	if !strings.Contains(err.Error(), "unknown integration") {
		t.Errorf("error should mention 'unknown integration', got: %v", err)
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
	restore := config.OverrideIntegration("stubeditor", stub)
	defer restore()

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command, inv LauncherInvocation) {})
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
	restore := config.OverrideIntegration("stubapp", stub)
	defer restore()

	oldSelector := config.DefaultSingleSelector
	defer func() { config.DefaultSingleSelector = oldSelector }()

	var gotCurrent string
	config.DefaultSingleSelector = func(title string, items []config.ModelItem, current string) (string, error) {
		gotCurrent = current
		return "qwen3:8b", nil
	}

	cmd := LaunchCmd(func(cmd *cobra.Command, args []string) error { return nil }, func(cmd *cobra.Command, inv LauncherInvocation) {})
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
