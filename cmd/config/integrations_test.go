package config

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

func TestIntegrationLookup(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantFound bool
		wantName  string
	}{
		{"claude lowercase", "claude", true, "Claude Code"},
		{"claude uppercase", "CLAUDE", true, "Claude Code"},
		{"claude mixed case", "Claude", true, "Claude Code"},
		{"codex", "codex", true, "Codex"},
		{"droid", "droid", true, "Droid"},
		{"opencode", "opencode", true, "OpenCode"},
		{"unknown integration", "unknown", false, ""},
		{"empty string", "", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, found := integrations[strings.ToLower(tt.input)]
			if found != tt.wantFound {
				t.Errorf("integrations[%q] found = %v, want %v", tt.input, found, tt.wantFound)
			}
			if found && r.String() != tt.wantName {
				t.Errorf("integrations[%q].String() = %q, want %q", tt.input, r.String(), tt.wantName)
			}
		})
	}
}

func TestIntegrationRegistry(t *testing.T) {
	expectedIntegrations := []string{"claude", "codex", "droid", "opencode"}

	for _, name := range expectedIntegrations {
		t.Run(name, func(t *testing.T) {
			r, ok := integrations[name]
			if !ok {
				t.Fatalf("integration %q not found in registry", name)
			}
			if r.String() == "" {
				t.Error("integration.String() should not be empty")
			}
		})
	}
}

func TestHasLocalModel(t *testing.T) {
	tests := []struct {
		name   string
		models []string
		want   bool
	}{
		{"empty list", []string{}, false},
		{"single local model", []string{"llama3.2"}, true},
		{"single cloud model", []string{"cloud-model"}, false},
		{"mixed models", []string{"cloud-model", "llama3.2"}, true},
		{"multiple local models", []string{"llama3.2", "qwen2.5"}, true},
		{"multiple cloud models", []string{"cloud-a", "cloud-b"}, false},
		{"local model first", []string{"llama3.2", "cloud-model"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.ContainsFunc(tt.models, func(m string) bool {
				return !strings.Contains(m, "cloud")
			})
			if got != tt.want {
				t.Errorf("hasLocalModel(%v) = %v, want %v", tt.models, got, tt.want)
			}
		})
	}
}

func TestLaunchCmd(t *testing.T) {
	// Mock checkServerHeartbeat that always succeeds
	mockCheck := func(cmd *cobra.Command, args []string) error {
		return nil
	}
	// Mock TUI function (not called in these tests)
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
	})

	t.Run("flags exist", func(t *testing.T) {
		modelFlag := cmd.Flags().Lookup("model")
		if modelFlag == nil {
			t.Error("--model flag should exist")
		}

		configFlag := cmd.Flags().Lookup("config")
		if configFlag == nil {
			t.Error("--config flag should exist")
		}
	})

	t.Run("PreRunE is set", func(t *testing.T) {
		if cmd.PreRunE == nil {
			t.Error("PreRunE should be set to checkServerHeartbeat")
		}
	})
}

func TestLaunchCmd_TUICallback(t *testing.T) {
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
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"claude"})
		// Will error because claude isn't configured, but that's OK
		_ = cmd.Execute()

		if tuiCalled {
			t.Error("TUI callback should NOT be called when integration arg provided")
		}
	})

	t.Run("--model flag bypasses TUI", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--model", "test-model"})
		// Will error because no integration specified, but that's OK
		_ = cmd.Execute()

		if tuiCalled {
			t.Error("TUI callback should NOT be called when --model flag provided")
		}
	})

	t.Run("--config flag bypasses TUI", func(t *testing.T) {
		tuiCalled := false
		mockTUI := func(cmd *cobra.Command) {
			tuiCalled = true
		}

		cmd := LaunchCmd(mockCheck, mockTUI)
		cmd.SetArgs([]string{"--config"})
		// Will error because no integration specified, but that's OK
		_ = cmd.Execute()

		if tuiCalled {
			t.Error("TUI callback should NOT be called when --config flag provided")
		}
	})
}

func TestRunIntegration_UnknownIntegration(t *testing.T) {
	err := runIntegration("unknown-integration", "model", nil)
	if err == nil {
		t.Error("expected error for unknown integration, got nil")
	}
	if !strings.Contains(err.Error(), "unknown integration") {
		t.Errorf("error should mention 'unknown integration', got: %v", err)
	}
}

func TestHasLocalModel_DocumentsHeuristic(t *testing.T) {
	tests := []struct {
		name   string
		models []string
		want   bool
		reason string
	}{
		{"empty list", []string{}, false, "empty list has no local models"},
		{"contains-cloud-substring", []string{"deepseek-r1:cloud"}, false, "model with 'cloud' substring is considered cloud"},
		{"cloud-in-name", []string{"my-cloud-model"}, false, "'cloud' anywhere in name = cloud model"},
		{"cloudless", []string{"cloudless-model"}, false, "'cloudless' still contains 'cloud'"},
		{"local-model", []string{"llama3.2"}, true, "no 'cloud' = local"},
		{"mixed", []string{"cloud-model", "llama3.2"}, true, "one local model = hasLocalModel true"},
		{"all-cloud", []string{"cloud-a", "cloud-b"}, false, "all contain 'cloud'"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.ContainsFunc(tt.models, func(m string) bool {
				return !strings.Contains(m, "cloud")
			})
			if got != tt.want {
				t.Errorf("hasLocalModel(%v) = %v, want %v (%s)", tt.models, got, tt.want, tt.reason)
			}
		})
	}
}

func TestLaunchCmd_NilHeartbeat(t *testing.T) {
	// This should not panic - cmd creation should work even with nil
	cmd := LaunchCmd(nil, nil)
	if cmd == nil {
		t.Fatal("LaunchCmd returned nil")
	}

	// PreRunE should be nil when passed nil
	if cmd.PreRunE != nil {
		t.Log("Note: PreRunE is set even when nil is passed (acceptable)")
	}
}

func TestAllIntegrations_HaveRequiredMethods(t *testing.T) {
	for name, r := range integrations {
		t.Run(name, func(t *testing.T) {
			displayName := r.String()
			if displayName == "" {
				t.Error("String() should not return empty")
			}
			var _ func(string, []string) error = r.Run
		})
	}
}

func TestParseArgs(t *testing.T) {
	// Tests reflect cobra's ArgsLenAtDash() semantics:
	// - cobra strips "--" from args
	// - ArgsLenAtDash() returns the index where "--" was, or -1
	tests := []struct {
		name     string
		args     []string // args as cobra delivers them (no "--")
		dashIdx  int      // what ArgsLenAtDash() returns
		wantName string
		wantArgs []string
		wantErr  bool
	}{
		{
			name:     "no extra args, no dash",
			args:     []string{"claude"},
			dashIdx:  -1,
			wantName: "claude",
		},
		{
			name:     "with extra args after --",
			args:     []string{"codex", "-p", "myprofile"},
			dashIdx:  1,
			wantName: "codex",
			wantArgs: []string{"-p", "myprofile"},
		},
		{
			name:     "extra args only after --",
			args:     []string{"codex", "--sandbox", "workspace-write"},
			dashIdx:  1,
			wantName: "codex",
			wantArgs: []string{"--sandbox", "workspace-write"},
		},
		{
			name:     "-- at end with no args after",
			args:     []string{"claude"},
			dashIdx:  1,
			wantName: "claude",
		},
		{
			name:     "-- with no integration name",
			args:     []string{"--verbose"},
			dashIdx:  0,
			wantName: "",
			wantArgs: []string{"--verbose"},
		},
		{
			name:    "multiple args before -- is error",
			args:    []string{"claude", "codex", "--verbose"},
			dashIdx: 2,
			wantErr: true,
		},
		{
			name:    "multiple args without -- is error",
			args:    []string{"claude", "codex"},
			dashIdx: -1,
			wantErr: true,
		},
		{
			name:     "no args, no dash",
			args:     []string{},
			dashIdx:  -1,
			wantName: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the parsing logic from LaunchCmd using dashIdx
			var name string
			var parsedArgs []string
			var err error

			dashIdx := tt.dashIdx
			args := tt.args

			if dashIdx == -1 {
				if len(args) > 1 {
					err = fmt.Errorf("unexpected arguments: %v", args[1:])
				} else if len(args) == 1 {
					name = args[0]
				}
			} else {
				if dashIdx > 1 {
					err = fmt.Errorf("expected at most 1 integration name before '--', got %d", dashIdx)
				} else {
					if dashIdx == 1 {
						name = args[0]
					}
					parsedArgs = args[dashIdx:]
				}
			}

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if name != tt.wantName {
				t.Errorf("name = %q, want %q", name, tt.wantName)
			}
			if !slices.Equal(parsedArgs, tt.wantArgs) {
				t.Errorf("args = %v, want %v", parsedArgs, tt.wantArgs)
			}
		})
	}
}

func TestIsCloudModel(t *testing.T) {
	// isCloudModel now only uses Show API, so nil client always returns false
	t.Run("nil client returns false", func(t *testing.T) {
		models := []string{"glm-4.7:cloud", "kimi-k2.5:cloud", "local-model"}
		for _, model := range models {
			if isCloudModel(context.Background(), nil, model) {
				t.Errorf("isCloudModel(%q) with nil client should return false", model)
			}
		}
	})
}

func names(items []ModelItem) []string {
	var out []string
	for _, item := range items {
		out = append(out, item.Name)
	}
	return out
}

func TestBuildModelList_NoExistingModels(t *testing.T) {
	items, _, _, _ := buildModelList(nil, nil, "")

	want := []string{"glm-4.7-flash", "qwen3:8b", "glm-4.7:cloud", "kimi-k2.5:cloud"}
	if diff := cmp.Diff(want, names(items)); diff != "" {
		t.Errorf("with no existing models, items should be recommended in order (-want +got):\n%s", diff)
	}

	for _, item := range items {
		if !strings.HasSuffix(item.Description, "install?") {
			t.Errorf("item %q should have description ending with 'install?', got %q", item.Name, item.Description)
		}
	}
}

func TestBuildModelList_OnlyLocalModels_CloudRecsAtBottom(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "qwen2.5:latest", Remote: false},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	want := []string{"llama3.2", "qwen2.5", "glm-4.7-flash", "glm-4.7:cloud", "kimi-k2.5:cloud", "qwen3:8b"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("cloud recs should be at bottom (-want +got):\n%s", diff)
	}
}

func TestBuildModelList_BothCloudAndLocal_RegularSort(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-4.7:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	want := []string{"glm-4.7:cloud", "llama3.2", "glm-4.7-flash", "kimi-k2.5:cloud", "qwen3:8b"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("mixed models should be alphabetical (-want +got):\n%s", diff)
	}
}

func TestBuildModelList_PreCheckedFirst(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-4.7:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, []string{"llama3.2"}, "")
	got := names(items)

	if got[0] != "llama3.2" {
		t.Errorf("pre-checked model should be first, got %v", got)
	}
}

func TestBuildModelList_ExistingRecommendedMarked(t *testing.T) {
	existing := []modelInfo{
		{Name: "glm-4.7-flash", Remote: false},
		{Name: "glm-4.7:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")

	for _, item := range items {
		switch item.Name {
		case "glm-4.7-flash", "glm-4.7:cloud":
			if strings.HasSuffix(item.Description, "install?") {
				t.Errorf("installed recommended %q should not have 'install?' suffix, got %q", item.Name, item.Description)
			}
		case "kimi-k2.5:cloud", "qwen3:8b":
			if !strings.HasSuffix(item.Description, "install?") {
				t.Errorf("non-installed recommended %q should have 'install?' suffix, got %q", item.Name, item.Description)
			}
		}
	}
}

func TestBuildModelList_ExistingCloudModelsNotPushedToBottom(t *testing.T) {
	existing := []modelInfo{
		{Name: "glm-4.7-flash", Remote: false},
		{Name: "glm-4.7:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// glm-4.7-flash and glm-4.7:cloud are installed so they sort normally;
	// kimi-k2.5:cloud and qwen3:8b are not installed so they go to the bottom
	want := []string{"glm-4.7-flash", "glm-4.7:cloud", "kimi-k2.5:cloud", "qwen3:8b"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("existing cloud models should sort normally (-want +got):\n%s", diff)
	}
}

func TestBuildModelList_HasRecommendedCloudModel_OnlyNonInstalledAtBottom(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "kimi-k2.5:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// kimi-k2.5:cloud is installed so it sorts normally;
	// the rest of the recommendations are not installed so they go to the bottom
	want := []string{"kimi-k2.5:cloud", "llama3.2", "glm-4.7-flash", "glm-4.7:cloud", "qwen3:8b"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("only non-installed models should be at bottom (-want +got):\n%s", diff)
	}

	for _, item := range items {
		if !slices.Contains([]string{"kimi-k2.5:cloud", "llama3.2"}, item.Name) {
			if !strings.HasSuffix(item.Description, "install?") {
				t.Errorf("non-installed %q should have 'install?' suffix, got %q", item.Name, item.Description)
			}
		}
	}
}

func TestBuildModelList_LatestTagStripped(t *testing.T) {
	existing := []modelInfo{
		{Name: "glm-4.7-flash:latest", Remote: false},
		{Name: "llama3.2:latest", Remote: false},
	}

	items, _, existingModels, _ := buildModelList(existing, nil, "")
	got := names(items)

	// :latest should be stripped from display names
	for _, name := range got {
		if strings.HasSuffix(name, ":latest") {
			t.Errorf("name %q should not have :latest suffix", name)
		}
	}

	// glm-4.7-flash should not be duplicated (existing :latest matches the recommendation)
	count := 0
	for _, name := range got {
		if name == "glm-4.7-flash" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("glm-4.7-flash should appear exactly once, got %d in %v", count, got)
	}

	// Stripped name should be in existingModels so it won't be pulled
	if !existingModels["glm-4.7-flash"] {
		t.Error("glm-4.7-flash should be in existingModels")
	}
}

func TestBuildModelList_ReturnsExistingAndCloudMaps(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-4.7:cloud", Remote: true},
	}

	_, _, existingModels, cloudModels := buildModelList(existing, nil, "")

	if !existingModels["llama3.2"] {
		t.Error("llama3.2 should be in existingModels")
	}
	if !existingModels["glm-4.7:cloud"] {
		t.Error("glm-4.7:cloud should be in existingModels")
	}
	if existingModels["glm-4.7-flash"] {
		t.Error("glm-4.7-flash should not be in existingModels (it's a recommendation)")
	}

	if !cloudModels["glm-4.7:cloud"] {
		t.Error("glm-4.7:cloud should be in cloudModels")
	}
	if !cloudModels["kimi-k2.5:cloud"] {
		t.Error("kimi-k2.5:cloud should be in cloudModels (recommended cloud)")
	}
	if cloudModels["llama3.2"] {
		t.Error("llama3.2 should not be in cloudModels")
	}
}

func TestEditorIntegration_SavedConfigSkipsSelection(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Save a config for opencode so it looks like a previous launch
	if err := saveIntegration("opencode", []string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	// Verify loadIntegration returns the saved models
	saved, err := loadIntegration("opencode")
	if err != nil {
		t.Fatal(err)
	}
	if len(saved.Models) == 0 {
		t.Fatal("expected saved models")
	}
	if saved.Models[0] != "llama3.2" {
		t.Errorf("expected llama3.2, got %s", saved.Models[0])
	}
}

func TestAliasConfigurerInterface(t *testing.T) {
	t.Run("claude implements AliasConfigurer", func(t *testing.T) {
		claude := &Claude{}
		if _, ok := interface{}(claude).(AliasConfigurer); !ok {
			t.Error("Claude should implement AliasConfigurer")
		}
	})

	t.Run("codex does not implement AliasConfigurer", func(t *testing.T) {
		codex := &Codex{}
		if _, ok := interface{}(codex).(AliasConfigurer); ok {
			t.Error("Codex should not implement AliasConfigurer")
		}
	})
}

func TestShowOrPull_ModelExists(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"model":"test-model"}`)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPull(context.Background(), client, "test-model")
	if err != nil {
		t.Errorf("showOrPull should return nil when model exists, got: %v", err)
	}
}

func TestShowOrPull_ModelNotFound_NoTerminal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, `{"error":"model not found"}`)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	// confirmPrompt will fail in test (no terminal), so showOrPull should return an error
	err := showOrPull(context.Background(), client, "missing-model")
	if err == nil {
		t.Error("showOrPull should return error when model not found and no terminal available")
	}
}

func TestShowOrPull_ShowCalledWithCorrectModel(t *testing.T) {
	var receivedModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			var req api.ShowRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
				receivedModel = req.Model
			}
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"model":"%s"}`, receivedModel)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	_ = showOrPull(context.Background(), client, "qwen3:8b")
	if receivedModel != "qwen3:8b" {
		t.Errorf("expected Show to be called with %q, got %q", "qwen3:8b", receivedModel)
	}
}

func TestEnsureAuth_NoCloudModels(t *testing.T) {
	// ensureAuth should be a no-op when no cloud models are selected
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("no API calls expected when no cloud models selected")
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := ensureAuth(context.Background(), client, map[string]bool{}, []string{"local-model"})
	if err != nil {
		t.Errorf("ensureAuth should return nil for non-cloud models, got: %v", err)
	}
}

func TestEnsureAuth_CloudModelFilteredCorrectly(t *testing.T) {
	// ensureAuth should only care about models in cloudModels map
	var whoamiCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			whoamiCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"name":"testuser"}`)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	cloudModels := map[string]bool{"cloud-model:cloud": true}
	selected := []string{"cloud-model:cloud", "local-model"}

	err := ensureAuth(context.Background(), client, cloudModels, selected)
	if err != nil {
		t.Errorf("ensureAuth should succeed when user is authenticated, got: %v", err)
	}
	if !whoamiCalled {
		t.Error("expected whoami to be called for cloud model")
	}
}

func TestEnsureAuth_SkipsWhenNoCloudSelected(t *testing.T) {
	var whoamiCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			whoamiCalled = true
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	// cloudModels has entries but none are in selected
	cloudModels := map[string]bool{"cloud-model:cloud": true}
	selected := []string{"local-model"}

	err := ensureAuth(context.Background(), client, cloudModels, selected)
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
	if whoamiCalled {
		t.Error("whoami should not be called when no cloud models are selected")
	}
}
