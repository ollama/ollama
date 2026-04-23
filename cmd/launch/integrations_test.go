package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
)

type stubEditorRunner struct {
	edited   [][]string
	ranModel string
	editErr  error
}

func (s *stubEditorRunner) Run(model string, args []string) error {
	s.ranModel = model
	return nil
}

func (s *stubEditorRunner) String() string { return "StubEditor" }

func (s *stubEditorRunner) Paths() []string { return nil }

func (s *stubEditorRunner) Edit(models []string) error {
	if s.editErr != nil {
		return s.editErr
	}
	cloned := append([]string(nil), models...)
	s.edited = append(s.edited, cloned)
	return nil
}

func (s *stubEditorRunner) Models() []string { return nil }

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
		{"kimi", "kimi", true, "Kimi Code CLI"},
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
	expectedIntegrations := []string{"claude", "codex", "kimi", "droid", "opencode", "hermes"}

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

func TestHiddenIntegrationsExcludedFromVisibleLists(t *testing.T) {
	for _, info := range ListIntegrationInfos() {
		switch info.Name {
		case "cline", "vscode", "kimi":
			t.Fatalf("hidden integration %q should not appear in ListIntegrationInfos", info.Name)
		}
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

func TestLookupIntegration_UnknownIntegration(t *testing.T) {
	_, _, err := LookupIntegration("unknown-integration")
	if err == nil {
		t.Error("expected error for unknown integration, got nil")
	}
	if !strings.Contains(err.Error(), "unknown integration") {
		t.Errorf("error should mention 'unknown integration', got: %v", err)
	}
}

func TestIsIntegrationInstalled_UnknownIntegrationReturnsFalse(t *testing.T) {
	stderr := captureStderr(t, func() {
		if IsIntegrationInstalled("unknown-integration") {
			t.Fatal("expected unknown integration to report not installed")
		}
	})
	if !strings.Contains(stderr, `Ollama couldn't find integration "unknown-integration", so it'll show up as not installed.`) {
		t.Fatalf("expected unknown-integration warning, got stderr: %q", stderr)
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
		models := []string{"glm-5.1:cloud", "kimi-k2.6:cloud", "local-model"}
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

	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "gemma4", "qwen3.5"}
	if diff := cmp.Diff(want, names(items)); diff != "" {
		t.Errorf("with no existing models, items should be recommended in order (-want +got):\n%s", diff)
	}

	for _, item := range items {
		if strings.HasSuffix(item.Name, ":cloud") {
			if strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("cloud model %q should not have '(not downloaded)' suffix, got %q", item.Name, item.Description)
			}
		} else {
			if !strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("item %q should have description ending with '(not downloaded)', got %q", item.Name, item.Description)
			}
		}
	}
}

func TestBuildModelList_OnlyLocalModels_CloudRecsStillFirst(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "qwen2.5:latest", Remote: false},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// Cloud recs always come first among recommended, regardless of installed inventory.
	// Cloud disablement is handled upstream in loadSelectableModels via filterCloudItems.
	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "gemma4", "qwen3.5", "llama3.2", "qwen2.5"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("cloud recs pinned first even when no cloud models installed (-want +got):\n%s", diff)
	}
}

func TestBuildModelList_BothCloudAndLocal_RegularSort(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// All recs pinned at top (cloud before local in mixed case), then non-recs
	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "gemma4", "qwen3.5", "llama3.2"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("recs pinned at top, cloud recs first in mixed case (-want +got):\n%s", diff)
	}
}

func TestBuildModelList_PreCheckedFirst(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, []string{"llama3.2"}, "")
	got := names(items)

	if got[0] != "llama3.2" {
		t.Errorf("pre-checked model should be first, got %v", got)
	}
}

func TestBuildModelList_CurrentDefaultFirstAmongCheckedNonRec(t *testing.T) {
	existing := []modelInfo{
		{Name: "alpha", Remote: false},
		{Name: "zebra", Remote: false},
		{Name: "middle", Remote: false},
	}

	// "zebra" is the current/default; all three are checked, none are recommended.
	// Expected non-rec order: zebra (default), alpha, middle (alphabetical).
	items, _, _, _ := buildModelList(existing, []string{"zebra", "alpha", "middle"}, "zebra")
	got := names(items)

	// Skip recommended items to find the non-rec portion.
	var nonRec []string
	for _, item := range items {
		if !item.Recommended {
			nonRec = append(nonRec, item.Name)
		}
	}
	if len(nonRec) < 3 {
		t.Fatalf("expected 3 non-rec items, got %v", nonRec)
	}
	if nonRec[0] != "zebra" {
		t.Errorf("current/default model should be first among checked non-rec, got %v (full: %v)", nonRec, got)
	}
	if nonRec[1] != "alpha" {
		t.Errorf("remaining checked should be alphabetical, expected alpha second, got %v", nonRec)
	}
	if nonRec[2] != "middle" {
		t.Errorf("remaining checked should be alphabetical, expected middle third, got %v", nonRec)
	}
}

func TestBuildModelList_ExistingRecommendedMarked(t *testing.T) {
	existing := []modelInfo{
		{Name: "gemma4", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")

	for _, item := range items {
		switch item.Name {
		case "gemma4", "glm-5.1:cloud":
			if strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("installed recommended %q should not have '(not downloaded)' suffix, got %q", item.Name, item.Description)
			}
		case "qwen3.5":
			if !strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("non-installed recommended %q should have '(not downloaded)' suffix, got %q", item.Name, item.Description)
			}
		case "minimax-m2.7:cloud", "kimi-k2.6:cloud", "qwen3.5:cloud":
			if strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("cloud model %q should not have '(not downloaded)' suffix, got %q", item.Name, item.Description)
			}
		}
	}
}

func TestBuildModelList_ExistingCloudModelsNotPushedToBottom(t *testing.T) {
	existing := []modelInfo{
		{Name: "gemma4", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// gemma4 and glm-5.1:cloud are installed so they sort normally;
	// qwen3.5:cloud and qwen3.5 are not installed so they go to the bottom
	// All recs: cloud first in mixed case, then local, in rec order within each
	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "gemma4", "qwen3.5"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("all recs, cloud first in mixed case (-want +got):\n%s", diff)
	}
}

func TestBuildModelList_HasRecommendedCloudModel_OnlyNonInstalledAtBottom(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "kimi-k2.6:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// kimi-k2.6:cloud is installed so it sorts normally;
	// the rest of the recommendations are not installed so they go to the bottom
	// All recs pinned at top (cloud first in mixed case), then non-recs
	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "gemma4", "qwen3.5", "llama3.2"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("recs pinned at top, cloud first in mixed case (-want +got):\n%s", diff)
	}

	for _, item := range items {
		isCloud := strings.HasSuffix(item.Name, ":cloud")
		isInstalled := slices.Contains([]string{"kimi-k2.6:cloud", "llama3.2"}, item.Name)
		if isInstalled || isCloud {
			if strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("installed or cloud model %q should not have '(not downloaded)' suffix, got %q", item.Name, item.Description)
			}
		} else {
			if !strings.HasSuffix(item.Description, "(not downloaded)") {
				t.Errorf("non-installed %q should have '(not downloaded)' suffix, got %q", item.Name, item.Description)
			}
		}
	}
}

func TestBuildModelList_LatestTagStripped(t *testing.T) {
	existing := []modelInfo{
		{Name: "gemma4:latest", Remote: false},
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

	// gemma4 should not be duplicated (existing :latest matches the recommendation)
	count := 0
	for _, name := range got {
		if name == "gemma4" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("gemma4 should appear exactly once, got %d in %v", count, got)
	}

	// Stripped name should be in existingModels so it won't be pulled
	if !existingModels["gemma4"] {
		t.Error("gemma4 should be in existingModels")
	}
}

func TestBuildModelList_ReturnsExistingAndCloudMaps(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	_, _, existingModels, cloudModels := buildModelList(existing, nil, "")

	if !existingModels["llama3.2"] {
		t.Error("llama3.2 should be in existingModels")
	}
	if !existingModels["glm-5.1:cloud"] {
		t.Error("glm-5.1:cloud should be in existingModels")
	}
	if existingModels["gemma4"] {
		t.Error("gemma4 should not be in existingModels (it's a recommendation)")
	}

	if !cloudModels["glm-5.1:cloud"] {
		t.Error("glm-5.1:cloud should be in cloudModels")
	}
	if !cloudModels["kimi-k2.6:cloud"] {
		t.Error("kimi-k2.6:cloud should be in cloudModels (recommended cloud)")
	}
	if !cloudModels["qwen3.5:cloud"] {
		t.Error("qwen3.5:cloud should be in cloudModels (recommended cloud)")
	}
	if cloudModels["llama3.2"] {
		t.Error("llama3.2 should not be in cloudModels")
	}
}

func TestBuildModelList_RecommendedFieldSet(t *testing.T) {
	existing := []modelInfo{
		{Name: "gemma4", Remote: false},
		{Name: "llama3.2:latest", Remote: false},
	}

	items, _, _, _ := buildModelList(existing, nil, "")

	for _, item := range items {
		switch item.Name {
		case "gemma4", "qwen3.5", "glm-5.1:cloud", "kimi-k2.6:cloud", "qwen3.5:cloud":
			if !item.Recommended {
				t.Errorf("%q should have Recommended=true", item.Name)
			}
		case "llama3.2":
			if item.Recommended {
				t.Errorf("%q should have Recommended=false", item.Name)
			}
		}
	}
}

func TestBuildModelList_MixedCase_CloudRecsFirst(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// Cloud recs should sort before local recs in mixed case
	cloudIdx := slices.Index(got, "glm-5.1:cloud")
	localIdx := slices.Index(got, "gemma4")
	if cloudIdx > localIdx {
		t.Errorf("cloud recs should be before local recs in mixed case, got %v", got)
	}
}

func TestBuildModelList_OnlyLocal_CloudRecsStillFirst(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// Cloud recs sort before local recs regardless of installed inventory.
	localIdx := slices.Index(got, "gemma4")
	cloudIdx := slices.Index(got, "glm-5.1:cloud")
	if cloudIdx > localIdx {
		t.Errorf("cloud recs should be before local recs even when only local models installed, got %v", got)
	}
}

func TestBuildModelList_RecsAboveNonRecs(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "custom-model", Remote: false},
	}

	items, _, _, _ := buildModelList(existing, nil, "")
	got := names(items)

	// All recommended models should appear before non-recommended installed models
	lastRecIdx := -1
	firstNonRecIdx := len(got)
	for i, name := range got {
		isRec := name == "gemma4" || name == "qwen3.5" || name == "minimax-m2.7:cloud" || name == "glm-5.1:cloud" || name == "kimi-k2.6:cloud" || name == "qwen3.5:cloud"
		if isRec && i > lastRecIdx {
			lastRecIdx = i
		}
		if !isRec && i < firstNonRecIdx {
			firstNonRecIdx = i
		}
	}
	if lastRecIdx > firstNonRecIdx {
		t.Errorf("all recs should be above non-recs, got %v", got)
	}
}

func TestBuildModelList_CheckedBeforeRecs(t *testing.T) {
	existing := []modelInfo{
		{Name: "llama3.2:latest", Remote: false},
		{Name: "glm-5.1:cloud", Remote: true},
	}

	items, _, _, _ := buildModelList(existing, []string{"llama3.2"}, "")
	got := names(items)

	if got[0] != "llama3.2" {
		t.Errorf("checked model should be first even before recs, got %v", got)
	}
}

func TestBuildModelList_CurrentPrefersExactLocalOverCloudPrefix(t *testing.T) {
	existing := []modelInfo{
		{Name: "qwen3.5:cloud", Remote: true},
		{Name: "qwen3.5", Remote: false},
	}

	_, orderedChecked, _, _ := buildModelList(existing, []string{"qwen3.5", "qwen3.5:cloud"}, "qwen3.5")
	if len(orderedChecked) < 2 {
		t.Fatalf("expected orderedChecked to preserve both selections, got %v", orderedChecked)
	}
	if orderedChecked[0] != "qwen3.5" {
		t.Fatalf("expected exact local current to stay first, got %v", orderedChecked)
	}
}

func TestBuildModelList_CurrentPrefersExactCloudOverLocalPrefix(t *testing.T) {
	existing := []modelInfo{
		{Name: "qwen3.5", Remote: false},
		{Name: "qwen3.5:cloud", Remote: true},
	}

	_, orderedChecked, _, _ := buildModelList(existing, []string{"qwen3.5:cloud", "qwen3.5"}, "qwen3.5:cloud")
	if len(orderedChecked) < 2 {
		t.Fatalf("expected orderedChecked to preserve both selections, got %v", orderedChecked)
	}
	if orderedChecked[0] != "qwen3.5:cloud" {
		t.Fatalf("expected exact cloud current to stay first, got %v", orderedChecked)
	}
}

func TestEditorIntegration_SavedConfigSkipsSelection(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Save a config for opencode so it looks like a previous launch
	if err := SaveIntegration("opencode", []string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	// Verify loadIntegration returns the saved models
	saved, err := LoadIntegration("opencode")
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

func TestLauncherClientFilterDisabledCloudModels_ChecksStatusOncePerInvocation(t *testing.T) {
	var statusCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			statusCalls++
			fmt.Fprintf(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := &launcherClient{
		apiClient: api.NewClient(u, srv.Client()),
	}

	filtered := client.filterDisabledCloudModels(context.Background(), []string{"llama3.2", "glm-5.1:cloud", "qwen3.5:cloud"})
	if diff := cmp.Diff([]string{"llama3.2"}, filtered); diff != "" {
		t.Fatalf("filtered models mismatch (-want +got):\n%s", diff)
	}
	if statusCalls != 1 {
		t.Fatalf("expected one cloud status lookup, got %d", statusCalls)
	}
}

func TestSavedMatchesModels(t *testing.T) {
	tests := []struct {
		name   string
		saved  *config.IntegrationConfig
		models []string
		want   bool
	}{
		{
			name:   "nil saved",
			saved:  nil,
			models: []string{"llama3.2"},
			want:   false,
		},
		{
			name:   "identical order",
			saved:  &config.IntegrationConfig{Models: []string{"llama3.2", "qwen3:8b"}},
			models: []string{"llama3.2", "qwen3:8b"},
			want:   true,
		},
		{
			name:   "different order",
			saved:  &config.IntegrationConfig{Models: []string{"llama3.2", "qwen3:8b"}},
			models: []string{"qwen3:8b", "llama3.2"},
			want:   false,
		},
		{
			name:   "subset",
			saved:  &config.IntegrationConfig{Models: []string{"llama3.2", "qwen3:8b"}},
			models: []string{"llama3.2"},
			want:   false,
		},
		{
			name:   "nil models in saved with non-nil models",
			saved:  &config.IntegrationConfig{Models: nil},
			models: []string{"llama3.2"},
			want:   false,
		},
		{
			name:   "empty both",
			saved:  &config.IntegrationConfig{Models: nil},
			models: nil,
			want:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := savedMatchesModels(tt.saved, tt.models); got != tt.want {
				t.Fatalf("savedMatchesModels = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPrepareEditorIntegration_SavesOnlyAfterSuccessfulEdit(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	if err := SaveIntegration("droid", []string{"existing-model"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	editor := &stubEditorRunner{editErr: errors.New("boom")}
	err := prepareEditorIntegration("droid", editor, editor, []string{"new-model"})
	if err == nil || !strings.Contains(err.Error(), "setup failed") {
		t.Fatalf("expected setup failure, got %v", err)
	}

	saved, err := LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := cmp.Diff([]string{"existing-model"}, saved.Models); diff != "" {
		t.Fatalf("saved models mismatch (-want +got):\n%s", diff)
	}
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

	err := showOrPullWithPolicy(context.Background(), client, "test-model", missingModelPromptPull, false)
	if err != nil {
		t.Errorf("showOrPull should return nil when model exists, got: %v", err)
	}
}

func TestShowOrPullWithPolicy_ModelExists(t *testing.T) {
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

	err := showOrPullWithPolicy(context.Background(), client, "test-model", missingModelFail, false)
	if err != nil {
		t.Errorf("showOrPullWithPolicy should return nil when model exists, got: %v", err)
	}
}

func TestShowOrPullWithPolicy_ModelNotFound_FailDoesNotPromptOrPull(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatal("confirm prompt should not be called with fail policy")
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "missing-model", missingModelFail, false)
	if err == nil {
		t.Fatal("expected fail policy to return an error for missing model")
	}
	if !strings.Contains(err.Error(), "ollama pull missing-model") {
		t.Fatalf("expected actionable pull guidance, got: %v", err)
	}
	if pullCalled {
		t.Fatal("expected pull not to be called with fail policy")
	}
}

func TestShowOrPullWithPolicy_ModelNotFound_PromptPolicyPulls(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if !strings.Contains(prompt, "missing-model") {
			t.Fatalf("expected prompt to mention missing model, got %q", prompt)
		}
		return true, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "missing-model", missingModelPromptPull, false)
	if err != nil {
		t.Fatalf("expected prompt policy to pull and succeed, got %v", err)
	}
	if !pullCalled {
		t.Fatal("expected pull to be called with prompt policy")
	}
}

func TestShowOrPullWithPolicy_ModelNotFound_AutoPullPolicyPullsWithoutPrompt(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("confirm prompt should not be called with auto-pull policy: %q", prompt)
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "missing-model", missingModelAutoPull, false)
	if err != nil {
		t.Fatalf("expected auto-pull policy to pull and succeed, got %v", err)
	}
	if !pullCalled {
		t.Fatal("expected pull to be called with auto-pull policy")
	}
}

func TestShowOrPullWithPolicy_CloudModelNotFound_FailsEarlyForAllPolicies(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatal("confirm prompt should not be called for explicit cloud models")
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	for _, policy := range []missingModelPolicy{missingModelPromptPull, missingModelAutoPull, missingModelFail} {
		t.Run(fmt.Sprintf("policy=%d", policy), func(t *testing.T) {
			var pullCalled bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					w.WriteHeader(http.StatusNotFound)
					fmt.Fprintf(w, `{"error":"model not found"}`)
				case "/api/status":
					w.WriteHeader(http.StatusNotFound)
					fmt.Fprintf(w, `{"error":"not found"}`)
				case "/api/pull":
					pullCalled = true
					w.WriteHeader(http.StatusOK)
					fmt.Fprintf(w, `{"status":"success"}`)
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer srv.Close()

			u, _ := url.Parse(srv.URL)
			client := api.NewClient(u, srv.Client())

			err := showOrPullWithPolicy(context.Background(), client, "glm-5.1:cloud", policy, true)
			if err == nil {
				t.Fatalf("expected cloud model not-found error for policy %d", policy)
			}
			if !strings.Contains(err.Error(), `model "glm-5.1:cloud" not found`) {
				t.Fatalf("expected not-found error for policy %d, got %v", policy, err)
			}
			if pullCalled {
				t.Fatalf("expected pull not to be called for cloud model with policy %d", policy)
			}
		})
	}
}

func TestShowOrPullWithPolicy_CloudModelDisabled_FailsWithCloudDisabledError(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatal("confirm prompt should not be called for explicit cloud models")
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	for _, policy := range []missingModelPolicy{missingModelPromptPull, missingModelAutoPull, missingModelFail} {
		t.Run(fmt.Sprintf("policy=%d", policy), func(t *testing.T) {
			var pullCalled bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					w.WriteHeader(http.StatusNotFound)
					fmt.Fprintf(w, `{"error":"model not found"}`)
				case "/api/status":
					w.WriteHeader(http.StatusOK)
					fmt.Fprintf(w, `{"cloud":{"disabled":true,"source":"config"}}`)
				case "/api/pull":
					pullCalled = true
					w.WriteHeader(http.StatusOK)
					fmt.Fprintf(w, `{"status":"success"}`)
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer srv.Close()

			u, _ := url.Parse(srv.URL)
			client := api.NewClient(u, srv.Client())

			err := showOrPullWithPolicy(context.Background(), client, "glm-5.1:cloud", policy, true)
			if err == nil {
				t.Fatalf("expected cloud disabled error for policy %d", policy)
			}
			if !strings.Contains(err.Error(), "remote inference is unavailable") {
				t.Fatalf("expected cloud disabled error for policy %d, got %v", policy, err)
			}
			if pullCalled {
				t.Fatalf("expected pull not to be called for cloud model with policy %d", policy)
			}
		})
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
	err := showOrPullWithPolicy(context.Background(), client, "missing-model", missingModelPromptPull, false)
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

	_ = showOrPullWithPolicy(context.Background(), client, "qwen3.5", missingModelPromptPull, false)
	if receivedModel != "qwen3.5" {
		t.Errorf("expected Show to be called with %q, got %q", "qwen3.5", receivedModel)
	}
}

func TestShowOrPull_ModelNotFound_ConfirmYes_Pulls(t *testing.T) {
	// Set up hook so confirmPrompt doesn't need a terminal
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if !strings.Contains(prompt, "missing-model") {
			t.Errorf("expected prompt to contain model name, got %q", prompt)
		}
		return true, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "missing-model", missingModelPromptPull, false)
	if err != nil {
		t.Errorf("ShowOrPull should succeed after pull, got: %v", err)
	}
	if !pullCalled {
		t.Error("expected pull to be called when user confirms download")
	}
}

func TestShowOrPull_ModelNotFound_ConfirmNo_Cancelled(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return false, ErrCancelled
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			t.Error("pull should not be called when user declines")
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "missing-model", missingModelPromptPull, false)
	if err == nil {
		t.Error("ShowOrPull should return error when user declines")
	}
}

func TestShowOrPull_CloudModel_NotFoundDoesNotPull(t *testing.T) {
	// Confirm prompt should NOT be called for explicit cloud models
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Error("confirm prompt should not be called for cloud models")
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "glm-5.1:cloud", missingModelPromptPull, true)
	if err == nil {
		t.Error("ShowOrPull should return not-found error for cloud model")
	}
	if !strings.Contains(err.Error(), `model "glm-5.1:cloud" not found`) {
		t.Errorf("expected cloud model not-found error, got: %v", err)
	}
	if pullCalled {
		t.Error("expected pull not to be called for cloud model")
	}
}

func TestShowOrPull_CloudLegacySuffix_NotFoundDoesNotPull(t *testing.T) {
	// Confirm prompt should NOT be called for explicit cloud models
	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Error("confirm prompt should not be called for cloud models")
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, `{"status":"success"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := showOrPullWithPolicy(context.Background(), client, "gpt-oss:20b-cloud", missingModelPromptPull, true)
	if err == nil {
		t.Error("ShowOrPull should return not-found error for cloud model")
	}
	if !strings.Contains(err.Error(), `model "gpt-oss:20b-cloud" not found`) {
		t.Errorf("expected cloud model not-found error, got: %v", err)
	}
	if pullCalled {
		t.Error("expected pull not to be called for cloud model")
	}
}

func TestConfirmPrompt_DelegatesToHook(t *testing.T) {
	oldHook := DefaultConfirmPrompt
	var hookCalled bool
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		hookCalled = true
		if prompt != "test prompt?" {
			t.Errorf("expected prompt %q, got %q", "test prompt?", prompt)
		}
		return true, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	ok, err := ConfirmPrompt("test prompt?")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("expected true from hook")
	}
	if !hookCalled {
		t.Error("expected DefaultConfirmPrompt hook to be called")
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

func TestEnsureAuth_PreservesCancelledSignInHook(t *testing.T) {
	oldSignIn := DefaultSignIn
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		return "", ErrCancelled
	}
	defer func() { DefaultSignIn = oldSignIn }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"not found"}`)
		case "/api/me":
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintf(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := ensureAuth(context.Background(), client, map[string]bool{"cloud-model:cloud": true}, []string{"cloud-model:cloud"})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("expected ErrCancelled, got %v", err)
	}
}

func TestEnsureAuth_DeclinedFallbackReturnsCancelled(t *testing.T) {
	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return false, nil
	}
	defer func() { DefaultConfirmPrompt = oldConfirm }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"not found"}`)
		case "/api/me":
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintf(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	client := api.NewClient(u, srv.Client())

	err := ensureAuth(context.Background(), client, map[string]bool{"cloud-model:cloud": true}, []string{"cloud-model:cloud"})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("expected ErrCancelled, got %v", err)
	}
}

func TestHyperlink(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		text     string
		wantURL  string
		wantText string
	}{
		{
			name:     "basic link",
			url:      "https://example.com",
			text:     "click here",
			wantURL:  "https://example.com",
			wantText: "click here",
		},
		{
			name:     "url with path",
			url:      "https://example.com/docs/install",
			text:     "install docs",
			wantURL:  "https://example.com/docs/install",
			wantText: "install docs",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hyperlink(tt.url, tt.text)

			// Should contain OSC 8 escape sequences
			if !strings.Contains(got, "\033]8;;") {
				t.Error("should contain OSC 8 open sequence")
			}
			if !strings.Contains(got, tt.wantURL) {
				t.Errorf("should contain URL %q", tt.wantURL)
			}
			if !strings.Contains(got, tt.wantText) {
				t.Errorf("should contain text %q", tt.wantText)
			}

			// Should have closing OSC 8 sequence
			wantSuffix := "\033]8;;\033\\"
			if !strings.HasSuffix(got, wantSuffix) {
				t.Error("should end with OSC 8 close sequence")
			}
		})
	}
}

func TestIntegration_InstallHint(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantEmpty bool
		wantURL   string
	}{
		{
			name:    "claude has hint",
			input:   "claude",
			wantURL: "https://code.claude.com/docs/en/quickstart",
		},
		{
			name:    "codex has hint",
			input:   "codex",
			wantURL: "https://developers.openai.com/codex/cli/",
		},
		{
			name:    "openclaw has hint",
			input:   "openclaw",
			wantURL: "https://docs.openclaw.ai",
		},
		{
			name:      "unknown has no hint",
			input:     "unknown",
			wantEmpty: true,
		},
		{
			name:      "empty name has no hint",
			input:     "",
			wantEmpty: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ""
			integration, err := integrationFor(tt.input)
			if err == nil {
				got = integration.installHint
			}
			if tt.wantEmpty {
				if got != "" {
					t.Errorf("expected empty hint, got %q", got)
				}
				return
			}
			if !strings.Contains(got, "Install from") {
				t.Errorf("hint should start with 'Install from', got %q", got)
			}
			if !strings.Contains(got, tt.wantURL) {
				t.Errorf("hint should contain URL %q, got %q", tt.wantURL, got)
			}
			// Should be a clickable hyperlink
			if !strings.Contains(got, "\033]8;;") {
				t.Error("hint URL should be wrapped in OSC 8 hyperlink")
			}
		})
	}
}

func TestListIntegrationInfos(t *testing.T) {
	infos := ListIntegrationInfos()

	t.Run("excludes aliases", func(t *testing.T) {
		for _, info := range infos {
			if integrationAliases[info.Name] {
				t.Errorf("alias %q should not appear in ListIntegrationInfos", info.Name)
			}
		}
	})

	t.Run("follows launcher order", func(t *testing.T) {
		got := make([]string, 0, len(infos))
		for _, info := range infos {
			got = append(got, info.Name)
		}
		if diff := compareStrings(got, integrationOrder); diff != "" {
			t.Fatalf("launcher integration order mismatch: %s", diff)
		}
	})

	t.Run("all fields populated", func(t *testing.T) {
		for _, info := range infos {
			if info.Name == "" {
				t.Error("Name should not be empty")
			}
			if info.DisplayName == "" {
				t.Errorf("DisplayName for %q should not be empty", info.Name)
			}
		}
	})

	t.Run("includes known integrations", func(t *testing.T) {
		known := map[string]bool{"claude": false, "codex": false, "opencode": false}
		for _, info := range infos {
			if _, ok := known[info.Name]; ok {
				known[info.Name] = true
			}
		}
		for name, found := range known {
			if !found {
				t.Errorf("expected %q in ListIntegrationInfos", name)
			}
		}
	})

	t.Run("includes hermes", func(t *testing.T) {
		for _, info := range infos {
			if info.Name == "hermes" {
				return
			}
		}
		t.Fatal("expected hermes to be included in ListIntegrationInfos")
	})

	t.Run("hermes still resolves explicitly", func(t *testing.T) {
		name, runner, err := LookupIntegration("hermes")
		if err != nil {
			t.Fatalf("expected explicit hermes integration lookup to work, got %v", err)
		}
		if name != "hermes" {
			t.Fatalf("expected canonical name hermes, got %q", name)
		}
		if runner.String() == "" {
			t.Fatal("expected hermes integration runner to be present")
		}
	})
}

func TestBuildModelList_Descriptions(t *testing.T) {
	t.Run("installed recommended has base description", func(t *testing.T) {
		existing := []modelInfo{
			{Name: "qwen3.5", Remote: false},
		}
		items, _, _, _ := buildModelList(existing, nil, "")

		for _, item := range items {
			if item.Name == "qwen3.5" {
				if strings.HasSuffix(item.Description, "install?") {
					t.Errorf("installed model should not have 'install?' suffix, got %q", item.Description)
				}
				if item.Description == "" {
					t.Error("installed recommended model should have a description")
				}
				return
			}
		}
		t.Error("qwen3.5 not found in items")
	})

	t.Run("not-installed local rec has VRAM in description", func(t *testing.T) {
		items, _, _, _ := buildModelList(nil, nil, "")

		for _, item := range items {
			if item.Name == "qwen3.5" {
				if !strings.Contains(item.Description, "~11GB") {
					t.Errorf("not-installed qwen3.5 should show VRAM hint, got %q", item.Description)
				}
				return
			}
		}
		t.Error("qwen3.5 not found in items")
	})

	t.Run("installed local rec omits VRAM", func(t *testing.T) {
		existing := []modelInfo{
			{Name: "qwen3.5", Remote: false},
		}
		items, _, _, _ := buildModelList(existing, nil, "")

		for _, item := range items {
			if item.Name == "qwen3.5" {
				if strings.Contains(item.Description, "~11GB") {
					t.Errorf("installed qwen3.5 should not show VRAM hint, got %q", item.Description)
				}
				return
			}
		}
		t.Error("qwen3.5 not found in items")
	})
}

func TestIntegration_Editor(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"droid", true},
		{"opencode", true},
		{"openclaw", true},
		{"claude", false},
		{"codex", false},
		{"nonexistent", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := false
			integration, err := integrationFor(tt.name)
			if err == nil {
				got = integration.editor
			}
			if got != tt.want {
				t.Errorf("integrationFor(%q).editor = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

func TestIntegration_AutoInstallable(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"openclaw", true},
		{"pi", true},
		{"hermes", true},
		{"claude", false},
		{"codex", false},
		{"opencode", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := false
			integration, err := integrationFor(tt.name)
			if err == nil {
				got = integration.autoInstallable
			}
			if got != tt.want {
				t.Errorf("integrationFor(%q).autoInstallable = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

func TestIntegrationModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("returns nil when not configured", func(t *testing.T) {
		if got := IntegrationModels("droid"); got != nil {
			t.Errorf("expected nil, got %v", got)
		}
	})

	t.Run("returns all saved models", func(t *testing.T) {
		if err := SaveIntegration("droid", []string{"llama3.2", "qwen3.5"}); err != nil {
			t.Fatal(err)
		}
		got := IntegrationModels("droid")
		want := []string{"llama3.2", "qwen3.5"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("IntegrationModels mismatch (-want +got):\n%s", diff)
		}
	})
}
