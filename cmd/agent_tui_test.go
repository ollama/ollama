package cmd

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	agentchat "github.com/ollama/ollama/cmd/tui/chat"
)

func TestAgentSystemPromptIncludesSessionWorkingDirOnce(t *testing.T) {
	workingDir := t.TempDir()
	prompt := agentSystemPromptAtWithWorkingDir(
		time.Date(2026, time.July, 14, 0, 0, 0, 0, time.UTC),
		"test-model",
		"model instruction",
		"caller instruction",
		workingDir,
	)

	workingDirInstruction := "Current working directory: " + strconv.Quote(workingDir) + "."
	if got := strings.Count(prompt, workingDirInstruction); got != 1 {
		t.Fatalf("working directory instruction count = %d, want 1:\n%s", got, prompt)
	}
	for _, want := range []string{"model instruction", "caller instruction"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestAgentWorkingDirIgnoresGetwdFailure(t *testing.T) {
	original := agentGetwd
	agentGetwd = func() (string, error) {
		return "", errors.New("getwd failed")
	}
	t.Cleanup(func() {
		agentGetwd = original
	})

	if got := agentWorkingDir(); got != "" {
		t.Fatalf("working directory = %q, want empty on getwd failure", got)
	}
}

func TestAgentSystemPromptIncludesSkillCatalog(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "release-notes"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "release-notes", "SKILL.md"), []byte("---\nname: release-notes\ndescription: Draft releases.\n---\nUse bullets."), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := coreagent.DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := agentSystemPromptAt(time.Date(2026, 7, 14, 0, 0, 0, 0, time.UTC), "model", "", catalog.SystemContext())
	if !strings.Contains(got, "release-notes: Draft releases.") || !strings.Contains(got, "normal approval rules") {
		t.Fatalf("system prompt missing skill context: %q", got)
	}
}

func TestAgentSelectionItemsUseLaunchSections(t *testing.T) {
	items := agentSelectionItems([]agentchat.ModelOption{
		{Name: "glm-5.2:cloud", Description: "cloud", Recommended: true, Cloud: true},
		{Name: "llama3.2", Description: "local"},
	})

	if len(items) != 2 {
		t.Fatalf("items = %d, want 2", len(items))
	}
	if !items[0].Recommended {
		t.Fatalf("cloud recommendation should be pinned: %#v", items[0])
	}
	if items[1].Recommended {
		t.Fatalf("local selected model should stay in launch More section: %#v", items[1])
	}
	if items[1].Description != "local" {
		t.Fatalf("selected model description = %q, want plain description", items[1].Description)
	}
}

func TestContextWindowFromRecommendationsMatchesCloudModel(t *testing.T) {
	got := contextWindowFromRecommendations("glm-5.2:cloud", []api.ModelRecommendation{
		{Model: "gemma4:cloud", ContextLength: 32768},
		{Model: "glm-5.2:cloud", ContextLength: 1048576},
	})
	if got != 1048576 {
		t.Fatalf("context window = %d, want 1048576", got)
	}
}

func TestShowResponseContextWindowReadsArchitectureContextLength(t *testing.T) {
	got := showResponseContextWindow(&api.ShowResponse{
		ModelInfo: map[string]any{
			"qwen3.context_length":                       uint32(262144),
			"qwen3.rope.scaling.original_context_length": uint32(32768),
		},
	})
	if got != 262144 {
		t.Fatalf("context window = %d, want 262144", got)
	}
}

func TestProcessContextWindowForModelMatchesLatestAlias(t *testing.T) {
	got := processContextWindowForModel("ornith", &api.ProcessResponse{
		Models: []api.ProcessModelResponse{
			{Name: "other:latest", Model: "other:latest", ContextLength: 32768},
			{Name: "ornith:latest", Model: "ornith:latest", ContextLength: 262144},
		},
	})
	if got != 262144 {
		t.Fatalf("context window = %d, want 262144", got)
	}
}

func TestSaveLastAgentModel(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	if err := saveLastAgentModel(" qwen3:8b "); err != nil {
		t.Fatalf("saveLastAgentModel returned error: %v", err)
	}
	if got := config.LastModel(); got != "qwen3:8b" {
		t.Fatalf("last model = %q, want qwen3:8b", got)
	}

	if err := saveLastAgentModel(" "); err != nil {
		t.Fatalf("saveLastAgentModel blank returned error: %v", err)
	}
	if got := config.LastModel(); got != "qwen3:8b" {
		t.Fatalf("blank save changed last model to %q", got)
	}
}

func TestApplyAgentFlagsNoTools(t *testing.T) {
	cmd := &cobra.Command{}
	registerAgentFlags(cmd)
	if err := cmd.Flags().Set("no-tools", "true"); err != nil {
		t.Fatal(err)
	}

	var opts agentTUIOptions
	if _, err := applyAgentFlags(cmd, &opts); err != nil {
		t.Fatalf("applyAgentFlags returned error: %v", err)
	}
	if !opts.ToolsDisabled {
		t.Fatal("--no-tools should disable tools")
	}
}
