package chat

import (
	"context"
	"slices"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
	apptui "github.com/ollama/ollama/cmd/tui"
)

func TestChatModelCommandOpensPicker(t *testing.T) {
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/model"),
		width:  100,
		height: 20,
		opts: Options{
			Model:               "llama3.2",
			ContextWindowTokens: 131072,
			ModelOptions: func(context.Context) ([]ModelOption, error) {
				return []ModelOption{
					{Name: "kimi-k2.6:cloud", Description: "cloud coding"},
					{Name: "llama3.2", Description: "local"},
				}, nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	m = updated.(chatModel)
	if m.modelPicker == nil {
		t.Fatal("model picker was not opened")
	}
	view := stripANSI(m.View())
	if !strings.Contains(view, "Select model") ||
		!strings.Contains(view, "Type to filter") ||
		!strings.Contains(view, "kimi-k2.6:cloud") ||
		!strings.Contains(view, "llama3.2") {
		t.Fatalf("model picker view missing content: %q", view)
	}
	if strings.Contains(view, "Search...") {
		t.Fatalf("model picker should render inline without full search box: %q", view)
	}
	if strings.Contains(view, "local") || strings.Contains(view, "cloud coding") {
		t.Fatalf("inline model picker should stay compact without descriptions: %q", view)
	}
	if !strings.Contains(view, "│ █") {
		t.Fatalf("inline model picker should keep input box visible: %q", view)
	}
}

func TestChatModelCommandShowsRecommendedFirstWithoutSections(t *testing.T) {
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/model"),
		width:  100,
		height: 20,
		opts: Options{
			Model: "llama3.2",
			ModelOptions: func(context.Context) ([]ModelOption, error) {
				return []ModelOption{
					{Name: "llama3.2", Description: "selected local"},
					{Name: "gemma4", Description: "local"},
					{Name: "glm-5.2:cloud", Description: "recommended cloud", Recommended: true, Cloud: true},
					{Name: "kimi-k2.7-code:cloud", Description: "another recommended cloud", Recommended: true, Cloud: true},
				}, nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	view := stripANSI(updated.(chatModel).View())
	for _, unwanted := range []string{"Recommended", "More", "recommended cloud", "selected local"} {
		if strings.Contains(view, unwanted) {
			t.Fatalf("compact model picker should be flat and description-free; found %q in %q", unwanted, view)
		}
	}
	firstRecommended := strings.Index(view, "glm-5.2:cloud")
	secondRecommended := strings.Index(view, "kimi-k2.7-code:cloud")
	current := strings.Index(view, "llama3.2")
	local := strings.Index(view, "gemma4")
	if firstRecommended < 0 || secondRecommended < 0 || current < 0 || local < 0 {
		t.Fatalf("compact model picker missing expected models: %q", view)
	}
	if !(firstRecommended < current && secondRecommended < current && current < local) {
		t.Fatalf("compact model picker order should be recommended, current, local: %q", view)
	}
}

func TestChatModelCommandOpensSmallPicker(t *testing.T) {
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/model"),
		width:  100,
		height: 24,
		opts: Options{
			Model: "model-1",
			ModelOptions: func(context.Context) ([]ModelOption, error) {
				return []ModelOption{
					{Name: "model-1"},
					{Name: "model-2"},
					{Name: "model-3"},
					{Name: "model-4"},
					{Name: "model-5"},
					{Name: "model-6"},
					{Name: "model-7"},
				}, nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	m = updated.(chatModel)
	view := stripANSI(m.View())
	for _, want := range []string{"model-1", "model-5", "... and 2 more"} {
		if !strings.Contains(view, want) {
			t.Fatalf("small model picker missing %q: %q", want, view)
		}
	}
	if strings.Contains(view, "model-6") || strings.Contains(view, "model-7") {
		t.Fatalf("small model picker rendered too many items: %q", view)
	}
}

func TestChatModelPickerStaysInlineWhenSmall(t *testing.T) {
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/model"),
		width:  44,
		height: 10,
		opts: Options{
			Model: "llama3.2",
			ModelOptions: func(context.Context) ([]ModelOption, error) {
				return []ModelOption{
					{Name: "kimi-k2.6:cloud", Description: "cloud coding"},
					{Name: "llama3.2", Description: "local"},
				}, nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	m = updated.(chatModel)
	view := stripANSI(m.View())
	if !strings.Contains(view, "Select model") || !strings.Contains(view, "Type to filter") {
		t.Fatalf("small model picker should stay inline: %q", view)
	}
	if strings.Contains(view, "Search...") {
		t.Fatalf("small model picker should not use bespoke full-frame search: %q", view)
	}
}

func TestChatModelPickerShowsRecommendedModelsFirst(t *testing.T) {
	models := normalizeModelOptions([]ModelOption{
		{Name: "llama3.2", Description: "local"},
		{Name: "kimi-k2.6:cloud", Description: "cloud coding", Recommended: true},
		{Name: "qwen3.5:cloud", Description: "cloud reasoning", Recommended: true},
		{Name: "gemma4", Description: "local"},
	})
	got := make([]string, 0, len(models))
	for _, model := range models {
		got = append(got, model.Name)
	}
	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "llama3.2", "gemma4"}
	if !slices.Equal(got, want) {
		t.Fatalf("model order = %#v, want %#v", got, want)
	}
}

func TestChatModelPickerPinsCurrentThenRecommendedModels(t *testing.T) {
	models := normalizeModelOptions([]ModelOption{
		{Name: "llama3.2", Description: "local"},
		{Name: "glm-5.2:cloud", Description: "cloud selected", Recommended: true, Cloud: true},
		{Name: "kimi-k2.7-code:cloud", Description: "cloud coding", Recommended: true, Cloud: true},
		{Name: "gemma4", Description: "local"},
	})

	items := modelSelectorItems(models, "glm-5.2:cloud")
	got := make([]string, 0, len(items))
	for _, item := range items {
		got = append(got, item.Name)
	}
	want := []string{"glm-5.2:cloud", "kimi-k2.7-code:cloud", "llama3.2", "gemma4"}
	if !slices.Equal(got, want) {
		t.Fatalf("selector item order = %#v, want %#v", got, want)
	}
	for _, item := range items[:3] {
		if !item.Recommended {
			t.Fatalf("%q should be pinned in the first picker section", item.Name)
		}
	}
	if items[0].Description != "cloud selected" {
		t.Fatalf("current model description = %q, want plain model description", items[0].Description)
	}
}

func TestInitialModelPickerRendersBeforeChatShell(t *testing.T) {
	models := normalizeModelOptions([]ModelOption{
		{Name: "glm-5.2:cloud", Description: "cloud selected", Recommended: true, Cloud: true},
		{Name: "llama3.2", Description: "local"},
	})
	picker := apptui.NewModelSelectorModel("Select model", modelSelectorItems(models, "glm-5.2:cloud"), "glm-5.2:cloud", "")
	m := chatModel{
		width:           100,
		height:          20,
		openModelOnInit: true,
		modelPicker:     &picker,
		entries:         []chatEntry{{role: "assistant", content: "old chat content"}},
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "Select model") || !strings.Contains(view, "llama3.2") {
		t.Fatalf("initial picker view missing model content: %q", view)
	}
	if strings.Contains(view, "old chat content") || strings.Contains(view, "│ █") {
		t.Fatalf("initial picker should render before chat shell: %q", view)
	}
}

func TestChatModelPickerRanksClosestFilteredModelFirst(t *testing.T) {
	models := normalizeModelOptions([]ModelOption{
		{Name: "gemma3:27b", Description: "recommended but longer", Recommended: true},
		{Name: "llama3.2", Description: "mentions gemm in description"},
		{Name: "gemma4:27b", Description: "longer local"},
		{Name: "gemma4", Description: "short local"},
	})
	picker := apptui.NewModelSelectorModel("Select model", modelSelectorItems(models, ""), "", "gemm")

	filtered := picker.FilteredItems()
	got := make([]string, 0, len(filtered))
	for _, model := range filtered {
		got = append(got, model.Name)
	}
	want := []string{"gemma4", "gemma3:27b", "gemma4:27b", "llama3.2"}
	if !slices.Equal(got, want) {
		t.Fatalf("filtered model order = %#v, want %#v", got, want)
	}
}

func TestChatModelPickerFiltersAndSwitchesModel(t *testing.T) {
	var savedModel string
	originalMessages := []api.Message{{Role: "user", Content: "keep me"}}
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "chat-1",
		input:    []rune("/model qwen"),
		width:    100,
		height:   20,
		messages: slices.Clone(originalMessages),
		opts: Options{
			Model: "llama3.2",
			ModelOptions: func(context.Context) ([]ModelOption, error) {
				return []ModelOption{
					{Name: "llama3.2", Description: "local"},
					{Name: "qwen3.5:cloud", Description: "cloud reasoning"},
				}, nil
			},
			ToolRegistryForModel: func(ctx context.Context, model string) *coreagent.Registry {
				if model != "qwen3.5:cloud" {
					t.Fatalf("tool registry model = %q, want qwen3.5:cloud", model)
				}
				registry := &coreagent.Registry{}
				registry.Register(chatTestTool{})
				return registry
			},
			ContextWindowTokensForModel: func(ctx context.Context, model string, fallback int) int {
				if model != "qwen3.5:cloud" {
					t.Fatalf("context model = %q, want qwen3.5:cloud", model)
				}
				if fallback != 0 {
					t.Fatalf("context fallback = %d, want 0 after model switch", fallback)
				}
				return 262144
			},
			SystemPromptForModel: func(ctx context.Context, model string, registry *coreagent.Registry, toolsDisabled bool) string {
				if model != "qwen3.5:cloud" {
					t.Fatalf("system prompt model = %q, want qwen3.5:cloud", model)
				}
				if registry == nil {
					t.Fatalf("system prompt registry missing fake tool: %#v", registry)
				}
				if _, ok := registry.Get("fake_tool"); !ok {
					t.Fatalf("system prompt registry missing fake tool: %#v", registry)
				}
				return "system for " + model
			},
			OnModelSelected: func(ctx context.Context, model string) error {
				savedModel = model
				return nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	m = updated.(chatModel)
	if m.modelPicker == nil || m.modelPicker.Filter() != "qwen" {
		t.Fatalf("model picker = %#v, want qwen filter", m.modelPicker)
	}
	if view := stripANSI(m.View()); !strings.Contains(view, "qwen3.5:cloud") || strings.Contains(view, "llama3.2") {
		t.Fatalf("filtered model picker view = %q", view)
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("switching models should not start a command")
	}
	if m.modelPicker != nil {
		t.Fatal("model picker should close after selection")
	}
	if m.status != "ready" || m.notificationLine() != "" {
		t.Fatalf("model switch should not show action status, status=%q notification=%q", m.status, m.notificationLine())
	}
	if m.opts.Model != "qwen3.5:cloud" {
		t.Fatalf("model = %q, want qwen3.5:cloud", m.opts.Model)
	}
	if m.chatID != "chat-1" {
		t.Fatalf("chatID = %q, want chat-1", m.chatID)
	}
	if len(m.messages) != len(originalMessages) || m.messages[0].Content != originalMessages[0].Content {
		t.Fatalf("messages changed on model switch: %#v", m.messages)
	}
	if len(m.entries) != 0 {
		t.Fatalf("model switch should not append transcript entries: %#v", m.entries)
	}
	if savedModel != "qwen3.5:cloud" {
		t.Fatalf("saved model = %q, want qwen3.5:cloud", savedModel)
	}
	if m.opts.Tools == nil {
		t.Fatalf("tools registry was not rebuilt for model: %#v", m.opts.Tools)
	}
	if _, ok := m.opts.Tools.Get("fake_tool"); !ok {
		t.Fatalf("tools registry was not rebuilt for model: %#v", m.opts.Tools)
	}
	if m.opts.ContextWindowTokens != 262144 {
		t.Fatalf("context window = %d, want 262144", m.opts.ContextWindowTokens)
	}
	if m.opts.SystemPrompt != "system for qwen3.5:cloud" {
		t.Fatalf("system prompt = %q", m.opts.SystemPrompt)
	}
}

func TestChatModelSelectionStartsBackgroundPreload(t *testing.T) {
	m := chatModel{
		ctx: context.Background(),
		opts: Options{
			Model: "llama3.2",
			PreloadModel: func(context.Context, string, *api.ThinkValue) (int, error) {
				return 0, nil
			},
		},
	}

	if err := m.applyModelSelection("qwen3", false); err != nil {
		t.Fatal(err)
	}
	cmd := m.startModelPreload("qwen3")
	if cmd == nil {
		t.Fatal("model switch should start background preload when configured")
	}
	if m.preloadingModel != "qwen3" {
		t.Fatalf("preloadingModel = %q, want qwen3", m.preloadingModel)
	}
}

func TestChatModelSwitchNextRunKeepsHistory(t *testing.T) {
	client := &chatCaptureClient{}
	history := []api.Message{
		{Role: "user", Content: "old question"},
		{Role: "assistant", Content: "old answer"},
	}
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "chat-1",
		messages: slices.Clone(history),
		input:    []rune("continue"),
		opts: Options{
			Model:  "llama3.2",
			Client: client,
			SystemPromptForModel: func(_ context.Context, model string, _ *coreagent.Registry, _ bool) string {
				return "system for " + model
			},
		},
	}
	if err := m.applyModelSelection("qwen3", true); err != nil {
		t.Fatal(err)
	}

	updated, cmd := m.handleSubmit()
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("next prompt should start a model run")
	}
	done := waitForRunDone(t, m.events)
	if done.err != nil {
		t.Fatal(done.err)
	}

	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	req := client.requests[0]
	if req.Model != "qwen3" {
		t.Fatalf("request model = %q, want qwen3", req.Model)
	}
	if len(req.Messages) != 4 {
		t.Fatalf("request messages = %#v, want system + 2 history + new user", req.Messages)
	}
	if req.Messages[0].Role != "system" || req.Messages[0].Content != "system for qwen3" {
		t.Fatalf("system message = %#v", req.Messages[0])
	}
	for i, want := range history {
		got := req.Messages[i+1]
		if got.Role != want.Role || got.Content != want.Content {
			t.Fatalf("history message %d = %#v, want %#v", i, got, want)
		}
	}
	if req.Messages[3].Role != "user" || req.Messages[3].Content != "continue" {
		t.Fatalf("new user message = %#v", req.Messages[3])
	}
}
