package chat

import (
	"context"
	"fmt"
	"slices"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	apptui "github.com/ollama/ollama/cmd/tui"
)

type chatModelPicker = apptui.SelectorModel

func (m *chatModel) openModelPicker(filter string) (tea.Model, tea.Cmd) {
	if m.opts.ModelOptions == nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "Model picker is unavailable.", err: "Model picker is unavailable."}))
		m.status = "error"
		return *m, nil
	}

	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	models, err := m.opts.ModelOptions(ctx)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not list models: %v", err), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	models = normalizeModelOptions(models)
	if len(models) == 0 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "No models available."}))
		m.status = "ready"
		return *m, nil
	}

	items := modelSelectorItems(models, m.opts.Model)
	current := m.opts.Model
	if !m.openModelOnInit {
		items = compactModelSelectorItems(models, m.opts.Model)
		current = ""
	}
	picker := apptui.NewModelSelectorModel("Select model", items, current, filter)
	picker.SetHelpText("↑/↓ navigate • enter select • type search • esc cancel")
	m.modelPicker = &picker
	m.modelPickerModels = models
	m.status = "model"
	return *m, nil
}

func normalizeModelOptions(models []ModelOption) []ModelOption {
	seen := make(map[string]struct{}, len(models))
	out := make([]ModelOption, 0, len(models))
	for _, model := range models {
		model.Name = strings.TrimSpace(model.Name)
		model.Description = strings.TrimSpace(model.Description)
		if model.Name == "" {
			continue
		}
		key := strings.ToLower(model.Name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, model)
	}
	slices.SortStableFunc(out, func(a, b ModelOption) int {
		if a.Recommended == b.Recommended {
			return 0
		}
		if a.Recommended {
			return -1
		}
		return 1
	})
	return out
}

func modelSelectorItems(models []ModelOption, current string) []apptui.SelectItem {
	return modelSelectorItemsWithCurrentPriority(models, current, true)
}

func compactModelSelectorItems(models []ModelOption, current string) []apptui.SelectItem {
	return modelSelectorItemsWithCurrentPriority(models, current, false)
}

func modelSelectorItemsWithCurrentPriority(models []ModelOption, current string, pinCurrent bool) []apptui.SelectItem {
	ordered := slices.Clone(models)
	slices.SortStableFunc(ordered, func(a, b ModelOption) int {
		if cmp := compareModelPickerGroup(modelPickerGroup(a, current, pinCurrent), modelPickerGroup(b, current, pinCurrent)); cmp != 0 {
			return cmp
		}
		return 0
	})

	items := make([]apptui.SelectItem, 0, len(ordered))
	for _, model := range ordered {
		items = append(items, apptui.SelectItem{
			Name:              model.Name,
			Description:       modelOptionMeta(model),
			Recommended:       model.Name == current || !model.Cloud || model.Recommended,
			AvailabilityBadge: model.AvailabilityBadge,
		})
	}
	return items
}

func modelPickerGroup(model ModelOption, current string, pinCurrent bool) int {
	if pinCurrent && model.Name == current {
		return 0
	}
	if model.Recommended {
		return 1
	}
	if model.Name == current {
		return 2
	}
	if !model.Cloud {
		return 3
	}
	return 4
}

func compareModelPickerGroup(a, b int) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func (m chatModel) updateModelPicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.modelPicker == nil {
		return m, nil
	}
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.modelPicker = nil
		m.modelPickerModels = nil
		m.openModelOnInit = false
		m.status = "ready"
		return m, nil
	case tea.KeyEnter:
		return m.selectModel()
	default:
		m.modelPicker.UpdateNavigation(msg)
	}
	return m, nil
}

func (m chatModel) selectModel() (tea.Model, tea.Cmd) {
	if m.modelPicker == nil {
		return m, nil
	}
	selectedItem, ok := m.modelPicker.SelectedItem()
	if !ok {
		return m, nil
	}
	selected, ok := m.modelOptionForSelection(selectedItem.Name)
	if !ok {
		return m, nil
	}

	// Cloud models need auth + plan check before switching. If we already
	// know the badge state from the model list, go directly to the right
	// prompt — no "checking" spinner.
	if selected.Cloud && m.opts.CheckCloudModel != nil {
		switch selected.AvailabilityBadge {
		case "Sign in required":
			return m.startCloudAuthSignIn(selected.Name, selected.RequiredPlan, selected.SignInURL)
		case "Upgrade required":
			return m.startCloudAuthUpgrade(selected.Name, selected.RequiredPlan)
		}
		// Badge is empty — auth is satisfied (confirmed via Whoami when the
		// list was built). Apply directly.
	}

	m.modelPicker = nil
	m.modelPickerModels = nil
	m.openModelOnInit = false
	if err := m.applyModelSelection(selected.Name, true); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", err), err: err.Error()}))
		m.status = "error"
		return m, nil
	}
	m.status = "ready"
	return m, tea.Batch(m.startModelPreload(selected.Name), cloudModelPreflightCmd(m.ctx, m.opts, selected.Name, selected.RequiredPlan))
}

func (m chatModel) modelOptionForSelection(name string) (ModelOption, bool) {
	for _, model := range m.modelPickerModels {
		if model.Name == name {
			return model, true
		}
	}
	return ModelOption{}, false
}

func (m *chatModel) applyModelSelection(modelName string, persist bool) error {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return nil
	}
	m.opts.Model = modelName
	m.opts.ContextWindowTokens = 0
	if m.opts.ToolRegistryForModel != nil {
		m.opts.Tools = m.opts.ToolRegistryForModel(m.ctx, modelName)
	}
	if m.opts.SystemPromptForModel != nil {
		m.opts.SystemPrompt = m.opts.SystemPromptForModel(m.ctx, modelName, m.opts.Tools, m.opts.ToolsDisabled)
	}
	if m.opts.MultiModalForModel != nil {
		ctx := m.ctx
		if ctx == nil {
			ctx = context.Background()
		}
		m.opts.MultiModal = m.opts.MultiModalForModel(ctx, modelName)
	}
	m.refreshContextWindowTokens(modelName)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	if persist && m.opts.OnModelSelected != nil {
		ctx := m.ctx
		if ctx == nil {
			ctx = context.Background()
		}
		return m.opts.OnModelSelected(ctx, modelName)
	}
	return nil
}

func (m *chatModel) startModelPreload(modelName string) tea.Cmd {
	modelName = strings.TrimSpace(modelName)
	if m == nil || modelName == "" || m.opts.PreloadModel == nil {
		return nil
	}
	m.preloadingModel = modelName
	m.spinner = 0
	return tea.Batch(preloadModelCmd(m.ctx, m.opts.PreloadModel, modelName, m.opts.Think), m.scheduleTick())
}

func (m chatModel) renderModelPicker(width int) string {
	return m.modelPicker.RenderContent()
}

func (m chatModel) renderInlineModelPicker(width int) []string {
	rendered := m.modelPicker.RenderCompactContent(maxInlineModelPickerItems)
	lines := strings.Split(strings.TrimRight(rendered, "\n"), "\n")
	for i := range lines {
		lines[i] = truncateRenderedLine(lines[i], width)
	}
	return lines
}

func modelOptionMeta(model ModelOption) string {
	return strings.TrimSpace(model.Description)
}
