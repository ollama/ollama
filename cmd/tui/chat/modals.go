package chat

import (
	"context"
	"fmt"
	"slices"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

type chatModelPicker = chatPicker[ModelOption]

type chatPicker[T any] struct {
	items  []T
	filter string
	cursor int
	scroll int
	match  func(T, string) bool
	less   func(T, T, string) int

	// Display configuration shared by the full-frame and inline renderers.
	title       string
	inlineTitle string
	emptyLabel  string
	fullFooter  string
	itemTitle   func(T) string
	itemMeta    func(T) string
	itemBadge   func(T) string
}

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

	m.modelPicker = newChatModelPicker(models, m.opts.Model, filter)
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

func newChatModelPicker(models []ModelOption, current, filter string) *chatModelPicker {
	picker := newChatPicker(models, filter, func(model ModelOption) bool {
		return model.Name == current
	}, func(model ModelOption, filter string) bool {
		return modelOptionMatchScore(model, filter).ok
	})
	picker.less = compareModelOptionsForFilter
	picker.title = "Switch model"
	picker.inlineTitle = "Select model"
	picker.emptyLabel = "No matching models"
	picker.fullFooter = "↑/↓ move • enter switch • type search • esc cancel"
	picker.itemTitle = func(model ModelOption) string { return model.Name }
	picker.itemMeta = func(model ModelOption) string { return modelOptionMeta(model, current) }
	picker.itemBadge = func(model ModelOption) string { return model.AvailabilityBadge }
	return picker
}

type modelOptionScore struct {
	ok          bool
	rank        int
	index       int
	lengthDelta int
	recommended int
	name        string
}

func compareModelOptionsForFilter(a, b ModelOption, filter string) int {
	aScore := modelOptionMatchScore(a, filter)
	bScore := modelOptionMatchScore(b, filter)
	for _, cmp := range []int{
		compareInt(aScore.rank, bScore.rank),
		compareInt(aScore.index, bScore.index),
		compareInt(aScore.lengthDelta, bScore.lengthDelta),
		compareInt(aScore.recommended, bScore.recommended),
		strings.Compare(aScore.name, bScore.name),
	} {
		if cmp != 0 {
			return cmp
		}
	}
	return 0
}

func modelOptionMatchScore(model ModelOption, filter string) modelOptionScore {
	filter = strings.ToLower(strings.TrimSpace(filter))
	name := strings.ToLower(strings.TrimSpace(model.Name))
	description := strings.ToLower(strings.TrimSpace(model.Description))
	score := modelOptionScore{
		rank:        4,
		index:       1 << 20,
		lengthDelta: 1 << 20,
		name:        name,
	}
	if model.Recommended {
		score.recommended = -1
	}
	if filter == "" {
		score.ok = true
		return score
	}
	nameRunes := len([]rune(name))
	filterRunes := len([]rune(filter))
	if name == filter {
		score.ok = true
		score.rank = 0
		score.index = 0
		score.lengthDelta = 0
		return score
	}
	if strings.HasPrefix(name, filter) {
		score.ok = true
		score.rank = 1
		score.index = 0
		score.lengthDelta = max(0, nameRunes-filterRunes)
		return score
	}
	if index := strings.Index(name, filter); index >= 0 {
		score.ok = true
		score.rank = 2
		score.index = len([]rune(name[:index]))
		score.lengthDelta = max(0, nameRunes-filterRunes)
		return score
	}
	if index := strings.Index(description, filter); index >= 0 {
		score.ok = true
		score.rank = 3
		score.index = len([]rune(description[:index]))
		score.lengthDelta = max(0, nameRunes-filterRunes)
	}
	return score
}

func compareInt(a, b int) int {
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
	clear, enter := m.modelPicker.handleKey(msg)
	if clear {
		m.modelPicker = nil
		m.status = "ready"
		return m, nil
	}
	if enter {
		return m.selectModel()
	}
	return m, nil
}

func (m chatModel) selectModel() (tea.Model, tea.Cmd) {
	if m.modelPicker == nil {
		return m, nil
	}
	selected, ok := m.modelPicker.selected()
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
	if err := m.applyModelSelection(selected.Name, true); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", err), err: err.Error()}))
		m.status = "error"
		return m, nil
	}
	m.status = "ready"
	return m, m.startModelPreload(selected.Name)
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
		m.opts.SystemPrompt = m.opts.SystemPromptForModel(m.ctx, modelName, m.opts.Tools)
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
	return m.modelPicker.render(width)
}

func (m chatModel) shouldRenderPickerFullFrame(width, height int) bool {
	return width < 48 || height < 12
}

func (m chatModel) renderInlineModelPicker(width int) []string {
	return m.modelPicker.renderInline(width)
}

func renderPickerSearchBox(filter string, width int) string {
	if width < 20 {
		width = 20
	}
	placeholder := "Search..."
	value := filter
	if value == "" {
		value = placeholder
	}
	line := "⌕ " + value
	lines := []string{
		chatPickerBorderStyle.Render(strings.Repeat("─", width)),
		chatPickerTextStyle.Render(truncateRunes(line, max(20, width))),
		chatPickerBorderStyle.Render(strings.Repeat("─", width)),
	}
	return strings.Join(lines, "\n")
}

func newChatPicker[T any](items []T, filter string, selected func(T) bool, match func(T, string) bool) *chatPicker[T] {
	picker := &chatPicker[T]{
		items:  slices.Clone(items),
		filter: strings.TrimSpace(filter),
		match:  match,
	}
	if picker.filter != "" || selected == nil {
		return picker
	}
	for i, item := range picker.items {
		if selected(item) {
			picker.cursor = i
			picker.updateScroll()
			break
		}
	}
	return picker
}

func (p *chatPicker[T]) filtered() []T {
	if p == nil {
		return nil
	}
	filter := strings.ToLower(strings.TrimSpace(p.filter))
	if filter == "" {
		return p.items
	}
	var out []T
	for _, item := range p.items {
		if p.match == nil || p.match(item, filter) {
			out = append(out, item)
		}
	}
	if p.less != nil {
		slices.SortStableFunc(out, func(a, b T) int {
			return p.less(a, b, filter)
		})
	}
	return out
}

func (p *chatPicker[T]) move(delta int) {
	if p == nil {
		return
	}
	filtered := p.filtered()
	if len(filtered) == 0 {
		p.cursor = 0
		p.scroll = 0
		return
	}
	p.cursor = clamp(p.cursor+delta, 0, len(filtered)-1)
	p.updateScroll()
}

func (p *chatPicker[T]) updateScroll() {
	if p == nil {
		return
	}
	if p.cursor < p.scroll {
		p.scroll = p.cursor
	}
	if p.cursor >= p.scroll+maxPickerItems {
		p.scroll = p.cursor - maxPickerItems + 1
	}
	if p.scroll < 0 {
		p.scroll = 0
	}
}

func (p *chatPicker[T]) selected() (T, bool) {
	var zero T
	if p == nil {
		return zero, false
	}
	filtered := p.filtered()
	if len(filtered) == 0 || p.cursor < 0 || p.cursor >= len(filtered) {
		return zero, false
	}
	return filtered[p.cursor], true
}

// handleKey processes a key event for any chatPicker. It returns clear when the
// picker should be closed and enter when the selected item should be activated.
func (p *chatPicker[T]) handleKey(msg tea.KeyMsg) (clear, enter bool) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		return true, false
	case tea.KeyEnter:
		return false, true
	case tea.KeyUp:
		p.move(-1)
	case tea.KeyDown:
		p.move(1)
	case tea.KeyPgUp:
		p.move(-maxPickerItems)
	case tea.KeyPgDown:
		p.move(maxPickerItems)
	case tea.KeyBackspace:
		if len(p.filter) > 0 {
			runes := []rune(p.filter)
			p.filter = string(runes[:len(runes)-1])
			p.cursor = 0
			p.scroll = 0
		}
	case tea.KeyCtrlU:
		p.filter = ""
		p.cursor = 0
		p.scroll = 0
	case tea.KeySpace:
		p.filter += " "
		p.cursor = 0
		p.scroll = 0
	case tea.KeyRunes:
		p.filter += string(msg.Runes)
		p.cursor = 0
		p.scroll = 0
	}
	return false, false
}

// render produces the full-frame picker view with a search box.
func (p *chatPicker[T]) render(width int) string {
	if p == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}

	var b strings.Builder
	b.WriteString(truncateRenderedLine(chatPickerTitleStyle.Render(p.title), width))
	b.WriteString("\n\n")
	b.WriteString(renderPickerSearchBox(p.filter, width))
	b.WriteString("\n\n")

	filtered := p.filtered()
	if len(filtered) == 0 {
		b.WriteString(chatPickerMetaStyle.Render(p.emptyLabel))
		b.WriteString("\n")
	} else {
		start := clamp(p.scroll, 0, max(0, len(filtered)-1))
		end := min(len(filtered), start+maxPickerItems)
		for i := start; i < end; i++ {
			item := filtered[i]
			selected := i == p.cursor
			badge := p.itemBadge(item)
			for j, line := range wrapChatText(p.itemTitle(item), max(10, width-2)) {
				marker := "  "
				if selected && j == 0 {
					marker = "› "
				}
				if j == 0 && badge != "" {
					line += " " + chatPickerMetaStyle.Render("("+badge+")")
				}
				if selected {
					b.WriteString(chatPickerSelectedStyle.Render(marker + line))
				} else {
					b.WriteString("  " + chatPickerTextStyle.Render(line))
				}
				b.WriteByte('\n')
			}
			if meta := p.itemMeta(item); meta != "" {
				for _, line := range wrapChatText(meta, max(10, width-2)) {
					b.WriteString(chatPickerMetaStyle.Render("  " + line))
					b.WriteByte('\n')
				}
			}
			if i < end-1 {
				b.WriteByte('\n')
			}
		}
		if end < len(filtered) {
			b.WriteString(chatPickerMetaStyle.Render(fmt.Sprintf("\n  +%d more", len(filtered)-end)))
			b.WriteByte('\n')
		}
	}

	b.WriteString("\n")
	b.WriteString(truncateRenderedLine(chatPickerMetaStyle.Render(p.fullFooter), width))
	return b.String()
}

// renderInline produces the compact inline picker view shown above the input.
func (p *chatPicker[T]) renderInline(width int) []string {
	if p == nil {
		return nil
	}
	if width <= 0 {
		width = 80
	}

	filter := strings.TrimSpace(p.filter)
	title := p.inlineTitle
	if filter != "" {
		title += ": " + filter
	} else {
		title += ": type to filter"
	}

	lines := []string{truncateRenderedLine(chatPickerTitleStyle.Render(title), width)}
	filtered := p.filtered()
	if len(filtered) == 0 {
		lines = append(lines, truncateRenderedLine(chatPickerMetaStyle.Render("  "+p.emptyLabel), width))
	} else {
		start, end := completionWindow(len(filtered), p.cursor, maxInlineModelPickerItems)
		for i := start; i < end; i++ {
			item := filtered[i]
			selected := i == p.cursor
			badge := p.itemBadge(item)
			for j, line := range wrapChatText(p.itemTitle(item), max(10, width-2)) {
				marker := "  "
				if selected && j == 0 {
					marker = "› "
				}
				if j == 0 && badge != "" {
					line += " " + chatPickerMetaStyle.Render("("+badge+")")
				}
				if selected {
					lines = append(lines, truncateRenderedLine(chatPickerSelectedStyle.Render(marker+line), width))
				} else {
					lines = append(lines, truncateRenderedLine("  "+chatPickerTextStyle.Render(line), width))
				}
			}
			if meta := p.itemMeta(item); meta != "" {
				for _, line := range wrapChatText(meta, max(10, width-2)) {
					lines = append(lines, truncateRenderedLine(chatPickerMetaStyle.Render("  "+line), width))
				}
			}
		}
		if end < len(filtered) {
			lines = append(lines, truncateRenderedLine(chatPickerMetaStyle.Render(fmt.Sprintf("  +%d more", len(filtered)-end)), width))
		}
	}
	return lines
}

func modelOptionMeta(model ModelOption, current string) string {
	var parts []string
	if model.Name == current {
		parts = append(parts, "current")
	}
	if model.Description != "" {
		parts = append(parts, model.Description)
	}
	return strings.Join(parts, " · ")
}
