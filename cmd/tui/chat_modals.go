package tui

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ollama/ollama/agent/chatstore"
)

type chatResumeStore interface {
	ListChats(context.Context, int) ([]chatstore.ChatSummary, error)
	Chat(context.Context, string) (*chatstore.Chat, error)
}

type chatPromptHistoryStore interface {
	ListUserMessages(context.Context, int) ([]string, error)
}

type chatResumePicker struct {
	chats  []chatstore.ChatSummary
	filter string
	cursor int
	scroll int
}

type chatModelPicker struct {
	models []ChatModelOption
	filter string
	cursor int
	scroll int
}

type chatHistoryPopup struct {
	content       string
	scroll        int
	stickToBottom bool
}

func (m *chatModel) openHistoryPopup() (tea.Model, tea.Cmd) {
	m.historyPopup = &chatHistoryPopup{content: m.historySummary(), stickToBottom: true}
	m.status = "history"
	return *m, nil
}

func (m chatModel) updateHistoryPopup(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.historyPopup = nil
		m.status = "ready"
		return m, nil
	case tea.KeyUp:
		m.moveHistoryPopup(-1)
	case tea.KeyDown:
		m.moveHistoryPopup(1)
	case tea.KeyPgUp:
		m.moveHistoryPopup(-m.historyPopupVisibleHeight())
	case tea.KeyPgDown:
		m.moveHistoryPopup(m.historyPopupVisibleHeight())
	case tea.KeyHome, tea.KeyCtrlHome:
		if m.historyPopup != nil {
			m.historyPopup.scroll = 0
			m.historyPopup.stickToBottom = false
		}
	case tea.KeyEnd, tea.KeyCtrlEnd:
		if m.historyPopup != nil {
			m.historyPopup.scroll = m.historyPopupMaxScroll()
			m.historyPopup.stickToBottom = true
		}
	}
	return m, nil
}

func (m *chatModel) moveHistoryPopup(delta int) {
	if m.historyPopup == nil || delta == 0 {
		return
	}
	if m.historyPopup.stickToBottom {
		m.historyPopup.scroll = m.historyPopupMaxScroll()
		m.historyPopup.stickToBottom = false
	}
	m.historyPopup.scroll = clamp(m.historyPopup.scroll+delta, 0, m.historyPopupMaxScroll())
}

func (m chatModel) historyPopupMaxScroll() int {
	return max(0, len(m.historyPopupBodyLines())-m.historyPopupVisibleHeight())
}

func (m chatModel) historyPopupVisibleHeight() int {
	height := m.height
	if height <= 0 {
		height = 24
	}
	return max(1, height-4)
}

func (m chatModel) historyPopupBodyLines() []string {
	width := m.width
	if width <= 0 {
		width = 80
	}
	return m.historyPopupBodyLinesForWidth(width)
}

func (m chatModel) historyPopupBodyLinesForWidth(width int) []string {
	if m.historyPopup == nil {
		return nil
	}
	if width <= 0 {
		width = 80
	}
	content := strings.TrimPrefix(m.historyPopup.content, "**Message History**\n\n")
	return renderHistoryLines(content, width)
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

func normalizeModelOptions(models []ChatModelOption) []ChatModelOption {
	seen := make(map[string]struct{}, len(models))
	out := make([]ChatModelOption, 0, len(models))
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
	return out
}

func newChatModelPicker(models []ChatModelOption, current, filter string) *chatModelPicker {
	picker := &chatModelPicker{models: slices.Clone(models), filter: strings.TrimSpace(filter)}
	if picker.filter != "" {
		return picker
	}
	for i, model := range picker.models {
		if model.Name == current {
			picker.cursor = i
			picker.updateScroll()
			break
		}
	}
	return picker
}

func (m chatModel) updateModelPicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.modelPicker = nil
		m.status = "ready"
		return m, nil
	case tea.KeyEnter:
		return m.selectModel()
	case tea.KeyUp:
		m.modelPicker.move(-1)
	case tea.KeyDown:
		m.modelPicker.move(1)
	case tea.KeyPgUp:
		m.modelPicker.move(-maxResumePickerItems)
	case tea.KeyPgDown:
		m.modelPicker.move(maxResumePickerItems)
	case tea.KeyBackspace:
		if len(m.modelPicker.filter) > 0 {
			runes := []rune(m.modelPicker.filter)
			m.modelPicker.filter = string(runes[:len(runes)-1])
			m.modelPicker.cursor = 0
			m.modelPicker.scroll = 0
		}
	case tea.KeyCtrlU:
		m.modelPicker.filter = ""
		m.modelPicker.cursor = 0
		m.modelPicker.scroll = 0
	case tea.KeySpace:
		m.modelPicker.filter += " "
		m.modelPicker.cursor = 0
		m.modelPicker.scroll = 0
	case tea.KeyRunes:
		m.modelPicker.filter += string(msg.Runes)
		m.modelPicker.cursor = 0
		m.modelPicker.scroll = 0
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

	m.modelPicker = nil
	previous := m.opts.Model
	if err := m.applyModelSelection(selected.Name, true); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", err), err: err.Error()}))
		m.status = "error"
		return m, nil
	}
	if selected.Name == previous {
		m.status = "model unchanged"
	} else {
		m.status = "model " + selected.Name
	}
	return m, nil
}

func (m *chatModel) applyModelSelection(modelName string, persist bool) error {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return nil
	}
	m.opts.Model = modelName
	if m.opts.ToolRegistryForModel != nil {
		m.opts.Tools = m.opts.ToolRegistryForModel(m.ctx, modelName)
	}
	if m.opts.SystemPromptForModel != nil {
		m.opts.SystemPrompt = m.opts.SystemPromptForModel(m.ctx, modelName, m.opts.Tools)
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

func (m *chatModel) openResumePicker() (tea.Model, tea.Cmd) {
	store, ok := m.opts.Store.(chatResumeStore)
	if !ok || store == nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "Chat resume is unavailable because persistence is disabled."}))
		m.status = "error"
		return *m, nil
	}

	chats, err := store.ListChats(m.ctx, 50)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not list saved chats: %v", err)}))
		m.status = "error"
		return *m, nil
	}
	if len(chats) == 0 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "No saved chats to resume."}))
		m.status = "ready"
		return *m, nil
	}

	m.resumePicker = newChatResumePicker(chats, m.chatID)
	m.status = "resume"
	return *m, nil
}

func newChatResumePicker(chats []chatstore.ChatSummary, currentChatID string) *chatResumePicker {
	picker := &chatResumePicker{chats: slices.Clone(chats)}
	for i, chat := range picker.chats {
		if chat.ID == currentChatID {
			picker.cursor = i
			picker.updateScroll()
			break
		}
	}
	return picker
}

func (m chatModel) updateResumePicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.resumePicker = nil
		m.status = "ready"
		return m, nil
	case tea.KeyEnter:
		return m.resumeSelectedChat()
	case tea.KeyUp:
		m.resumePicker.move(-1)
	case tea.KeyDown:
		m.resumePicker.move(1)
	case tea.KeyPgUp:
		m.resumePicker.move(-maxResumePickerItems)
	case tea.KeyPgDown:
		m.resumePicker.move(maxResumePickerItems)
	case tea.KeyBackspace:
		if len(m.resumePicker.filter) > 0 {
			runes := []rune(m.resumePicker.filter)
			m.resumePicker.filter = string(runes[:len(runes)-1])
			m.resumePicker.cursor = 0
			m.resumePicker.scroll = 0
		}
	case tea.KeyCtrlU:
		m.resumePicker.filter = ""
		m.resumePicker.cursor = 0
		m.resumePicker.scroll = 0
	case tea.KeySpace:
		m.resumePicker.filter += " "
		m.resumePicker.cursor = 0
		m.resumePicker.scroll = 0
	case tea.KeyRunes:
		m.resumePicker.filter += string(msg.Runes)
		m.resumePicker.cursor = 0
		m.resumePicker.scroll = 0
	}
	return m, nil
}

func (m *chatModel) resumeSelectedChat() (tea.Model, tea.Cmd) {
	if m.resumePicker == nil {
		return *m, nil
	}
	selected, ok := m.resumePicker.selected()
	if !ok {
		return *m, nil
	}
	store, ok := m.opts.Store.(chatResumeStore)
	if !ok || store == nil {
		m.resumePicker = nil
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "Chat resume is unavailable because persistence is disabled."}))
		m.status = "error"
		return *m, nil
	}

	chat, err := store.Chat(m.ctx, selected.ID)
	if err != nil {
		m.resumePicker = nil
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not resume chat: %v", err)}))
		m.status = "error"
		return *m, nil
	}

	m.resumePicker = nil
	m.chatID = chat.ID
	if chat.Model != "" && chat.Model != m.opts.Model {
		_ = m.applyModelSelection(chat.Model, false)
	}
	m.messages = slices.Clone(chat.Messages)
	m.entries = entriesFromMessages(m.messages)
	m.input = nil
	m.queued = nil
	m.resetWorkingDir()
	m.complete = 0
	m.thinking = false
	m.thinkingTokens = 0
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	m.scroll = 0
	m.status = "resumed"
	return *m, nil
}

func (m chatModel) renderResumePicker(width int) string {
	picker := m.resumePicker
	if picker == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}

	var b strings.Builder
	b.WriteString(chatResumeTitleStyle.Render("Resume session"))
	b.WriteString("\n\n")
	b.WriteString(renderResumeSearchBox(picker.filter, width))
	b.WriteString("\n\n")

	filtered := picker.filtered()
	if len(filtered) == 0 {
		b.WriteString(chatResumeMetaStyle.Render("No matching chats"))
		b.WriteString("\n")
	} else {
		start := clamp(picker.scroll, 0, max(0, len(filtered)-1))
		end := min(len(filtered), start+maxResumePickerItems)
		for i := start; i < end; i++ {
			chat := filtered[i]
			selected := i == picker.cursor
			if selected {
				b.WriteString(chatResumeSelectedStyle.Render("› " + resumeChatTitle(chat)))
			} else {
				b.WriteString("  ")
				b.WriteString(chatResumeTextStyle.Render(resumeChatTitle(chat)))
			}
			b.WriteByte('\n')
			b.WriteString(chatResumeMetaStyle.Render("  " + resumeChatMeta(chat)))
			b.WriteByte('\n')
			if i < end-1 {
				b.WriteByte('\n')
			}
		}
		if end < len(filtered) {
			b.WriteString(chatResumeMetaStyle.Render(fmt.Sprintf("\n  +%d more", len(filtered)-end)))
			b.WriteByte('\n')
		}
	}

	b.WriteString("\n")
	b.WriteString(chatResumeMetaStyle.Render("↑/↓ move • enter resume • type search • esc cancel"))
	return b.String()
}

func (m chatModel) renderModelPicker(width int) string {
	picker := m.modelPicker
	if picker == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}

	var b strings.Builder
	b.WriteString(chatResumeTitleStyle.Render("Switch model"))
	b.WriteString("\n\n")
	b.WriteString(renderResumeSearchBox(picker.filter, width))
	b.WriteString("\n\n")

	filtered := picker.filtered()
	if len(filtered) == 0 {
		b.WriteString(chatResumeMetaStyle.Render("No matching models"))
		b.WriteString("\n")
	} else {
		start := clamp(picker.scroll, 0, max(0, len(filtered)-1))
		end := min(len(filtered), start+maxResumePickerItems)
		for i := start; i < end; i++ {
			model := filtered[i]
			selected := i == picker.cursor
			if selected {
				b.WriteString(chatResumeSelectedStyle.Render("› " + model.Name))
			} else {
				b.WriteString("  ")
				b.WriteString(chatResumeTextStyle.Render(model.Name))
			}
			b.WriteByte('\n')
			if meta := modelOptionMeta(model, m.opts.Model); meta != "" {
				b.WriteString(chatResumeMetaStyle.Render("  " + meta))
				b.WriteByte('\n')
			}
			if i < end-1 {
				b.WriteByte('\n')
			}
		}
		if end < len(filtered) {
			b.WriteString(chatResumeMetaStyle.Render(fmt.Sprintf("\n  +%d more", len(filtered)-end)))
			b.WriteByte('\n')
		}
	}

	b.WriteString("\n")
	b.WriteString(chatResumeMetaStyle.Render("↑/↓ move • enter switch • type search • esc cancel"))
	return b.String()
}

func (m chatModel) renderHistoryPopup(width, height int) string {
	popup := m.historyPopup
	if popup == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}
	if height <= 0 {
		height = 24
	}

	bodyLines := m.historyPopupBodyLinesForWidth(width)
	visibleHeight := max(1, height-4)
	maxScroll := max(0, len(bodyLines)-visibleHeight)
	scroll := clamp(popup.scroll, 0, maxScroll)
	if popup.stickToBottom {
		scroll = maxScroll
	}
	end := min(len(bodyLines), scroll+visibleHeight)

	var b strings.Builder
	b.WriteString(chatResumeTitleStyle.Render("Message history"))
	b.WriteString("\n\n")
	if len(bodyLines) == 0 {
		b.WriteString(chatResumeMetaStyle.Render("No messages yet."))
		b.WriteByte('\n')
	} else {
		for _, line := range bodyLines[scroll:end] {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	b.WriteString("\n")
	help := "↑/↓ scroll • pgup/pgdn page • home/end jump • esc close"
	b.WriteString(chatResumeMetaStyle.Render(truncateRenderedLine(help, width)))
	return b.String()
}

func renderResumeSearchBox(filter string, width int) string {
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
		chatResumeBorderStyle.Render(strings.Repeat("─", width)),
		chatResumeTextStyle.Render(truncateRunes(line, max(20, width))),
		chatResumeBorderStyle.Render(strings.Repeat("─", width)),
	}
	return strings.Join(lines, "\n")
}

func (p *chatResumePicker) filtered() []chatstore.ChatSummary {
	if p == nil {
		return nil
	}
	filter := strings.ToLower(strings.TrimSpace(p.filter))
	if filter == "" {
		return p.chats
	}
	var out []chatstore.ChatSummary
	for _, chat := range p.chats {
		haystack := strings.ToLower(strings.Join([]string{
			chat.Title,
			chat.Model,
			chat.ID,
		}, " "))
		if strings.Contains(haystack, filter) {
			out = append(out, chat)
		}
	}
	return out
}

func (p *chatResumePicker) move(delta int) {
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

func (p *chatResumePicker) updateScroll() {
	if p == nil {
		return
	}
	if p.cursor < p.scroll {
		p.scroll = p.cursor
	}
	if p.cursor >= p.scroll+maxResumePickerItems {
		p.scroll = p.cursor - maxResumePickerItems + 1
	}
	if p.scroll < 0 {
		p.scroll = 0
	}
}

func (p *chatResumePicker) selected() (chatstore.ChatSummary, bool) {
	if p == nil {
		return chatstore.ChatSummary{}, false
	}
	filtered := p.filtered()
	if len(filtered) == 0 || p.cursor < 0 || p.cursor >= len(filtered) {
		return chatstore.ChatSummary{}, false
	}
	return filtered[p.cursor], true
}

func (p *chatModelPicker) filtered() []ChatModelOption {
	if p == nil {
		return nil
	}
	filter := strings.ToLower(strings.TrimSpace(p.filter))
	if filter == "" {
		return p.models
	}
	var out []ChatModelOption
	for _, model := range p.models {
		haystack := strings.ToLower(strings.Join([]string{
			model.Name,
			model.Description,
		}, " "))
		if strings.Contains(haystack, filter) {
			out = append(out, model)
		}
	}
	return out
}

func (p *chatModelPicker) move(delta int) {
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

func (p *chatModelPicker) updateScroll() {
	if p == nil {
		return
	}
	if p.cursor < p.scroll {
		p.scroll = p.cursor
	}
	if p.cursor >= p.scroll+maxResumePickerItems {
		p.scroll = p.cursor - maxResumePickerItems + 1
	}
	if p.scroll < 0 {
		p.scroll = 0
	}
}

func (p *chatModelPicker) selected() (ChatModelOption, bool) {
	if p == nil {
		return ChatModelOption{}, false
	}
	filtered := p.filtered()
	if len(filtered) == 0 || p.cursor < 0 || p.cursor >= len(filtered) {
		return ChatModelOption{}, false
	}
	return filtered[p.cursor], true
}

func modelOptionMeta(model ChatModelOption, current string) string {
	var parts []string
	if model.Name == current {
		parts = append(parts, "current")
	}
	if model.Description != "" {
		parts = append(parts, model.Description)
	}
	return strings.Join(parts, " · ")
}

func resumeChatTitle(chat chatstore.ChatSummary) string {
	title := strings.TrimSpace(chat.Title)
	if title == "" {
		title = shortChatID(chat.ID)
	}
	return truncateRunes(title, 96)
}

func resumeChatMeta(chat chatstore.ChatSummary) string {
	var parts []string
	if !chat.UpdatedAt.IsZero() {
		parts = append(parts, relativeTime(chat.UpdatedAt))
	}
	if chat.Model != "" {
		parts = append(parts, chat.Model)
	}
	if chat.MessageCount > 0 {
		parts = append(parts, fmt.Sprintf("%d messages", chat.MessageCount))
	}
	if chat.ApproxBytes > 0 {
		parts = append(parts, formatByteSize(chat.ApproxBytes))
	}
	return strings.Join(parts, " · ")
}

func relativeTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	elapsed := time.Since(t)
	if elapsed < 0 {
		elapsed = 0
	}
	switch {
	case elapsed < time.Minute:
		return "just now"
	case elapsed < time.Hour:
		return fmt.Sprintf("%d min ago", int(elapsed/time.Minute))
	case elapsed < 24*time.Hour:
		hours := int(elapsed / time.Hour)
		if hours == 1 {
			return "1 hour ago"
		}
		return fmt.Sprintf("%d hours ago", hours)
	case elapsed < 14*24*time.Hour:
		days := int(elapsed / (24 * time.Hour))
		if days == 1 {
			return "1 day ago"
		}
		return fmt.Sprintf("%d days ago", days)
	default:
		return t.Format("Jan 2")
	}
}

func formatByteSize(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%dB", bytes)
	}
	kb := float64(bytes) / 1024
	if kb < 1024 {
		return fmt.Sprintf("%.1fKB", kb)
	}
	return fmt.Sprintf("%.1fMB", kb/1024)
}

func shortChatID(id string) string {
	if len(id) <= 8 {
		return id
	}
	return id[:8]
}
