package chat

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ollama/ollama/api"
	appstore "github.com/ollama/ollama/app/store"
)

type chatResumeStore interface {
	ListChats(context.Context, int) ([]appstore.ChatSummary, error)
	AgentChat(context.Context, string) (*appstore.AgentChat, error)
}

type chatPromptHistoryStore interface {
	ListUserMessages(context.Context, int) ([]string, error)
}

type chatCurrentModelStore interface {
	SetChatModel(context.Context, string, string) error
}

type (
	chatResumePicker = chatPicker[appstore.ChatSummary]
	chatModelPicker  = chatPicker[ModelOption]
)

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
}

type chatHistoryPopup struct {
	title         string
	empty         string
	header        []string
	raw           string
	messages      []api.Message
	scroll        int
	stickToBottom bool
	selection     chatSelection
}

func (m *chatModel) openHistoryPopup() (tea.Model, tea.Cmd) {
	preview := m.requestPreview(m.messages)
	m.historyPopup = &chatHistoryPopup{
		title:         "Message history",
		empty:         "No messages yet.",
		header:        m.promptTokenHeader(preview.PromptTokens),
		messages:      m.historyMessages(),
		stickToBottom: true,
	}
	m.status = "history"
	return *m, tea.EnterAltScreen
}

func (m chatModel) updateHistoryPopup(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.historyPopup = nil
		m.status = "ready"
		if m.fullScreen {
			return m, nil
		}
		return m, tea.ExitAltScreen
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

func (m chatModel) updateHistoryPopupMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.MouseWheelUp:
		m.moveHistoryPopup(-3)
	case tea.MouseWheelDown:
		m.moveHistoryPopup(3)
	case tea.MouseLeft:
		switch msg.Action {
		case tea.MouseActionPress:
			m.startHistoryPopupSelection(msg)
		case tea.MouseActionMotion:
			m.dragHistoryPopupSelection(msg)
		default:
			if msg.Action == 0 {
				m.startHistoryPopupSelection(msg)
			}
		}
	case tea.MouseMotion:
		m.dragHistoryPopupSelection(msg)
	case tea.MouseRelease:
		return m.finishHistoryPopupSelection(msg)
	}
	return m, nil
}

func (m chatModel) historyPopupLayout() (top, height int) {
	return 2 + m.historyPopupHeaderHeight(), m.historyPopupVisibleHeight()
}

func (m chatModel) mouseInHistoryPopupBody(msg tea.MouseMsg) bool {
	top, height := m.historyPopupLayout()
	if msg.X < 0 || msg.X >= m.viewWidth() {
		return false
	}
	return msg.Y >= top && msg.Y < top+height
}

func (m chatModel) mouseHistoryPopupPoint(msg tea.MouseMsg) chatSelectionPoint {
	top, height := m.historyPopupLayout()
	visibleY := clamp(msg.Y-top, 0, max(0, height-1))
	line := m.historyPopupVisibleStartLine() + visibleY
	col := max(0, msg.X)
	return chatSelectionPoint{line: line, col: col}
}

func (m *chatModel) startHistoryPopupSelection(msg tea.MouseMsg) {
	if m.historyPopup == nil {
		return
	}
	startChatSelection(&m.historyPopup.selection, msg, m.mouseInHistoryPopupBody, m.mouseHistoryPopupPoint)
}

func (m *chatModel) dragHistoryPopupSelection(msg tea.MouseMsg) {
	if m.historyPopup == nil {
		return
	}
	dragChatSelection(&m.historyPopup.selection, msg, m.mouseHistoryPopupPoint, func(msg tea.MouseMsg) {
		top, height := m.historyPopupLayout()
		if msg.Y <= top {
			m.moveHistoryPopup(-1)
		} else if msg.Y >= top+height-1 {
			m.moveHistoryPopup(1)
		}
	})
}

func (m chatModel) finishHistoryPopupSelection(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.historyPopup == nil {
		return m, nil
	}
	return finishChatSelection(m, &m.historyPopup.selection, msg, m.mouseHistoryPopupPoint, m.selectedHistoryPopupText)
}

func (m chatModel) historyPopupMaxScroll() int {
	return max(0, len(m.historyPopupBodyLines())-m.historyPopupVisibleHeight())
}

func (m chatModel) historyPopupVisibleStartLine() int {
	if m.historyPopup == nil {
		return 0
	}
	bodyLines := m.historyPopupBodyLines()
	visibleHeight := m.historyPopupVisibleHeight()
	maxScroll := max(0, len(bodyLines)-visibleHeight)
	scroll := clamp(m.historyPopup.scroll, 0, maxScroll)
	if m.historyPopup.stickToBottom {
		return maxScroll
	}
	return scroll
}

func (m chatModel) historyPopupVisibleHeight() int {
	height := m.height
	if height <= 0 {
		height = 24
	}
	return max(1, height-4-m.historyPopupHeaderHeight())
}

func (m chatModel) historyPopupHeaderHeight() int {
	if m.historyPopup == nil || len(m.historyPopup.header) == 0 {
		return 0
	}
	return len(m.historyPopup.header) + 1
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
	popup := m.historyPopup
	var body []string
	if popup.raw != "" {
		body = renderRawRequestLines(popup.raw, width)
	} else {
		body = renderHistoryMessages(popup.messages, width)
	}
	if len(body) == 0 && popup.empty != "" {
		body = []string{chatResumeMetaStyle.Render(popup.empty)}
	}
	return body
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
	if persist && m.chatID != "" {
		if store, ok := m.opts.Store.(chatCurrentModelStore); ok && store != nil {
			ctx := m.ctx
			if ctx == nil {
				ctx = context.Background()
			}
			if err := store.SetChatModel(ctx, m.chatID, modelName); err != nil {
				return err
			}
		}
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

func newChatResumePicker(chats []appstore.ChatSummary, currentChatID string) *chatResumePicker {
	picker := newChatPicker(chats, "", func(chat appstore.ChatSummary) bool {
		return chat.ID == currentChatID
	}, func(chat appstore.ChatSummary, filter string) bool {
		return strings.Contains(strings.ToLower(strings.Join([]string{
			chat.Title,
			chat.Model,
			chat.ID,
		}, " ")), filter)
	})
	picker.title = "Resume session"
	picker.inlineTitle = "Resume chat"
	picker.emptyLabel = "No matching chats"
	picker.fullFooter = "↑/↓ move • enter resume • type search • esc cancel"
	picker.itemTitle = resumeChatTitle
	picker.itemMeta = resumeChatMeta
	return picker
}

func (m chatModel) updateResumePicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.resumePicker == nil {
		return m, nil
	}
	clear, enter := m.resumePicker.handleKey(msg)
	if clear {
		m.resumePicker = nil
		m.status = "ready"
		return m, nil
	}
	if enter {
		return m.resumeSelectedChat()
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

	chat, err := store.AgentChat(m.ctx, selected.ID)
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
	m.inputAttachments = nil
	m.inputPastedTexts = nil
	m.queued = nil
	m.queuedAttachments = nil
	m.queuedPastedTexts = nil
	m.nextImageID, m.nextAudioID = nextInputAttachmentIDsFromMessages(m.messages)
	m.nextPastedTextID = nextInputPastedTextIDFromMessages(m.messages)
	m.resetWorkingDir()
	m.complete = 0
	m.thinking = false
	m.thinkingTokens = 0
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	m.scroll = 0
	m.boundedFrame = true
	m.flowPrintedLines = 0
	m.status = "resumed"
	return *m, tea.ClearScreen
}

func (m chatModel) renderResumePicker(width int) string {
	return m.resumePicker.render(width)
}

func (m chatModel) renderModelPicker(width int) string {
	return m.modelPicker.render(width)
}

func (m chatModel) shouldRenderPickerFullFrame(width, height int) bool {
	return width < 48 || height < 12
}

func (m chatModel) renderInlineResumePicker(width int) []string {
	return m.resumePicker.renderInline(width)
}

func (m chatModel) renderInlineModelPicker(width int) []string {
	return m.modelPicker.renderInline(width)
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
	visibleHeight := m.historyPopupVisibleHeight()
	maxScroll := max(0, len(bodyLines)-visibleHeight)
	scroll := clamp(popup.scroll, 0, maxScroll)
	if popup.stickToBottom {
		scroll = maxScroll
	}
	end := min(len(bodyLines), scroll+visibleHeight)
	visibleLines := m.applyHistoryPopupSelection(bodyLines[scroll:end], scroll)

	var b strings.Builder
	title := popup.title
	if title == "" {
		title = "Message history"
	}
	b.WriteString(chatResumeTitleStyle.Render(title))
	b.WriteString("\n\n")
	for _, line := range popup.header {
		b.WriteString(line)
		b.WriteByte('\n')
	}
	if len(popup.header) > 0 {
		b.WriteByte('\n')
	}
	if len(bodyLines) == 0 {
		empty := popup.empty
		if empty == "" {
			empty = "No messages yet."
		}
		b.WriteString(chatResumeMetaStyle.Render(empty))
		b.WriteByte('\n')
	} else {
		for _, line := range visibleLines {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	b.WriteString("\n")
	help := "↑/↓ scroll • pgup/pgdn page • home/end jump • esc close"
	b.WriteString(chatResumeMetaStyle.Render(truncateRenderedLine(help, width)))
	return b.String()
}

func (m chatModel) applyHistoryPopupSelection(lines []string, offset int) []string {
	if m.historyPopup == nil || len(lines) == 0 {
		return lines
	}
	start, end, ok := normalizedSelectionRangeFor(m.historyPopup.selection)
	if !ok {
		return lines
	}
	out := slices.Clone(lines)
	for i, line := range out {
		lineIndex := offset + i
		if lineIndex < start.line || lineIndex > end.line {
			continue
		}
		text := stripChatANSI(line)
		startCol, endCol := 0, len([]rune(text))
		if lineIndex == start.line {
			startCol = clamp(start.col, 0, len([]rune(text)))
		}
		if lineIndex == end.line {
			endCol = clamp(end.col, 0, len([]rune(text)))
		}
		if startCol > endCol {
			startCol, endCol = endCol, startCol
		}
		out[i] = renderSelectedTranscriptLine(text, startCol, endCol)
	}
	return out
}

func (m chatModel) selectedHistoryPopupText() string {
	if m.historyPopup == nil {
		return ""
	}
	start, end, ok := normalizedSelectionRangeFor(m.historyPopup.selection)
	if !ok {
		return ""
	}
	lines := m.historyPopupBodyLines()
	if len(lines) == 0 {
		return ""
	}
	start.line = clamp(start.line, 0, len(lines)-1)
	end.line = clamp(end.line, 0, len(lines)-1)
	var selected []string
	for lineIndex := start.line; lineIndex <= end.line; lineIndex++ {
		text := stripChatANSI(lines[lineIndex])
		runes := []rune(text)
		startCol, endCol := 0, len(runes)
		if lineIndex == start.line {
			startCol = clamp(start.col, 0, len(runes))
		}
		if lineIndex == end.line {
			endCol = clamp(end.col, 0, len(runes))
		}
		if startCol > endCol {
			startCol, endCol = endCol, startCol
		}
		selected = append(selected, string(runes[startCol:endCol]))
	}
	return strings.TrimRight(strings.Join(selected, "\n"), "\n")
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
	if p.cursor >= p.scroll+maxResumePickerItems {
		p.scroll = p.cursor - maxResumePickerItems + 1
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
		p.move(-maxResumePickerItems)
	case tea.KeyPgDown:
		p.move(maxResumePickerItems)
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
	b.WriteString(truncateRenderedLine(chatResumeTitleStyle.Render(p.title), width))
	b.WriteString("\n\n")
	b.WriteString(renderResumeSearchBox(p.filter, width))
	b.WriteString("\n\n")

	filtered := p.filtered()
	if len(filtered) == 0 {
		b.WriteString(chatResumeMetaStyle.Render(p.emptyLabel))
		b.WriteString("\n")
	} else {
		start := clamp(p.scroll, 0, max(0, len(filtered)-1))
		end := min(len(filtered), start+maxResumePickerItems)
		for i := start; i < end; i++ {
			item := filtered[i]
			selected := i == p.cursor
			for j, line := range wrapChatText(p.itemTitle(item), max(10, width-2)) {
				marker := "  "
				if selected && j == 0 {
					marker = "› "
				}
				if selected {
					b.WriteString(chatResumeSelectedStyle.Render(marker + line))
				} else {
					b.WriteString("  " + chatResumeTextStyle.Render(line))
				}
				b.WriteByte('\n')
			}
			if meta := p.itemMeta(item); meta != "" {
				for _, line := range wrapChatText(meta, max(10, width-2)) {
					b.WriteString(chatResumeMetaStyle.Render("  " + line))
					b.WriteByte('\n')
				}
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
	b.WriteString(truncateRenderedLine(chatResumeMetaStyle.Render(p.fullFooter), width))
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

	lines := []string{truncateRenderedLine(chatResumeTitleStyle.Render(title), width)}
	filtered := p.filtered()
	if len(filtered) == 0 {
		lines = append(lines, truncateRenderedLine(chatResumeMetaStyle.Render("  "+p.emptyLabel), width))
	} else {
		start, end := completionWindow(len(filtered), p.cursor, maxInlineModelPickerItems)
		for i := start; i < end; i++ {
			item := filtered[i]
			selected := i == p.cursor
			for j, line := range wrapChatText(p.itemTitle(item), max(10, width-2)) {
				marker := "  "
				if selected && j == 0 {
					marker = "› "
				}
				if selected {
					lines = append(lines, truncateRenderedLine(chatResumeSelectedStyle.Render(marker+line), width))
				} else {
					lines = append(lines, truncateRenderedLine("  "+chatResumeTextStyle.Render(line), width))
				}
			}
			if meta := p.itemMeta(item); meta != "" {
				for _, line := range wrapChatText(meta, max(10, width-2)) {
					lines = append(lines, truncateRenderedLine(chatResumeMetaStyle.Render("  "+line), width))
				}
			}
		}
		if end < len(filtered) {
			lines = append(lines, truncateRenderedLine(chatResumeMetaStyle.Render(fmt.Sprintf("  +%d more", len(filtered)-end)), width))
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

func resumeChatTitle(chat appstore.ChatSummary) string {
	title := strings.TrimSpace(chat.Title)
	if title == "" {
		title = shortChatID(chat.ID)
	}
	return truncateRunes(title, 96)
}

func resumeChatMeta(chat appstore.ChatSummary) string {
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
