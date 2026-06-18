package tui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ollama/ollama/agent/skills"
	agenttools "github.com/ollama/ollama/agent/tools"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/filedata"
)

type chatSlashCommand struct {
	name        string
	description string
}

type chatCompletion struct {
	value       string
	label       string
	description string
	directory   bool
}

const maxInputBoxBodyLines = 6
const (
	pastedTextPlaceholderMinRunes = 1000
	pastedTextPlaceholderMinLines = 8
)

var chatSlashCommands = []chatSlashCommand{
	{name: "/copy", description: "copy latest model output"},
	{name: "/copy-all", description: "copy all model output"},
	{name: "/clear", description: "clear this chat"},
	{name: "/tools", description: "show available tools"},
	{name: "/model", description: "switch models"},
	{name: "/launch", description: "open launch flow"},
	{name: "/history", description: "show prompt message history"},
	{name: "/skills", description: "show or import installed skills"},
	{name: "/new", description: "start a new chat"},
	{name: "/resume", description: "resume a saved chat"},
	{name: "/system", description: "toggle or set system prompt"},
	{name: "/think", description: "set thinking mode"},
	{name: "/verbose", description: "toggle model metrics"},
	{name: "/compact", description: "summarize older context"},
	{name: "/help", description: "show commands"},
	{name: "/bye", description: "exit"},
}

func (m *chatModel) handleSubmit() (tea.Model, tea.Cmd) {
	m.syncInputPlaceholders()
	input := strings.TrimSpace(string(m.input))
	if selected, ok := m.selectedSlashCommand(); ok {
		input = selected
	}
	attachments := cloneInputAttachments(m.inputAttachments)
	pastedTexts := cloneInputPastedTexts(m.inputPastedTexts)
	m.input = nil
	m.inputCursor = 0
	m.inputCursorSet = false
	m.inputAttachments = nil
	m.inputPastedTexts = nil
	m.complete = 0
	m.resetPromptHistoryCursor()
	if input == "" {
		return *m, nil
	}

	if m.running || m.compacting {
		m.queued = append(m.queued, input)
		m.queuedAttachments = append(m.queuedAttachments, attachments)
		m.queuedPastedTexts = append(m.queuedPastedTexts, pastedTexts)
		m.status = "queued"
		return *m, nil
	}

	m.inputAttachments = attachments
	m.inputPastedTexts = pastedTexts
	return m.submitInput(input)
}

func (m chatModel) selectedSlashCommand() (string, bool) {
	input := strings.TrimSpace(string(m.input))
	if !strings.HasPrefix(input, "/") {
		return "", false
	}
	completions := m.slashCompletions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return "", false
	}
	return completions[clamp(m.complete, 0, len(completions)-1)].value, true
}

func (m *chatModel) submitInput(input string) (tea.Model, tea.Cmd) {
	switch {
	case input == "/bye" || input == "/exit":
		m.quitting = true
		return *m, tea.Quit
	case input == "/?" || input == "/help":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.helpSummary()}))
		return *m, nil
	case strings.HasPrefix(input, "/? ") || strings.HasPrefix(input, "/help "):
		return m.handleLegacyHelpCommand(input)
	case input == "/copy":
		return m.copyModelOutput(false)
	case input == "/copy-all":
		return m.copyModelOutput(true)
	case input == "/clear":
		return m.resetChat("cleared")
	case input == "/tools":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.toolsSummary()}))
		return *m, nil
	case input == "/model" || strings.HasPrefix(input, "/model "):
		filter := strings.TrimSpace(strings.TrimPrefix(input, "/model"))
		return m.openModelPicker(filter)
	case input == "/launch":
		m.launchRequested = true
		m.quitting = true
		return *m, tea.Quit
	case input == "/load" || strings.HasPrefix(input, "/load "):
		return m.handleLegacyLoadCommand(input)
	case input == "/think":
		return m.openThinkPicker()
	case strings.HasPrefix(input, "/think "):
		return m.handleThinkCommand(strings.TrimSpace(strings.TrimPrefix(input, "/think")))
	case input == "/set" || strings.HasPrefix(input, "/set "):
		return m.handleLegacySetCommand(input)
	case input == "/show" || strings.HasPrefix(input, "/show "):
		return m.handleLegacyShowCommand(input)
	case input == "/history":
		return m.openHistoryPopup()
	case input == "/skills" || strings.HasPrefix(input, "/skills "):
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.handleSkillsCommand(input)}))
		return *m, nil
	case input == "/new":
		return m.resetChat("new chat")
	case input == "/resume":
		return m.openResumePicker()
	case input == "/system" || strings.HasPrefix(input, "/system "):
		return m.handleSystemCommand(input)
	case input == "/verbose" || strings.HasPrefix(input, "/verbose "):
		return m.handleVerboseCommand(input)
	case input == "/compact":
		return m.startManualCompaction()
	case strings.HasPrefix(input, "/") && m.slashInputIsMultimodalFile(input):
		return m.startRun(input)
	case strings.HasPrefix(input, "/"):
		if skill, request, ok := m.skillTrigger(input); ok {
			displayInput, userMessage, err := m.userMessageFromInput(input, request)
			if err != nil {
				m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
				return *m, nil
			}
			manualMessages, err := agenttools.ManualSkillMessages(skill, request, len(m.messages)+1)
			if err != nil {
				m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
				return *m, nil
			}
			if strings.TrimSpace(userMessage.Content) != "" {
				manualMessages[0].Content = userMessage.Content
			}
			manualMessages[0].Images = userMessage.Images
			return m.startRunWithMessages(displayInput, manualMessages, "")
		}
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Unknown command %q", strings.Fields(input)[0])}))
		return *m, nil
	}

	return m.startRun(input)
}

func (m chatModel) slashInputIsMultimodalFile(input string) bool {
	if !m.opts.MultiModal {
		return false
	}
	fields := strings.Fields(input)
	if len(fields) == 0 {
		return false
	}
	for _, file := range filedata.ExtractNames(input) {
		if strings.HasPrefix(file, fields[0]) {
			return true
		}
	}
	return false
}

func (m *chatModel) handleSystemCommand(input string) (tea.Model, tea.Cmd) {
	arg := strings.TrimSpace(strings.TrimPrefix(input, "/system"))
	if arg == "" {
		m.systemPromptDisabled = !m.systemPromptDisabled
		state := "on"
		if m.systemPromptDisabled {
			state = "off"
		}
		m.status = "cache will break by turning system prompt " + state
		return *m, nil
	}
	m.opts.SystemPrompt = arg
	m.systemPromptDisabled = false
	m.status = "system prompt set; cache will break"
	return *m, nil
}

func (m *chatModel) handleVerboseCommand(input string) (tea.Model, tea.Cmd) {
	arg := strings.TrimSpace(strings.TrimPrefix(input, "/verbose"))
	switch strings.ToLower(arg) {
	case "":
		m.opts.Verbose = !m.opts.Verbose
	case "on", "true", "1":
		m.opts.Verbose = true
	case "off", "false", "0":
		m.opts.Verbose = false
	default:
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: `Usage: /verbose [on|off]`}))
		m.status = "error"
		return *m, nil
	}
	if m.opts.Verbose {
		m.status = "verbose on"
	} else {
		m.status = "verbose off"
	}
	return *m, nil
}

func initialPromptHistory(ctx context.Context, opts ChatOptions) []string {
	if ctx == nil {
		ctx = context.Background()
	}
	if store, ok := opts.Store.(chatPromptHistoryStore); ok && store != nil {
		prompts, err := store.ListUserMessages(ctx, maxPromptHistory)
		if err == nil {
			return normalizePromptHistory(prompts)
		}
	}

	var prompts []string
	for _, msg := range opts.Messages {
		if msg.Role == "user" {
			prompts = append(prompts, msg.Content)
		}
	}
	return normalizePromptHistory(prompts)
}

func normalizePromptHistory(prompts []string) []string {
	history := make([]string, 0, min(len(prompts), maxPromptHistory))
	for _, prompt := range prompts {
		prompt = strings.TrimSpace(prompt)
		if prompt == "" || strings.HasPrefix(prompt, chatCompactionSummaryPrefix) {
			continue
		}
		history = append(history, prompt)
	}
	if len(history) > maxPromptHistory {
		history = history[len(history)-maxPromptHistory:]
	}
	return history
}

func (m *chatModel) addPromptHistory(prompt string) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return
	}
	m.promptHistory = append(m.promptHistory, prompt)
	if len(m.promptHistory) > maxPromptHistory {
		m.promptHistory = m.promptHistory[len(m.promptHistory)-maxPromptHistory:]
	}
	m.resetPromptHistoryCursor()
}

func (m *chatModel) movePromptHistory(delta int) bool {
	if len(m.promptHistory) == 0 || delta == 0 {
		return false
	}
	if !m.promptActive {
		if delta > 0 {
			return false
		}
		m.promptDraft = slices.Clone(m.input)
		m.promptCursor = len(m.promptHistory) - 1
		m.promptActive = true
	} else {
		m.promptCursor += delta
		if m.promptCursor >= len(m.promptHistory) {
			m.input = slices.Clone(m.promptDraft)
			m.inputCursor = len(m.input)
			m.inputCursorSet = true
			m.inputAttachments = nil
			m.resetPromptHistoryCursor()
			m.complete = 0
			return true
		}
		if m.promptCursor < 0 {
			m.promptCursor = 0
		}
	}

	m.input = []rune(m.promptHistory[m.promptCursor])
	m.inputCursor = len(m.input)
	m.inputCursorSet = true
	m.inputAttachments = nil
	m.complete = 0
	return true
}

func (m *chatModel) resetPromptHistoryCursor() {
	m.promptActive = false
	m.promptCursor = 0
	m.promptDraft = nil
}

func (m *chatModel) insertInputNewline() {
	m.insertInputRunes([]rune{'\n'})
}

func (m *chatModel) insertInputRunesFromKey(runes []rune, pasted bool) {
	if len(runes) == 0 {
		return
	}
	if m.opts.MultiModal && (pasted || len(runes) > 1) && m.insertInputFilePlaceholders(string(runes)) {
		return
	}
	if pasted && m.insertPastedTextPlaceholder(string(runes)) {
		return
	}
	m.insertInputRunes(runes)
}

func (m *chatModel) insertInputFilePlaceholders(input string) bool {
	cleaned, files, err := filedata.ExtractWithFiles(input)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "attachment failed"
		return true
	}
	if len(files) == 0 {
		return false
	}

	var parts []string
	if strings.TrimSpace(cleaned) != "" {
		if placeholder, ok := m.pastedTextPlaceholder(cleaned); ok {
			parts = append(parts, placeholder)
		} else {
			parts = append(parts, cleaned)
		}
	}
	for _, file := range files {
		kind := filedata.Kind(file.Path)
		placeholder := m.nextInputAttachmentPlaceholder(kind)
		m.inputAttachments = append(m.inputAttachments, chatInputAttachment{
			placeholder: placeholder,
			kind:        kind,
			data:        file.Data,
		})
		parts = append(parts, placeholder)
	}
	m.insertInputRunes([]rune(strings.Join(parts, " ")))
	m.status = inputAttachmentStatus(files)
	return true
}

func (m *chatModel) insertPastedTextPlaceholder(input string) bool {
	placeholder, ok := m.pastedTextPlaceholder(input)
	if !ok {
		return false
	}
	m.insertInputRunes([]rune(placeholder))
	m.status = "pasted text"
	return true
}

func (m *chatModel) pastedTextPlaceholder(input string) (string, bool) {
	if !shouldCollapsePastedText(input) {
		return "", false
	}
	placeholder := m.nextInputPastedTextPlaceholder(input)
	m.inputPastedTexts = append(m.inputPastedTexts, chatInputPastedText{
		placeholder: placeholder,
		content:     input,
	})
	return placeholder, true
}

func shouldCollapsePastedText(input string) bool {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return false
	}
	return len([]rune(trimmed)) >= pastedTextPlaceholderMinRunes || pastedTextLineCount(trimmed) >= pastedTextPlaceholderMinLines
}

func pastedTextLineCount(input string) int {
	if input == "" {
		return 0
	}
	return strings.Count(input, "\n") + 1
}

func (m *chatModel) insertInputRunes(runes []rune) {
	if len(runes) == 0 {
		return
	}
	m.resetPromptHistoryCursor()
	m.disarmQuit()
	cursor := m.normalizedInputCursor()
	next := make([]rune, 0, len(m.input)+len(runes))
	next = append(next, m.input[:cursor]...)
	next = append(next, runes...)
	next = append(next, m.input[cursor:]...)
	m.input = next
	m.inputCursor = cursor + len(runes)
	m.inputCursorSet = true
	m.complete = 0
}

func (m *chatModel) deleteInputBackward() {
	cursor := m.normalizedInputCursor()
	if cursor <= 0 {
		return
	}
	start, end, ok := m.placeholderRangeForBackspace(cursor)
	if !ok {
		start, end = cursor-1, cursor
	}
	m.deleteInputRange(start, end)
}

func (m *chatModel) deleteInputWordBackward() {
	cursor := m.normalizedInputCursor()
	if cursor <= 0 {
		return
	}
	start, end, ok := m.placeholderRangeForWordDelete(cursor)
	if !ok {
		start, end = previousInputWordStart(m.input, cursor), cursor
	}
	m.deleteInputRange(start, end)
}

func (m *chatModel) deleteInputRange(start, end int) {
	start = clamp(start, 0, len(m.input))
	end = clamp(end, start, len(m.input))
	m.input = append(slices.Clone(m.input[:start]), m.input[end:]...)
	m.inputCursor = start
	m.inputCursorSet = true
	m.complete = 0
	m.syncInputPlaceholders()
}

func (m chatModel) placeholderRangeForBackspace(cursor int) (int, int, bool) {
	cursor = clamp(cursor, 0, len(m.input))
	input := string(m.input)
	for _, placeholder := range m.inputPlaceholders() {
		if placeholder == "" {
			continue
		}
		start, end, ok := inputPlaceholderRuneRange(input, placeholder)
		if ok && cursor > start && cursor <= end {
			return start, end, true
		}
	}
	return 0, 0, false
}

func (m chatModel) placeholderRangeForWordDelete(cursor int) (int, int, bool) {
	cursor = clamp(cursor, 0, len(m.input))
	end := cursor
	for end > 0 && unicode.IsSpace(m.input[end-1]) {
		end--
	}
	input := string(m.input)
	for _, placeholder := range m.inputPlaceholders() {
		if placeholder == "" {
			continue
		}
		start, placeholderEnd, ok := inputPlaceholderRuneRange(input, placeholder)
		if ok && end > start && end <= placeholderEnd {
			return start, cursor, true
		}
	}
	return 0, 0, false
}

func (m chatModel) inputPlaceholders() []string {
	placeholders := make([]string, 0, len(m.inputAttachments)+len(m.inputPastedTexts))
	for _, attachment := range m.inputAttachments {
		placeholders = append(placeholders, attachment.placeholder)
	}
	for _, pastedText := range m.inputPastedTexts {
		placeholders = append(placeholders, pastedText.placeholder)
	}
	return placeholders
}

func inputPlaceholderRuneRange(input, placeholder string) (int, int, bool) {
	byteStart := strings.Index(input, placeholder)
	if byteStart < 0 {
		return 0, 0, false
	}
	start := len([]rune(input[:byteStart]))
	end := start + len([]rune(placeholder))
	return start, end, true
}

func previousInputWordStart(input []rune, cursor int) int {
	end := clamp(cursor, 0, len(input))
	for end > 0 && unicode.IsSpace(input[end-1]) {
		end--
	}
	start := end
	for start > 0 && !unicode.IsSpace(input[start-1]) {
		start--
	}
	return start
}

func (m *chatModel) syncInputPlaceholders() {
	m.inputAttachments = m.activeInputAttachmentsFor(string(m.input))
	m.inputPastedTexts = m.activeInputPastedTextsFor(string(m.input))
}

func (m *chatModel) syncInputAttachments() {
	m.syncInputPlaceholders()
}

func (m chatModel) activeInputAttachmentsFor(input string) []chatInputAttachment {
	if len(m.inputAttachments) == 0 {
		return nil
	}
	active := make([]chatInputAttachment, 0, len(m.inputAttachments))
	for _, attachment := range m.inputAttachments {
		if strings.Contains(input, attachment.placeholder) {
			active = append(active, attachment)
		}
	}
	return active
}

func cloneInputAttachments(in []chatInputAttachment) []chatInputAttachment {
	return slices.Clone(in)
}

type chatInputPastedText struct {
	placeholder string
	content     string
}

func (m chatModel) activeInputPastedTextsFor(input string) []chatInputPastedText {
	if len(m.inputPastedTexts) == 0 {
		return nil
	}
	active := make([]chatInputPastedText, 0, len(m.inputPastedTexts))
	for _, pastedText := range m.inputPastedTexts {
		if strings.Contains(input, pastedText.placeholder) {
			active = append(active, pastedText)
		}
	}
	return active
}

func cloneInputPastedTexts(in []chatInputPastedText) []chatInputPastedText {
	return slices.Clone(in)
}

func (m chatModel) expandPastedTextPlaceholders(input string) string {
	for _, pastedText := range m.activeInputPastedTextsFor(input) {
		input = strings.ReplaceAll(input, pastedText.placeholder, pastedText.content)
	}
	return input
}

func (m *chatModel) nextInputPastedTextPlaceholder(content string) string {
	if m.nextPastedTextID <= 0 {
		m.nextPastedTextID = 1
	}
	id := m.nextPastedTextID
	m.nextPastedTextID++
	return fmt.Sprintf("[Pasted text #%d +%d lines]", id, pastedTextLineCount(strings.TrimSpace(content)))
}

func (m *chatModel) nextInputAttachmentPlaceholder(kind string) string {
	label := inputAttachmentLabel(kind)
	switch kind {
	case "audio":
		id := m.nextAudioID
		m.nextAudioID++
		return fmt.Sprintf("[%s #%d]", label, id)
	default:
		id := m.nextImageID
		m.nextImageID++
		return fmt.Sprintf("[%s #%d]", label, id)
	}
}

func inputAttachmentLabel(kind string) string {
	switch kind {
	case "audio":
		return "Audio"
	default:
		return "Image"
	}
}

func inputAttachmentStatus(files []filedata.File) string {
	if len(files) == 0 {
		return ""
	}
	imageCount, audioCount := 0, 0
	for _, file := range files {
		switch filedata.Kind(file.Path) {
		case "audio":
			audioCount++
		default:
			imageCount++
		}
	}
	switch {
	case imageCount > 0 && audioCount == 0:
		return "attached image"
	case audioCount > 0 && imageCount == 0:
		return "attached audio"
	default:
		return "attached file"
	}
}

var inputAttachmentPlaceholderPattern = regexp.MustCompile(`\[(Image|Audio) #([0-9]+)\]`)
var inputPastedTextPlaceholderPattern = regexp.MustCompile(`\[Pasted text #([0-9]+) \+[0-9]+ lines?\]`)

func nextInputAttachmentIDsFromMessages(messages []api.Message) (imageID int, audioID int) {
	for _, msg := range messages {
		for _, match := range inputAttachmentPlaceholderPattern.FindAllStringSubmatch(msg.Content, -1) {
			if len(match) != 3 {
				continue
			}
			id, err := strconv.Atoi(match[2])
			if err != nil {
				continue
			}
			switch match[1] {
			case "Image":
				imageID = max(imageID, id+1)
			case "Audio":
				audioID = max(audioID, id+1)
			}
		}
	}
	return imageID, audioID
}

func nextInputPastedTextIDFromMessages(messages []api.Message) int {
	nextID := 1
	for _, msg := range messages {
		for _, match := range inputPastedTextPlaceholderPattern.FindAllStringSubmatch(msg.Content, -1) {
			if len(match) != 2 {
				continue
			}
			id, err := strconv.Atoi(match[1])
			if err != nil {
				continue
			}
			nextID = max(nextID, id+1)
		}
	}
	return nextID
}

func (m *chatModel) moveInputCursorHorizontal(delta int) bool {
	if delta == 0 {
		return false
	}
	cursor := clamp(m.normalizedInputCursor()+delta, 0, len(m.input))
	if cursor == m.normalizedInputCursor() {
		return false
	}
	m.inputCursor = cursor
	m.inputCursorSet = true
	m.resetPromptHistoryCursor()
	m.complete = 0
	return true
}

func (m *chatModel) moveInputCursorVertical(delta int) bool {
	if delta == 0 || len(m.input) == 0 {
		return false
	}
	cursor := m.normalizedInputCursor()
	lineStart, lineEnd := inputLineBounds(m.input, cursor)
	column := cursor - lineStart
	var targetStart, targetEnd int
	if delta < 0 {
		if lineStart == 0 {
			return false
		}
		targetEnd = lineStart - 1
		targetStart, _ = inputLineBounds(m.input, targetEnd)
	} else {
		if lineEnd >= len(m.input) {
			return false
		}
		targetStart = lineEnd + 1
		_, targetEnd = inputLineBounds(m.input, targetStart)
	}
	target := min(targetStart+column, targetEnd)
	m.inputCursor = target
	m.inputCursorSet = true
	m.resetPromptHistoryCursor()
	m.complete = 0
	return true
}

func inputLineBounds(input []rune, cursor int) (int, int) {
	cursor = clamp(cursor, 0, len(input))
	start := cursor
	for start > 0 && input[start-1] != '\n' {
		start--
	}
	end := cursor
	for end < len(input) && input[end] != '\n' {
		end++
	}
	return start, end
}

func (m chatModel) normalizedInputCursor() int {
	if !m.inputCursorSet {
		return len(m.input)
	}
	return clamp(m.inputCursor, 0, len(m.input))
}

func inputWithCursor(input []rune, cursor int) string {
	cursor = clamp(cursor, 0, len(input))
	next := make([]rune, 0, len(input)+1)
	next = append(next, input[:cursor]...)
	next = append(next, '█')
	next = append(next, input[cursor:]...)
	return string(next)
}

func isShiftEnterCSI(msg tea.Msg) bool {
	switch fmt.Sprint(msg) {
	case "?CSI[49 51 59 50 117]?", // \x1b[13;2u
		"?CSI[49 51 59 50 126]?",          // \x1b[13;2~
		"?CSI[50 55 59 50 59 49 51 126]?": // \x1b[27;2;13~
		return true
	default:
		return false
	}
}

func renderInputBoxLines(input string, cursor int, width, maxBodyLines int) []string {
	if width < 1 {
		width = 1
	}
	if maxBodyLines < 1 {
		maxBodyLines = 1
	}
	body := wrapChatText("> "+inputWithCursor([]rune(input), cursor), width)
	if len(body) > maxBodyLines {
		body = slices.Clone(body[len(body)-maxBodyLines:])
		body[0] = truncateInputLine("> ... "+body[0], width)
	}
	for i, line := range body {
		body[i] = chatUserStyle.Render(line)
	}

	lines := make([]string, 0, len(body)+2)
	lines = append(lines, chatInputBorderStyle.Render(strings.Repeat("─", width)))
	lines = append(lines, body...)
	lines = append(lines, chatInputBorderStyle.Render(strings.Repeat("─", width)))
	return lines
}

func truncateInputLine(line string, width int) string {
	if width <= 0 {
		return line
	}
	runes := []rune(line)
	if len(runes) <= width {
		return line
	}
	return string(runes[:width])
}

func renderPromptRow(text string, width int) []string {
	if width < 20 {
		width = 20
	}
	lines := wrapChatText(text, width)
	for i, line := range lines {
		lines[i] = chatUserStyle.Render(line)
	}
	return lines
}

func (m chatModel) slashCommandLines(width int) []string {
	return m.renderCompletions(m.slashCompletions(), width)
}

func (m chatModel) completionLines(width int) []string {
	return m.renderCompletions(m.completions(), width)
}

func (m chatModel) renderCompletions(completions []chatCompletion, width int) []string {
	if len(completions) == 0 {
		return nil
	}
	selected := clamp(m.complete, 0, len(completions)-1)
	start, end := completionWindow(len(completions), selected, m.completionVisibleLimit(len(completions)))
	completions = completions[start:end]

	nameWidth := 0
	for _, completion := range completions {
		nameWidth = max(nameWidth, lipgloss.Width(completion.label))
	}

	lines := make([]string, 0, len(completions))
	for i, completion := range completions {
		marker := "  "
		if start+i == selected {
			marker = "› "
		}
		name := chatCommandNameStyle.Render(completion.label)
		padding := strings.Repeat(" ", max(1, nameWidth-lipgloss.Width(completion.label)+2))
		line := marker + name + padding + chatMetaStyle.Render(completion.description)
		lines = append(lines, truncateRenderedLine(line, width))
	}
	return lines
}

func (m chatModel) completionVisibleLimit(total int) int {
	if strings.HasPrefix(strings.TrimSpace(string(m.input)), "/") {
		return min(maxSlashCompletions, total)
	}
	return total
}

func completionWindow(total, selected, limit int) (int, int) {
	if total <= 0 || limit <= 0 || limit >= total {
		return 0, total
	}
	selected = clamp(selected, 0, total-1)
	start := selected - limit + 1
	if start < 0 {
		start = 0
	}
	end := start + limit
	if end > total {
		end = total
		start = max(0, end-limit)
	}
	return start, end
}

func (m chatModel) completions() []chatCompletion {
	if completions := m.slashCompletions(); len(completions) > 0 {
		return completions
	}
	return m.mentionCompletions()
}

func (m chatModel) slashCompletions() []chatCompletion {
	input := strings.TrimSpace(string(m.input))
	if !strings.HasPrefix(input, "/") {
		return nil
	}

	commands := matchingSlashCommands(input)
	skillCompletions := m.skillSlashCompletions(input)
	if len(commands) == 0 && len(skillCompletions) == 0 {
		return []chatCompletion{{label: "No matching commands"}}
	}

	completions := make([]chatCompletion, 0, len(commands)+len(skillCompletions))
	for _, command := range commands {
		completions = append(completions, chatCompletion{
			value:       command.name,
			label:       command.name,
			description: command.description,
		})
	}
	completions = append(completions, skillCompletions...)
	return completions
}

func matchingSlashCommands(input string) []chatSlashCommand {
	prefix := strings.ToLower(strings.TrimSpace(input))
	if prefix == "" {
		return nil
	}

	var commands []chatSlashCommand
	for _, command := range chatSlashCommands {
		if strings.HasPrefix(command.name, prefix) {
			commands = append(commands, command)
		}
	}
	return commands
}

func (m chatModel) mentionCompletions() []chatCompletion {
	input := string(m.input)
	_, query, ok := activeMentionToken(input)
	if !ok {
		return nil
	}

	workingDir := m.currentWorkingDir()
	if strings.TrimSpace(workingDir) == "" {
		var err error
		workingDir, err = os.Getwd()
		if err != nil {
			return []chatCompletion{{label: "No working directory"}}
		}
	}

	dirPart, prefix := splitMentionQuery(query)
	dir, err := resolveCompletionDir(workingDir, dirPart)
	if err != nil {
		return []chatCompletion{{label: "No matching files"}}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []chatCompletion{{label: "No matching files"}}
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].IsDir() != entries[j].IsDir() {
			return entries[i].IsDir()
		}
		return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
	})

	includeHidden := strings.HasPrefix(prefix, ".")
	completions := make([]chatCompletion, 0, 8)
	for _, entry := range entries {
		name := entry.Name()
		if !includeHidden && strings.HasPrefix(name, ".") {
			continue
		}
		if !strings.HasPrefix(strings.ToLower(name), strings.ToLower(prefix)) {
			continue
		}
		value := filepath.ToSlash(filepath.Join(dirPart, name))
		label := "@" + value
		description := "file"
		if entry.IsDir() {
			value += "/"
			label += "/"
			description = "directory"
		}
		completions = append(completions, chatCompletion{
			value:       value,
			label:       label,
			description: description,
			directory:   entry.IsDir(),
		})
		if len(completions) >= 8 {
			break
		}
	}
	if len(completions) == 0 {
		return []chatCompletion{{label: "No matching files"}}
	}
	return completions
}

func activeMentionToken(input string) (int, string, bool) {
	runes := []rune(input)
	start := len(runes)
	for start > 0 && !unicode.IsSpace(runes[start-1]) {
		start--
	}
	token := string(runes[start:])
	if !strings.HasPrefix(token, "@") {
		return 0, "", false
	}
	return start, token[1:], true
}

func splitMentionQuery(query string) (string, string) {
	query = filepath.ToSlash(query)
	index := strings.LastIndex(query, "/")
	if index < 0 {
		return ".", query
	}
	return query[:index+1], query[index+1:]
}

func resolveCompletionDir(workingDir, dir string) (string, error) {
	if filepath.IsAbs(dir) {
		return "", fmt.Errorf("absolute paths are not allowed")
	}
	base := workingDir
	if base == "" {
		base = "."
	}
	base, err := filepath.Abs(base)
	if err != nil {
		return "", err
	}
	resolved := filepath.Clean(filepath.Join(base, dir))
	rel, err := filepath.Rel(base, resolved)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes working directory")
	}
	return resolved, nil
}

func (m *chatModel) moveCompletion(delta int) bool {
	completions := m.completions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return false
	}
	m.complete = (m.complete + delta) % len(completions)
	if m.complete < 0 {
		m.complete += len(completions)
	}
	return true
}

func (m *chatModel) applyCompletion() bool {
	completions := m.completions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return false
	}
	m.resetPromptHistoryCursor()
	selected := completions[clamp(m.complete, 0, len(completions)-1)]
	input := string(m.input)
	if strings.HasPrefix(strings.TrimSpace(input), "/") {
		m.input = []rune(selected.value)
		m.inputCursor = len(m.input)
		m.inputCursorSet = true
		m.complete = 0
		return true
	}

	start, _, ok := activeMentionToken(input)
	if !ok {
		return false
	}
	suffix := ""
	if !selected.directory {
		suffix = " "
	}
	next := string([]rune(input)[:start]) + "@" + selected.value + suffix
	m.input = []rune(next)
	m.inputCursor = len(m.input)
	m.inputCursorSet = true
	m.complete = 0
	return true
}

func completionIsSelectable(completions []chatCompletion) bool {
	return len(completions) > 0 && completions[0].value != ""
}

func (m chatModel) helpSummary() string {
	return strings.Join([]string{
		"**Commands**",
		"",
		"- `/copy`: copy latest model output",
		"- `/copy-all`: copy all model output",
		"- `/tools`: show available tools",
		"- `/model`: switch models",
		"- `/launch`: open launch flow",
		"- `/think`: set thinking mode",
		"- `/history`: show prompt message history",
		"- `/skills`: show or import skills",
		"- `/<skill>`: run the next message with a skill",
		"- `/new`: start a new chat",
		"- `/resume`: resume a saved chat",
		"- `/system`: toggle or set system prompt",
		"- `/verbose`: toggle model metrics",
		"- `/compact`: summarize older context",
		"- `/clear`: clear this chat",
		"- `/bye`: exit",
		"",
		"**Shortcuts**",
		"",
		"- `ctrl+o`: toggle tool output and details",
		"- `ctrl+g`: open your editor to compose a prompt",
		"- `shift+enter`: insert a newline",
		"- `shift+tab`: toggle permission mode",
		"- `↑/↓`: previous or next prompt",
		"- `mouse wheel`, `pgup/pgdn`, `home/end`: scroll transcript",
		"- `drag`: select transcript text and copy on release",
	}, "\n")
}

func (m chatModel) toolsSummary() string {
	if m.opts.Tools == nil || len(m.opts.Tools.Names()) == 0 {
		return "No tools are available for this model."
	}
	var b strings.Builder
	b.WriteString("Available tools:\n\n")
	for _, name := range m.opts.Tools.Names() {
		tool, _ := m.opts.Tools.Get(name)
		b.WriteString("- **")
		b.WriteString(name)
		b.WriteString("**")
		if tool != nil && tool.Description() != "" {
			b.WriteString(": ")
			b.WriteString(tool.Description())
		}
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

func (m *chatModel) copyModelOutput(all bool) (tea.Model, tea.Cmd) {
	content := m.modelOutputContent(all)
	if strings.TrimSpace(content) == "" {
		m.status = "nothing to copy"
		return *m, nil
	}
	if m.opts.Clipboard == nil {
		m.status = "copy unavailable"
		return *m, nil
	}
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if err := m.opts.Clipboard(ctx, content); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not copy output: %v", err), err: err.Error()}))
		m.status = "copy failed"
		return *m, nil
	}
	if all {
		m.status = "copied all output"
	} else {
		m.status = "copied latest output"
	}
	return *m, nil
}

func (m chatModel) modelOutputContent(all bool) string {
	if all {
		return strings.Join(m.assistantOutputs(), "\n\n")
	}
	outputs := m.assistantOutputs()
	if len(outputs) == 0 {
		return ""
	}
	return outputs[len(outputs)-1]
}

func (m chatModel) assistantOutputs() []string {
	var outputs []string
	for _, msg := range m.messages {
		if msg.Role != "assistant" {
			continue
		}
		content := strings.TrimRight(msg.Content, "\n")
		if strings.TrimSpace(content) != "" {
			outputs = append(outputs, content)
		}
	}
	if len(outputs) > 0 {
		return outputs
	}

	for _, entry := range m.entries {
		if entry.role != "assistant" {
			continue
		}
		content := strings.TrimRight(entry.content, "\n")
		if strings.TrimSpace(content) != "" {
			outputs = append(outputs, content)
		}
	}
	return outputs
}

func (m chatModel) historyMessages() []api.Message {
	var messages []api.Message
	if systemPrompt := strings.TrimSpace(m.systemPrompt("")); systemPrompt != "" {
		messages = append(messages, api.Message{Role: "system", Content: systemPrompt})
	}
	return append(messages, m.messages...)
}

func (m chatModel) skillsSummary() string {
	if m.opts.Skills == nil || m.opts.Skills.Empty() {
		return "No skills are installed.\n\nImport skills with `/skills import claude`, `/skills import codex`, `/skills import pi`, or `/skills import all`."
	}
	return m.opts.Skills.SummaryMarkdown() + "\n\nImport more with `/skills import claude`, `/skills import codex`, `/skills import pi`, or `/skills import all`."
}

func (m *chatModel) handleSkillsCommand(input string) string {
	fields := strings.Fields(input)
	if len(fields) == 1 {
		return m.skillsSummary()
	}
	if len(fields) >= 2 && fields[1] == "import" {
		return m.importSkills(fields[2:])
	}
	return "Usage:\n\n/skills\n/skills import claude|codex|pi|agents|all [--force]"
}

func (m *chatModel) importSkills(args []string) string {
	source := "all"
	force := false
	for _, arg := range args {
		switch arg {
		case "--force":
			force = true
		default:
			if strings.TrimSpace(arg) != "" {
				source = arg
			}
		}
	}

	results, err := skills.Import(source, force)
	if err != nil {
		return err.Error()
	}
	catalog, err := skills.LoadDefault()
	if err != nil {
		return fmt.Sprintf("imported skills but could not reload catalog: %v", err)
	}
	if m.opts.Skills != nil {
		*m.opts.Skills = *catalog
		catalog = m.opts.Skills
	} else {
		m.opts.Skills = catalog
	}
	if m.opts.Tools != nil && !catalog.Empty() {
		m.opts.Tools.Register(agenttools.NewSkill(catalog))
	}
	if m.opts.SystemPromptForModel != nil {
		m.opts.SystemPrompt = m.opts.SystemPromptForModel(m.ctx, m.opts.Model, m.opts.Tools)
	}

	if len(results) == 0 {
		return fmt.Sprintf("No skills found for %s.", source)
	}
	var imported, skipped int
	var lines []string
	for _, result := range results {
		name := result.Skill.Name
		if name == "" {
			name = result.From
		}
		if result.Skipped {
			skipped++
			line := "skipped " + name
			if result.Error != "" {
				line += " (" + result.Error + ")"
			}
			lines = append(lines, line)
			continue
		}
		imported++
		line := "imported " + name
		if result.Error != "" {
			line += " (" + result.Error + ")"
		}
		lines = append(lines, line)
	}
	lines = append(lines, "", fmt.Sprintf("%d imported, %d skipped", imported, skipped))
	return strings.Join(lines, "\n")
}

func (m chatModel) systemPrompt(extra string) string {
	var parts []string
	if !m.systemPromptDisabled && strings.TrimSpace(m.opts.SystemPrompt) != "" {
		parts = append(parts, strings.TrimSpace(m.opts.SystemPrompt))
	}
	if strings.TrimSpace(extra) != "" {
		parts = append(parts, strings.TrimSpace(extra))
	}
	return strings.Join(parts, "\n\n")
}

func (m chatModel) skillTrigger(input string) (skills.Skill, string, bool) {
	if m.opts.Skills == nil || m.opts.Skills.Empty() {
		return skills.Skill{}, "", false
	}
	command, rest, _ := strings.Cut(strings.TrimSpace(input), " ")
	name := strings.TrimPrefix(command, "/")
	skill, ok := m.opts.Skills.Find(name)
	return skill, strings.TrimSpace(rest), ok
}

func (m chatModel) skillSlashCompletions(input string) []chatCompletion {
	if m.opts.Skills == nil || m.opts.Skills.Empty() {
		return nil
	}
	prefix := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(input)), "/")
	var completions []chatCompletion
	for _, skill := range m.opts.Skills.Skills {
		if !strings.HasPrefix(skill.Name, prefix) {
			continue
		}
		completions = append(completions, chatCompletion{
			value:       "/" + skill.Name,
			label:       "/" + skill.Name,
			description: skill.Description,
		})
	}
	return completions
}
