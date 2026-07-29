package chat

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
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mattn/go-runewidth"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/filedata"
)

type chatSlashCommand struct {
	name        string
	usage       string
	description string
	aliases     []string
	hidden      bool
}

type chatCompletion struct {
	value       string
	label       string
	description string
	directory   bool
}

const (
	chatPromptPrefix          = ""
	inputBoxHorizontalPadding = 1
	inputCursorGlyph          = "█"
	inputCursorMarker         = "\x00"
	maxInputBoxBodyLines      = 12
)

const (
	pastedTextPlaceholderMinRunes = 1000
	pastedTextPlaceholderMinLines = 8
)

var chatSlashCommands = []chatSlashCommand{
	{name: "/model", description: "switch models"},
	{name: "/new", description: "start a new chat"},
	{name: "/think", description: "set thinking mode"},
	{name: "/tools", description: "toggle tools on or off"},
	{name: "/system", usage: "/system [on|off]", description: "show or set the built-in system prompt"},
	{name: "/skills", usage: "/skills [import codex|claude|pi]", description: "list or import skills"},
	{name: "/compact", description: "summarize older context"},
	{name: "/help", description: "show commands", aliases: []string{"/?"}},
	{name: "/bye", description: "exit", aliases: []string{"/exit"}},
	{name: "/prompt", description: "show full prompt, tools, and messages"},
	{name: "/save", usage: "/save <filename>", description: "save request JSON; saved as <filename>.json"},
}

var skillsImportCompletions = []chatCompletion{
	{value: "/skills import codex", label: "/skills import codex", description: "import from ~/.codex/skills"},
	{value: "/skills import claude", label: "/skills import claude", description: "import from ~/.claude/skills"},
	{value: "/skills import pi", label: "/skills import pi", description: "import from ~/.pi/agent/skills"},
}

// BuiltinSlashCommandNames returns the names reserved by built-in slash
// commands, including aliases.
func BuiltinSlashCommandNames() []string {
	names := make(map[string]struct{})
	for _, command := range chatSlashCommands {
		names[strings.TrimPrefix(command.name, "/")] = struct{}{}
		for _, alias := range command.aliases {
			names[strings.TrimPrefix(alias, "/")] = struct{}{}
		}
	}
	reserved := make([]string, 0, len(names))
	for name := range names {
		reserved = append(reserved, name)
	}
	sort.Strings(reserved)
	return reserved
}

func (m *chatModel) handleSubmit() (tea.Model, tea.Cmd) {
	m.syncInputPlaceholders()
	input := strings.TrimSpace(string(m.input))
	if input == "" {
		return *m, nil
	}
	_, _, hasSlashCommand := slashCommandInvocation(input)
	if (m.running || m.compacting) && !hasSlashCommand {
		m.status = "wait for current response"
		return *m, nil
	}

	attachments := cloneInputAttachments(m.inputAttachments)
	pastedTexts := cloneInputPastedTexts(m.inputPastedTexts)
	m.input = nil
	m.inputCursor = 0
	m.inputCursorSet = false
	m.inputAttachments = attachments
	m.inputPastedTexts = pastedTexts
	m.complete = 0
	m.resetPromptHistoryCursor()
	return m.submitInput(input)
}

func (m *chatModel) applySlashCompletion() bool {
	rawInput := string(m.input)
	input := strings.TrimSpace(rawInput)
	if !strings.HasPrefix(input, "/") {
		return false
	}
	if _, _, known := slashCommandInvocation(input); known && !hasSystemCommandArgument(rawInput) {
		return false
	}
	completions := m.slashCompletions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return false
	}
	selected := completions[clamp(m.complete, 0, len(completions)-1)]
	if strings.EqualFold(selected.value, input) {
		return false
	}
	// Reset prompt-history state: Up/Down is shared between history recall and
	// slash completion, and a recalled prompt may start with "/" and trigger
	// completion. Keep the two in sync when we accept a completion.
	m.resetPromptHistoryCursor()
	m.input = []rune(selected.value)
	m.inputCursor = len(m.input)
	m.inputCursorSet = true
	m.complete = 0
	return true
}

func (m *chatModel) submitInput(input string) (tea.Model, tea.Cmd) {
	command, args, _ := slashCommandInvocation(input)
	if command != "" {
		input = strings.TrimSpace(command + " " + args)
	}
	skillName, skillPrompt, skillOK := m.skillSlashInvocation(input)

	switch {
	case command == "/bye":
		m.quitting = true
		return *m, m.quitCmd()
	case command == "/help":
		m.entries = append(m.entries, newSlashEntry(m.helpSummary()))
		return *m, nil
	case command == "/model":
		return m.openModelPicker(args)
	case command == "/think" && args == "":
		return m.openThinkPicker()
	case command == "/think":
		return m.handleThinkCommand(args)
	case command == "/tools":
		return m.handleToolsCommand(args)
	case command == "/system":
		return m.handleSystemCommand(args)
	case command == "/skills":
		return m.handleSkillsCommand(args)
	case command == "/prompt":
		return m.handlePromptCommand(args)
	case command == "/save":
		return m.handleSaveCommand(args)
	case command == "/new" && args == "":
		return m.resetChat("new chat")
	case command == "/compact" && args == "":
		return m.startManualCompaction()
	case skillOK:
		return m.startSkillRun(skillName, skillPrompt)
	case strings.HasPrefix(input, "/") && m.slashInputIsMultimodalFile(input):
		return m.startRun(input)
	case strings.HasPrefix(input, "/"):
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Unknown command %q", strings.Fields(input)[0])}))
		return *m, nil
	}

	return m.startRun(input)
}

func (m *chatModel) handleSkillsCommand(args string) (tea.Model, tea.Cmd) {
	if fields := strings.Fields(args); len(fields) == 2 && fields[0] == "import" {
		return m.handleSkillsImport(fields[1])
	} else if len(fields) != 0 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "usage: /skills [import codex|claude|pi]"}))
		return *m, nil
	}
	skills := m.opts.Skills.List()
	if len(skills) == 0 {
		m.entries = append(m.entries, newSlashEntry("No skills found. Add directories containing SKILL.md under "+skillsDirForDisplay(m.opts.Skills)+"."))
		return *m, nil
	}
	lines := []string{"Available skills:"}
	for _, skill := range skills {
		description := skill.Description
		if description == "" {
			description = "No description provided."
		}
		lines = append(lines, fmt.Sprintf("- `%s`: %s", skill.Name, description))
	}
	lines = append(lines, "\nType `/<name>` to load a skill into the conversation.")
	m.entries = append(m.entries, newSlashEntry(strings.Join(lines, "\n")))
	return *m, nil
}

func (m *chatModel) handleSkillsImport(source string) (tea.Model, tea.Cmd) {
	importSkills := m.opts.ImportSkills
	if importSkills == nil {
		importSkills = coreagent.ImportSkills
	}
	result, err := importSkills(source)
	if err != nil {
		m.status = "error"
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not import %s skills: %v", source, err)}))
		return *m, nil
	}

	if len(result.Imported) != 0 || len(result.Existing) != 0 {
		reload := m.opts.ReloadSkills
		if reload == nil {
			reload = func() (*coreagent.SkillCatalog, error) {
				return coreagent.LoadDefaultSkills(m.currentWorkingDir())
			}
		}
		catalog, err := reload()
		if err != nil {
			m.status = "error"
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("%s\n\nCould not reload skills: %v", skillsImportSummary(result), err)}))
			return *m, nil
		}
		m.opts.Skills = catalog
		if m.opts.ToolRegistryForModel != nil && m.opts.Model != "" {
			m.opts.Tools = m.opts.ToolRegistryForModel(m.ctx, m.opts.Model)
		}
		if m.opts.SystemPromptForModel != nil {
			m.opts.SystemPrompt = m.opts.SystemPromptForModel(m.ctx, m.opts.Model, m.opts.Tools, m.opts.ToolsDisabled)
		}
		m.status = "skills reloaded"
	}
	m.entries = append(m.entries, newSlashEntry(skillsImportSummary(result)))
	return *m, nil
}

func skillsImportSummary(result coreagent.SkillImportResult) string {
	if len(result.Imported) == 0 && len(result.Existing) == 0 && len(result.Failures) == 0 {
		return fmt.Sprintf("No %s skills found at %s.", result.Source, result.SourceDir)
	}
	var lines []string
	if len(result.Imported) != 0 {
		lines = append(lines, fmt.Sprintf("Imported %d skill%s from %s.", len(result.Imported), pluralSuffix(len(result.Imported)), result.SourceDir))
	}
	if len(result.Existing) != 0 {
		lines = append(lines, "Already present (left unchanged): "+strings.Join(result.Existing, ", ")+".")
	}
	for _, failure := range result.Failures {
		lines = append(lines, fmt.Sprintf("Skipped %s: %v.", failure.Name, failure.Err))
	}
	return strings.Join(lines, "\n")
}

func skillsDirForDisplay(catalog *coreagent.SkillCatalog) string {
	if catalog != nil && catalog.Dir() != "" {
		return catalog.Dir()
	}
	dir, err := coreagent.SkillsDir()
	if err != nil {
		return "the Ollama skills directory"
	}
	return dir
}

// skillSlashInvocation parses "/<skill-name>" or "/<skill-name> <prompt>".
// It returns the skill name, any trailing prompt, and ok when the first token is
// a catalog skill. Built-in slash commands take precedence over same-named
// skills, so they are never claimed here.
func (m *chatModel) skillSlashInvocation(input string) (name, prompt string, ok bool) {
	input = strings.TrimSpace(input)
	if !strings.HasPrefix(input, "/") {
		return "", "", false
	}
	token, args, _ := strings.Cut(input, " ")
	name = strings.TrimPrefix(token, "/")
	if name == "" {
		return "", "", false
	}
	if _, _, known := slashCommandInvocation(input); known {
		return "", "", false
	}
	if _, err := m.opts.Skills.Load(name); err != nil {
		return "", "", false
	}
	return name, strings.TrimSpace(args), true
}

func (m *chatModel) handleToolsCommand(args string) (tea.Model, tea.Cmd) {
	if strings.TrimSpace(args) != "" {
		m.status = "error"
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "usage: /tools"}))
		return *m, nil
	}
	if m.opts.ToolsDisabled {
		m.opts.ToolsDisabled = false
		if m.opts.ToolRegistryForModel != nil {
			m.opts.Tools = m.opts.ToolRegistryForModel(m.ctx, m.opts.Model)
		}
		m.status = "tools on"
	} else {
		m.opts.ToolsDisabled = true
		m.status = "tools off"
	}
	if m.opts.SystemPromptForModel != nil {
		m.opts.SystemPrompt = m.opts.SystemPromptForModel(m.ctx, m.opts.Model, m.opts.Tools, m.opts.ToolsDisabled)
	}
	return *m, nil
}

func (m *chatModel) handleSystemCommand(args string) (tea.Model, tea.Cmd) {
	switch strings.ToLower(strings.TrimSpace(args)) {
	case "":
		m.entries = append(m.entries, newSlashEntry(m.systemCommandOutput()))
	case "on":
		m.systemPromptDisabled = false
		m.status = "system prompt on"
		m.entries = append(m.entries, newSlashEntry(m.systemCommandOutput()))
	case "off":
		m.systemPromptDisabled = true
		m.status = "system prompt off"
		m.entries = append(m.entries, newSlashEntry(m.systemCommandOutput()))
	default:
		m.status = "error"
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "usage: /system [on|off]"}))
	}
	return *m, nil
}

func (m chatModel) systemPromptState() string {
	if m.systemPromptDisabled {
		return "off"
	}
	return "on"
}

func (m chatModel) systemCommandOutput() string {
	prompt := strings.TrimSpace(m.opts.SystemPrompt)
	if prompt == "" {
		prompt = "(empty)"
	}
	return "Built-in system prompt is " + m.systemPromptState() + ".\n\n" + prompt + "\n\nWarning: Changing the system prompt during a session breaks the prompt cache."
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

func initialPromptHistory(ctx context.Context, opts Options) []string {
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
		if prompt == "" || coreagent.IsCompactionSummary(api.Message{Role: "user", Content: prompt}) {
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
			m.inputPastedTexts = nil
			m.resetPromptHistoryCursor()
			m.complete = 0
			return true
		}
		if m.promptCursor < 0 {
			m.promptCursor = 0
		}
	}

	m.inputPastedTexts = nil
	input := m.promptHistory[m.promptCursor]
	if placeholder, ok := m.pastedTextPlaceholder(input); ok {
		input = placeholder
	}
	m.input = []rune(input)
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
	return true
}

func (m *chatModel) insertPastedTextPlaceholder(input string) bool {
	placeholder, ok := m.pastedTextPlaceholder(input)
	if !ok {
		return false
	}
	m.insertInputRunes([]rune(placeholder))
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
	runes = normalizeInputRunes(runes)
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

func normalizeInputRunes(runes []rune) []rune {
	for _, r := range runes {
		if r == '\r' {
			return normalizeInputRunesSlow(runes)
		}
	}
	return runes
}

func normalizeInputRunesSlow(runes []rune) []rune {
	out := make([]rune, 0, len(runes))
	for i := 0; i < len(runes); i++ {
		if runes[i] != '\r' {
			out = append(out, runes[i])
			continue
		}
		out = append(out, '\n')
		if i+1 < len(runes) && runes[i+1] == '\n' {
			i++
		}
	}
	return out
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

func (m *chatModel) deleteInputForward() {
	cursor := m.normalizedInputCursor()
	if cursor >= len(m.input) {
		return
	}
	start, end, ok := m.placeholderRangeForForwardDelete(cursor)
	if !ok {
		start, end = cursor, cursor+1
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

func (m chatModel) placeholderRangeForForwardDelete(cursor int) (int, int, bool) {
	cursor = clamp(cursor, 0, len(m.input))
	input := string(m.input)
	for _, placeholder := range m.inputPlaceholders() {
		if placeholder == "" {
			continue
		}
		start, end, ok := inputPlaceholderRuneRange(input, placeholder)
		if ok && cursor >= start && cursor < end {
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

var (
	inputAttachmentPlaceholderPattern = regexp.MustCompile(`\[(Image|Audio) #([0-9]+)\]`)
	inputPastedTextPlaceholderPattern = regexp.MustCompile(`\[Pasted text #([0-9]+) \+[0-9]+ lines?\]`)
)

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

func (m *chatModel) moveInputCursorToLineStart() bool {
	cursor := m.normalizedInputCursor()
	start, _ := inputLineBounds(m.input, cursor)
	if start == cursor {
		return false
	}
	m.inputCursor = start
	m.inputCursorSet = true
	m.resetPromptHistoryCursor()
	m.complete = 0
	return true
}

func (m *chatModel) moveInputCursorToLineEnd() bool {
	cursor := m.normalizedInputCursor()
	_, end := inputLineBounds(m.input, cursor)
	if end == cursor {
		return false
	}
	m.inputCursor = end
	m.inputCursorSet = true
	m.resetPromptHistoryCursor()
	m.complete = 0
	return true
}

func (m *chatModel) moveInputCursorWord(delta int) bool {
	if delta == 0 || len(m.input) == 0 {
		return false
	}
	cursor := m.normalizedInputCursor()
	target := cursor
	if delta < 0 {
		for target > 0 && unicode.IsSpace(m.input[target-1]) {
			target--
		}
		for target > 0 && !unicode.IsSpace(m.input[target-1]) {
			target--
		}
	} else {
		for target < len(m.input) && !unicode.IsSpace(m.input[target]) {
			target++
		}
		for target < len(m.input) && unicode.IsSpace(m.input[target]) {
			target++
		}
	}
	if target == cursor {
		return false
	}
	m.inputCursor = target
	m.inputCursorSet = true
	m.resetPromptHistoryCursor()
	m.complete = 0
	return true
}

func (m *chatModel) handleInputAltRunes(runes []rune) bool {
	if len(runes) != 1 {
		return false
	}
	switch runes[0] {
	case 'b', 'B':
		m.moveInputCursorWord(-1)
		return true
	case 'f', 'F':
		m.moveInputCursorWord(1)
		return true
	default:
		return false
	}
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
	next = append(next, []rune(inputCursorMarker)...)
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

func (m chatModel) emptyInputPlaceholder() string {
	if m.promptDebug != nil || m.modelPicker != nil || m.thinkPicker != nil || m.approvalPrompt != nil || m.cloudAuthPrompt != nil {
		return ""
	}
	if len(m.entries) > 0 || len(m.messages) > 0 || len(m.input) > 0 || len(m.inputAttachments) > 0 || len(m.inputPastedTexts) > 0 {
		return ""
	}
	return m.emptyChatHint()
}

func renderInputBoxLines(input string, cursor int, width, maxBodyLines int, placeholder string) []string {
	if width < 12 {
		width = 12
	}
	if maxBodyLines < 1 {
		maxBodyLines = 1
	}

	prefix := chatPromptPrefix
	continuationPrefix := strings.Repeat(" ", lipgloss.Width(prefix))
	prefixWidth := lipgloss.Width(prefix)
	innerWidth := max(1, width-2)
	contentWidth := max(1, innerWidth-inputBoxHorizontalPadding*2)
	bodyWidth := max(1, contentWidth-prefixWidth)

	var raw []string
	placeholderMode := input == "" && strings.TrimSpace(placeholder) != ""
	if input == "" && strings.TrimSpace(placeholder) != "" {
		raw = renderInputPromptRawLines(inputWithCursor([]rune(placeholder), 0), prefix, continuationPrefix, bodyWidth)
	} else {
		if cursor >= 0 {
			input = inputWithCursor([]rune(input), cursor)
		}
		raw = renderInputPromptRawLines(input, prefix, continuationPrefix, bodyWidth)
	}
	if len(raw) > maxBodyLines {
		raw = slices.Clone(raw[len(raw)-maxBodyLines:])
		raw[0] = truncateInputLine(continuationPrefix+trimInputPromptPrefix(raw[0]), contentWidth)
	}

	lines := make([]string, 0, len(raw)+2)
	lines = append(lines, chatInputBorderStyle.Render(inputBoxTopBorderLine(width)))
	for i, line := range raw {
		rendered := line
		if placeholderMode {
			if i == 0 && strings.HasPrefix(line, prefix) {
				rest := strings.TrimPrefix(line, prefix)
				if strings.HasPrefix(rest, inputCursorMarker) {
					rendered = chatUserStyle.Render(prefix) + renderInputTextWithCursorStyle(rest, chatInputPlaceholderStyle)
					lines = append(lines, renderInputBoxBodyLine(rendered, contentWidth))
					continue
				}
				rendered = chatUserStyle.Render(prefix) + chatInputPlaceholderStyle.Render(rest)
				lines = append(lines, renderInputBoxBodyLine(rendered, contentWidth))
				continue
			}
			if strings.HasPrefix(line, continuationPrefix) {
				rendered = chatInputPlaceholderStyle.Render(continuationPrefix + strings.TrimPrefix(line, continuationPrefix))
				lines = append(lines, renderInputBoxBodyLine(rendered, contentWidth))
				continue
			}
			rendered = chatInputPlaceholderStyle.Render(line)
			lines = append(lines, renderInputBoxBodyLine(rendered, contentWidth))
			continue
		}
		lines = append(lines, renderInputBoxBodyLine(renderInputTextWithCursor(rendered), contentWidth))
	}
	lines = append(lines, chatInputBorderStyle.Render(inputBoxBottomBorderLine(width)))
	return lines
}

func renderInputTextWithCursor(line string) string {
	return renderInputTextWithCursorStyle(line, chatUserStyle)
}

func renderInputTextWithCursorStyle(line string, style lipgloss.Style) string {
	before, after, ok := strings.Cut(line, inputCursorMarker)
	if !ok {
		return style.Render(line)
	}
	cell, rest := inputCursorCell(after)
	return style.Render(before) + renderInputCursorCell(cell) + style.Render(rest)
}

func inputCursorCell(after string) (string, string) {
	if after == "" {
		return "", ""
	}
	r, size := utf8.DecodeRuneInString(after)
	if r == '\n' {
		return "", after
	}
	return after[:size], after[size:]
}

func renderInputCursorCell(cell string) string {
	if cell == "" {
		return chatBlankCursorStyle.Render(inputCursorGlyph)
	}
	return chatCursorStyle.Render(cell)
}

func renderInputPromptRawLines(text, prefix, continuationPrefix string, width int) []string {
	if width <= 0 {
		width = 1
	}
	text = string(normalizeInputRunes([]rune(text)))
	body := wrapChatText(text, width)
	for i, line := range body {
		if i == 0 {
			body[i] = prefix + line
			continue
		}
		body[i] = continuationPrefix + line
	}
	if len(body) == 0 {
		return []string{prefix}
	}
	return body
}

func trimInputPromptPrefix(line string) string {
	for _, prefix := range []string{chatPromptPrefix, strings.Repeat(" ", lipgloss.Width(chatPromptPrefix))} {
		if prefix == "" {
			continue
		}
		if strings.HasPrefix(line, prefix) {
			return strings.TrimPrefix(line, prefix)
		}
	}
	return strings.TrimSpace(line)
}

func inputBoxTopBorderLine(width int) string {
	return inputBoxBorderLine(width, "╭", "╮")
}

func inputBoxBottomBorderLine(width int) string {
	return inputBoxBorderLine(width, "╰", "╯")
}

func inputBoxBorderLine(width int, left, right string) string {
	if width < 4 {
		width = 4
	}
	return left + strings.Repeat("─", max(0, width-2)) + right
}

func renderInputBoxBodyLine(line string, width int) string {
	padding := strings.Repeat(" ", inputBoxHorizontalPadding)
	return chatInputBorderStyle.Render("│") + padding + padRenderedLine(line, width) + padding + chatInputBorderStyle.Render("│")
}

func padRenderedLine(line string, width int) string {
	if width <= 0 {
		return line
	}
	if renderedWidth := lipgloss.Width(line); renderedWidth < width {
		return line + strings.Repeat(" ", width-renderedWidth)
	}
	return line
}

func truncateInputLine(line string, width int) string {
	if width <= 0 {
		return line
	}
	if runewidth.StringWidth(line) <= width {
		return line
	}
	return runewidth.Truncate(line, width, "")
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
	rawInput := string(m.input)
	input := strings.TrimLeftFunc(rawInput, unicode.IsSpace)
	if !strings.HasPrefix(input, "/") {
		return nil
	}
	if argument, ok := systemCommandArgument(rawInput); ok {
		return systemCommandCompletions(argument)
	}
	if m.skillSlashPromptStarted(rawInput) {
		return nil
	}
	if completions := matchingSkillsImportCompletions(input); completions != nil {
		return completions
	}

	commands := matchingSlashCommands(input)
	completions := make([]chatCompletion, 0, len(commands))
	for _, command := range commands {
		completions = append(completions, chatCompletion{
			value:       command.name,
			label:       command.name,
			description: command.description,
		})
	}
	if strings.EqualFold(input, "/skills") {
		completions = append(completions, chatCompletion{
			value:       "/skills import",
			label:       "/skills import",
			description: "import skills from Codex, Claude, or Pi",
		})
	}
	// Each catalog skill is also invocable as "/<skill-name>"; surface them as
	// completions so they are discoverable by typing.
	if m.opts.Skills != nil {
		prefix := strings.ToLower(input)
		for _, skill := range m.opts.Skills.List() {
			name := "/" + skill.Name
			if !strings.HasPrefix(name, prefix) {
				continue
			}
			if _, _, known := slashCommandInvocation(name); known {
				continue // built-in command wins; don't shadow it
			}
			description := skill.Description
			if description == "" {
				description = "No description provided."
			}
			completions = append(completions, chatCompletion{
				value:       name,
				label:       name,
				description: description,
			})
		}
	}
	if len(completions) == 0 {
		return []chatCompletion{{label: "No matching commands"}}
	}
	return completions
}

func matchingSkillsImportCompletions(input string) []chatCompletion {
	const importCommand = "/skills import"
	lower := strings.ToLower(input)
	if lower == "/skills" {
		return nil // Preserve Enter on /skills as the listing command.
	}
	if !strings.HasPrefix(lower, "/skills ") {
		return nil
	}
	if strings.HasPrefix(importCommand, lower) {
		return []chatCompletion{{
			value:       importCommand,
			label:       importCommand,
			description: "import skills from Codex, Claude, or Pi",
		}}
	}
	if !strings.HasPrefix(lower, importCommand) {
		return nil
	}
	prefix := strings.TrimSpace(strings.TrimPrefix(lower, importCommand))
	completions := make([]chatCompletion, 0, len(skillsImportCompletions))
	for _, completion := range skillsImportCompletions {
		if strings.HasPrefix(strings.TrimPrefix(completion.value, importCommand+" "), prefix) {
			completions = append(completions, completion)
		}
	}
	if len(completions) == 0 {
		return []chatCompletion{{label: "No matching skill sources"}}
	}
	return completions
}

func hasSystemCommandArgument(input string) bool {
	_, ok := systemCommandArgument(input)
	return ok
}

func systemCommandArgument(input string) (string, bool) {
	input = strings.TrimLeftFunc(input, unicode.IsSpace)
	end := strings.IndexFunc(input, unicode.IsSpace)
	if end < 0 {
		return "", false
	}
	command, _, known := slashCommandInvocation(input[:end])
	if !known || command != "/system" {
		return "", false
	}
	return strings.TrimSpace(input[end:]), true
}

func systemCommandCompletions(argument string) []chatCompletion {
	argument = strings.ToLower(argument)
	options := []chatCompletion{
		{value: "/system on", label: "on", description: "enable the built-in system prompt"},
		{value: "/system off", label: "off", description: "disable the built-in system prompt"},
	}
	completions := make([]chatCompletion, 0, len(options))
	for _, option := range options {
		if strings.HasPrefix(option.label, argument) {
			completions = append(completions, option)
		}
	}
	if len(completions) == 0 {
		return []chatCompletion{{label: "No matching options"}}
	}
	return completions
}

func (m chatModel) skillSlashPromptStarted(input string) bool {
	input = strings.TrimLeftFunc(input, unicode.IsSpace)
	end := strings.IndexFunc(input, unicode.IsSpace)
	if end < 0 {
		return false
	}
	_, _, ok := m.skillSlashInvocation(input[:end])
	return ok
}

func matchingSlashCommands(input string) []chatSlashCommand {
	prefix := strings.ToLower(strings.TrimSpace(input))
	if prefix == "" {
		return nil
	}

	var commands []chatSlashCommand
	for _, command := range chatSlashCommands {
		if command.hidden {
			continue
		}
		if command.matchesPrefix(prefix) {
			commands = append(commands, command)
		}
	}
	return commands
}

func (c chatSlashCommand) matchesPrefix(prefix string) bool {
	if strings.HasPrefix(c.name, prefix) {
		return true
	}
	for _, alias := range c.aliases {
		if strings.HasPrefix(alias, prefix) {
			return true
		}
	}
	return false
}

func slashCommandInvocation(input string) (string, string, bool) {
	input = strings.TrimSpace(input)
	if !strings.HasPrefix(input, "/") {
		return "", "", false
	}
	token, args, _ := strings.Cut(input, " ")
	token = strings.ToLower(token)
	for _, command := range chatSlashCommands {
		if command.name == token || slices.Contains(command.aliases, token) {
			return command.name, strings.TrimSpace(args), true
		}
	}
	return "", "", false
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
	lines := []string{
		"**Commands**",
		"",
	}
	for _, command := range chatSlashCommands {
		if command.hidden || strings.TrimSpace(command.description) == "" {
			continue
		}
		usage := command.name
		if command.usage != "" {
			usage = command.usage
		}
		lines = append(lines, fmt.Sprintf("- `%s`: %s", usage, command.description))
	}
	lines = append(lines,
		"",
		"**Shortcuts**",
		"",
		"- `shift+enter`: insert a newline",
		"- `shift+tab`: toggle permission mode",
		"- `↑/↓`: previous or next prompt",
		"- `ctrl+a/e`: move to line start or end",
	)
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
