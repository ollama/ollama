package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

type chatPromptDebug struct {
	request api.ChatRequest
	tokens  int
	scroll  int
}

const maxPromptDebugToolResultRunes = 400

func (m *chatModel) handleSaveCommand(args string) (tea.Model, tea.Cmd) {
	filename, err := saveRequestFilename(args)
	if err != nil {
		return m.addDebugError(err)
	}
	raw, err := m.rawRequestJSON()
	if err != nil {
		return m.addDebugError(err)
	}

	dir, err := m.debugWorkingDir()
	if err != nil {
		return m.addDebugError(err)
	}
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, []byte(raw+"\n"), 0o644); err != nil {
		return m.addDebugError(err)
	}
	m.entries = append(m.entries, newSlashEntry(fmt.Sprintf("saved as %s", filename)))
	m.status = "saved"
	return *m, nil
}

func (m *chatModel) handlePromptCommand(args string) (tea.Model, tea.Cmd) {
	if strings.TrimSpace(args) != "" {
		return m.addDebugError(fmt.Errorf("usage: /prompt"))
	}
	req, tokens := m.requestPreview()
	m.promptDebug = &chatPromptDebug{
		request: req,
		tokens:  tokens,
	}
	m.flowPrintedLines = 0
	m.selection = chatSelection{}
	m.status = "prompt"
	return *m, tea.Batch(tea.ClearScreen, tea.EnableMouseCellMotion)
}

func (m chatModel) updatePromptDebug(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.promptDebug == nil {
		return m, nil
	}
	switch msg.Type {
	case tea.KeyEsc, tea.KeyCtrlC, tea.KeyEnter:
		return m.closePromptDebug()
	case tea.KeyUp, tea.KeyCtrlP:
		m.promptDebug.scroll--
	case tea.KeyDown, tea.KeyCtrlN:
		m.promptDebug.scroll++
	case tea.KeyPgUp:
		m.promptDebug.scroll -= max(1, m.promptDebugPageSize())
	case tea.KeyPgDown:
		m.promptDebug.scroll += max(1, m.promptDebugPageSize())
	case tea.KeyHome, tea.KeyCtrlHome:
		m.promptDebug.scroll = 0
	case tea.KeyEnd, tea.KeyCtrlEnd:
		m.promptDebug.scroll = m.promptDebugMaxScroll()
	}
	if m.promptDebug != nil {
		m.promptDebug.scroll = clamp(m.promptDebug.scroll, 0, m.promptDebugMaxScroll())
	}
	return m, nil
}

func (m chatModel) closePromptDebug() (tea.Model, tea.Cmd) {
	m.promptDebug = nil
	m.status = "ready"
	m.flowPrintedLines = 0
	next, printCmd := m.flowTranscriptFlushCmd()
	return next, tea.Sequence(tea.DisableMouse, tea.ClearScreen, printCmd)
}

func (m chatModel) renderPromptDebug(width, height int) string {
	if width <= 0 {
		width = 80
	}
	if height <= 0 {
		height = 24
	}
	if m.promptDebug == nil {
		return renderFullFrame("", width, height)
	}

	header := []string{
		chatPickerTitleStyle.Render("Prompt"),
		chatPickerMetaStyle.Render("full request preview • /save <filename> saved as <filename>.json"),
		"",
	}
	footer := chatPickerMetaStyle.Render("↑/↓ scroll • pgup/pgdn page • enter/esc close")
	bodyHeight := max(0, height-len(header)-1)
	body := m.promptDebugLines(width)
	maxScroll := max(0, len(body)-bodyHeight)
	scroll := clamp(m.promptDebug.scroll, 0, maxScroll)
	if bodyHeight < len(body) {
		body = body[scroll:min(len(body), scroll+bodyHeight)]
	}

	lines := slices.Clone(header)
	lines = append(lines, body...)
	for len(lines) < height-1 {
		lines = append(lines, "")
	}
	lines = append(lines, footer)
	return renderFrameLines(lines, width, height)
}

func (m chatModel) promptDebugPageSize() int {
	height := m.height
	if height <= 0 {
		height = 24
	}
	return max(1, height-5)
}

func (m chatModel) promptDebugMaxScroll() int {
	if m.promptDebug == nil {
		return 0
	}
	width := m.viewWidth()
	height := m.height
	if height <= 0 {
		height = 24
	}
	bodyHeight := max(0, height-4)
	return max(0, len(m.promptDebugLines(width))-bodyHeight)
}

func (m *chatModel) addDebugError(err error) (tea.Model, tea.Cmd) {
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
	m.status = "error"
	return *m, nil
}

func (m chatModel) rawRequestJSON() (string, error) {
	req, _ := m.requestPreview()
	data, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (m chatModel) requestPreview() (api.ChatRequest, int) {
	opts := m.previewRunOptions()
	messages := m.previewMessages()
	req := m.previewChatRequest(opts, messages)
	return req, m.estimatePromptTokens(messages, opts.SystemPrompt)
}

func (m chatModel) previewRunOptions() coreagent.RunOptions {
	return coreagent.RunOptions{
		ChatID:       m.chatID,
		Model:        m.opts.Model,
		SystemPrompt: m.systemPrompt(""),
		Format:       m.opts.Format,
		Options:      m.opts.Options,
		Think:        m.opts.Think,
		KeepAlive:    m.opts.KeepAlive,
	}
}

func (m chatModel) previewMessages() []api.Message {
	if len(m.liveMessages) > 0 {
		return slices.Clone(m.liveMessages)
	}
	return slices.Clone(m.messages)
}

func (m chatModel) previewChatRequest(opts coreagent.RunOptions, messages []api.Message) api.ChatRequest {
	requestMessages := slices.Clone(messages)
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		withSystem := make([]api.Message, 0, len(requestMessages)+1)
		withSystem = append(withSystem, api.Message{Role: "system", Content: opts.SystemPrompt})
		requestMessages = append(withSystem, requestMessages...)
	}

	format := opts.Format
	if format == "json" {
		format = `"` + format + `"`
	}

	req := api.ChatRequest{
		Model:    opts.Model,
		Messages: requestMessages,
		Format:   json.RawMessage(format),
		Options:  opts.Options,
		Think:    opts.Think,
	}
	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}
	if m.opts.Tools != nil && !m.opts.ToolsDisabled {
		req.Tools = m.opts.Tools.Tools()
	}
	return req
}

func (m chatModel) promptDebugLines(width int) []string {
	if m.promptDebug == nil {
		return nil
	}
	req := m.promptDebug.request
	innerWidth := max(20, width-2)
	lines := []string{
		chatHeaderStyle.Render("Request"),
		promptDebugFieldLine("model", req.Model, innerWidth),
		promptDebugFieldLine("estimated prompt", m.promptTokenText(m.promptDebug.tokens), innerWidth),
		promptDebugFieldLine("messages", fmt.Sprint(len(req.Messages)), innerWidth),
		promptDebugFieldLine("tools", fmt.Sprint(len(req.Tools)), innerWidth),
	}
	if len(req.Format) > 0 {
		lines = append(lines, promptDebugFieldLine("format", strings.TrimSpace(string(req.Format)), innerWidth))
	}
	if req.Options != nil {
		lines = append(lines, promptDebugMapLines("options", req.Options, innerWidth)...)
	}
	if req.Think != nil {
		lines = append(lines, promptDebugBlockLines("think", req.Think.String(), innerWidth, chatHistoryTextStyle)...)
	}
	if req.KeepAlive != nil {
		lines = append(lines, promptDebugFieldLine("keep_alive", req.KeepAlive.String(), innerWidth))
	}
	lines = append(lines, "", chatHeaderStyle.Render("Messages"))
	if len(req.Messages) == 0 {
		lines = append(lines, chatMetaStyle.Render("none"))
	} else {
		for i, msg := range req.Messages {
			if i > 0 {
				lines = append(lines, "")
			}
			lines = append(lines, promptDebugMessageLines(i+1, msg, innerWidth)...)
		}
	}
	lines = append(lines, "", chatHeaderStyle.Render("Tools"))
	if len(req.Tools) == 0 {
		lines = append(lines, chatMetaStyle.Render("none"))
		return lines
	}
	for i, tool := range req.Tools {
		if i > 0 {
			lines = append(lines, "")
		}
		lines = append(lines, promptDebugToolLines(i+1, tool, innerWidth)...)
	}
	return lines
}

func promptDebugFieldLine(label, value string, width int) string {
	labelText := label + ":"
	value = strings.TrimSpace(value)
	if value == "" {
		value = "_empty_"
	}
	line := chatHistoryLabelStyle.Render(labelText) + " " + chatHistoryTextStyle.Render(value)
	return truncateRenderedLine(line, width)
}

func promptDebugMessageLines(index int, msg api.Message, width int) []string {
	role := promptMessageLabel(msg)
	header := fmt.Sprintf("%d. %s", index, role)
	lines := []string{historyRoleStyle(msg.Role).Render(header)}

	if strings.TrimSpace(msg.Thinking) != "" {
		lines = append(lines, promptDebugBlockLines("thinking", msg.Thinking, width, chatHistoryTextStyle)...)
	}
	if msg.Role != "tool" && (strings.TrimSpace(msg.Content) != "" || (msg.Role != "assistant" && len(msg.ToolCalls) == 0 && len(msg.Images) == 0 && msg.Thinking == "")) {
		lines = append(lines, promptDebugBlockLines("content", msg.Content, width, chatHistoryTextStyle)...)
	}
	if len(msg.ToolCalls) > 0 {
		for i, call := range msg.ToolCalls {
			lines = append(lines, promptDebugToolCallLines(i+1, call, width)...)
		}
	}
	if msg.Role == "tool" {
		if msg.ToolName != "" {
			lines = append(lines, "  "+chatHistoryLabelStyle.Render("tool_name:")+" "+chatHistoryTextStyle.Render(msg.ToolName))
		}
		if msg.ToolCallID != "" {
			lines = append(lines, "  "+chatHistoryLabelStyle.Render("tool_call_id:")+" "+chatHistoryTextStyle.Render(msg.ToolCallID))
		}
		lines = append(lines, promptDebugBlockLines("tool result", promptDebugToolResult(msg.Content), width, chatHistoryTextStyle)...)
	}
	if len(msg.Images) > 0 {
		lines = append(lines, "  "+chatHistoryLabelStyle.Render(fmt.Sprintf("%d image%s", len(msg.Images), pluralSuffix(len(msg.Images)))))
	}
	return lines
}

func promptDebugToolResult(content string) string {
	runes := []rune(content)
	if len(runes) <= maxPromptDebugToolResultRunes {
		return content
	}
	return string(runes[:maxPromptDebugToolResultRunes-3]) + "..."
}

func promptDebugMapLines(label string, values map[string]any, width int) []string {
	lines := []string{"  " + chatHistoryLabelStyle.Render(label+":")}
	if len(values) == 0 {
		return append(lines, "    "+chatMetaStyle.Render("_empty_"))
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		lines = append(lines, promptDebugValueLine(4, key, values[key], width)...)
	}
	return lines
}

func promptDebugToolLines(index int, tool api.Tool, width int) []string {
	name := strings.TrimSpace(tool.Function.Name)
	if name == "" {
		name = "_unnamed_"
	}
	lines := []string{historyRoleStyle("tool").Render(fmt.Sprintf("%d. %s", index, name))}
	if strings.TrimSpace(tool.Function.Description) != "" {
		lines = append(lines, promptDebugBlockLines("description", tool.Function.Description, width, chatHistoryTextStyle)...)
	}

	params := tool.Function.Parameters
	if params.Type != "" || params.Properties != nil {
		kind := params.Type
		if kind == "" {
			kind = "object"
		}
		lines = append(lines, "  "+chatHistoryLabelStyle.Render("parameters:")+" "+chatHistoryTextStyle.Render(kind))
	}
	if params.Properties == nil || params.Properties.Len() == 0 {
		return lines
	}

	lines = append(lines, "  "+chatHistoryLabelStyle.Render("properties:"))
	required := map[string]bool{}
	for _, name := range params.Required {
		required[name] = true
	}
	for name, property := range params.Properties.All() {
		label := name
		propertyType := property.ToTypeScriptType()
		switch {
		case propertyType != "" && required[name]:
			label += " (" + propertyType + ", required)"
		case propertyType != "":
			label += " (" + propertyType + ")"
		case required[name]:
			label += " (required)"
		}
		value := strings.TrimSpace(property.Description)
		if value == "" {
			value = promptDebugPropertyDetails(property)
		}
		lines = append(lines, promptDebugTextLine(4, label, value, width)...)
	}
	return lines
}

func promptDebugToolCallLines(index int, call api.ToolCall, width int) []string {
	name := strings.TrimSpace(call.Function.Name)
	if name == "" {
		name = "_unnamed_"
	}
	lines := []string{"  " + chatHistoryLabelStyle.Render(fmt.Sprintf("tool call %d:", index)) + " " + chatHistoryTextStyle.Render(name)}
	if strings.TrimSpace(call.ID) != "" {
		lines = append(lines, promptDebugTextLine(4, "id", call.ID, width)...)
	}
	if call.Function.Arguments.Len() == 0 {
		lines = append(lines, "    "+chatHistoryLabelStyle.Render("arguments:")+" "+chatMetaStyle.Render("none"))
		return lines
	}
	lines = append(lines, "    "+chatHistoryLabelStyle.Render("arguments:"))
	for key, value := range call.Function.Arguments.All() {
		lines = append(lines, promptDebugValueLine(6, key, value, width)...)
	}
	return lines
}

func promptDebugPropertyDetails(property api.ToolProperty) string {
	var parts []string
	if len(property.Enum) > 0 {
		values := make([]string, 0, len(property.Enum))
		for _, value := range property.Enum {
			values = append(values, promptDebugValueText(value))
		}
		parts = append(parts, "one of "+strings.Join(values, ", "))
	}
	if property.Properties != nil && property.Properties.Len() > 0 {
		count := property.Properties.Len()
		noun := "property"
		if count != 1 {
			noun = "properties"
		}
		parts = append(parts, fmt.Sprintf("%d nested %s", count, noun))
	}
	if property.Items != nil {
		parts = append(parts, "array items: "+promptDebugValueText(property.Items))
	}
	if len(parts) == 0 {
		return "_empty_"
	}
	return strings.Join(parts, "; ")
}

func promptDebugValueLine(indent int, label string, value any, width int) []string {
	return promptDebugTextLine(indent, label, promptDebugValueText(value), width)
}

func promptDebugTextLine(indent int, label, value string, width int) []string {
	prefix := strings.Repeat(" ", indent) + chatHistoryLabelStyle.Render(label+":")
	value = strings.TrimSpace(value)
	if value == "" {
		value = "_empty_"
	}
	wrapWidth := max(20, width-indent-lipgloss.Width(label)-2)
	wrapped := wrapChatText(value, wrapWidth)
	if len(wrapped) == 0 {
		return []string{prefix + " " + chatMetaStyle.Render("_empty_")}
	}
	lines := []string{prefix + " " + chatHistoryTextStyle.Render(wrapped[0])}
	for _, line := range wrapped[1:] {
		lines = append(lines, strings.Repeat(" ", indent+2)+chatHistoryTextStyle.Render(line))
	}
	return lines
}

func promptDebugValueText(value any) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			parts = append(parts, promptDebugValueText(item))
		}
		return strings.Join(parts, ", ")
	case map[string]any:
		keys := make([]string, 0, len(v))
		for key := range v {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			parts = append(parts, key+": "+promptDebugValueText(v[key]))
		}
		return strings.Join(parts, ", ")
	default:
		return fmt.Sprint(value)
	}
}

func promptDebugBlockLines(label, value string, width int, style lipgloss.Style) []string {
	lines := []string{"  " + chatHistoryLabelStyle.Render(label+":")}
	if value == "" {
		return append(lines, "    "+chatMetaStyle.Render("_empty_"))
	}
	for _, raw := range strings.Split(strings.TrimRight(value, "\n"), "\n") {
		if raw == "" {
			lines = append(lines, "")
			continue
		}
		for _, wrapped := range wrapChatText(raw, max(20, width-4)) {
			lines = append(lines, "    "+style.Render(wrapped))
		}
	}
	return lines
}

func (m chatModel) promptTokenText(tokens int) string {
	window := m.displayContextWindowTokens()
	if window > 0 {
		return fmt.Sprintf("%d / %d tokens", max(tokens, 0), window)
	}
	return formatTokenCount(tokens)
}

func promptMessageLabel(msg api.Message) string {
	if msg.Role == "tool" && msg.ToolName != "" {
		return msg.Role + ":" + msg.ToolName
	}
	return msg.Role
}

func saveRequestFilename(args string) (string, error) {
	args = strings.TrimSpace(args)
	if args == "" {
		return "", fmt.Errorf("usage: /save <filename>")
	}
	if strings.HasPrefix(args, ">") {
		args = strings.TrimSpace(strings.TrimPrefix(args, ">"))
	}
	fields := strings.Fields(args)
	if len(fields) != 1 {
		return "", fmt.Errorf("usage: /save <filename>")
	}
	filename := strings.TrimSpace(fields[0])
	if filename == "" || filename == "." || filename == ".." || strings.ContainsAny(filename, `/\`) || filepath.IsAbs(filename) {
		return "", fmt.Errorf("save filename must be a file name, not a path")
	}
	if !strings.HasSuffix(strings.ToLower(filename), ".json") {
		filename += ".json"
	}
	return filename, nil
}

func (m chatModel) debugWorkingDir() (string, error) {
	dir := strings.TrimSpace(m.currentWorkingDir())
	if dir != "" {
		return dir, nil
	}
	return os.Getwd()
}
