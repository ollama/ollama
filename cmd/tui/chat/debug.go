package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

type rawRequestDisplay struct {
	Model     string              `json:"model"`
	Messages  []rawRequestMessage `json:"messages"`
	Format    any                 `json:"format,omitempty"`
	KeepAlive *api.Duration       `json:"keep_alive,omitempty"`
	Tools     api.Tools           `json:"tools,omitempty"`
	Options   map[string]any      `json:"options"`
	Think     *api.ThinkValue     `json:"think,omitempty"`
}

type rawRequestMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	Thinking   string         `json:"thinking,omitempty"`
	Images     []string       `json:"images,omitempty"`
	ToolCalls  []api.ToolCall `json:"tool_calls,omitempty"`
	ToolName   string         `json:"tool_name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

func (m chatModel) requestPreview(messages []api.Message) coreagent.ChatRequestPreview {
	var tools api.Tools
	if m.opts.Tools != nil {
		tools = m.opts.Tools.Tools()
	}
	return coreagent.BuildChatRequestPreview(coreagent.RunOptions{
		Model:        m.opts.Model,
		SystemPrompt: m.systemPrompt(""),
		Format:       m.opts.Format,
		Options:      m.opts.Options,
		Think:        m.opts.Think,
		KeepAlive:    m.opts.KeepAlive,
		UseTools:     m.opts.Tools != nil,
	}, messages, tools)
}

func (m *chatModel) openRawRequestPopup() (tea.Model, tea.Cmd) {
	raw, tokens, err := m.rawRequestPreviewJSON()
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	m.historyPopup = &chatHistoryPopup{
		title:         "Raw request",
		empty:         "No request preview.",
		header:        m.promptTokenHeader(tokens),
		raw:           raw,
		stickToBottom: false,
	}
	m.status = "raw"
	return *m, tea.EnterAltScreen
}

func (m *chatModel) handleRawCommand(args string) (tea.Model, tea.Cmd) {
	if strings.TrimSpace(args) == "" {
		return m.openRawRequestPopup()
	}
	filename, err := rawRedirectFilename(args)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	return m.saveRawRequest(filename)
}

func (m *chatModel) saveRawRequest(filename string) (tea.Model, tea.Cmd) {
	raw, _, err := m.rawRequestPreviewJSON()
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}

	dir := m.currentWorkingDir()
	if strings.TrimSpace(dir) == "" {
		dir, err = os.Getwd()
		if err != nil {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
			m.status = "error"
			return *m, nil
		}
	}
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, []byte(raw+"\n"), 0o644); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}

	m.entries = append(m.entries, newSlashEntry(fmt.Sprintf("Saved raw request to %s", filename)))
	m.status = "raw saved"
	return *m, nil
}

func (m chatModel) rawRequestPreviewJSON() (string, int, error) {
	preview := m.requestPreview(m.messages)
	raw, err := rawRequestJSON(preview.Request)
	if err != nil {
		return "", 0, err
	}
	return raw, preview.PromptTokens, nil
}

func rawRedirectFilename(args string) (string, error) {
	args = strings.TrimSpace(args)
	if !strings.HasPrefix(args, ">") {
		return "", fmt.Errorf("usage: /raw > filename")
	}
	fields := strings.Fields(strings.TrimSpace(strings.TrimPrefix(args, ">")))
	if len(fields) != 1 {
		return "", fmt.Errorf("usage: /raw > filename")
	}
	filename := strings.TrimSpace(fields[0])
	if filename == "" || filename == "." || filename == ".." || strings.ContainsAny(filename, `/\`) {
		return "", fmt.Errorf("raw filename must be a file name, not a path")
	}
	if !strings.HasSuffix(strings.ToLower(filename), ".json") {
		filename += ".json"
	}
	return filename, nil
}

func (m chatModel) promptTokenHeader(tokens int) []string {
	window := coreagent.ResolveContextWindowTokens(m.opts.Options, m.opts.ContextWindowTokens)
	line := "estimated prompt: " + formatTokenCount(tokens)
	if window > 0 {
		line = fmt.Sprintf("estimated prompt: %d / %d tokens", max(tokens, 0), window)
	}
	return []string{chatResumeMetaStyle.Render(line)}
}

func rawRequestJSON(req api.ChatRequest) (string, error) {
	display := rawRequestDisplay{
		Model:     req.Model,
		Messages:  rawRequestMessages(req.Messages),
		KeepAlive: req.KeepAlive,
		Tools:     req.Tools,
		Options:   req.Options,
		Think:     req.Think,
	}
	if format := rawRequestFormat(req.Format); format != nil {
		display.Format = format
	}
	data, err := json.MarshalIndent(display, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func rawRequestFormat(format json.RawMessage) any {
	if len(format) == 0 {
		return nil
	}
	var value any
	if err := json.Unmarshal(format, &value); err == nil {
		return value
	}
	return string(format)
}

func rawRequestMessages(messages []api.Message) []rawRequestMessage {
	out := make([]rawRequestMessage, 0, len(messages))
	for _, msg := range messages {
		out = append(out, rawRequestMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			Thinking:   msg.Thinking,
			Images:     rawImagePlaceholders(msg.Images),
			ToolCalls:  msg.ToolCalls,
			ToolName:   msg.ToolName,
			ToolCallID: msg.ToolCallID,
		})
	}
	return out
}

func rawImagePlaceholders(images []api.ImageData) []string {
	if len(images) == 0 {
		return nil
	}
	out := make([]string, 0, len(images))
	for _, image := range images {
		out = append(out, fmt.Sprintf("[image data: %d bytes]", len(image)))
	}
	return out
}

func renderRawRequestLines(raw string, width int) []string {
	raw = strings.TrimRight(raw, "\n")
	if raw == "" {
		return nil
	}
	var lines []string
	for _, line := range strings.Split(raw, "\n") {
		lines = append(lines, renderHistoryCodeLine("", line, width)...)
	}
	return stringsTrimTrailingEmptyLines(lines)
}
