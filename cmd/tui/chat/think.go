package chat

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ollama/ollama/api"
)

type chatThinkOption struct {
	value       string
	label       string
	description string
}

type chatThinkPicker struct {
	options []chatThinkOption
	cursor  int
}

var chatThinkOptions = []chatThinkOption{
	{value: "auto", label: "auto", description: "use the model default"},
	{value: "on", label: "on", description: "enable thinking"},
	{value: "off", label: "off", description: "disable thinking"},
	{value: "low", label: "low", description: "use low thinking effort"},
	{value: "medium", label: "medium", description: "use medium thinking effort"},
	{value: "high", label: "high", description: "use high thinking effort"},
	{value: "max", label: "max", description: "use maximum thinking effort"},
}

func (m *chatModel) openThinkPicker() (tea.Model, tea.Cmd) {
	m.thinkPicker = newChatThinkPicker(m.opts.Think)
	m.status = "think"
	return *m, nil
}

func newChatThinkPicker(current *api.ThinkValue) *chatThinkPicker {
	picker := &chatThinkPicker{options: append([]chatThinkOption(nil), chatThinkOptions...)}
	currentValue := thinkValueLabel(current)
	for i, option := range picker.options {
		if option.value == currentValue {
			picker.cursor = i
			break
		}
	}
	return picker
}

func (m chatModel) updateThinkPicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.thinkPicker = nil
		m.status = "ready"
	case tea.KeyEnter:
		return m.selectThinkOption()
	case tea.KeyUp:
		m.thinkPicker.move(-1)
	case tea.KeyDown:
		m.thinkPicker.move(1)
	}
	return m, nil
}

func (p *chatThinkPicker) move(delta int) {
	if p == nil || len(p.options) == 0 || delta == 0 {
		return
	}
	p.cursor = clamp(p.cursor+delta, 0, len(p.options)-1)
}

func (p *chatThinkPicker) selected() (chatThinkOption, bool) {
	if p == nil || len(p.options) == 0 {
		return chatThinkOption{}, false
	}
	return p.options[clamp(p.cursor, 0, len(p.options)-1)], true
}

func (m chatModel) selectThinkOption() (tea.Model, tea.Cmd) {
	option, ok := m.thinkPicker.selected()
	if !ok {
		return m, nil
	}
	m.thinkPicker = nil
	return m.applyThinkValue(option.value)
}

func (m *chatModel) handleThinkCommand(value string) (tea.Model, tea.Cmd) {
	return m.applyThinkValue(value)
}

func (m *chatModel) handleLegacySetThinkCommand(input string) (tea.Model, tea.Cmd) {
	value := strings.TrimSpace(strings.TrimPrefix(input, "/set think"))
	if value == "" {
		value = "on"
	}
	return m.applyThinkValue(value)
}

func (m *chatModel) applyThinkValue(value string) (tea.Model, tea.Cmd) {
	think, label, err := parseThinkValue(value)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	m.opts.Think = think
	m.status = "think " + label
	return *m, nil
}

func parseThinkValue(value string) (*api.ThinkValue, string, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "auto", "default", "unset":
		return nil, "auto", nil
	case "on", "true", "think", "thinking":
		return &api.ThinkValue{Value: true}, "on", nil
	case "off", "false", "nothink", "no-think":
		return &api.ThinkValue{Value: false}, "off", nil
	case "low", "medium", "high", "max":
		value = strings.ToLower(strings.TrimSpace(value))
		return &api.ThinkValue{Value: value}, value, nil
	default:
		return nil, "", fmt.Errorf("Usage: /think [auto|on|off|low|medium|high|max]")
	}
}

func thinkValueLabel(value *api.ThinkValue) string {
	if value == nil || value.Value == nil {
		return "auto"
	}
	switch v := value.Value.(type) {
	case bool:
		if v {
			return "on"
		}
		return "off"
	case string:
		return strings.ToLower(v)
	default:
		return "auto"
	}
}

func (m chatModel) renderThinkPicker(width int) string {
	picker := m.thinkPicker
	if picker == nil {
		return ""
	}

	var b strings.Builder
	b.WriteString(chatPickerTitleStyle.Render("Thinking mode"))
	b.WriteString("\n\n")
	for i, option := range picker.options {
		selected := i == picker.cursor
		if selected {
			b.WriteString(chatPickerSelectedStyle.Render("› " + option.label))
		} else {
			b.WriteString("  ")
			b.WriteString(chatPickerTextStyle.Render(option.label))
		}
		b.WriteByte('\n')
		b.WriteString(chatPickerMetaStyle.Render("  " + option.description))
		b.WriteByte('\n')
		if i < len(picker.options)-1 {
			b.WriteByte('\n')
		}
	}

	b.WriteString("\n")
	b.WriteString(chatPickerMetaStyle.Render("↑/↓ move • enter select • esc cancel"))
	return b.String()
}
