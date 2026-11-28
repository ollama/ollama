package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/launch"
)

type claudeDesktopMappingModel struct {
	rows      []launch.ClaudeDesktopMapping
	cursor    int
	width     int
	choice    launch.ClaudeDesktopMappingChoice
	done      bool
	cancelled bool
}

func newClaudeDesktopMappingModel(rows []launch.ClaudeDesktopMapping, current string) claudeDesktopMappingModel {
	m := claudeDesktopMappingModel{rows: rows}
	for i, row := range rows {
		if row.ID == current {
			m.cursor = i
			break
		}
	}
	return m
}

func (m claudeDesktopMappingModel) canSave() bool {
	for _, row := range m.rows {
		if strings.TrimSpace(row.Model) != "" || strings.TrimSpace(row.DefaultModel) != "" {
			return true
		}
	}
	return false
}

func (m claudeDesktopMappingModel) Init() tea.Cmd { return nil }

func (m claudeDesktopMappingModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.done || m.cancelled {
		return m, nil
	}
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc, tea.KeyLeft:
			m.cancelled = true
			return m, tea.Quit
		case tea.KeyUp:
			m.cursor = max(0, m.cursor-1)
		case tea.KeyDown:
			m.cursor = min(max(0, len(m.rows)-1), m.cursor+1)
		case tea.KeyBackspace, tea.KeyDelete:
			if m.cursor < len(m.rows) && m.rows[m.cursor].Model != "" {
				m.choice = launch.ClaudeDesktopMappingChoice{RouteID: m.rows[m.cursor].ID, Clear: true}
				m.done = true
				return m, tea.Quit
			}
		case tea.KeyRight:
			if m.cursor < len(m.rows) {
				m.choice.RouteID = m.rows[m.cursor].ID
				m.done = true
				return m, tea.Quit
			}
		case tea.KeyEnter:
			if !m.canSave() {
				return m, nil
			}
			m.done = true
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m claudeDesktopMappingModel) View() string {
	if m.done || m.cancelled {
		return ""
	}
	var s strings.Builder
	s.WriteString(selectorTitleStyle.Render("Claude Desktop model mappings"))
	s.WriteString("\n\n")
	var labels []string
	for _, row := range m.rows {
		model := row.Model
		if row.DefaultModel != "" && (model == "" || model == row.DefaultModel) {
			model = row.DefaultModel + " (Default)"
		} else if model == "" {
			model = "Default unavailable"
		}
		labels = append(labels, fmt.Sprintf("%-12s %s", row.Name, model))
	}
	for i, label := range labels {
		if i == m.cursor {
			s.WriteString(selectorSelectedItemStyle.Render("▸ " + label))
		} else {
			s.WriteString(selectorItemStyle.Render(label))
		}
		s.WriteString("\n")
	}
	if !m.canSave() {
		s.WriteString("\n" + selectorDescStyle.Render("Map at least one model before saving.") + "\n")
	}
	help := "↑/↓ navigate • → change model • backspace use default\nenter save and launch • esc cancel"
	s.WriteString("\n" + selectorHelpStyle.Render(help))
	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s.String())
	}
	return s.String()
}

func SelectClaudeDesktopMapping(rows []launch.ClaudeDesktopMapping, current string) (launch.ClaudeDesktopMappingChoice, error) {
	final, err := tea.NewProgram(newClaudeDesktopMappingModel(rows, current)).Run()
	if err != nil {
		return launch.ClaudeDesktopMappingChoice{}, fmt.Errorf("edit Claude Desktop mappings: %w", err)
	}
	m := final.(claudeDesktopMappingModel)
	if m.cancelled || !m.done {
		return launch.ClaudeDesktopMappingChoice{}, ErrCancelled
	}
	return m.choice, nil
}
