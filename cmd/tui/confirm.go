package tui

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	confirmActiveStyle = lipgloss.NewStyle().
				Bold(true).
				Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	confirmInactiveStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})
)

type confirmModel struct {
	prompt    string
	yes       bool
	confirmed bool
	cancelled bool
	width     int
}

func (m confirmModel) Init() tea.Cmd {
	return nil
}

func (m confirmModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		wasSet := m.width > 0
		m.width = msg.Width
		if wasSet {
			return m, tea.EnterAltScreen
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "esc", "n":
			m.cancelled = true
			return m, tea.Quit
		case "y":
			m.yes = true
			m.confirmed = true
			return m, tea.Quit
		case "enter":
			m.confirmed = true
			return m, tea.Quit
		case "left", "h":
			m.yes = true
		case "right", "l":
			m.yes = false
		case "tab":
			m.yes = !m.yes
		}
	}

	return m, nil
}

func (m confirmModel) View() string {
	if m.confirmed || m.cancelled {
		return ""
	}

	var yesBtn, noBtn string
	if m.yes {
		yesBtn = confirmActiveStyle.Render(" Yes ")
		noBtn = confirmInactiveStyle.Render(" No ")
	} else {
		yesBtn = confirmInactiveStyle.Render(" Yes ")
		noBtn = confirmActiveStyle.Render(" No ")
	}

	s := selectorTitleStyle.Render(m.prompt) + "\n\n"
	s += "  " + yesBtn + "  " + noBtn + "\n\n"
	s += selectorHelpStyle.Render("←/→ navigate • enter confirm • esc cancel")

	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s)
	}
	return s
}

// RunConfirm shows a bubbletea yes/no confirmation prompt.
// Returns true if the user confirmed, false if cancelled.
func RunConfirm(prompt string) (bool, error) {
	m := confirmModel{
		prompt: prompt,
		yes:    true, // default to yes
	}

	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return false, fmt.Errorf("error running confirm: %w", err)
	}

	fm := finalModel.(confirmModel)
	if fm.cancelled {
		return false, ErrCancelled
	}

	return fm.yes, nil
}
