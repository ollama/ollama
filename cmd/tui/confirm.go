package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/launch"
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
	yesLabel  string
	noLabel   string
	yes       bool
	confirmed bool
	cancelled bool
	width     int
	plain     bool
}

type ConfirmOptions = launch.ConfirmOptions

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
		case "ctrl+c", "esc":
			m.cancelled = true
			return m, tea.Quit
		case "enter":
			m.confirmed = true
			return m, tea.Quit
		case "left":
			m.yes = true
		case "right":
			m.yes = false
		}
	}

	return m, nil
}

func (m confirmModel) View() string {
	if m.confirmed || m.cancelled {
		return ""
	}

	var yesBtn, noBtn string
	yesLabel := m.yesLabel
	if yesLabel == "" {
		yesLabel = "Yes"
	}
	noLabel := m.noLabel
	if noLabel == "" {
		noLabel = "No"
	}
	if m.yes {
		yesBtn = confirmActiveStyle.Render(" " + yesLabel + " ")
		noBtn = confirmInactiveStyle.Render(" " + noLabel + " ")
	} else {
		yesBtn = confirmInactiveStyle.Render(" " + yesLabel + " ")
		noBtn = confirmActiveStyle.Render(" " + noLabel + " ")
	}

	prompt := renderConfirmPrompt(m.prompt, m.width, m.plain)
	s := prompt + "\n\n"
	s += "  " + yesBtn + "  " + noBtn + "\n\n"
	s += renderConfirmHelp(m.width)

	return s
}

func renderConfirmPrompt(prompt string, width int, plain bool) string {
	lines := []string{prompt}
	if width > 0 {
		lines = wrapChatText(prompt, width)
	}
	if plain {
		return strings.Join(lines, "\n")
	}
	for i, line := range lines {
		lines[i] = selectorTitleStyle.Render(line)
	}
	return strings.Join(lines, "\n")
}

func renderConfirmHelp(width int) string {
	help := "←/→ navigate • enter confirm • esc cancel"
	lines := []string{help}
	if width > 0 {
		lines = wrapChatText(help, width)
	}
	for i, line := range lines {
		lines[i] = selectorHelpStyle.Render(line)
	}
	return strings.Join(lines, "\n")
}

// RunConfirm shows a bubbletea yes/no confirmation prompt.
// Returns true if the user confirmed, false if cancelled.
func RunConfirm(prompt string) (bool, error) {
	return RunConfirmWithOptions(prompt, ConfirmOptions{})
}

// RunConfirmWithOptions shows a bubbletea yes/no confirmation prompt with
// optional custom button labels.
func RunConfirmWithOptions(prompt string, options ConfirmOptions) (bool, error) {
	yesLabel := options.YesLabel
	if yesLabel == "" {
		yesLabel = "Yes"
	}
	noLabel := options.NoLabel
	if noLabel == "" {
		noLabel = "No"
	}

	m := confirmModel{
		prompt:   prompt,
		yesLabel: yesLabel,
		noLabel:  noLabel,
		yes:      true, // default to yes
		plain:    options.PlainPrompt,
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
