package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/config"
)

type signInModel struct {
	modelName string
	signInURL string
	spinner   int
	width     int
	userName  string
	cancelled bool
}

func (m signInModel) Init() tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return signInTickMsg{}
	})
}

func (m signInModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		wasSet := m.width > 0
		m.width = msg.Width
		if wasSet {
			return m, tea.EnterAltScreen
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			m.cancelled = true
			return m, tea.Quit
		}

	case signInTickMsg:
		m.spinner++
		if m.spinner%5 == 0 {
			return m, tea.Batch(
				tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
					return signInTickMsg{}
				}),
				checkSignIn,
			)
		}
		return m, tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
			return signInTickMsg{}
		})

	case signInCheckMsg:
		if msg.signedIn {
			m.userName = msg.userName
			return m, tea.Quit
		}
	}

	return m, nil
}

func (m signInModel) View() string {
	if m.userName != "" {
		return ""
	}
	return renderSignIn(m.modelName, m.signInURL, m.spinner, m.width)
}

func renderSignIn(modelName, signInURL string, spinner, width int) string {
	spinnerFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := spinnerFrames[spinner%len(spinnerFrames)]

	urlColor := lipgloss.NewStyle().
		Foreground(lipgloss.Color("117"))
	urlWrap := lipgloss.NewStyle().PaddingLeft(2)
	if width > 4 {
		urlWrap = urlWrap.Width(width - 4)
	}

	var s strings.Builder

	fmt.Fprintf(&s, "To use %s, please sign in.\n\n", selectorSelectedItemStyle.Render(modelName))

	// Wrap in OSC 8 hyperlink so the entire URL is clickable even when wrapped.
	// Padding is outside the hyperlink so spaces don't get underlined.
	link := fmt.Sprintf("\033]8;;%s\033\\%s\033]8;;\033\\", signInURL, urlColor.Render(signInURL))
	s.WriteString("Navigate to:\n")
	s.WriteString(urlWrap.Render(link))
	s.WriteString("\n\n")

	s.WriteString(lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).Render(
		frame + " Waiting for sign in to complete..."))
	s.WriteString("\n\n")

	s.WriteString(selectorHelpStyle.Render("esc cancel"))

	return lipgloss.NewStyle().PaddingLeft(2).Render(s.String())
}

// RunSignIn shows a bubbletea sign-in dialog and polls until the user signs in or cancels.
func RunSignIn(modelName, signInURL string) (string, error) {
	config.OpenBrowser(signInURL)

	m := signInModel{
		modelName: modelName,
		signInURL: signInURL,
	}

	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return "", fmt.Errorf("error running sign-in: %w", err)
	}

	fm := finalModel.(signInModel)
	if fm.cancelled {
		return "", ErrCancelled
	}

	return fm.userName, nil
}
