package tui

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type vscodeTickMsg struct{}

type vscodeCheckMsg struct {
	done bool
}

type vscodeModel struct {
	spinner   int
	done      bool
	timedOut  bool
	cancelled bool
	checkDone func() bool
}

func (m vscodeModel) Init() tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return vscodeTickMsg{}
	})
}

func (m vscodeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			m.cancelled = true
			return m, tea.Quit
		}

	case vscodeTickMsg:
		m.spinner++
		if m.spinner >= 150 { // 150 × 200ms = 30s timeout
			m.timedOut = true
			return m, tea.Quit
		}
		if m.spinner%5 == 0 { // check every ~1s
			return m, tea.Batch(
				tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
					return vscodeTickMsg{}
				}),
				m.checkDoneCmd(),
			)
		}
		return m, tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
			return vscodeTickMsg{}
		})

	case vscodeCheckMsg:
		if msg.done {
			m.done = true
			return m, tea.Quit
		}
	}

	return m, nil
}

func (m vscodeModel) checkDoneCmd() tea.Cmd {
	checkDone := m.checkDone
	return func() tea.Msg {
		return vscodeCheckMsg{done: checkDone()}
	}
}

func (m vscodeModel) View() string {
	if m.done || m.timedOut {
		return ""
	}
	return renderVSCodeRestart(m.spinner)
}

func renderVSCodeRestart(spinner int) string {
	spinnerFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := spinnerFrames[spinner%len(spinnerFrames)]

	return lipgloss.NewStyle().
		Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
		Render(fmt.Sprintf("%s Restarting VS Code...", frame))
}

// RunVSCodeRestart shows a bubble tea spinner while waiting for VS Code to
// exit. It polls checkDone every ~1s and times out after 30s. Returns
// ErrCancelled if the user presses Esc/Ctrl+C.
func RunVSCodeRestart(checkDone func() bool) error {
	m := vscodeModel{
		checkDone: checkDone,
	}

	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return fmt.Errorf("error running vscode restart: %w", err)
	}

	fm := finalModel.(vscodeModel)
	if fm.cancelled {
		return ErrCancelled
	}

	return nil
}
