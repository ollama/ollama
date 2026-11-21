package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type WelcomeOptions struct {
	IsCompleted func() bool
}

type welcomeModel struct {
	options   WelcomeOptions
	continued bool
	cancelled bool
	width     int
}

type welcomeCompletedMsg bool

func (m welcomeModel) Init() tea.Cmd {
	if m.options.IsCompleted == nil {
		return nil
	}
	return tea.Tick(time.Second, func(time.Time) tea.Msg {
		return welcomeCompletedMsg(m.options.IsCompleted())
	})
}

func (m welcomeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.continued || m.cancelled {
		return m, nil
	}
	switch msg := msg.(type) {
	case welcomeCompletedMsg:
		if msg {
			m.continued = true
			return m, tea.Quit
		}
		return m, m.Init()
	case tea.WindowSizeMsg:
		m.width = msg.Width
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyEnter:
			m.continued = true
			return m, tea.Quit
		case tea.KeyCtrlC, tea.KeyEsc:
			m.cancelled = true
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m welcomeModel) View() string {
	if m.continued || m.cancelled {
		return ""
	}
	style := lipgloss.NewStyle().Padding(1, 2)
	if m.width > 0 {
		style = style.Width(min(m.width, 80))
	}
	return style.Render(m.introView())
}

func (m welcomeModel) introView() string {
	var s strings.Builder
	s.WriteString(selectorTitleStyle.Render("Welcome to Ollama!"))
	s.WriteString("\n\nRun open models with your coding agents so you can spend less\nwhile keeping your data private.\n\n")
	s.WriteString(selectorTitleStyle.Render("Connect your apps"))
	s.WriteString("\nPower your existing coding apps with open models\n\n")
	s.WriteString(selectorTitleStyle.Render("Easily switch models"))
	s.WriteString("\nSwap between frontier models in one click.\n\n")
	s.WriteString(selectorTitleStyle.Render("Your data stays yours"))
	s.WriteString("\nYour prompt data is never logged or trained on.\n\n")
	s.WriteString(selectorTitleStyle.Render("Press Enter to continue"))
	return s.String()
}

// RunWelcome introduces Ollama. Returning successfully leads to the regular launcher.
func RunWelcome(options WelcomeOptions) error {
	finalModel, err := tea.NewProgram(welcomeModel{options: options}).Run()
	if err != nil {
		return fmt.Errorf("show welcome: %w", err)
	}
	if !finalModel.(welcomeModel).continued {
		return ErrCancelled
	}
	return nil
}
