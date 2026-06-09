package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
)

type signInTickMsg struct{}

type signInCheckMsg struct {
	signedIn bool
	userName string
}

type upgradeTickMsg struct{}

type upgradeCheckMsg struct {
	upgraded bool
	plan     string
	err      error
}

type signInModel struct {
	modelName string
	signInURL string
	spinner   int
	width     int
	userName  string
	cancelled bool
}

type upgradeModel struct {
	modelName    string
	requiredPlan string
	spinner      int
	width        int
	openNow      bool
	polling      bool
	plan         string
	cancelled    bool
	err          error
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

func (m upgradeModel) Init() tea.Cmd {
	if m.polling {
		return upgradeTickCmd()
	}
	return nil
}

func (m upgradeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
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
		case tea.KeyLeft:
			if !m.polling {
				m.openNow = true
			}
		case tea.KeyRight:
			if !m.polling {
				m.openNow = false
			}
		case tea.KeyEnter:
			if !m.polling {
				if !m.openNow {
					m.cancelled = true
					return m, tea.Quit
				}
				launch.OpenBrowser(launch.DefaultUpgradeURL)
				m.polling = true
				return m, upgradeTickCmd()
			}
		}

	case upgradeTickMsg:
		if !m.polling {
			return m, nil
		}
		m.spinner++
		if m.spinner%5 == 0 {
			return m, tea.Batch(
				upgradeTickCmd(),
				checkUpgrade(m.requiredPlan),
			)
		}
		return m, upgradeTickCmd()

	case upgradeCheckMsg:
		if msg.err != nil {
			m.err = msg.err
			return m, tea.Quit
		}
		if msg.upgraded {
			m.plan = msg.plan
			return m, tea.Quit
		}
	}

	return m, nil
}

func (m upgradeModel) View() string {
	if m.plan != "" {
		return ""
	}
	if m.err != nil {
		return ""
	}
	return renderUpgrade(m.modelName, m.spinner, m.width, m.polling, m.openNow)
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

	s.WriteString("Navigate to:\n")
	s.WriteString(urlWrap.Render(urlColor.Render(signInURL)))
	s.WriteString("\n\n")

	s.WriteString(lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).Render(
		frame + " Waiting for sign in to complete..."))
	s.WriteString("\n\n")

	s.WriteString(selectorHelpStyle.Render("esc cancel"))

	return lipgloss.NewStyle().PaddingLeft(2).Render(s.String())
}

func upgradeTickCmd() tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return upgradeTickMsg{}
	})
}

func renderUpgrade(modelName string, spinner, width int, polling, openNow bool) string {
	spinnerFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := spinnerFrames[spinner%len(spinnerFrames)]

	urlColor := lipgloss.NewStyle().
		Foreground(lipgloss.Color("117"))
	urlWrap := lipgloss.NewStyle().PaddingLeft(2)
	if width > 4 {
		urlWrap = urlWrap.Width(width - 4)
	}

	var s strings.Builder

	fmt.Fprintf(&s, "To use %s, upgrade your Ollama plan.\n\n", selectorSelectedItemStyle.Render(modelName))

	s.WriteString("Navigate to:\n")
	s.WriteString(urlWrap.Render(urlColor.Render(launch.DefaultUpgradeURL)))
	s.WriteString("\n\n")

	if !polling {
		var yesBtn, noBtn string
		if openNow {
			yesBtn = confirmActiveStyle.Render(" Yes ")
			noBtn = confirmInactiveStyle.Render(" No ")
		} else {
			yesBtn = confirmInactiveStyle.Render(" Yes ")
			noBtn = confirmActiveStyle.Render(" No ")
		}

		s.WriteString("Open now?\n")
		s.WriteString("  " + yesBtn + "  " + noBtn)
		s.WriteString("\n\n")
		s.WriteString(selectorHelpStyle.Render("←/→ navigate • enter confirm • esc cancel"))
	} else {
		s.WriteString(lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).Render(
			frame + " Waiting for upgrade to complete..."))
		s.WriteString("\n\n")
		s.WriteString(selectorHelpStyle.Render("esc cancel"))
	}

	return lipgloss.NewStyle().PaddingLeft(2).Render(s.String())
}

func checkSignIn() tea.Msg {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return signInCheckMsg{signedIn: false}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return signInCheckMsg{signedIn: true, userName: user.Name}
	}
	return signInCheckMsg{signedIn: false}
}

func checkUpgrade(requiredPlan string) tea.Cmd {
	return func() tea.Msg {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			return upgradeCheckMsg{err: launch.ErrPlanVerificationUnavailable}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		user, err := client.Whoami(ctx)
		if err != nil {
			return upgradeCheckMsg{err: launch.ErrPlanVerificationUnavailable}
		}
		if err == nil && user != nil && user.Name != "" && launch.PlanSatisfies(user.Plan, requiredPlan) {
			return upgradeCheckMsg{upgraded: true, plan: user.Plan}
		}
		return upgradeCheckMsg{upgraded: false}
	}
}

// RunSignIn shows a bubbletea sign-in dialog and polls until the user signs in or cancels.
func RunSignIn(modelName, signInURL string) (string, error) {
	launch.OpenBrowser(signInURL)

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

// RunUpgrade shows a bubbletea upgrade dialog and polls until the user's plan is updated or cancelled.
func RunUpgrade(modelName, requiredPlan string) (string, error) {
	m := upgradeModel{
		modelName:    modelName,
		requiredPlan: requiredPlan,
		openNow:      true,
	}

	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return "", fmt.Errorf("error running upgrade: %w", err)
	}

	fm := finalModel.(upgradeModel)
	if fm.cancelled {
		return "", ErrCancelled
	}
	if fm.err != nil {
		return "", fm.err
	}

	return fm.plan, nil
}
