package tui

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type WelcomeAccount struct {
	CloudDisabled bool
	SignedIn      bool
	SigninURL     string
	Err           error
}

type WelcomeOptions struct {
	CheckAccount func() WelcomeAccount
	OpenBrowser  func(string)
	IsCompleted  func() bool
}

type welcomeStep int

const (
	welcomeIntro welcomeStep = iota
	welcomeAccount
	welcomeSignIn
)

type welcomeModel struct {
	options   WelcomeOptions
	step      welcomeStep
	checking  bool
	account   WelcomeAccount
	signIn    signInModel
	cursor    int
	continued bool
	cancelled bool
	width     int
}

type (
	welcomeCompletedMsg bool
	welcomeAccountMsg   WelcomeAccount
)

func (m welcomeModel) Init() tea.Cmd {
	if m.options.IsCompleted == nil {
		return nil
	}
	return tea.Tick(time.Second, func(time.Time) tea.Msg {
		return welcomeCompletedMsg(m.options.IsCompleted())
	})
}

func (m welcomeModel) checkAccount() (tea.Model, tea.Cmd) {
	if m.checking {
		return m, nil
	}
	m.checking = true
	return m, func() tea.Msg {
		if m.options.CheckAccount == nil {
			return welcomeAccountMsg{Err: fmt.Errorf("account check unavailable")}
		}
		return welcomeAccountMsg(m.options.CheckAccount())
	}
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
	case welcomeAccountMsg:
		m.account, m.checking = WelcomeAccount(msg), false
		m.cursor = 0
		if m.account.SignedIn || m.account.CloudDisabled {
			m.continued = true
			return m, tea.Quit
		}
		m.step = welcomeAccount
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.signIn.width = msg.Width
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC || (msg.Type == tea.KeyEsc && m.step != welcomeSignIn) {
			m.cancelled = true
			return m, tea.Quit
		}
	}

	if m.step == welcomeSignIn {
		updated, cmd := m.signIn.Update(msg)
		m.signIn = updated.(signInModel)
		if m.signIn.cancelled {
			m.step = welcomeAccount
			return m, nil // Esc returns to the account choices, not out of onboarding.
		}
		if m.signIn.userName != "" {
			m.continued = true
			return m, tea.Quit
		}
		return m, cmd
	}

	if key, ok := msg.(tea.KeyMsg); ok {
		switch key.String() {
		case "up", "k":
			if m.step == welcomeAccount && !m.checking {
				m.cursor = max(0, m.cursor-1)
			}
		case "down", "j":
			if m.step == welcomeAccount && !m.checking {
				m.cursor = min(len(m.accountChoices())-1, m.cursor+1)
			}
		case "enter":
			if m.step == welcomeIntro {
				return m.checkAccount()
			}
			if m.checking {
				return m, nil
			}
			choices := m.accountChoices()
			if m.cursor == len(choices)-1 {
				m.continued = true
				return m, tea.Quit
			}
			if m.account.Err != nil || m.account.SigninURL == "" {
				return m.checkAccount()
			}
			signInURL := m.account.SigninURL
			u, err := url.Parse(signInURL)
			if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || m.options.OpenBrowser == nil {
				m.account.Err = fmt.Errorf("sign-in link unavailable")
				m.cursor = 0
				return m, nil
			}
			query := u.Query()
			query.Del("launch")
			query.Del("signup")
			u.RawQuery = query.Encode()
			signInURL = u.String()
			m.step = welcomeSignIn
			m.signIn = signInModel{modelName: "Ollama Cloud", signInURL: signInURL, width: m.width}
			return m, tea.Batch(
				func() tea.Msg { m.options.OpenBrowser(signInURL); return nil },
				m.signIn.Init(),
			)
		}
	}
	return m, nil
}

func (m welcomeModel) View() string {
	if m.continued || m.cancelled {
		return ""
	}
	content := m.introView()
	if m.step == welcomeAccount || m.step == welcomeSignIn {
		content = m.accountView()
	}
	style := lipgloss.NewStyle().Padding(1, 2)
	if m.width > 0 {
		style = style.Width(min(m.width, 80))
	}
	return style.Render(content)
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
	if m.checking {
		s.WriteString(selectorDescStyle.Render("Checking your account…"))
	} else {
		s.WriteString(selectorTitleStyle.Render("Press Enter to continue"))
	}
	return s.String()
}

func (m welcomeModel) accountChoices() []string {
	if m.account.Err != nil || m.account.SigninURL == "" {
		return []string{"Try again", "No thanks, I'll use Ollama locally"}
	}
	return []string{"Sign up / sign in", "No thanks, I'll use Ollama locally"}
}

func (m welcomeModel) accountView() string {
	var s strings.Builder
	s.WriteString(selectorTitleStyle.Render("Create an account"))
	s.WriteString("\n\nCreate your account for access to faster, larger open models.\n")
	s.WriteString("Your data is never logged or trained on.\n\n")
	if m.step == welcomeSignIn {
		s.WriteString(selectorDescStyle.Render("Finish in your browser…"))
		s.WriteString("\n\n" + m.signIn.signInURL)
		s.WriteString("\n\n" + selectorHelpStyle.Render("esc back"))
		return s.String()
	}
	if m.checking {
		s.WriteString("Checking your account…\n\n")
		s.WriteString(selectorHelpStyle.Render("esc quit"))
		return s.String()
	}
	if m.account.Err != nil || m.account.SigninURL == "" {
		s.WriteString("Unable to check your account. Please try again.\n\n")
	}
	for i, choice := range m.accountChoices() {
		if i == m.cursor {
			s.WriteString(menuSelectedItemStyle.Render("▸ " + choice))
		} else {
			s.WriteString("  " + choice)
		}
		s.WriteString("\n")
	}
	s.WriteString("\n\n" + selectorHelpStyle.Render("↑/↓ navigate • enter select • esc quit"))
	return s.String()
}

// RunWelcome introduces Ollama, then offers account setup if needed. Returning
// successfully leads to the regular launcher (or an explicitly requested app).
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
