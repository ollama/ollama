package tui

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/version"
)

type homeAction int

const (
	homeActionNone homeAction = iota
	homeActionRunModel
	homeActionLaunchAgent
)

var homeItems = []struct {
	title  string
	action homeAction
}{
	{"Run a model", homeActionRunModel},
	{"Run a model with an agent", homeActionLaunchAgent},
}

type homeModel struct {
	cursor   int
	width    int
	selected bool
	quitting bool
}

func (m homeModel) Init() tea.Cmd { return nil }

func (m homeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			m.quitting = true
			return m, tea.Quit
		case "up", "k":
			m.cursor = max(0, m.cursor-1)
		case "down", "j":
			m.cursor = min(len(homeItems)-1, m.cursor+1)
		case "enter", " ", "right", "l":
			m.selected = true
			m.quitting = true
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m homeModel) View() string {
	if m.quitting {
		return ""
	}

	s := selectorTitleStyle.Render("Ollama "+versionStyle.Render(version.Version)) + "\n\n"
	s += selectorTitleStyle.Render("What would you like to do?") + "\n\n"
	for i, item := range homeItems {
		style := menuItemStyle
		cursor := ""
		if i == m.cursor {
			style = menuSelectedItemStyle
			cursor = "▸ "
		}
		s += style.Render(cursor+item.title) + "\n"
	}
	s += "\n" + selectorHelpStyle.Render("↑/↓ navigate • enter select • esc quit")
	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s)
	}
	return s
}

// HomeMenu keeps navigation in the home screen or app list between actions.
// Its zero value starts at the home screen.
type HomeMenu struct {
	showApps bool
	cursor   int
}

func (m *HomeMenu) Run(state *launch.LauncherState) (TUIAction, error) {
	for {
		if !m.showApps {
			final, err := tea.NewProgram(homeModel{cursor: m.cursor}).Run()
			if err != nil {
				return TUIAction{}, fmt.Errorf("error running home menu: %w", err)
			}
			home := final.(homeModel)
			m.cursor = home.cursor
			if !home.selected {
				return TUIAction{}, nil
			}
			if homeItems[home.cursor].action == homeActionRunModel {
				return TUIAction{Kind: TUIActionRunModel, ForceConfigure: true}, nil
			}
			m.showApps = true
		}

		action, back, err := runMenu(state, true)
		if err != nil || !back {
			return action, err
		}
		m.showApps = false
	}
}
