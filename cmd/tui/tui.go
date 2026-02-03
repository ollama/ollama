package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/version"
)

const (
	logoNormal = ` ▆▁▂▃▂▁▆
▟███████▙
█▙▛▄ ▄▜▟█
▟█▙▀▀▀▟█▙
█████████
▟███████▙
▀▀▀▀▀▀▀▀▀`

	logoBlink = ` ▆▁▂▃▂▁▆
▟███████▙
██▛▄ ▄▜██
▟█▙▀▀▀▟█▙
█████████
▟███████▙
▀▀▀▀▀▀▀▀▀`

	blinkInterval = 15 * time.Second
	blinkDuration = 250 * time.Millisecond
)

type (
	blinkMsg   struct{}
	unblinkMsg struct{}
)

var (
	logoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("255")).
			Background(lipgloss.Color("0"))

	titleStyle = lipgloss.NewStyle().
			Bold(true).
			MarginBottom(1)

	versionStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245"))

	itemStyle = lipgloss.NewStyle().
			PaddingLeft(2)

	selectedStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Foreground(lipgloss.Color("147")).
			Bold(true)

	greyedStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Foreground(lipgloss.Color("241"))

	greyedSelectedStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Foreground(lipgloss.Color("243"))

	descStyle = lipgloss.NewStyle().
			PaddingLeft(4).
			Foreground(lipgloss.Color("241"))

	modelStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245"))

	notInstalledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("241")).
				Italic(true)
)

type menuItem struct {
	title       string
	description string
	integration string // integration name for loading model config, empty if not an integration
	isRunModel  bool   // true for the "Run a model" option
	isOthers    bool   // true for the "Others..." toggle item
}

var mainMenuItems = []menuItem{
	{
		title:       "Run a model",
		description: "Start an interactive chat with a local model",
		isRunModel:  true,
	},
	{
		title:       "Launch Claude Code",
		description: "Open Claude Code AI assistant",
		integration: "claude",
	},
	{
		title:       "Launch Open Claw",
		description: "Open the Open Claw integration",
		integration: "openclaw",
	},
}

var othersMenuItem = menuItem{
	title:       "Others...",
	description: "Show additional integrations",
	isOthers:    true,
}

// getOtherIntegrations returns the list of other integrations, filtering out
// Codex if it's not installed (since it requires npm install).
func getOtherIntegrations() []menuItem {
	items := []menuItem{
		{
			title:       "Launch Droid",
			description: "Open Droid integration",
			integration: "droid",
		},
		{
			title:       "Launch Open Code",
			description: "Open Open Code integration",
			integration: "opencode",
		},
	}

	// Only show Codex if it's already installed
	if config.IsIntegrationInstalled("codex") {
		items = append([]menuItem{{
			title:       "Launch Codex",
			description: "Open Codex CLI",
			integration: "codex",
		}}, items...)
	}

	return items
}

type model struct {
	items           []menuItem
	cursor          int
	quitting        bool
	selected        bool            // true if user made a selection (enter/space)
	changeModel     bool            // true if user pressed 'm' to change model
	showOthers      bool            // true if "Others..." is expanded
	availableModels map[string]bool // cache of available model names
	blinking        bool            // true when showing blink logo
	err             error
}

// modelExists checks if a model exists in the cached available models.
func (m *model) modelExists(name string) bool {
	if m.availableModels == nil || name == "" {
		return false
	}
	if m.availableModels[name] {
		return true
	}
	// Check for prefix match (e.g., "llama2" matches "llama2:latest")
	for modelName := range m.availableModels {
		if strings.HasPrefix(modelName, name+":") {
			return true
		}
	}
	return false
}

// loadAvailableModels fetches and caches the list of available models.
func (m *model) loadAvailableModels() {
	m.availableModels = make(map[string]bool)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return
	}
	models, err := client.List(context.Background())
	if err != nil {
		return
	}
	for _, mdl := range models.Models {
		m.availableModels[mdl.Name] = true
	}
}

func (m *model) buildItems() {
	others := getOtherIntegrations()
	m.items = make([]menuItem, 0, len(mainMenuItems)+1+len(others))
	m.items = append(m.items, mainMenuItems...)

	if m.showOthers {
		// Change "Others..." to "Hide others..."
		hideItem := menuItem{
			title:       "Hide others...",
			description: "Hide additional integrations",
			isOthers:    true,
		}
		m.items = append(m.items, hideItem)
		m.items = append(m.items, others...)
	} else {
		m.items = append(m.items, othersMenuItem)
	}
}

// isOthersIntegration returns true if the integration is in the "Others" menu
func isOthersIntegration(name string) bool {
	switch name {
	case "codex", "droid", "opencode":
		return true
	}
	return false
}

func initialModel() model {
	m := model{
		cursor: 0,
	}
	m.loadAvailableModels()

	// Check last selection to determine if we need to expand "Others"
	lastSelection := config.LastSelection()
	if isOthersIntegration(lastSelection) {
		m.showOthers = true
	}

	m.buildItems()

	// Position cursor on last selection
	if lastSelection != "" {
		for i, item := range m.items {
			if lastSelection == "run" && item.isRunModel {
				m.cursor = i
				break
			} else if item.integration == lastSelection {
				m.cursor = i
				break
			}
		}
	}

	return m
}

func (m model) Init() tea.Cmd {
	return tea.Tick(blinkInterval, func(t time.Time) tea.Msg {
		return blinkMsg{}
	})
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case blinkMsg:
		m.blinking = true
		return m, tea.Tick(blinkDuration, func(t time.Time) tea.Msg {
			return unblinkMsg{}
		})

	case unblinkMsg:
		m.blinking = false
		return m, tea.Tick(blinkInterval, func(t time.Time) tea.Msg {
			return blinkMsg{}
		})

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			m.quitting = true
			return m, tea.Quit

		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}

		case "down", "j":
			if m.cursor < len(m.items)-1 {
				m.cursor++
			}

		case "enter", " ":
			item := m.items[m.cursor]

			// Handle "Others..." toggle
			if item.isOthers {
				m.showOthers = !m.showOthers
				m.buildItems()
				// Keep cursor on the Others/Hide item
				if m.cursor >= len(m.items) {
					m.cursor = len(m.items) - 1
				}
				return m, nil
			}

			// Don't allow selecting uninstalled integrations
			if item.integration != "" && !config.IsIntegrationInstalled(item.integration) {
				return m, nil
			}

			m.selected = true
			m.quitting = true
			return m, tea.Quit

		case "m":
			// Allow model change for integrations and run model
			item := m.items[m.cursor]
			if item.integration != "" || item.isRunModel {
				// Don't allow for uninstalled integrations
				if item.integration != "" && !config.IsIntegrationInstalled(item.integration) {
					return m, nil
				}
				m.changeModel = true
				m.quitting = true
				return m, tea.Quit
			}
		}
	}

	return m, nil
}

func (m model) View() string {
	if m.quitting {
		return ""
	}

	logo := logoNormal
	if m.blinking {
		logo = logoBlink
	}

	versionText := "\n\n  Ollama " + versionStyle.Render("v"+version.Version)

	logoRendered := logoStyle.Render(logo)
	logoBlock := lipgloss.NewStyle().Padding(0, 1).MarginLeft(2).Background(lipgloss.Color("0")).Render(logoRendered)
	versionBlock := titleStyle.Render(versionText)
	header := lipgloss.JoinHorizontal(lipgloss.Top, logoBlock, versionBlock)

	s := header + "\n\n"

	for i, item := range m.items {
		cursor := "  "
		style := itemStyle
		isInstalled := true

		if item.integration != "" {
			isInstalled = config.IsIntegrationInstalled(item.integration)
		}

		if m.cursor == i {
			cursor = "▸ "
			if isInstalled {
				style = selectedStyle
			} else {
				style = greyedSelectedStyle
			}
		} else if !isInstalled && item.integration != "" {
			style = greyedStyle
		}

		title := item.title
		if item.integration != "" {
			if !isInstalled {
				title += " " + notInstalledStyle.Render("(not installed)")
			} else if mdl := config.IntegrationModel(item.integration); mdl != "" && m.modelExists(mdl) {
				title += " " + modelStyle.Render("("+mdl+")")
			}
		} else if item.isRunModel {
			if mdl := config.LastModel(); mdl != "" && m.modelExists(mdl) {
				title += " " + modelStyle.Render("("+mdl+")")
			}
		}

		s += style.Render(cursor+title) + "\n"
		s += descStyle.Render(item.description) + "\n\n"
	}

	s += "\n" + lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Render("↑/↓ navigate • enter select • m change model • esc quit")

	return s
}

// Selection represents what the user selected
type Selection int

const (
	SelectionNone Selection = iota
	SelectionRunModel
	SelectionChangeRunModel
	SelectionIntegration       // Generic integration selection
	SelectionChangeIntegration // Generic change model for integration
)

// Result contains the selection and any associated data
type Result struct {
	Selection   Selection
	Integration string // integration name if applicable
}

// Run starts the TUI and returns the user's selection
func Run() (Result, error) {
	m := initialModel()
	p := tea.NewProgram(m)

	finalModel, err := p.Run()
	if err != nil {
		return Result{Selection: SelectionNone}, fmt.Errorf("error running TUI: %w", err)
	}

	fm := finalModel.(model)
	if fm.err != nil {
		return Result{Selection: SelectionNone}, fm.err
	}

	// User quit without selecting
	if !fm.selected && !fm.changeModel {
		return Result{Selection: SelectionNone}, nil
	}

	item := fm.items[fm.cursor]

	// Handle model change request
	if fm.changeModel {
		if item.isRunModel {
			return Result{Selection: SelectionChangeRunModel}, nil
		}
		return Result{
			Selection:   SelectionChangeIntegration,
			Integration: item.integration,
		}, nil
	}

	// Handle selection
	if item.isRunModel {
		return Result{Selection: SelectionRunModel}, nil
	}

	return Result{
		Selection:   SelectionIntegration,
		Integration: item.integration,
	}, nil
}
