package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/version"
)

var (
	versionStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "243", Dark: "250"})

	menuItemStyle = lipgloss.NewStyle().
			PaddingLeft(2)

	menuSelectedItemStyle = lipgloss.NewStyle().
				Bold(true).
				Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	menuDescStyle = selectorDescStyle.
			PaddingLeft(4)

	greyedStyle = menuItemStyle.
			Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	greyedSelectedStyle = menuSelectedItemStyle.
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	modelStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "243", Dark: "250"})

	notInstalledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
				Italic(true)
)

type menuItem struct {
	title       string
	description string
	integration string // integration name for loading model config, empty if not an integration
	isRunModel  bool
	isOthers    bool
}

var mainMenuItems = []menuItem{
	{
		title:       "Run a model",
		description: "Start an interactive chat with a model",
		isRunModel:  true,
	},
	{
		title:       "Launch Claude Code",
		description: "Agentic coding across large codebases",
		integration: "claude",
	},
	{
		title:       "Launch Codex",
		description: "OpenAI's open-source coding agent",
		integration: "codex",
	},
	{
		title:       "Launch OpenClaw",
		description: "Personal AI with 100+ skills",
		integration: "openclaw",
	},
}

var othersMenuItem = menuItem{
	title:       "More...",
	description: "Show additional integrations",
	isOthers:    true,
}

// getOtherIntegrations dynamically builds the "Others" list from the integration
// registry, excluding any integrations already present in the pinned mainMenuItems.
func getOtherIntegrations() []menuItem {
	pinned := map[string]bool{
		"run": true, // not an integration but in the pinned list
	}
	for _, item := range mainMenuItems {
		if item.integration != "" {
			pinned[item.integration] = true
		}
	}

	var others []menuItem
	for _, info := range config.ListIntegrationInfos() {
		if pinned[info.Name] {
			continue
		}
		desc := info.Description
		if desc == "" {
			desc = "Open " + info.DisplayName + " integration"
		}
		others = append(others, menuItem{
			title:       "Launch " + info.DisplayName,
			description: desc,
			integration: info.Name,
		})
	}
	return others
}

type model struct {
	items           []menuItem
	cursor          int
	quitting        bool
	selected        bool
	changeModel     bool
	showOthers      bool
	availableModels map[string]bool
	err             error

	showingModal  bool
	modalSelector selectorModel
	modalItems    []SelectItem

	showingSignIn   bool
	signInURL       string
	signInModel     string
	signInSpinner   int
	signInFromModal bool   // true if sign-in was triggered from modal (not main menu)

	width     int    // terminal width from WindowSizeMsg
	statusMsg string // temporary status message shown near help text
}

type signInTickMsg struct{}

type signInCheckMsg struct {
	signedIn bool
	userName string
}

type clearStatusMsg struct{}

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

func (m *model) buildModalItems() []SelectItem {
	modelItems, _ := config.GetModelItems(context.Background())
	var items []SelectItem
	for _, item := range modelItems {
		items = append(items, SelectItem{Name: item.Name, Description: item.Description, Recommended: item.Recommended})
	}
	return items
}

func (m *model) openModelModal() {
	m.modalItems = m.buildModalItems()
	m.modalSelector = selectorModel{
		title:    "Select model:",
		items:    m.modalItems,
		helpText: "↑/↓ navigate • enter select • ← back",
	}
	m.showingModal = true
}

func isCloudModel(name string) bool {
	return strings.HasSuffix(name, ":cloud")
}

// checkCloudSignIn checks if a cloud model needs sign-in.
// Returns a command to start sign-in if needed, or nil if already signed in.
func (m *model) checkCloudSignIn(modelName string, fromModal bool) tea.Cmd {
	if modelName == "" || !isCloudModel(modelName) {
		return nil
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil
	}
	user, err := client.Whoami(context.Background())
	if err == nil && user != nil && user.Name != "" {
		return nil
	}
	var aErr api.AuthorizationError
	if errors.As(err, &aErr) && aErr.SigninURL != "" {
		return m.startSignIn(modelName, aErr.SigninURL, fromModal)
	}
	return nil
}

// startSignIn initiates the sign-in flow for a cloud model.
// fromModal indicates if this was triggered from the model picker modal.
func (m *model) startSignIn(modelName, signInURL string, fromModal bool) tea.Cmd {
	m.showingModal = false
	m.showingSignIn = true
	m.signInURL = signInURL
	m.signInModel = modelName
	m.signInSpinner = 0
	m.signInFromModal = fromModal

	config.OpenBrowser(signInURL)

	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return signInTickMsg{}
	})
}

func checkSignIn() tea.Msg {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return signInCheckMsg{signedIn: false}
	}
	user, err := client.Whoami(context.Background())
	if err == nil && user != nil && user.Name != "" {
		return signInCheckMsg{signedIn: true, userName: user.Name}
	}
	return signInCheckMsg{signedIn: false}
}

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
		m.items = append(m.items, others...)
	} else {
		m.items = append(m.items, othersMenuItem)
	}
}

func isOthersIntegration(name string) bool {
	for _, item := range getOtherIntegrations() {
		if item.integration == name {
			return true
		}
	}
	return false
}

func initialModel() model {
	m := model{
		cursor: 0,
	}
	m.loadAvailableModels()

	lastSelection := config.LastSelection()
	if isOthersIntegration(lastSelection) {
		m.showOthers = true
	}

	m.buildItems()

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
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if wmsg, ok := msg.(tea.WindowSizeMsg); ok {
		wasSet := m.width > 0
		m.width = wmsg.Width
		if wasSet {
			return m, tea.EnterAltScreen
		}
		return m, nil
	}

	if _, ok := msg.(clearStatusMsg); ok {
		m.statusMsg = ""
		return m, nil
	}

	if m.showingSignIn {
		switch msg := msg.(type) {
		case tea.KeyMsg:
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				m.showingSignIn = false
				if m.signInFromModal {
					m.showingModal = true
				}
				return m, nil
			}

		case signInTickMsg:
			m.signInSpinner++
			// Check sign-in status every 5th tick (~1 second)
			if m.signInSpinner%5 == 0 {
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
				if m.signInFromModal {
					m.modalSelector.selected = m.signInModel
					m.changeModel = true
				} else {
					m.selected = true
				}
				m.quitting = true
				return m, tea.Quit
			}
		}
		return m, nil
	}

	if m.showingModal {
		switch msg := msg.(type) {
		case tea.KeyMsg:
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc, tea.KeyLeft:
				m.showingModal = false
				return m, nil

			case tea.KeyEnter:
				filtered := m.modalSelector.filteredItems()
				if len(filtered) > 0 && m.modalSelector.cursor < len(filtered) {
					m.modalSelector.selected = filtered[m.modalSelector.cursor].Name
				}
				if m.modalSelector.selected != "" {
					if cmd := m.checkCloudSignIn(m.modalSelector.selected, true); cmd != nil {
						return m, cmd
					}
					m.changeModel = true
					m.quitting = true
					return m, tea.Quit
				}
				return m, nil

			default:
				// Delegate navigation (up/down/pgup/pgdown/filter/backspace) to selectorModel
				m.modalSelector.updateNavigation(msg)
			}
		}
		return m, nil
	}

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			m.quitting = true
			return m, tea.Quit

		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
			// Auto-collapse "Others" when cursor moves back into pinned items
			if m.showOthers && m.cursor < len(mainMenuItems) {
				m.showOthers = false
				m.buildItems()
			}

		case "down", "j":
			if m.cursor < len(m.items)-1 {
				m.cursor++
			}
			// Auto-expand "Others..." when cursor lands on it
			if m.cursor < len(m.items) && m.items[m.cursor].isOthers && !m.showOthers {
				m.showOthers = true
				m.buildItems()
				// cursor now points at the first "other" integration
			}

		case "enter", " ":
			item := m.items[m.cursor]

			if item.integration != "" && !config.IsIntegrationInstalled(item.integration) {
				return m, nil
			}

			var configuredModel string
			if item.isRunModel {
				configuredModel = config.LastModel()
			} else if item.integration != "" {
				configuredModel = config.IntegrationModel(item.integration)
			}
			if cmd := m.checkCloudSignIn(configuredModel, false); cmd != nil {
				return m, cmd
			}

			m.selected = true
			m.quitting = true
			return m, tea.Quit

		case "right", "l":
			item := m.items[m.cursor]
			if item.integration != "" || item.isRunModel {
				if item.integration != "" && !config.IsIntegrationInstalled(item.integration) {
					return m, nil
				}
				m.openModelModal()
			}
		}
	}

	return m, nil
}

func (m model) View() string {
	if m.quitting {
		return ""
	}

	if m.showingSignIn {
		return m.renderSignInDialog()
	}

	if m.showingModal {
		return m.renderModal()
	}

	s := selectorTitleStyle.Render("Ollama "+versionStyle.Render(version.Version)) + "\n\n"

	for i, item := range m.items {
		cursor := ""
		style := menuItemStyle
		isInstalled := true

		if item.integration != "" {
			isInstalled = config.IsIntegrationInstalled(item.integration)
		}

		if m.cursor == i {
			cursor = "▸ "
			if isInstalled {
				style = menuSelectedItemStyle
			} else {
				style = greyedSelectedStyle
			}
		} else if !isInstalled && item.integration != "" {
			style = greyedStyle
		}

		title := item.title
		var modelSuffix string
		if item.integration != "" {
			if !isInstalled {
				title += " " + notInstalledStyle.Render("(not installed)")
			} else if m.cursor == i {
				if mdl := config.IntegrationModel(item.integration); mdl != "" && m.modelExists(mdl) {
					modelSuffix = " " + modelStyle.Render("("+mdl+")")
				}
			}
		} else if item.isRunModel && m.cursor == i {
			if mdl := config.LastModel(); mdl != "" && m.modelExists(mdl) {
				modelSuffix = " " + modelStyle.Render("("+mdl+")")
			}
		}

		s += style.Render(cursor+title) + modelSuffix + "\n"

		desc := item.description
		if !isInstalled && item.integration != "" && m.cursor == i {
			if hint := config.IntegrationInstallHint(item.integration); hint != "" {
				desc = hint
			} else {
				desc = "not installed"
			}
		}
		s += menuDescStyle.Render(desc) + "\n\n"
	}

	if m.statusMsg != "" {
		s += "\n" + lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "124", Dark: "210"}).Render(m.statusMsg) + "\n"
	}

	s += "\n" + selectorHelpStyle.Render("↑/↓ navigate • enter launch • → change model • esc quit")

	return s
}

func (m model) renderModal() string {
	modalStyle := lipgloss.NewStyle().
		PaddingBottom(1).
		PaddingRight(2)

	return modalStyle.Render(m.modalSelector.renderContent())
}

func (m model) renderSignInDialog() string {
	return renderSignIn(m.signInModel, m.signInURL, m.signInSpinner, m.width)
}

type Selection int

const (
	SelectionNone Selection = iota
	SelectionRunModel
	SelectionChangeRunModel
	SelectionIntegration       // Generic integration selection
	SelectionChangeIntegration // Generic change model for integration
)

type Result struct {
	Selection   Selection
	Integration string // integration name if applicable
	Model       string // model name if selected from modal
}

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

	if !fm.selected && !fm.changeModel {
		return Result{Selection: SelectionNone}, nil
	}

	item := fm.items[fm.cursor]

	if fm.changeModel {
		if item.isRunModel {
			return Result{
				Selection: SelectionChangeRunModel,
				Model:     fm.modalSelector.selected,
			}, nil
		}
		return Result{
			Selection:   SelectionChangeIntegration,
			Integration: item.integration,
			Model:       fm.modalSelector.selected,
		}, nil
	}

	if item.isRunModel {
		return Result{Selection: SelectionRunModel}, nil
	}

	return Result{
		Selection:   SelectionIntegration,
		Integration: item.integration,
	}, nil
}
