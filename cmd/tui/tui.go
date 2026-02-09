package tui

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/version"
)

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			MarginBottom(1)

	versionStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "243", Dark: "250"})

	itemStyle = lipgloss.NewStyle().
			PaddingLeft(4)

	selectedStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Bold(true).
			Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	greyedStyle = lipgloss.NewStyle().
			PaddingLeft(4).
			Foreground(lipgloss.AdaptiveColor{Light: "249", Dark: "240"})

	greyedSelectedStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Foreground(lipgloss.AdaptiveColor{Light: "240", Dark: "248"}).
				Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	descStyle = lipgloss.NewStyle().
			PaddingLeft(6).
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
	isRunModel  bool   // true for the "Run a model" option
	isOthers    bool   // true for the "Others..." toggle item
}

var mainMenuItems = []menuItem{
	{
		title:       "Run a model",
		description: "Start an interactive chat with a model",
		isRunModel:  true,
	},
	{
		title:       "Launch Claude Code",
		description: "Open Claude Code AI assistant",
		integration: "claude",
	},
	{
		title:       "Launch Codex",
		description: "Open Codex CLI",
		integration: "codex",
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
	return []menuItem{
		{
			title:       "Launch Droid",
			description: "Open Droid integration",
			integration: "droid",
		},
		{
			title:       "Launch OpenCode",
			description: "Open OpenCode integration",
			integration: "opencode",
		},
		{
			title:       "Launch Pi",
			description: "Open Pi coding agent",
			integration: "pi",
		},
	}
}

type model struct {
	items           []menuItem
	cursor          int
	quitting        bool
	selected        bool            // true if user made a selection (enter/space)
	changeModel     bool            // true if user pressed right arrow to change model
	showOthers      bool            // true if "Others..." is expanded
	availableModels map[string]bool // cache of available model names
	err             error

	// Modal state
	showingModal  bool          // true when model picker modal is visible
	modalSelector selectorModel // the selector model for the modal
	modalItems    []SelectItem  // cached items for the modal

	// Sign-in dialog state
	showingSignIn   bool   // true when sign-in dialog is visible
	signInURL       string // URL for sign-in
	signInModel     string // model that requires sign-in
	signInSpinner   int    // spinner frame index
	signInFromModal bool   // true if sign-in was triggered from modal (not main menu)

	// Status message state
	statusMsg string // temporary status message shown near help text
}

// signInTickMsg is sent to animate the sign-in spinner
type signInTickMsg struct{}

// signInCheckMsg is sent to check if sign-in is complete
type signInCheckMsg struct {
	signedIn bool
	userName string
}

// clearStatusMsg is sent to clear the temporary status message
type clearStatusMsg struct{}

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

// buildModalItems creates the list of models for the modal selector.
func (m *model) buildModalItems() []SelectItem {
	modelItems, _ := config.GetModelItems(context.Background())
	var items []SelectItem
	for _, item := range modelItems {
		items = append(items, SelectItem{Name: item.Name, Description: item.Description, Recommended: item.Recommended})
	}
	return items
}

// openModelModal opens the model picker modal.
func (m *model) openModelModal() {
	m.modalItems = m.buildModalItems()
	m.modalSelector = selectorModel{
		title:    "Select model:",
		items:    m.modalItems,
		helpText: "↑/↓ navigate • enter select • ← back",
	}
	m.showingModal = true
}

// isCloudModel returns true if the model name indicates a cloud model.
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
		return nil // Already signed in
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

	// Open browser (best effort)
	switch runtime.GOOS {
	case "darwin":
		_ = exec.Command("open", signInURL).Start()
	case "linux":
		_ = exec.Command("xdg-open", signInURL).Start()
	case "windows":
		_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", signInURL).Start()
	}

	// Start the spinner tick
	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return signInTickMsg{}
	})
}

// checkSignIn checks if the user has completed sign-in.
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
		m.items = append(m.items, others...)
	} else {
		m.items = append(m.items, othersMenuItem)
	}
}

// isOthersIntegration returns true if the integration is in the "Others" menu
func isOthersIntegration(name string) bool {
	switch name {
	case "droid", "opencode":
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
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Handle clearStatusMsg
	if _, ok := msg.(clearStatusMsg); ok {
		m.statusMsg = ""
		return m, nil
	}

	// Handle sign-in dialog
	if m.showingSignIn {
		switch msg := msg.(type) {
		case tea.KeyMsg:
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				// Cancel sign-in and go back
				m.showingSignIn = false
				if m.signInFromModal {
					m.showingModal = true
				}
				// If from main menu, just return to main menu (default state)
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
				// Sign-in complete - proceed with selection
				if m.signInFromModal {
					// Came from modal - set changeModel
					m.modalSelector.selected = m.signInModel
					m.changeModel = true
				} else {
					// Came from main menu - just select
					m.selected = true
				}
				m.quitting = true
				return m, tea.Quit
			}
		}
		return m, nil
	}

	// Handle modal input if modal is showing
	if m.showingModal {
		switch msg := msg.(type) {
		case tea.KeyMsg:
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc, tea.KeyLeft:
				// Close modal without selection
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
					// Selection made - exit with changeModel
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

			// Don't allow selecting uninstalled integrations
			if item.integration != "" && !config.IsIntegrationInstalled(item.integration) {
				m.statusMsg = fmt.Sprintf("%s is not installed", item.title)
				if hint := config.IntegrationInstallHint(item.integration); hint != "" {
					m.statusMsg += " — " + hint
				}
				return m, tea.Tick(4*time.Second, func(t time.Time) tea.Msg { return clearStatusMsg{} })
			}

			// Check if a cloud model is configured and needs sign-in
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
			// Allow model change for integrations and run model
			item := m.items[m.cursor]
			if item.integration != "" || item.isRunModel {
				// Don't allow for uninstalled integrations
				if item.integration != "" && !config.IsIntegrationInstalled(item.integration) {
					m.statusMsg = fmt.Sprintf("%s is not installed", item.title)
					if hint := config.IntegrationInstallHint(item.integration); hint != "" {
						m.statusMsg += " — " + hint
					}
					return m, tea.Tick(4*time.Second, func(t time.Time) tea.Msg { return clearStatusMsg{} })
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

	// Render sign-in dialog if showing
	if m.showingSignIn {
		return m.renderSignInDialog()
	}

	// Render modal overlay if showing - replaces main view
	if m.showingModal {
		return m.renderModal()
	}

	s := titleStyle.Render("  Ollama "+versionStyle.Render(version.Version)) + "\n\n"

	for i, item := range m.items {
		cursor := ""
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
		s += descStyle.Render(item.description) + "\n\n"
	}

	if m.statusMsg != "" {
		s += "\n" + lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "124", Dark: "210"}).Render(m.statusMsg) + "\n"
	}

	s += "\n" + lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "244", Dark: "244"}).Render("↑/↓ navigate • enter select • → change model • esc quit")

	return s
}

// renderModal renders the model picker modal.
// Delegates to selectorModel.renderContent() for the actual item rendering.
func (m model) renderModal() string {
	modalStyle := lipgloss.NewStyle().
		Padding(1, 2).
		MarginLeft(2)

	return modalStyle.Render(m.modalSelector.renderContent())
}

// renderSignInDialog renders the sign-in dialog.
func (m model) renderSignInDialog() string {
	dialogStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("245")).
		Padding(1, 2).
		MarginLeft(2)

	spinnerFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	spinner := spinnerFrames[m.signInSpinner%len(spinnerFrames)]

	var content strings.Builder

	content.WriteString(selectorTitleStyle.Render("Sign in required"))
	content.WriteString("\n\n")

	content.WriteString(fmt.Sprintf("To use %s, please sign in.\n\n", selectedStyle.Render(m.signInModel)))

	content.WriteString("Navigate to:\n")
	content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("117")).Render("  " + m.signInURL))
	content.WriteString("\n\n")

	content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).Render(
		fmt.Sprintf("%s Waiting for sign in to complete...", spinner)))
	content.WriteString("\n\n")

	content.WriteString(selectorHelpStyle.Render("esc cancel"))

	return dialogStyle.Render(content.String())
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
	Model       string // model name if selected from modal
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

	// Handle selection
	if item.isRunModel {
		return Result{Selection: SelectionRunModel}, nil
	}

	return Result{
		Selection:   SelectionIntegration,
		Integration: item.integration,
	}, nil
}
