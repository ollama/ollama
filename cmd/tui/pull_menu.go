package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/api"
)

const pullMenuTimeout = 5 * time.Second

type pullMenuState int

const (
	pullStateLoading pullMenuState = iota
	pullStateBaseList
	pullStateTagList
	pullStateError
)

type remoteModelsMsg struct {
	baseModels []SelectItem
	tagsByBase map[string][]SelectItem
	err        error
}

type pullMenuModel struct {
	state      pullMenuState
	baseModels []SelectItem
	tagsByBase map[string][]SelectItem

	baseSelector selectorModel
	tagSelector  selectorModel
	selectedBase string

	err      error
	selected string
	width    int
}

func newPullMenuModel() pullMenuModel {
	return pullMenuModel{state: pullStateLoading}
}

func (m pullMenuModel) Init() tea.Cmd {
	return fetchRemoteModelsCmd
}

// fetchRemoteModelsCmd is a Bubble Tea command that calls the ollama.com /v1/models API
// Models are grouped client-side by base name; no additional network calls are made.
func fetchRemoteModelsCmd() tea.Msg {
	resp, err := api.ListRemote(context.Background(), pullMenuTimeout)
	if err != nil {
		return remoteModelsMsg{err: err}
	}

	baseOrder := []string{}
	seenBase := map[string]bool{}
	tagsByBase := map[string][]SelectItem{}

	for _, rm := range resp.Data {
		id := strings.TrimSpace(rm.ID)
		if id == "" {
			continue
		}

		base := id
		if idx := strings.LastIndex(id, ":"); idx >= 0 {
			base = id[:idx]
		}

		tagsByBase[base] = append(tagsByBase[base], SelectItem{Name: id})

		if !seenBase[base] {
			seenBase[base] = true
			baseOrder = append(baseOrder, base)
		}
	}

	baseModels := make([]SelectItem, 0, len(baseOrder))
	for _, base := range baseOrder {
		tags := tagsByBase[base]
		desc := ""
		if len(tags) > 1 {
			variants := make([]string, len(tags))
			for i, t := range tags {
				if idx := strings.LastIndex(t.Name, ":"); idx >= 0 {
					variants[i] = t.Name[idx+1:]
				} else {
					variants[i] = t.Name
				}
			}
			desc = strings.Join(variants, ", ")
		}
		baseModels = append(baseModels, SelectItem{Name: base, Description: desc})
	}

	return remoteModelsMsg{baseModels: baseModels, tagsByBase: tagsByBase}
}

func (m pullMenuModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		return m, nil

	case remoteModelsMsg:
		if msg.err != nil {
			m.state = pullStateError
			m.err = msg.err
			return m, nil
		}
		m.baseModels = msg.baseModels
		m.tagsByBase = msg.tagsByBase
		m.state = pullStateBaseList
		m.baseSelector = selectorModel{
			title:    "Pull a Model",
			items:    m.baseModels,
			helpText: "↑/↓ navigate • enter select • → view tags • esc cancel",
		}
		return m, nil

	case tea.KeyMsg:
		switch m.state {
		case pullStateError:
			switch msg.Type {
			case tea.KeyEsc, tea.KeyCtrlC:
				m.selected = ""
				return m, tea.Quit
			}
			return m, nil

		case pullStateBaseList:
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				m.selected = ""
				return m, tea.Quit

			case tea.KeyEnter:
				filtered := m.baseSelector.filteredItems()
				if len(filtered) == 0 {
					return m, nil
				}
				chosen := filtered[m.baseSelector.cursor]
				tags := m.tagsByBase[chosen.Name]
				if len(tags) == 1 {
					m.selected = tags[0].Name
					return m, tea.Quit
				}
				m.selectedBase = chosen.Name
				m.tagSelector = selectorModel{
					title:    "Pull a Model › " + chosen.Name,
					items:    tags,
					helpText: "↑/↓ navigate • enter pull • ← back • esc cancel",
				}
				m.state = pullStateTagList
				return m, nil

			case tea.KeyRight:
				filtered := m.baseSelector.filteredItems()
				if len(filtered) == 0 {
					return m, nil
				}
				chosen := filtered[m.baseSelector.cursor]
				tags := m.tagsByBase[chosen.Name]
				m.selectedBase = chosen.Name
				m.tagSelector = selectorModel{
					title:    "Pull a Model › " + chosen.Name,
					items:    tags,
					helpText: "↑/↓ navigate • enter pull • ← back • esc cancel",
				}
				m.state = pullStateTagList
				return m, nil

			default:
				m.baseSelector.updateNavigation(msg)
				return m, nil
			}

		case pullStateTagList:
			switch msg.Type {
			case tea.KeyCtrlC, tea.KeyEsc:
				m.selected = ""
				return m, tea.Quit

			case tea.KeyLeft:
				m.state = pullStateBaseList
				return m, nil

			case tea.KeyEnter:
				filtered := m.tagSelector.filteredItems()
				if len(filtered) == 0 {
					return m, nil
				}
				m.selected = filtered[m.tagSelector.cursor].Name
				return m, tea.Quit

			default:
				m.tagSelector.updateNavigation(msg)
				return m, nil
			}
		}
	}

	return m, nil
}

func (m pullMenuModel) View() string {
	if m.selected != "" {
		return ""
	}

	var s string

	switch m.state {
	case pullStateLoading:
		s = selectorTitleStyle.Render("Pull a Model") + "\n\n"
		s += selectorItemStyle.Render("Fetching available models...")
		s += "\n\n" + selectorHelpStyle.Render("esc cancel")

	case pullStateError:
		s = selectorTitleStyle.Render("Pull a Model") + "\n\n"
		s += lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "1", Dark: "9"}).
			PaddingLeft(2).
			Render("⚠  Could not fetch model list") + "\n\n"
		s += selectorItemStyle.Render(fmt.Sprintf("Run %s directly", selectorInputStyle.Render("`ollama pull <model>`"))) + "\n\n"
		s += selectorHelpStyle.Render("esc cancel")

	case pullStateBaseList:
		s = m.baseSelector.renderContent()

	case pullStateTagList:
		s = m.tagSelector.renderContent()
	}

	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s)
	}
	return s
}

// RunPullMenu runs the interactive pull menu and returns the selected model name.
// Returns ("", ErrCancelled) if the user cancels, or ("", err) on fetch failure.
func RunPullMenu() (string, error) {
	m := newPullMenuModel()
	p := tea.NewProgram(m)

	final, err := p.Run()
	if err != nil {
		return "", fmt.Errorf("pull menu: %w", err)
	}

	fm := final.(pullMenuModel)

	if fm.state == pullStateError {
		return "", ErrCancelled
	}

	if fm.selected == "" {
		return "", ErrCancelled
	}

	return fm.selected, nil
}
