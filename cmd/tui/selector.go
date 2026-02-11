package tui

import (
	"errors"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/config"
)

var (
	selectorTitleStyle = lipgloss.NewStyle().
				Bold(true)

	selectorItemStyle = lipgloss.NewStyle().
				PaddingLeft(4)

	selectorSelectedItemStyle = lipgloss.NewStyle().
					PaddingLeft(2).
					Bold(true).
					Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	selectorDescStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	selectorDescLineStyle = selectorDescStyle.
				PaddingLeft(6)

	selectorFilterStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
				Italic(true)

	selectorInputStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "235", Dark: "252"})

	selectorDefaultTagStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
				Italic(true)

	selectorHelpStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "244", Dark: "244"})

	selectorMoreStyle = lipgloss.NewStyle().
				PaddingLeft(6).
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
				Italic(true)

	sectionHeaderStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "240", Dark: "249"})
)

const maxSelectorItems = 10

// ErrCancelled is returned when the user cancels the selection.
var ErrCancelled = errors.New("cancelled")

type SelectItem struct {
	Name        string
	Description string
	Recommended bool
}

// ConvertItems converts config.ModelItem slice to SelectItem slice.
func ConvertItems(items []config.ModelItem) []SelectItem {
	out := make([]SelectItem, len(items))
	for i, item := range items {
		out[i] = SelectItem{Name: item.Name, Description: item.Description, Recommended: item.Recommended}
	}
	return out
}

// ReorderItems returns a copy with recommended items first, then non-recommended,
// preserving relative order within each group. This ensures the data order matches
// the visual section layout (Recommended / More).
func ReorderItems(items []SelectItem) []SelectItem {
	var rec, other []SelectItem
	for _, item := range items {
		if item.Recommended {
			rec = append(rec, item)
		} else {
			other = append(other, item)
		}
	}
	return append(rec, other...)
}

// selectorModel is the bubbletea model for single selection.
type selectorModel struct {
	title        string
	items        []SelectItem
	filter       string
	cursor       int
	scrollOffset int
	selected     string
	cancelled    bool
	helpText     string
	width        int
}

func (m selectorModel) filteredItems() []SelectItem {
	if m.filter == "" {
		return m.items
	}
	filterLower := strings.ToLower(m.filter)
	var result []SelectItem
	for _, item := range m.items {
		if strings.Contains(strings.ToLower(item.Name), filterLower) {
			result = append(result, item)
		}
	}
	return result
}

func (m selectorModel) Init() tea.Cmd {
	return nil
}

// otherStart returns the index of the first non-recommended item in the filtered list.
// When filtering, all items scroll together so this returns 0.
func (m selectorModel) otherStart() int {
	if m.filter != "" {
		return 0
	}
	filtered := m.filteredItems()
	for i, item := range filtered {
		if !item.Recommended {
			return i
		}
	}
	return len(filtered)
}

// updateNavigation handles navigation keys (up/down/pgup/pgdown/filter/backspace).
// It does NOT handle Enter, Esc, or CtrlC. This is used by both the standalone
// selector and the TUI modal (which intercepts Enter/Esc for its own logic).
func (m *selectorModel) updateNavigation(msg tea.KeyMsg) {
	filtered := m.filteredItems()
	otherStart := m.otherStart()

	switch msg.Type {
	case tea.KeyUp:
		if m.cursor > 0 {
			m.cursor--
			m.updateScroll(otherStart)
		}

	case tea.KeyDown:
		if m.cursor < len(filtered)-1 {
			m.cursor++
			m.updateScroll(otherStart)
		}

	case tea.KeyPgUp:
		m.cursor -= maxSelectorItems
		if m.cursor < 0 {
			m.cursor = 0
		}
		m.updateScroll(otherStart)

	case tea.KeyPgDown:
		m.cursor += maxSelectorItems
		if m.cursor >= len(filtered) {
			m.cursor = len(filtered) - 1
		}
		m.updateScroll(otherStart)

	case tea.KeyBackspace:
		if len(m.filter) > 0 {
			m.filter = m.filter[:len(m.filter)-1]
			m.cursor = 0
			m.scrollOffset = 0
		}

	case tea.KeyRunes:
		m.filter += string(msg.Runes)
		m.cursor = 0
		m.scrollOffset = 0
	}
}

// updateScroll adjusts scrollOffset based on cursor position.
// When not filtering, scrollOffset is relative to the "More" (non-recommended) section.
// When filtering, it's relative to the full filtered list.
func (m *selectorModel) updateScroll(otherStart int) {
	if m.filter != "" {
		if m.cursor < m.scrollOffset {
			m.scrollOffset = m.cursor
		}
		if m.cursor >= m.scrollOffset+maxSelectorItems {
			m.scrollOffset = m.cursor - maxSelectorItems + 1
		}
		return
	}

	// Cursor is in recommended section — reset "More" scroll to top
	if m.cursor < otherStart {
		m.scrollOffset = 0
		return
	}

	// Cursor is in "More" section — scroll relative to others
	posInOthers := m.cursor - otherStart
	maxOthers := maxSelectorItems - otherStart
	if maxOthers < 3 {
		maxOthers = 3
	}
	if posInOthers < m.scrollOffset {
		m.scrollOffset = posInOthers
	}
	if posInOthers >= m.scrollOffset+maxOthers {
		m.scrollOffset = posInOthers - maxOthers + 1
	}
}

func (m selectorModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
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

		case tea.KeyEnter:
			filtered := m.filteredItems()
			if len(filtered) > 0 && m.cursor < len(filtered) {
				m.selected = filtered[m.cursor].Name
			}
			return m, tea.Quit

		default:
			m.updateNavigation(msg)
		}
	}

	return m, nil
}

func (m selectorModel) renderItem(s *strings.Builder, item SelectItem, idx int) {
	if idx == m.cursor {
		s.WriteString(selectorSelectedItemStyle.Render("▸ " + item.Name))
	} else {
		s.WriteString(selectorItemStyle.Render(item.Name))
	}
	s.WriteString("\n")
	if item.Description != "" {
		s.WriteString(selectorDescLineStyle.Render(item.Description))
		s.WriteString("\n")
	}
}

// renderContent renders the selector content (title, items, help text) without
// checking the cancelled/selected state. This is used by both View() (standalone mode)
// and by the TUI modal which embeds a selectorModel.
func (m selectorModel) renderContent() string {
	var s strings.Builder

	s.WriteString(selectorTitleStyle.Render(m.title))
	s.WriteString(" ")
	if m.filter == "" {
		s.WriteString(selectorFilterStyle.Render("Type to filter..."))
	} else {
		s.WriteString(selectorInputStyle.Render(m.filter))
	}
	s.WriteString("\n\n")

	filtered := m.filteredItems()

	if len(filtered) == 0 {
		s.WriteString(selectorItemStyle.Render(selectorDescStyle.Render("(no matches)")))
		s.WriteString("\n")
	} else if m.filter != "" {
		s.WriteString(sectionHeaderStyle.Render("Top Results"))
		s.WriteString("\n")

		displayCount := min(len(filtered), maxSelectorItems)
		for i := range displayCount {
			idx := m.scrollOffset + i
			if idx >= len(filtered) {
				break
			}
			m.renderItem(&s, filtered[idx], idx)
		}

		if remaining := len(filtered) - m.scrollOffset - displayCount; remaining > 0 {
			s.WriteString(selectorMoreStyle.Render(fmt.Sprintf("... and %d more", remaining)))
			s.WriteString("\n")
		}
	} else {
		// Split into pinned recommended and scrollable others
		var recItems, otherItems []int
		for i, item := range filtered {
			if item.Recommended {
				recItems = append(recItems, i)
			} else {
				otherItems = append(otherItems, i)
			}
		}

		// Always render all recommended items (pinned)
		if len(recItems) > 0 {
			s.WriteString(sectionHeaderStyle.Render("Recommended"))
			s.WriteString("\n")
			for _, idx := range recItems {
				m.renderItem(&s, filtered[idx], idx)
			}
		}

		if len(otherItems) > 0 {
			s.WriteString("\n")
			s.WriteString(sectionHeaderStyle.Render("More"))
			s.WriteString("\n")

			maxOthers := maxSelectorItems - len(recItems)
			if maxOthers < 3 {
				maxOthers = 3
			}
			displayCount := min(len(otherItems), maxOthers)

			for i := range displayCount {
				idx := m.scrollOffset + i
				if idx >= len(otherItems) {
					break
				}
				m.renderItem(&s, filtered[otherItems[idx]], otherItems[idx])
			}

			if remaining := len(otherItems) - m.scrollOffset - displayCount; remaining > 0 {
				s.WriteString(selectorMoreStyle.Render(fmt.Sprintf("... and %d more", remaining)))
				s.WriteString("\n")
			}
		}
	}

	s.WriteString("\n")
	help := "↑/↓ navigate • enter select • esc cancel"
	if m.helpText != "" {
		help = m.helpText
	}
	s.WriteString(selectorHelpStyle.Render(help))

	return s.String()
}

func (m selectorModel) View() string {
	if m.cancelled || m.selected != "" {
		return ""
	}

	s := m.renderContent()
	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s)
	}
	return s
}

func SelectSingle(title string, items []SelectItem) (string, error) {
	if len(items) == 0 {
		return "", fmt.Errorf("no items to select from")
	}

	m := selectorModel{
		title: title,
		items: items,
	}

	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return "", fmt.Errorf("error running selector: %w", err)
	}

	fm := finalModel.(selectorModel)
	if fm.cancelled {
		return "", ErrCancelled
	}

	return fm.selected, nil
}

// multiSelectorModel is the bubbletea model for multi selection.
type multiSelectorModel struct {
	title        string
	items        []SelectItem
	itemIndex    map[string]int
	filter       string
	cursor       int
	scrollOffset int
	checked      map[int]bool
	checkOrder   []int
	cancelled    bool
	confirmed    bool
	width        int
}

func newMultiSelectorModel(title string, items []SelectItem, preChecked []string) multiSelectorModel {
	m := multiSelectorModel{
		title:     title,
		items:     items,
		itemIndex: make(map[string]int, len(items)),
		checked:   make(map[int]bool),
	}

	for i, item := range items {
		m.itemIndex[item.Name] = i
	}

	for _, name := range preChecked {
		if idx, ok := m.itemIndex[name]; ok {
			m.checked[idx] = true
			m.checkOrder = append(m.checkOrder, idx)
		}
	}

	return m
}

func (m multiSelectorModel) filteredItems() []SelectItem {
	if m.filter == "" {
		return m.items
	}
	filterLower := strings.ToLower(m.filter)
	var result []SelectItem
	for _, item := range m.items {
		if strings.Contains(strings.ToLower(item.Name), filterLower) {
			result = append(result, item)
		}
	}
	return result
}

// otherStart returns the index of the first non-recommended item in the filtered list.
func (m multiSelectorModel) otherStart() int {
	if m.filter != "" {
		return 0
	}
	filtered := m.filteredItems()
	for i, item := range filtered {
		if !item.Recommended {
			return i
		}
	}
	return len(filtered)
}

// updateScroll adjusts scrollOffset for section-based scrolling (matches single-select).
func (m *multiSelectorModel) updateScroll(otherStart int) {
	if m.filter != "" {
		if m.cursor < m.scrollOffset {
			m.scrollOffset = m.cursor
		}
		if m.cursor >= m.scrollOffset+maxSelectorItems {
			m.scrollOffset = m.cursor - maxSelectorItems + 1
		}
		return
	}

	if m.cursor < otherStart {
		m.scrollOffset = 0
		return
	}

	posInOthers := m.cursor - otherStart
	maxOthers := maxSelectorItems - otherStart
	if maxOthers < 3 {
		maxOthers = 3
	}
	if posInOthers < m.scrollOffset {
		m.scrollOffset = posInOthers
	}
	if posInOthers >= m.scrollOffset+maxOthers {
		m.scrollOffset = posInOthers - maxOthers + 1
	}
}

func (m *multiSelectorModel) toggleItem() {
	filtered := m.filteredItems()
	if len(filtered) == 0 || m.cursor >= len(filtered) {
		return
	}

	item := filtered[m.cursor]
	origIdx := m.itemIndex[item.Name]

	if m.checked[origIdx] {
		delete(m.checked, origIdx)
		for i, idx := range m.checkOrder {
			if idx == origIdx {
				m.checkOrder = append(m.checkOrder[:i], m.checkOrder[i+1:]...)
				break
			}
		}
	} else {
		m.checked[origIdx] = true
		m.checkOrder = append(m.checkOrder, origIdx)
	}
}

func (m multiSelectorModel) selectedCount() int {
	return len(m.checkOrder)
}

func (m multiSelectorModel) Init() tea.Cmd {
	return nil
}

func (m multiSelectorModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		wasSet := m.width > 0
		m.width = msg.Width
		if wasSet {
			return m, tea.EnterAltScreen
		}
		return m, nil

	case tea.KeyMsg:
		filtered := m.filteredItems()

		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			m.cancelled = true
			return m, tea.Quit

		case tea.KeyEnter:
			if len(m.checkOrder) > 0 {
				m.confirmed = true
				return m, tea.Quit
			}

		case tea.KeySpace:
			m.toggleItem()

		case tea.KeyUp:
			if m.cursor > 0 {
				m.cursor--
				m.updateScroll(m.otherStart())
			}

		case tea.KeyDown:
			if m.cursor < len(filtered)-1 {
				m.cursor++
				m.updateScroll(m.otherStart())
			}

		case tea.KeyPgUp:
			m.cursor -= maxSelectorItems
			if m.cursor < 0 {
				m.cursor = 0
			}
			m.updateScroll(m.otherStart())

		case tea.KeyPgDown:
			m.cursor += maxSelectorItems
			if m.cursor >= len(filtered) {
				m.cursor = len(filtered) - 1
			}
			m.updateScroll(m.otherStart())

		case tea.KeyBackspace:
			if len(m.filter) > 0 {
				m.filter = m.filter[:len(m.filter)-1]
				m.cursor = 0
				m.scrollOffset = 0
			}

		case tea.KeyRunes:
			m.filter += string(msg.Runes)
			m.cursor = 0
			m.scrollOffset = 0
		}
	}

	return m, nil
}

func (m multiSelectorModel) renderMultiItem(s *strings.Builder, item SelectItem, idx int) {
	origIdx := m.itemIndex[item.Name]

	var check string
	if m.checked[origIdx] {
		check = "[x] "
	} else {
		check = "[ ] "
	}

	suffix := ""
	if len(m.checkOrder) > 0 && m.checkOrder[0] == origIdx {
		suffix = " " + selectorDefaultTagStyle.Render("(default)")
	}

	if idx == m.cursor {
		s.WriteString(selectorSelectedItemStyle.Render("▸ " + check + item.Name))
	} else {
		s.WriteString(selectorItemStyle.Render(check + item.Name))
	}
	s.WriteString(suffix)
	s.WriteString("\n")
	if item.Description != "" {
		s.WriteString(selectorDescLineStyle.Render(item.Description))
		s.WriteString("\n")
	}
}

func (m multiSelectorModel) View() string {
	if m.cancelled || m.confirmed {
		return ""
	}

	var s strings.Builder

	s.WriteString(selectorTitleStyle.Render(m.title))
	s.WriteString(" ")
	if m.filter == "" {
		s.WriteString(selectorFilterStyle.Render("Type to filter..."))
	} else {
		s.WriteString(selectorInputStyle.Render(m.filter))
	}
	s.WriteString("\n\n")

	filtered := m.filteredItems()

	if len(filtered) == 0 {
		s.WriteString(selectorItemStyle.Render(selectorDescStyle.Render("(no matches)")))
		s.WriteString("\n")
	} else if m.filter != "" {
		// Filtering: flat scroll through all matches
		displayCount := min(len(filtered), maxSelectorItems)
		for i := range displayCount {
			idx := m.scrollOffset + i
			if idx >= len(filtered) {
				break
			}
			m.renderMultiItem(&s, filtered[idx], idx)
		}

		if remaining := len(filtered) - m.scrollOffset - displayCount; remaining > 0 {
			s.WriteString(selectorMoreStyle.Render(fmt.Sprintf("... and %d more", remaining)))
			s.WriteString("\n")
		}
	} else {
		// Split into pinned recommended and scrollable others (matches single-select layout)
		var recItems, otherItems []int
		for i, item := range filtered {
			if item.Recommended {
				recItems = append(recItems, i)
			} else {
				otherItems = append(otherItems, i)
			}
		}

		// Always render all recommended items (pinned)
		if len(recItems) > 0 {
			s.WriteString(sectionHeaderStyle.Render("Recommended"))
			s.WriteString("\n")
			for _, idx := range recItems {
				m.renderMultiItem(&s, filtered[idx], idx)
			}
		}

		if len(otherItems) > 0 {
			s.WriteString("\n")
			s.WriteString(sectionHeaderStyle.Render("More"))
			s.WriteString("\n")

			maxOthers := maxSelectorItems - len(recItems)
			if maxOthers < 3 {
				maxOthers = 3
			}
			displayCount := min(len(otherItems), maxOthers)

			for i := range displayCount {
				idx := m.scrollOffset + i
				if idx >= len(otherItems) {
					break
				}
				m.renderMultiItem(&s, filtered[otherItems[idx]], otherItems[idx])
			}

			if remaining := len(otherItems) - m.scrollOffset - displayCount; remaining > 0 {
				s.WriteString(selectorMoreStyle.Render(fmt.Sprintf("... and %d more", remaining)))
				s.WriteString("\n")
			}
		}
	}

	s.WriteString("\n")

	count := m.selectedCount()
	if count == 0 {
		s.WriteString(selectorDescStyle.Render("  Select at least one model."))
	} else {
		s.WriteString(selectorDescStyle.Render(fmt.Sprintf("  %d selected - press enter to continue", count)))
	}
	s.WriteString("\n\n")

	s.WriteString(selectorHelpStyle.Render("↑/↓ navigate • space toggle • enter confirm • esc cancel"))

	result := s.String()
	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(result)
	}
	return result
}

func SelectMultiple(title string, items []SelectItem, preChecked []string) ([]string, error) {
	if len(items) == 0 {
		return nil, fmt.Errorf("no items to select from")
	}

	m := newMultiSelectorModel(title, items, preChecked)

	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return nil, fmt.Errorf("error running selector: %w", err)
	}

	fm := finalModel.(multiSelectorModel)
	if fm.cancelled {
		return nil, ErrCancelled
	}

	if !fm.confirmed {
		return nil, ErrCancelled
	}

	var result []string
	for _, idx := range fm.checkOrder {
		result = append(result, fm.items[idx].Name)
	}

	return result, nil
}
