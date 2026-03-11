package tui

import (
	"context"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ollama/ollama/api"
)

// ── styles ─────────────────────────────────────────────────────────────────

var (
	fitTabActiveStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "235", Dark: "252"}).
				Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"}).
				PaddingLeft(1).
				PaddingRight(1)

	fitTabInactiveStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
				PaddingLeft(1).
				PaddingRight(1)

	fitTabBarStyle = lipgloss.NewStyle().
			PaddingBottom(1)

	fitModelSelectedStyle = lipgloss.NewStyle().
				Bold(true).
				Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	fitModelStyle = lipgloss.NewStyle().
			PaddingLeft(2)

	fitModelCheckedStyle = fitModelStyle.
				Foreground(lipgloss.AdaptiveColor{Light: "28", Dark: "120"})

	fitDescStyle = lipgloss.NewStyle().
			PaddingLeft(6).
			Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	fitInstalledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "28", Dark: "120"}).
				Bold(true)

	fitHelpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "244", Dark: "244"})

	fitHWStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "240", Dark: "249"}).
			PaddingLeft(1)

	fitErrorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "196", Dark: "196"})
)

// ── tiers ──────────────────────────────────────────────────────────────────

const (
	fitTabTooLarge = iota // index 0 = shown last
	fitTabPossible
	fitTabMarginal
	fitTabGood
	fitTabIdeal
)

// tabDefs defines the display order (Ideal first on screen → index 4 in API).
// We reverse the iota above so Ideal tab is leftmost (#0 on screen).
var fitTabDefs = []struct {
	label  string
	apiIdx int // api.FitModelCandidate.Tier value
}{
	{"✅ Ideal", 0},
	{"🟡 Good", 1},
	{"🟠 Marginal", 2},
	{"⬜ Possible", 3},
	{"🔴 Too Large", 4},
}

// ── messages ───────────────────────────────────────────────────────────────

type fitLoadedMsg struct {
	resp *api.FitResponse
}

type fitLoadErrMsg struct {
	err error
}

// ── model ──────────────────────────────────────────────────────────────────

// FitCheckModel is a bubbletea model for the interactive fit-check TUI.
// It is embedded in the main TUI model and never run standalone.
type FitCheckModel struct {
	// data
	resp *api.FitResponse
	tabs [5][]api.FitModelCandidate // indexed by fitTab* constants (0=Ideal…4=TooLarge)
	err  error

	// navigation
	activeTab int // 0 = Ideal
	cursor    int

	// selection (multi-select)
	checked   map[string]bool
	confirmed bool
	cancelled bool

	// ui
	width int
}

func NewFitCheckModel() FitCheckModel {
	return FitCheckModel{
		checked: make(map[string]bool),
	}
}

// ── init / load ─────────────────────────────────────────────────────────────

func (m FitCheckModel) Init() tea.Cmd {
	return loadFitData
}

func loadFitData() tea.Msg {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fitLoadErrMsg{err}
	}
	resp, err := client.Fit(context.Background(), api.FitRequest{All: true})
	if err != nil {
		return fitLoadErrMsg{err}
	}
	return fitLoadedMsg{resp}
}

// ── update ──────────────────────────────────────────────────────────────────

func (m FitCheckModel) Update(msg tea.Msg) (FitCheckModel, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width

	case fitLoadedMsg:
		m.resp = msg.resp
		// bucket models into tabs
		for i := range m.tabs {
			m.tabs[i] = nil
		}
		for _, c := range m.resp.Models {
			if c.Tier >= 0 && c.Tier < len(m.tabs) {
				m.tabs[c.Tier] = append(m.tabs[c.Tier], c)
			}
		}
		// start on first non-empty tab (ideally Ideal)
		m.activeTab = 0
		m.cursor = 0
		for i := range fitTabDefs {
			if len(m.tabs[i]) > 0 {
				m.activeTab = i
				break
			}
		}

	case fitLoadErrMsg:
		m.err = msg.err

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyEsc:
			m.cancelled = true

		case tea.KeyLeft, tea.KeyShiftLeft:
			if m.activeTab > 0 {
				m.activeTab--
				m.cursor = 0
			}

		case tea.KeyRight:
			if m.activeTab < len(fitTabDefs)-1 {
				m.activeTab++
				m.cursor = 0
			}

		case tea.KeyUp:
			if m.cursor > 0 {
				m.cursor--
			}

		case tea.KeyDown:
			if items := m.tabs[m.activeTab]; m.cursor < len(items)-1 {
				m.cursor++
			}

		case tea.KeySpace:
			if items := m.tabs[m.activeTab]; len(items) > 0 && m.cursor < len(items) {
				name := items[m.cursor].Req.Name
				if m.checked[name] {
					delete(m.checked, name)
				} else {
					m.checked[name] = true
				}
			}

		case tea.KeyEnter:
			if len(m.checked) > 0 {
				m.confirmed = true
			}

		case tea.KeyRunes:
			// ←/→ via h/l vim keys
			switch string(msg.Runes) {
			case "h":
				if m.activeTab > 0 {
					m.activeTab--
					m.cursor = 0
				}
			case "l":
				if m.activeTab < len(fitTabDefs)-1 {
					m.activeTab++
					m.cursor = 0
				}
			}
		}
	}
	return m, nil
}

// Selected returns the names of all checked models in selection order.
func (m FitCheckModel) Selected() []string {
	var out []string
	for name := range m.checked {
		out = append(out, name)
	}
	return out
}

// ── view ────────────────────────────────────────────────────────────────────

func (m FitCheckModel) View() string {
	if m.cancelled || m.confirmed {
		return ""
	}

	var s strings.Builder

	// title
	s.WriteString(selectorTitleStyle.Render("Ollama Fit Check"))
	s.WriteString("\n\n")

	if m.resp == nil && m.err == nil {
		s.WriteString("  Loading hardware profile...\n")
		s.WriteString("\n" + fitHelpStyle.Render("esc  cancel"))
		return m.maybeWrap(s.String())
	}

	if m.err != nil {
		s.WriteString(fitErrorStyle.Render("  Error: "+m.err.Error()) + "\n")
		s.WriteString("\n" + fitHelpStyle.Render("esc  back"))
		return m.maybeWrap(s.String())
	}

	// hardware summary line
	hw := m.resp.System
	if hw.BestGPU != nil {
		s.WriteString(fitHWStyle.Render(fmt.Sprintf(
			"GPU: %s %s  •  %.1f GB free",
			hw.BestGPU.Library, hw.BestGPU.Name,
			float64(hw.BestGPU.FreeMemory)/(1024*1024*1024),
		)) + "\n")
	} else {
		gpuMsg := "GPU: none detected"
		s.WriteString(fitHWStyle.Render(fmt.Sprintf(
			"%s   RAM: %.1f GB free   Disk: %.1f GB free",
			gpuMsg,
			float64(hw.RAMAvailableBytes)/(1024*1024*1024),
			float64(hw.DiskModelAvailBytes)/(1024*1024*1024),
		)) + "\n")
	}
	s.WriteString("\n")

	// ── tab bar ─────────────────────────────────────────────────────────────
	var tabs []string
	for i, td := range fitTabDefs {
		count := len(m.tabs[i])
		label := fmt.Sprintf("%s (%d)", td.label, count)
		if i == m.activeTab {
			tabs = append(tabs, fitTabActiveStyle.Render(label))
		} else {
			tabs = append(tabs, fitTabInactiveStyle.Render(label))
		}
	}
	s.WriteString(fitTabBarStyle.Render("  " + strings.Join(tabs, "  ")))
	s.WriteString("\n\n")

	// ── model list ───────────────────────────────────────────────────────────
	items := m.tabs[m.activeTab]
	if len(items) == 0 {
		s.WriteString(selectorItemStyle.Render(selectorDescStyle.Render("(none)")) + "\n")
	} else {
		const maxVisible = 14
		start := 0
		if m.cursor >= maxVisible {
			start = m.cursor - maxVisible + 1
		}
		end := start + maxVisible
		if end > len(items) {
			end = len(items)
		}

		for idx := start; idx < end; idx++ {
			c := items[idx]
			name := c.Req.Name

			// checkbox prefix
			check := "[ ] "
			if m.checked[name] {
				check = "[x] "
			}

			// installed marker
			suffix := ""
			if c.Installed {
				suffix = " " + fitInstalledStyle.Render("✓ installed")
			}

			// row
			row := check + name + suffix
			if idx == m.cursor {
				s.WriteString(fitModelSelectedStyle.Render("▸ "+row) + "\n")
			} else if m.checked[name] {
				s.WriteString(fitModelCheckedStyle.Render("  "+row) + "\n")
			} else {
				s.WriteString(fitModelStyle.Render(row) + "\n")
			}

			// description line
			desc := buildFitDesc(c)
			s.WriteString(fitDescStyle.Render(desc) + "\n")

			// warning notes
			for _, note := range c.Notes {
				s.WriteString(fitDescStyle.Render("  ⚠  " + note) + "\n")
			}
		}

		// scroll indicator
		if start > 0 || end < len(items) {
			remaining := len(items) - end
			if remaining > 0 {
				s.WriteString(selectorMoreStyle.Render(fmt.Sprintf("... and %d more ↓", remaining)) + "\n")
			}
		}
	}

	// ── footer ───────────────────────────────────────────────────────────────
	s.WriteString("\n")
	sel := len(m.checked)
	if sel > 0 {
		s.WriteString(selectorDescStyle.Render(fmt.Sprintf("  %d selected — press enter to pull", sel)) + "\n\n")
		s.WriteString(fitHelpStyle.Render("←/→ tabs • ↑/↓ navigate • space toggle • enter pull • esc back"))
	} else {
		s.WriteString(fitHelpStyle.Render("←/→ tabs • ↑/↓ navigate • space select • enter pull • esc back"))
	}

	return m.maybeWrap(s.String())
}

func (m FitCheckModel) maybeWrap(s string) string {
	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s)
	}
	return s
}

// buildFitDesc builds the single description line shown below a model row.
func buildFitDesc(c api.FitModelCandidate) string {
	size := fitFormatBytes(c.Req.DiskSizeMB * 1024 * 1024)
	if c.Installed {
		size = "installed"
	}
	return fmt.Sprintf("%s • %s • ~%d tok/s • %s",
		c.Req.Quant, size, c.EstTPS, c.RunMode)
}

// fitFormatBytes formats MB-based byte count into a human string.
func fitFormatBytes(b uint64) string {
	const (
		mb = uint64(1024 * 1024)
		gb = uint64(1024 * 1024 * 1024)
	)
	if b >= gb {
		return fmt.Sprintf("%.1f GB", float64(b)/float64(gb))
	}
	return fmt.Sprintf("%.0f MB", float64(b)/float64(mb))
}
