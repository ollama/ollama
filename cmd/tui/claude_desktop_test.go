package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/cmd/launch"
)

func TestClaudeDesktopMappingOverview(t *testing.T) {
	rows := []launch.ClaudeDesktopMapping{
		{ID: "fable", Name: "Fable 5", Model: "local-one", DefaultModel: "local-one"},
		{ID: "opus", Name: "Opus 5", DefaultModel: "local-two"},
	}
	m := newClaudeDesktopMappingModel(rows, "opus")
	view := m.View()
	for _, text := range []string{"Fable 5", "local-one (Default)", "Opus 5", "local-two (Default)"} {
		if !strings.Contains(view, text) {
			t.Fatalf("overview missing %q: %s", text, view)
		}
	}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRight})
	if got := updated.(claudeDesktopMappingModel); cmd == nil || got.choice.RouteID != "opus" || got.choice.Clear {
		t.Fatalf("selected mapping = %+v", got)
	}
	// Input queued after opening the picker must not turn that choice into a
	// save before the current program has exited.
	finished, _ := updated.(claudeDesktopMappingModel).Update(tea.KeyMsg{Type: tea.KeyEnter})
	if finished.(claudeDesktopMappingModel).choice.RouteID != "opus" {
		t.Fatal("queued Enter replaced the request to open the picker")
	}
	for i := range rows {
		m.cursor = i
		updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		got := updated.(claudeDesktopMappingModel)
		if cmd == nil || !got.done || got.cancelled || got.choice.RouteID != "" {
			t.Fatalf("Enter did not save from mapping row %d: %+v", i, got)
		}
	}
	m.cursor = len(rows) - 1
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	if updated.(claudeDesktopMappingModel).cursor != len(rows)-1 {
		t.Fatal("navigation continued beyond the mapping rows")
	}
}

func TestClaudeDesktopMappingCanSaveDefaults(t *testing.T) {
	m := newClaudeDesktopMappingModel([]launch.ClaudeDesktopMapping{{ID: "opus", Name: "Opus 5", DefaultModel: "local-two"}}, "")
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(claudeDesktopMappingModel)
	if cmd == nil || !got.done || got.cancelled || got.choice.RouteID != "" {
		t.Fatalf("default-only selection could not be saved: %+v", got)
	}
}

func TestClaudeDesktopMappingActions(t *testing.T) {
	rows := []launch.ClaudeDesktopMapping{{ID: "opus", Name: "Opus 5", Model: "local-one"}}
	for _, test := range []struct {
		name      string
		cursor    int
		key       tea.KeyType
		clear     bool
		cancelled bool
	}{
		{name: "clear", key: tea.KeyBackspace, clear: true},
		{name: "save", key: tea.KeyEnter},
		{name: "escape", key: tea.KeyEsc, cancelled: true},
		{name: "interrupt", key: tea.KeyCtrlC, cancelled: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			m := newClaudeDesktopMappingModel(rows, "")
			m.cursor = test.cursor
			updated, cmd := m.Update(tea.KeyMsg{Type: test.key})
			got := updated.(claudeDesktopMappingModel)
			if cmd == nil || got.cancelled != test.cancelled || got.choice.Clear != test.clear {
				t.Fatalf("action = %+v", got)
			}
			if test.clear && got.choice.RouteID != "opus" || !test.clear && got.choice.RouteID != "" {
				t.Fatalf("action route = %q", got.choice.RouteID)
			}
			if got.View() != "" {
				t.Fatal("finished editor remained visible")
			}
		})
	}
}

func TestClaudeDesktopMappingRequiresAtLeastOneModel(t *testing.T) {
	m := newClaudeDesktopMappingModel([]launch.ClaudeDesktopMapping{{ID: "opus", Name: "Opus 5"}}, "")
	// Clearing an unmapped row stays in the editor.
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	if cmd != nil || updated.(claudeDesktopMappingModel).done {
		t.Fatal("clearing an empty row left the editor")
	}
	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil || updated.(claudeDesktopMappingModel).done {
		t.Fatal("saved an empty model selection")
	}
}
