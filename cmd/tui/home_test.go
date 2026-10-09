package tui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestHomeMenuNavigation(t *testing.T) {
	for _, tt := range []struct {
		name     string
		keys     []tea.KeyType
		cursor   int
		selected bool
		action   homeAction
	}{
		{"run a model", []tea.KeyType{tea.KeyEnter}, 0, true, homeActionRunModel},
		{"run with an agent", []tea.KeyType{tea.KeyDown, tea.KeyEnter}, 1, true, homeActionLaunchAgent},
		{"back to run a model", []tea.KeyType{tea.KeyDown, tea.KeyUp, tea.KeyEnter}, 0, true, homeActionRunModel},
		{"first item boundary", []tea.KeyType{tea.KeyUp, tea.KeyEnter}, 0, true, homeActionRunModel},
		{"last item boundary", []tea.KeyType{tea.KeyDown, tea.KeyDown, tea.KeyEnter}, 1, true, homeActionLaunchAgent},
		{"escape", []tea.KeyType{tea.KeyEsc}, 0, false, homeActionNone},
		{"interrupt", []tea.KeyType{tea.KeyDown, tea.KeyCtrlC}, 1, false, homeActionNone},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := homeModel{}
			var cmd tea.Cmd
			for _, key := range tt.keys {
				var updated tea.Model
				updated, cmd = m.Update(tea.KeyMsg{Type: key})
				m = updated.(homeModel)
			}
			if m.cursor != tt.cursor || m.selected != tt.selected {
				t.Fatalf("cursor=%d selected=%v, want cursor=%d selected=%v", m.cursor, m.selected, tt.cursor, tt.selected)
			}
			if m.selected && homeItems[m.cursor].action != tt.action {
				t.Fatalf("selected action=%v, want %v", homeItems[m.cursor].action, tt.action)
			}
			if !m.quitting || cmd == nil {
				t.Fatal("selection or cancellation should close the menu")
			}
			if _, ok := cmd().(tea.QuitMsg); !ok {
				t.Fatal("expected the TUI to quit")
			}
		})
	}
}

func TestAppMenuBackNavigation(t *testing.T) {
	for _, tt := range []struct {
		name      string
		canGoBack bool
		key       tea.KeyType
		wantBack  bool
		wantQuit  bool
	}{
		{"nested escape", true, tea.KeyEsc, true, true},
		{"nested left", true, tea.KeyLeft, true, true},
		{"nested interrupt", true, tea.KeyCtrlC, false, true},
		{"direct escape", false, tea.KeyEsc, false, true},
		{"direct left", false, tea.KeyLeft, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := newModel(launcherTestState())
			m.canGoBack = tt.canGoBack
			updated, _ := m.Update(tea.KeyMsg{Type: tt.key})
			got := updated.(model)
			if got.goBack != tt.wantBack || got.quitting != tt.wantQuit || got.selected {
				t.Fatalf("goBack=%v quitting=%v selected=%v, want goBack=%v quitting=%v selected=false", got.goBack, got.quitting, got.selected, tt.wantBack, tt.wantQuit)
			}
		})
	}
}
