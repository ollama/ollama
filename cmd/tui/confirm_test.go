package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestConfirmModel_DefaultsToYes(t *testing.T) {
	m := confirmModel{prompt: "Download test?", yes: true}
	if !m.yes {
		t.Error("should default to yes")
	}
}

func TestConfirmModel_View_ContainsPrompt(t *testing.T) {
	m := confirmModel{prompt: "Download qwen3:8b?", yes: true}
	got := m.View()
	if !strings.Contains(got, "Download qwen3:8b?") {
		t.Error("should contain the prompt text")
	}
}

func TestConfirmModel_View_ContainsButtons(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}
	got := m.View()
	if !strings.Contains(got, "Yes") {
		t.Error("should contain Yes button")
	}
	if !strings.Contains(got, "No") {
		t.Error("should contain No button")
	}
}

func TestConfirmModel_View_ContainsHelp(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}
	got := m.View()
	if !strings.Contains(got, "enter confirm") {
		t.Error("should contain help text")
	}
}

func TestConfirmModel_View_ClearsAfterConfirm(t *testing.T) {
	m := confirmModel{prompt: "Download?", confirmed: true}
	if m.View() != "" {
		t.Error("View should return empty string after confirmation")
	}
}

func TestConfirmModel_View_ClearsAfterCancel(t *testing.T) {
	m := confirmModel{prompt: "Download?", cancelled: true}
	if m.View() != "" {
		t.Error("View should return empty string after cancellation")
	}
}

func TestConfirmModel_EnterConfirmsYes(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	fm := updated.(confirmModel)
	if !fm.confirmed {
		t.Error("enter should set confirmed=true")
	}
	if !fm.yes {
		t.Error("enter with yes selected should keep yes=true")
	}
	if cmd == nil {
		t.Error("enter should return tea.Quit")
	}
}

func TestConfirmModel_EnterConfirmsNo(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: false}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	fm := updated.(confirmModel)
	if !fm.confirmed {
		t.Error("enter should set confirmed=true")
	}
	if fm.yes {
		t.Error("enter with no selected should keep yes=false")
	}
	if cmd == nil {
		t.Error("enter should return tea.Quit")
	}
}

func TestConfirmModel_EscCancels(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	fm := updated.(confirmModel)
	if !fm.cancelled {
		t.Error("esc should set cancelled=true")
	}
	if cmd == nil {
		t.Error("esc should return tea.Quit")
	}
}

func TestConfirmModel_CtrlCCancels(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	fm := updated.(confirmModel)
	if !fm.cancelled {
		t.Error("ctrl+c should set cancelled=true")
	}
	if cmd == nil {
		t.Error("ctrl+c should return tea.Quit")
	}
}

func TestConfirmModel_NCancels(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'n'}})
	fm := updated.(confirmModel)
	if !fm.cancelled {
		t.Error("'n' should set cancelled=true")
	}
	if cmd == nil {
		t.Error("'n' should return tea.Quit")
	}
}

func TestConfirmModel_YConfirmsYes(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: false}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'y'}})
	fm := updated.(confirmModel)
	if !fm.confirmed {
		t.Error("'y' should set confirmed=true")
	}
	if !fm.yes {
		t.Error("'y' should set yes=true")
	}
	if cmd == nil {
		t.Error("'y' should return tea.Quit")
	}
}

func TestConfirmModel_ArrowKeysNavigate(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}

	// Right moves to No
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'l'}})
	fm := updated.(confirmModel)
	if fm.yes {
		t.Error("right/l should move to No")
	}
	if fm.confirmed || fm.cancelled {
		t.Error("navigation should not confirm or cancel")
	}

	// Left moves back to Yes
	updated, _ = fm.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'h'}})
	fm = updated.(confirmModel)
	if !fm.yes {
		t.Error("left/h should move to Yes")
	}
}

func TestConfirmModel_TabToggles(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	fm := updated.(confirmModel)
	if fm.yes {
		t.Error("tab should toggle from Yes to No")
	}

	updated, _ = fm.Update(tea.KeyMsg{Type: tea.KeyTab})
	fm = updated.(confirmModel)
	if !fm.yes {
		t.Error("tab should toggle from No to Yes")
	}
}

func TestConfirmModel_WindowSizeUpdatesWidth(t *testing.T) {
	m := confirmModel{prompt: "Download?"}
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	fm := updated.(confirmModel)
	if fm.width != 100 {
		t.Errorf("expected width 100, got %d", fm.width)
	}
}

func TestConfirmModel_ResizeEntersAltScreen(t *testing.T) {
	m := confirmModel{prompt: "Download?", width: 80}
	_, cmd := m.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	if cmd == nil {
		t.Error("resize (width already set) should return a command")
	}
}

func TestConfirmModel_InitialWindowSizeNoAltScreen(t *testing.T) {
	m := confirmModel{prompt: "Download?"}
	_, cmd := m.Update(tea.WindowSizeMsg{Width: 80, Height: 40})
	if cmd != nil {
		t.Error("initial WindowSizeMsg should not return a command")
	}
}

func TestConfirmModel_ViewMaxWidth(t *testing.T) {
	m := confirmModel{prompt: "Download?", yes: true, width: 40}
	got := m.View()
	// Just ensure it doesn't panic and returns content
	if got == "" {
		t.Error("View with width set should still return content")
	}
}
