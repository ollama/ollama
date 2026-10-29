package tui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func updateWelcome(m *welcomeModel, msg tea.Msg) tea.Cmd {
	updated, cmd := m.Update(msg)
	*m = updated.(welcomeModel)
	return cmd
}

func TestWelcomeContinue(t *testing.T) {
	m := welcomeModel{}
	quit := updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter})
	if !m.continued || m.cancelled || quit == nil {
		t.Fatal("Enter must complete welcome and continue to the launcher")
	}
	if _, ok := quit().(tea.QuitMsg); !ok {
		t.Fatal("completed welcome must exit the program")
	}
}

func TestWelcomeCancellation(t *testing.T) {
	for _, key := range []tea.KeyType{tea.KeyEsc, tea.KeyCtrlC} {
		m := welcomeModel{}
		quit := updateWelcome(&m, tea.KeyMsg{Type: key})
		if m.continued || !m.cancelled || quit == nil {
			t.Fatalf("cancellation incorrectly completed onboarding: %+v", m)
		}
	}
}

func TestWelcomeCompletedInApp(t *testing.T) {
	m := welcomeModel{options: WelcomeOptions{IsCompleted: func() bool { return true }}}
	if m.Init() == nil {
		t.Fatal("welcome must watch shared completion")
	}
	if quit := updateWelcome(&m, welcomeCompletedMsg(true)); !m.continued || quit == nil {
		t.Fatal("app completion must dismiss CLI onboarding")
	}
}
