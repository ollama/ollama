package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/cmd/launch"
)

func TestRenderSignIn_ContainsModelName(t *testing.T) {
	got := renderSignIn("glm-4.7:cloud", "https://example.com/signin", 0, 80)
	if !strings.Contains(got, "glm-4.7:cloud") {
		t.Error("should contain model name")
	}
	if !strings.Contains(got, "please sign in") {
		t.Error("should contain sign-in prompt")
	}
}

func TestRenderSignIn_ContainsURL(t *testing.T) {
	url := "https://ollama.com/connect?key=abc123"
	got := renderSignIn("test:cloud", url, 0, 120)
	if !strings.Contains(got, url) {
		t.Errorf("should contain URL %q", url)
	}
}

func TestRenderSignIn_ContainsSpinner(t *testing.T) {
	got := renderSignIn("test:cloud", "https://example.com", 0, 80)
	if !strings.Contains(got, "Waiting for sign in to complete") {
		t.Error("should contain waiting message")
	}
	if !strings.Contains(got, "⠋") {
		t.Error("should contain first spinner frame at spinner=0")
	}
}

func TestRenderSignIn_SpinnerAdvances(t *testing.T) {
	got0 := renderSignIn("test:cloud", "https://example.com", 0, 80)
	got1 := renderSignIn("test:cloud", "https://example.com", 1, 80)
	if got0 == got1 {
		t.Error("different spinner values should produce different output")
	}
}

func TestRenderSignIn_ContainsEscHelp(t *testing.T) {
	got := renderSignIn("test:cloud", "https://example.com", 0, 80)
	if !strings.Contains(got, "esc cancel") {
		t.Error("should contain esc cancel help text")
	}
}

func TestRenderUpgrade_AsksBeforeOpening(t *testing.T) {
	got := renderUpgrade("kimi-k2.6:cloud", 0, 80, false, true)
	if !strings.Contains(got, "kimi-k2.6:cloud") {
		t.Error("should contain model name")
	}
	if !strings.Contains(got, launch.DefaultUpgradeURL) {
		t.Error("should contain upgrade URL")
	}
	if !strings.Contains(got, "Open now?") {
		t.Error("should ask before opening")
	}
	if !strings.Contains(got, "Yes") || !strings.Contains(got, "No") {
		t.Error("should show yes/no selector")
	}
	if strings.Contains(got, "Waiting for upgrade to complete") {
		t.Error("should not start waiting before open choice is confirmed")
	}
}

func TestRenderUpgrade_PollingShowsWaiting(t *testing.T) {
	got := renderUpgrade("kimi-k2.6:cloud", 0, 80, true, true)
	if !strings.Contains(got, "Waiting for upgrade to complete") {
		t.Error("should contain waiting message")
	}
	if strings.Contains(got, "Open now?") {
		t.Error("should not show open prompt while polling")
	}
}

func TestSignInModel_EscCancels(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	fm := updated.(signInModel)
	if !fm.cancelled {
		t.Error("esc should set cancelled=true")
	}
	if cmd == nil {
		t.Error("esc should return tea.Quit")
	}
}

func TestUpgradeModel_NoCancelsWithoutPolling(t *testing.T) {
	m := upgradeModel{
		modelName:    "kimi-k2.6:cloud",
		requiredPlan: "pro",
		openNow:      true,
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRight})
	fm := updated.(upgradeModel)
	if fm.openNow {
		t.Error("right should select no")
	}
	if fm.polling {
		t.Error("right should not start polling")
	}

	updated, cmd := fm.Update(tea.KeyMsg{Type: tea.KeyEnter})
	fm = updated.(upgradeModel)
	if !fm.cancelled {
		t.Error("enter on no should cancel")
	}
	if fm.polling {
		t.Error("enter on no should not start polling")
	}
	if cmd == nil {
		t.Error("enter on no should quit")
	}
}

func TestSignInModel_CtrlCCancels(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	fm := updated.(signInModel)
	if !fm.cancelled {
		t.Error("ctrl+c should set cancelled=true")
	}
	if cmd == nil {
		t.Error("ctrl+c should return tea.Quit")
	}
}

func TestSignInModel_SignedInQuitsClean(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
	}

	updated, cmd := m.Update(signInCheckMsg{signedIn: true, userName: "alice"})
	fm := updated.(signInModel)
	if fm.userName != "alice" {
		t.Errorf("expected userName 'alice', got %q", fm.userName)
	}
	if cmd == nil {
		t.Error("successful sign-in should return tea.Quit")
	}
}

func TestSignInModel_SignedInViewClears(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
		userName:  "alice",
	}

	got := m.View()
	if got != "" {
		t.Errorf("View should return empty string after sign-in, got %q", got)
	}
}

func TestSignInModel_NotSignedInContinues(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
	}

	updated, _ := m.Update(signInCheckMsg{signedIn: false})
	fm := updated.(signInModel)
	if fm.userName != "" {
		t.Error("should not set userName when not signed in")
	}
	if fm.cancelled {
		t.Error("should not cancel when check returns not signed in")
	}
}

func TestSignInModel_WindowSizeUpdatesWidth(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
	}

	updated, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	fm := updated.(signInModel)
	if fm.width != 120 {
		t.Errorf("expected width 120, got %d", fm.width)
	}
}

func TestSignInModel_TickAdvancesSpinner(t *testing.T) {
	m := signInModel{
		modelName: "test:cloud",
		signInURL: "https://example.com",
		spinner:   0,
	}

	updated, cmd := m.Update(signInTickMsg{})
	fm := updated.(signInModel)
	if fm.spinner != 1 {
		t.Errorf("expected spinner=1, got %d", fm.spinner)
	}
	if cmd == nil {
		t.Error("tick should return a command")
	}
}
