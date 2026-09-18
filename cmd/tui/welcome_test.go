package tui

import (
	"errors"
	"net/url"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func updateWelcome(m *welcomeModel, msg tea.Msg) tea.Cmd {
	updated, cmd := m.Update(msg)
	*m = updated.(welcomeModel)
	return cmd
}

func TestWelcomeAccountRouting(t *testing.T) {
	for _, account := range []WelcomeAccount{
		{SignedIn: true},
		{CloudDisabled: true},
		{SigninURL: "https://ollama.com/connect?key=test"},
		{Err: errors.New("offline")},
	} {
		checks := 0
		m := welcomeModel{options: WelcomeOptions{
			CheckAccount: func() WelcomeAccount { checks++; return account },
			OpenBrowser:  func(string) { t.Fatal("Continue must not open the browser") },
		}}
		check := updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter})
		if check == nil || m.step != welcomeIntro || m.continued {
			t.Fatal("Continue must check the account before advancing")
		}
		if updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter}) != nil {
			t.Fatal("repeated Enter started another check")
		}
		updateWelcome(&m, check())
		if checks != 1 || m.continued != (account.SignedIn || account.CloudDisabled) {
			t.Fatalf("incorrect account routing: %+v", m)
		}
		if !account.SignedIn && !account.CloudDisabled {
			if m.step != welcomeAccount {
				t.Fatal("signed-out users must reach account choices")
			}
			for range len(m.accountChoices()) - 1 {
				updateWelcome(&m, tea.KeyMsg{Type: tea.KeyDown})
			}
			updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter})
			if !m.continued {
				t.Fatal("continuing without an account must finish onboarding, even offline")
			}
		}
	}
}

func TestWelcomeWaitsForSignIn(t *testing.T) {
	opened := ""
	m := welcomeModel{
		step: welcomeAccount, account: WelcomeAccount{SigninURL: "https://ollama.com/connect?key=test"},
		options: WelcomeOptions{OpenBrowser: func(raw string) { opened = raw }},
	}
	start := updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter})
	start().(tea.BatchMsg)[0]() // Open the browser without starting real account polling.
	updateWelcome(&m, signInCheckMsg{})
	if opened != m.signIn.signInURL || opened == "" || m.step != welcomeSignIn || m.continued {
		t.Fatal("opening the browser alone must not complete onboarding")
	}
	updateWelcome(&m, signInCheckMsg{signedIn: true, userName: "test-user"})
	if !m.continued {
		t.Fatal("confirmed sign-in must complete onboarding")
	}
}

func TestWelcomeAccountDestination(t *testing.T) {
	opened := ""
	m := welcomeModel{
		step:    welcomeAccount,
		account: WelcomeAccount{SigninURL: "https://ollama.com/connect?key=test&launch=claude&signup=true"},
		options: WelcomeOptions{OpenBrowser: func(raw string) { opened = raw }},
	}
	start := updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter})
	start().(tea.BatchMsg)[0]()
	u, err := url.Parse(opened)
	if err != nil {
		t.Fatal(err)
	}
	if u.Path != "/connect" || u.Query().Has("signup") || u.Query().Has("launch") || u.Query().Get("key") != "test" {
		t.Fatalf("incorrect account destination: %s", opened)
	}
}

func TestWelcomeCancellation(t *testing.T) {
	for _, step := range []welcomeStep{welcomeIntro, welcomeAccount, welcomeSignIn} {
		for _, key := range []tea.KeyType{tea.KeyEsc, tea.KeyCtrlC} {
			m := welcomeModel{step: step}
			quit := updateWelcome(&m, tea.KeyMsg{Type: key})
			back := step == welcomeSignIn && key == tea.KeyEsc
			if m.continued || m.cancelled == back || (quit == nil) != back || (back && m.step != welcomeAccount) {
				t.Fatalf("cancellation incorrectly completed onboarding: %+v", m)
			}
		}
	}
}

func TestWelcomeInvalidSignInLink(t *testing.T) {
	m := welcomeModel{
		step: welcomeAccount, account: WelcomeAccount{SigninURL: "file:///tmp/connect"},
		options: WelcomeOptions{OpenBrowser: func(string) { t.Fatal("invalid sign-in link opened") }},
	}
	if cmd := updateWelcome(&m, tea.KeyMsg{Type: tea.KeyEnter}); cmd != nil || m.account.Err == nil || m.continued {
		t.Fatal("invalid sign-in link must leave onboarding pending")
	}
}

func TestWelcomeCompletedInApp(t *testing.T) {
	for _, step := range []welcomeStep{welcomeIntro, welcomeAccount, welcomeSignIn} {
		m := welcomeModel{step: step, options: WelcomeOptions{IsCompleted: func() bool { return true }}}
		if m.Init() == nil {
			t.Fatal("welcome must watch shared completion")
		}
		if quit := updateWelcome(&m, welcomeCompletedMsg(true)); !m.continued || quit == nil {
			t.Fatal("app completion must dismiss CLI onboarding")
		}
	}
}
