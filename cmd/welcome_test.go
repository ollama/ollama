package cmd

import (
	"errors"
	"testing"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
)

func TestWelcomeOnceAfterCompletion(t *testing.T) {
	setCmdTestHome(t, t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	if err := config.SaveIntegration("claude", []string{"saved-model"}); err != nil {
		t.Fatal(err)
	}
	shows := 0
	cancelled := true
	show := func() error {
		shows++
		if cancelled {
			return launch.ErrCancelled
		}
		return nil
	}
	if err := ensureWelcome(show); !errors.Is(err, launch.ErrCancelled) || shows != 1 {
		t.Fatalf("cancelled welcome: shows=%d err=%v", shows, err)
	}
	cancelled = false
	if err := ensureWelcome(show); err != nil || shows != 2 {
		t.Fatalf("unfinished welcome must reappear: shows=%d err=%v", shows, err)
	}
	if err := ensureWelcome(show); err != nil || shows != 2 {
		t.Fatalf("completed welcome must not repeat: shows=%d err=%v", shows, err)
	}
}

func TestInstallerOnboarding(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		completed, interactive, want bool
		ci, noStart                  string
	}{
		{name: "first install", interactive: true, want: true},
		{name: "completed", completed: true, interactive: true},
		{name: "headless"},
		{name: "CI", interactive: true, ci: "true"},
		{name: "no-start", interactive: true, noStart: "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setCmdTestHome(t, t.TempDir())
			t.Setenv("LOCALAPPDATA", t.TempDir())
			t.Setenv("CI", tc.ci)
			t.Setenv("OLLAMA_NO_START", tc.noStart)
			if tc.completed {
				if err := config.CompleteWelcome(); err != nil {
					t.Fatal(err)
				}
			}
			if got, err := needsInstallerOnboarding(tc.interactive); err != nil || got != tc.want {
				t.Fatalf("onboarding=%v err=%v; want %v", got, err, tc.want)
			}
		})
	}
}
