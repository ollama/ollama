package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
)

func TestWelcomeCloudStatus(t *testing.T) {
	for _, status := range []string{`{"cloud":{"disabled":true}}`, `{"cloud":{"disabled":false}}`, "unavailable"} {
		t.Run(status, func(t *testing.T) {
			accountRequests := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/":
				case "/api/status":
					if status == "unavailable" {
						w.WriteHeader(http.StatusNotFound)
						return
					}
					fmt.Fprint(w, status)
				case "/api/me":
					accountRequests++
					fmt.Fprint(w, `{"name":"test-user"}`)
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			account := checkWelcomeAccount(context.Background())
			disabled := status == `{"cloud":{"disabled":true}}`
			if account.Err != nil || account.CloudDisabled != disabled || account.SignedIn == disabled {
				t.Fatalf("unexpected account state: %+v", account)
			}
			wantRequests := 1
			if disabled {
				wantRequests = 0
			}
			if accountRequests != wantRequests {
				t.Fatalf("account requests = %d, want %d", accountRequests, wantRequests)
			}
		})
	}
}

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
