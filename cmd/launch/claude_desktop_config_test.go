package launch

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"testing"
)

func claudeDesktopEditorClient(t *testing.T) *launcherClient {
	t.Helper()
	setTestHome(t, t.TempDir())
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	DefaultSingleSelectorWithUpdates = nil
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"local-one"},{"name":"local-two"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"capabilities":["completion","tools"]}`)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"fixture-user","plan":"pro"}`)
		case "/api/experimental/model-recommendations":
			if r.URL.Query().Get("app") == "claude-desktop" {
				if r.URL.Query().Get("ts") == "" || r.Header.Get("Authorization") == "" {
					t.Error("Claude defaults request must use the app's signed request format")
				}
				fmt.Fprint(w, `{"recommendations":[{"model":"test:cloud","required_plan":"pro"}],"mappings":{"claude-sonnet-5":{"model":"test:cloud"}}}`)
			} else {
				fmt.Fprint(w, `{"recommendations":[]}`)
			}
		default:
			t.Errorf("unexpected editor request: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	t.Setenv("OLLAMA_HOST", server.URL)
	client, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestClaudeDesktopEditorDraft(t *testing.T) {
	for _, cancel := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel_%t", cancel), func(t *testing.T) {
			client := claudeDesktopEditorClient(t)
			initial := map[string]string{"claude-opus-5": "local-one", "claude-sonnet-5": "local-one"}
			original := maps.Clone(initial)
			defaults := map[string]string{"claude-sonnet-5": "local-two", "claude-haiku-4-5-20251001": "local-two"}
			step, picks := 0, 0
			DefaultSingleSelector = func(_ string, _ []SelectionItem, current string) (string, error) {
				picks++
				if picks == 1 {
					return "", ErrCancelled // Cancelling one picker returns to the draft.
				}
				if current != "local-one" {
					t.Fatalf("picker current = %q", current)
				}
				return "local-two", nil
			}
			DefaultClaudeDesktopMappingSelector = func(rows []ClaudeDesktopMapping, current string) (ClaudeDesktopMappingChoice, error) {
				step++
				if !maps.Equal(initial, original) {
					t.Fatal("editing mutated the saved selection")
				}
				if len(rows) != 5 {
					t.Fatalf("editor has %d slots", len(rows))
				}
				switch step {
				case 1, 2:
					if rows[1].Model != "local-one" {
						t.Fatal("cancelled picker changed the draft")
					}
					return ClaudeDesktopMappingChoice{RouteID: "claude-opus-5"}, nil
				case 3:
					if rows[1].Model != "local-two" || current != "claude-opus-5" {
						t.Fatalf("edited draft = %+v, cursor %q", rows, current)
					}
					return ClaudeDesktopMappingChoice{RouteID: "claude-sonnet-5", Clear: true}, nil
				case 4:
					if rows[2].Model != "" || rows[2].DefaultModel != "local-two" {
						t.Fatal("reset did not inherit the default")
					}
					if cancel {
						return ClaudeDesktopMappingChoice{}, ErrCancelled
					}
					return ClaudeDesktopMappingChoice{}, nil
				default:
					t.Fatal("editor continued after save or cancel")
					return ClaudeDesktopMappingChoice{}, ErrCancelled
				}
			}
			got, err := client.configureClaudeDesktopMappings(context.Background(), initial, defaults)
			if cancel {
				if !errors.Is(err, ErrCancelled) || got != nil {
					t.Fatalf("cancel = %v, %v", got, err)
				}
			} else {
				want := map[string]string{"claude-opus-5": "local-two", "claude-sonnet-5": "local-two", "claude-haiku-4-5-20251001": "local-two"}
				if err != nil || !maps.Equal(got, want) {
					t.Fatalf("saved draft = %v, %v", got, err)
				}
			}
			if !maps.Equal(initial, original) {
				t.Fatal("editor changed the caller's selection")
			}
		})
	}
}

func TestClaudeDesktopEditorRequiresTerminal(t *testing.T) {
	withInteractiveSession(t, false)
	if _, err := (&launcherClient{}).configureClaudeDesktopMappings(context.Background(), nil, nil); err == nil {
		t.Fatal("opened a mapping editor without a terminal")
	}
}
