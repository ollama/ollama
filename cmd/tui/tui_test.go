package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/cmd/launch"
)

func launcherTestState() *launch.LauncherState {
	return &launch.LauncherState{
		LastSelection: "run",
		RunModel:      "qwen3:8b",
		Integrations: map[string]launch.LauncherIntegrationState{
			"claude": {
				Name:         "claude",
				DisplayName:  "Claude Code",
				Description:  "Anthropic's coding tool with subagents",
				Selectable:   true,
				Changeable:   true,
				CurrentModel: "glm-5:cloud",
			},
			"codex": {
				Name:        "codex",
				DisplayName: "Codex CLI",
				Description: "OpenAI's open-source coding agent",
				Selectable:  true,
				Changeable:  true,
			},
			"chatgpt": {
				Name:        "chatgpt",
				DisplayName: "ChatGPT",
				Description: "Complete work with ChatGPT",
				Selectable:  true,
				Changeable:  true,
			},
			"openclaw": {
				Name:            "openclaw",
				DisplayName:     "OpenClaw",
				Description:     "Personal AI with 100+ skills",
				Selectable:      true,
				Changeable:      true,
				AutoInstallable: true,
			},
			"opencode": {
				Name:        "opencode",
				DisplayName: "OpenCode",
				Description: "Anomaly's open-source coding agent",
				Selectable:  true,
				Changeable:  true,
			},
			"hermes": {
				Name:        "hermes",
				DisplayName: "Hermes Agent",
				Description: "Self-improving AI agent built by Nous Research",
				Selectable:  true,
				Changeable:  true,
			},
			"droid": {
				Name:        "droid",
				DisplayName: "Droid",
				Description: "Factory's coding agent across terminal and IDEs",
				Selectable:  true,
				Changeable:  true,
			},
			"pi": {
				Name:        "pi",
				DisplayName: "Pi",
				Description: "Minimal AI agent toolkit with plugin support",
				Selectable:  true,
				Changeable:  true,
			},
		},
	}
}

func findMenuCursorByIntegration(items []menuItem, name string) int {
	for i, item := range items {
		if item.integration == name {
			return i
		}
	}
	return -1
}

func integrationSequence(items []menuItem) []string {
	sequence := make([]string, 0, len(items))
	for _, item := range items {
		switch {
		case item.isOthers:
			sequence = append(sequence, "more")
		case item.integration != "":
			sequence = append(sequence, item.integration)
		}
	}
	return sequence
}

func compareStrings(got, want []string) string {
	return cmp.Diff(want, got)
}

func TestMenuPromotesInstalledAppsInDefaultPriorityOrder(t *testing.T) {
	for _, tc := range []struct {
		name         string
		installed    []string
		want         []string
		wantOverflow []string
	}{
		{"none installed", nil, []string{"claude", "opencode", "hermes", "openclaw", "codex", "more"}, []string{"droid", "pi"}},
		{"only OpenClaw", []string{"openclaw"}, []string{"openclaw", "claude", "opencode", "hermes", "codex", "more"}, []string{"droid", "pi"}},
		{"installed primary and additional apps", []string{"pi", "codex", "openclaw"}, []string{"openclaw", "codex", "pi", "claude", "opencode", "more"}, []string{"hermes", "droid"}},
		{"more than five installed", []string{"pi", "droid", "codex", "openclaw", "hermes", "opencode"}, []string{"opencode", "hermes", "openclaw", "codex", "droid", "more"}, []string{"pi", "claude"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := launcherTestState()
			// Keep these ordering fixtures independent of GUI platform support.
			delete(state.Integrations, "chatgpt")
			for _, name := range tc.installed {
				app := state.Integrations[name]
				app.Installed = true
				state.Integrations[name] = app
			}
			menu := newModel(state)
			if diff := compareStrings(integrationSequence(menu.items), tc.want); diff != "" {
				t.Fatalf("wrong installed-first order: %s", diff)
			}
			expanded := buildMenuItems(state, true)
			if diff := compareStrings(integrationSequence(expanded[launcherMenuLimit:]), tc.wantOverflow); diff != "" {
				t.Fatalf("remaining apps should stay under More in priority order: %s", diff)
			}
		})
	}
}

func TestMenuRemembersPromotedAppWithoutOpeningMore(t *testing.T) {
	state := launcherTestState()
	app := state.Integrations["codex"]
	app.Installed = true
	state.Integrations["codex"] = app
	state.LastSelection = "codex"
	menu := newModel(state)
	if menu.showOthers || menu.cursor != 0 || menu.items[menu.cursor].integration != "codex" {
		t.Fatal("previously selected installed app should be recalled in its promoted position")
	}
}

func TestMenuEmptyStateDoesNotPanic(t *testing.T) {
	menu := newModel(nil)
	if len(menu.items) != 0 {
		t.Fatal("empty state must not create selectable menu items")
	}
	_ = menu.View() // Rendering an empty menu must not panic, regardless of its copy.
	for _, key := range []tea.KeyType{tea.KeyEnter, tea.KeyRight, tea.KeyUp, tea.KeyDown} {
		updated, cmd := menu.Update(tea.KeyMsg{Type: key})
		menu = updated.(model)
		if menu.selected || menu.action.Kind != TUIActionNone || menu.cursor != 0 || cmd != nil {
			t.Fatal("empty menu must not select an action")
		}
		_ = menu.View()
	}
	if updated, quit := menu.Update(tea.KeyMsg{Type: tea.KeyEsc}); !updated.(model).quitting || quit == nil {
		t.Fatal("Escape must still exit an empty menu")
	}
}

func TestMenuUpToFiveAppsDoesNotNeedMore(t *testing.T) {
	for _, names := range [][]string{
		{"claude", "opencode", "hermes"},
		{"claude", "opencode", "hermes", "openclaw", "codex"},
	} {
		state := launcherTestState()
		apps := make(map[string]launch.LauncherIntegrationState)
		for _, name := range names {
			apps[name] = state.Integrations[name]
		}
		state.Integrations = apps
		menu := newModel(state)
		if diff := compareStrings(integrationSequence(menu.items), names); diff != "" {
			t.Fatalf("up to five apps should appear directly without More: %s", diff)
		}
	}
}

func TestMenuExpandsAndCollapsesMoreAfterPromotion(t *testing.T) {
	state := launcherTestState()
	app := state.Integrations["codex"]
	app.Installed = true
	state.Integrations["codex"] = app
	menu := newModel(state)
	menu.cursor = launcherMenuLimit - 1
	root := integrationSequence(menu.items)

	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyDown})
	expanded := updated.(model)
	if !expanded.showOthers || expanded.cursor != launcherMenuLimit || expanded.items[expanded.cursor].integration == "" {
		t.Fatal("Down must expand More and select the first overflow app")
	}
	updated, _ = expanded.Update(tea.KeyMsg{Type: tea.KeyUp})
	collapsed := updated.(model)
	if collapsed.showOthers || collapsed.cursor != launcherMenuLimit-1 {
		t.Fatal("Up must return to the last primary app")
	}
	if diff := compareStrings(integrationSequence(collapsed.items), root); diff != "" {
		t.Fatalf("collapsing More changed the promoted root menu: %s", diff)
	}
}

func TestMenuStartsExpandedForPreviousOverflowSelection(t *testing.T) {
	state := launcherTestState()
	overflow := buildMenuItems(state, true)[launcherMenuLimit:]
	if len(overflow) < 2 {
		t.Fatal("expected at least two additional integrations")
	}
	state.LastSelection = overflow[1].integration

	menu := newModel(state)
	if !menu.showOthers {
		t.Fatal("expected previous additional integration selection to start expanded")
	}
	if got := menu.items[menu.cursor].integration; got != state.LastSelection {
		t.Fatalf("initial cursor integration = %q, want %q", got, state.LastSelection)
	}
	for _, item := range menu.items {
		if item.isOthers {
			t.Fatal("expanded menu must contain apps instead of More")
		}
	}
}

func TestMenuPreviousRunSelectionFallsBackToFirstApp(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "run"
	menu := newModel(state)
	if menu.cursor != 0 || menu.items[menu.cursor].integration != "claude" {
		t.Fatal("previous chat selection must fall back to the first app")
	}
}

func TestMenuEnterOnIntegrationSelectsLaunch(t *testing.T) {
	menu := newModel(launcherTestState())
	menu.cursor = findMenuCursorByIntegration(menu.items, "claude")
	if menu.cursor == -1 {
		t.Fatal("expected claude menu item")
	}
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionLaunchIntegration, Integration: "claude"}
	if !got.selected || got.action != want {
		t.Fatalf("expected enter on integration to launch, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuRightOnIntegrationSelectsConfigure(t *testing.T) {
	menu := newModel(launcherTestState())
	menu.cursor = findMenuCursorByIntegration(menu.items, "claude")
	if menu.cursor == -1 {
		t.Fatal("expected claude menu item")
	}
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionLaunchIntegration, Integration: "claude", ForceConfigure: true}
	if !got.selected || got.action != want {
		t.Fatalf("expected right on integration to configure, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuIgnoresDisabledActions(t *testing.T) {
	state := launcherTestState()
	claude := state.Integrations["claude"]
	claude.Selectable = false
	claude.Changeable = false
	state.Integrations["claude"] = claude

	menu := newModel(state)
	menu.cursor = findMenuCursorByIntegration(menu.items, "claude")
	if menu.cursor == -1 {
		t.Fatal("expected claude menu item")
	}

	updatedEnter, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if updatedEnter.(model).selected {
		t.Fatal("expected non-selectable integration to ignore enter")
	}

	updatedRight, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	if updatedRight.(model).selected {
		t.Fatal("expected non-changeable integration to ignore right")
	}
}

func TestMenuShowsOnlySelectedAppsCurrentModel(t *testing.T) {
	state := launcherTestState()
	menu := newModel(state)
	for i, item := range menu.items {
		menu.cursor = i
		view := menu.View()
		if strings.Contains(view, state.RunModel) {
			t.Fatalf("removed chat model must not appear at cursor %d", i)
		}
		if got, want := strings.Contains(view, state.Integrations["claude"].CurrentModel), item.integration == "claude"; got != want {
			t.Fatalf("current model visibility at cursor %d = %v, want %v", i, got, want)
		}
	}
}

func TestMenuShowsInstallStatusAndHint(t *testing.T) {
	state := launcherTestState()
	opencode := state.Integrations["opencode"]
	opencode.Installed = false
	opencode.Selectable = false
	opencode.Changeable = false
	opencode.InstallHint = "Install from https://example.com/opencode"
	state.Integrations["opencode"] = opencode

	state.LastSelection = "opencode"
	menu := newModel(state)
	menu.cursor = findMenuCursorByIntegration(menu.items, "opencode")
	if menu.cursor == -1 {
		t.Fatal("expected opencode menu item")
	}
	view := menu.View()
	if !strings.Contains(view, "(not installed)") {
		t.Fatalf("expected not-installed marker\n%s", view)
	}
	if !strings.Contains(view, opencode.InstallHint) {
		t.Fatalf("expected install hint in description\n%s", view)
	}
}
