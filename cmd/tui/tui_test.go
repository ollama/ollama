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
				DisplayName: "Codex",
				Description: "OpenAI's open-source coding agent",
				Selectable:  true,
				Changeable:  true,
			},
			"codex-app": {
				Name:        "codex-app",
				DisplayName: "Codex App",
				Description: "An AI agent you can delegate real work to, by OpenAI",
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
		case item.isRunModel:
			sequence = append(sequence, "run")
		case item.integration != "":
			sequence = append(sequence, item.integration)
		}
	}
	return sequence
}

func compareStrings(got, want []string) string {
	return cmp.Diff(want, got)
}

func TestMenuRendersRootLaunchChoices(t *testing.T) {
	state := launcherTestState()
	menu := newModel(state)
	want := []string{"run", "claude", "opencode", "hermes", "openclaw"}
	if diff := compareStrings(integrationSequence(menu.items), want); diff != "" {
		t.Fatalf("unexpected root launch choices: %s", diff)
	}

	view := menu.View()
	for _, want := range []string{
		"Chat, Code, & Work",
		"Chat with models, code, search the web, and delegate real work",
		"Launch Claude Code",
		"Launch OpenCode",
		"Launch Hermes Agent",
		"Launch OpenClaw",
	} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected menu view to contain %q\n%s", want, view)
		}
	}
	for _, hidden := range []string{"Launch Codex App", "Launch Codex", "Launch Droid", "Launch Pi", "More..."} {
		if strings.Contains(view, hidden) {
			t.Fatalf("expected root menu to omit %q\n%s", hidden, view)
		}
	}
}

func TestMenuEnterOnRunSelectsRun(t *testing.T) {
	menu := newModel(launcherTestState())
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionRunModel}
	if !got.selected || got.action != want {
		t.Fatalf("expected enter on run to select run action, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuRightOnRunSelectsChangeRun(t *testing.T) {
	menu := newModel(launcherTestState())
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionRunModel, ForceConfigure: true}
	if !got.selected || got.action != want {
		t.Fatalf("expected right on run to select change-run action, got selected=%v action=%v", got.selected, got.action)
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

func TestMenuShowsCurrentModelSuffixes(t *testing.T) {
	menu := newModel(launcherTestState())
	runView := menu.View()
	if !strings.Contains(runView, "(qwen3:8b)") {
		t.Fatalf("expected run row to show current model suffix\n%s", runView)
	}

	menu.cursor = findMenuCursorByIntegration(menu.items, "claude")
	if menu.cursor == -1 {
		t.Fatal("expected claude menu item")
	}
	integrationView := menu.View()
	if !strings.Contains(integrationView, "(glm-5:cloud)") {
		t.Fatalf("expected integration row to show current model suffix\n%s", integrationView)
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
