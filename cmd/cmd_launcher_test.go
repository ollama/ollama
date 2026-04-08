package cmd

import (
	"context"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/cmd/tui"
)

func setCmdTestHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("HOME", dir)
	t.Setenv("USERPROFILE", dir)
}

func unexpectedRunModelResolution(t *testing.T) func(context.Context, launch.RunModelRequest) (string, error) {
	t.Helper()
	return func(ctx context.Context, req launch.RunModelRequest) (string, error) {
		t.Fatalf("did not expect run-model resolution: %+v", req)
		return "", nil
	}
}

func unexpectedIntegrationLaunch(t *testing.T) func(context.Context, launch.IntegrationLaunchRequest) error {
	t.Helper()
	return func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
		t.Fatalf("did not expect integration launch: %+v", req)
		return nil
	}
}

func unexpectedModelLaunch(t *testing.T) func(*cobra.Command, string) error {
	t.Helper()
	return func(cmd *cobra.Command, model string) error {
		t.Fatalf("did not expect chat launch: %s", model)
		return nil
	}
}

func TestRunInteractiveTUI_RunModelActionsUseResolveRunModel(t *testing.T) {
	tests := []struct {
		name      string
		action    tui.TUIAction
		wantForce bool
		wantModel string
	}{
		{
			name:      "enter uses saved model flow",
			action:    tui.TUIAction{Kind: tui.TUIActionRunModel},
			wantModel: "qwen3:8b",
		},
		{
			name:      "right forces picker",
			action:    tui.TUIAction{Kind: tui.TUIActionRunModel, ForceConfigure: true},
			wantForce: true,
			wantModel: "glm-5:cloud",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setCmdTestHome(t, t.TempDir())

			var menuCalls int
			runMenu := func(state *launch.LauncherState) (tui.TUIAction, error) {
				menuCalls++
				if menuCalls == 1 {
					return tt.action, nil
				}
				return tui.TUIAction{Kind: tui.TUIActionNone}, nil
			}

			var gotReq launch.RunModelRequest
			var launched string
			deps := launcherDeps{
				buildState: func(ctx context.Context) (*launch.LauncherState, error) {
					return &launch.LauncherState{}, nil
				},
				runMenu: runMenu,
				resolveRunModel: func(ctx context.Context, req launch.RunModelRequest) (string, error) {
					gotReq = req
					return tt.wantModel, nil
				},
				launchIntegration: unexpectedIntegrationLaunch(t),
				runModel: func(cmd *cobra.Command, model string) error {
					launched = model
					return nil
				},
			}

			cmd := &cobra.Command{}
			cmd.SetContext(context.Background())
			for {
				continueLoop, err := runInteractiveTUIStep(cmd, deps)
				if err != nil {
					t.Fatalf("unexpected step error: %v", err)
				}
				if !continueLoop {
					break
				}
			}

			if gotReq.ForcePicker != tt.wantForce {
				t.Fatalf("expected ForcePicker=%v, got %v", tt.wantForce, gotReq.ForcePicker)
			}
			if launched != tt.wantModel {
				t.Fatalf("expected interactive launcher to run %q, got %q", tt.wantModel, launched)
			}
			if got := config.LastSelection(); got != "run" {
				t.Fatalf("expected last selection to be run, got %q", got)
			}
		})
	}
}

func TestRunInteractiveTUI_IntegrationActionsUseLaunchIntegration(t *testing.T) {
	tests := []struct {
		name      string
		action    tui.TUIAction
		wantForce bool
	}{
		{
			name:   "enter launches integration",
			action: tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude"},
		},
		{
			name:      "right forces configure",
			action:    tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude", ForceConfigure: true},
			wantForce: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setCmdTestHome(t, t.TempDir())

			var menuCalls int
			runMenu := func(state *launch.LauncherState) (tui.TUIAction, error) {
				menuCalls++
				if menuCalls == 1 {
					return tt.action, nil
				}
				return tui.TUIAction{Kind: tui.TUIActionNone}, nil
			}

			var gotReq launch.IntegrationLaunchRequest
			deps := launcherDeps{
				buildState: func(ctx context.Context) (*launch.LauncherState, error) {
					return &launch.LauncherState{}, nil
				},
				runMenu:         runMenu,
				resolveRunModel: unexpectedRunModelResolution(t),
				launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
					gotReq = req
					return nil
				},
				runModel: unexpectedModelLaunch(t),
			}

			cmd := &cobra.Command{}
			cmd.SetContext(context.Background())
			for {
				continueLoop, err := runInteractiveTUIStep(cmd, deps)
				if err != nil {
					t.Fatalf("unexpected step error: %v", err)
				}
				if !continueLoop {
					break
				}
			}

			if gotReq.Name != "claude" {
				t.Fatalf("expected integration name to be passed through, got %q", gotReq.Name)
			}
			if gotReq.ForceConfigure != tt.wantForce {
				t.Fatalf("expected ForceConfigure=%v, got %v", tt.wantForce, gotReq.ForceConfigure)
			}
			if got := config.LastSelection(); got != "claude" {
				t.Fatalf("expected last selection to be claude, got %q", got)
			}
		})
	}
}

func TestRunLauncherAction_RunModelContinuesAfterCancellation(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	continueLoop, err := runLauncherAction(cmd, tui.TUIAction{Kind: tui.TUIActionRunModel}, launcherDeps{
		buildState: nil,
		runMenu:    nil,
		resolveRunModel: func(ctx context.Context, req launch.RunModelRequest) (string, error) {
			return "", launch.ErrCancelled
		},
		launchIntegration: unexpectedIntegrationLaunch(t),
		runModel:          unexpectedModelLaunch(t),
	})
	if err != nil {
		t.Fatalf("expected nil error on cancellation, got %v", err)
	}
	if !continueLoop {
		t.Fatal("expected cancellation to continue the menu loop")
	}
}

func TestRunLauncherAction_VSCodeExitsTUILoop(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	// VS Code should exit the TUI loop (return false) after a successful launch.
	continueLoop, err := runLauncherAction(cmd, tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "vscode"}, launcherDeps{
		resolveRunModel: unexpectedRunModelResolution(t),
		launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
			return nil
		},
		runModel: unexpectedModelLaunch(t),
	})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if continueLoop {
		t.Fatal("expected vscode launch to exit the TUI loop (return false)")
	}

	// Other integrations should continue the TUI loop (return true).
	continueLoop, err = runLauncherAction(cmd, tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude"}, launcherDeps{
		resolveRunModel: unexpectedRunModelResolution(t),
		launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
			return nil
		},
		runModel: unexpectedModelLaunch(t),
	})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if !continueLoop {
		t.Fatal("expected non-vscode integration to continue the TUI loop (return true)")
	}
}

func TestRunLauncherAction_IntegrationContinuesAfterCancellation(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	continueLoop, err := runLauncherAction(cmd, tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude"}, launcherDeps{
		buildState:      nil,
		runMenu:         nil,
		resolveRunModel: unexpectedRunModelResolution(t),
		launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
			return launch.ErrCancelled
		},
		runModel: unexpectedModelLaunch(t),
	})
	if err != nil {
		t.Fatalf("expected nil error on cancellation, got %v", err)
	}
	if !continueLoop {
		t.Fatal("expected cancellation to continue the menu loop")
	}
}
