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

func unexpectedRequestedRunModelResolution(t *testing.T) func(context.Context, string) (string, error) {
	t.Helper()
	return func(ctx context.Context, model string) (string, error) {
		t.Fatalf("did not expect requested run-model resolution: %s", model)
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
				resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
				launchIntegration:        unexpectedIntegrationLaunch(t),
				runModel: func(cmd *cobra.Command, model string) error {
					launched = model
					return nil
				},
			}

			cmd := &cobra.Command{}
			cmd.SetContext(context.Background())
			for {
				continueLoop, err := runInteractiveTUIStep(cmd, launch.LauncherInvocation{}, deps)
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
				runMenu:                  runMenu,
				resolveRunModel:          unexpectedRunModelResolution(t),
				resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
				launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
					gotReq = req
					return nil
				},
				runModel: unexpectedModelLaunch(t),
			}

			cmd := &cobra.Command{}
			cmd.SetContext(context.Background())
			for {
				continueLoop, err := runInteractiveTUIStep(cmd, launch.LauncherInvocation{}, deps)
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

	continueLoop, err := runLauncherAction(cmd, launch.LauncherInvocation{}, tui.TUIAction{Kind: tui.TUIActionRunModel}, launcherDeps{
		buildState: nil,
		runMenu:    nil,
		resolveRunModel: func(ctx context.Context, req launch.RunModelRequest) (string, error) {
			return "", config.ErrCancelled
		},
		resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
		launchIntegration:        unexpectedIntegrationLaunch(t),
		runModel:                 unexpectedModelLaunch(t),
	})

	if err != nil {
		t.Fatalf("expected nil error on cancellation, got %v", err)
	}
	if !continueLoop {
		t.Fatal("expected cancellation to continue the menu loop")
	}
}

func TestRunLauncherAction_IntegrationContinuesAfterCancellation(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	continueLoop, err := runLauncherAction(cmd, launch.LauncherInvocation{}, tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude"}, launcherDeps{
		buildState:               nil,
		runMenu:                  nil,
		resolveRunModel:          unexpectedRunModelResolution(t),
		resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
		launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
			return config.ErrCancelled
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

func TestRunLauncherAction_RunModelUsesInvocationOverrideOnEnter(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	var gotModel string
	var launched string
	continueLoop, err := runLauncherAction(cmd, launch.LauncherInvocation{ModelOverride: "qwen3.5:cloud"}, tui.TUIAction{Kind: tui.TUIActionRunModel}, launcherDeps{
		resolveRunModel: unexpectedRunModelResolution(t),
		resolveRequestedRunModel: func(ctx context.Context, model string) (string, error) {
			gotModel = model
			return model, nil
		},
		launchIntegration: unexpectedIntegrationLaunch(t),
		runModel: func(cmd *cobra.Command, model string) error {
			launched = model
			return nil
		},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !continueLoop {
		t.Fatal("expected menu loop to continue after launch")
	}
	if gotModel != "qwen3.5:cloud" {
		t.Fatalf("expected requested model override to be used, got %q", gotModel)
	}
	if launched != "qwen3.5:cloud" {
		t.Fatalf("expected launched model to use override, got %q", launched)
	}
	if got := config.LastSelection(); got != "run" {
		t.Fatalf("expected last selection to be run, got %q", got)
	}
}

func TestRunLauncherAction_RunModelIgnoresInvocationOverrideOnChange(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	var gotReq launch.RunModelRequest
	var launched string
	continueLoop, err := runLauncherAction(cmd, launch.LauncherInvocation{ModelOverride: "qwen3.5:cloud"}, tui.TUIAction{Kind: tui.TUIActionRunModel, ForceConfigure: true}, launcherDeps{
		resolveRunModel: func(ctx context.Context, req launch.RunModelRequest) (string, error) {
			gotReq = req
			return "llama3.2", nil
		},
		resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
		launchIntegration:        unexpectedIntegrationLaunch(t),
		runModel: func(cmd *cobra.Command, model string) error {
			launched = model
			return nil
		},
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !continueLoop {
		t.Fatal("expected menu loop to continue after launch")
	}
	if !gotReq.ForcePicker {
		t.Fatal("expected change action to force the picker")
	}
	if launched != "llama3.2" {
		t.Fatalf("expected launched model to come from picker flow, got %q", launched)
	}
}

func TestRunLauncherAction_IntegrationUsesInvocationOverrideOnEnter(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	var gotReq launch.IntegrationLaunchRequest
	continueLoop, err := runLauncherAction(cmd, launch.LauncherInvocation{
		ModelOverride: "qwen3.5:cloud",
		ExtraArgs:     []string{"--sandbox", "workspace-write"},
	}, tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude"}, launcherDeps{
		resolveRunModel:          unexpectedRunModelResolution(t),
		resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
		launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
			gotReq = req
			return nil
		},
		runModel: unexpectedModelLaunch(t),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !continueLoop {
		t.Fatal("expected menu loop to continue after launch")
	}
	if gotReq.Name != "claude" {
		t.Fatalf("expected integration name to be passed through, got %q", gotReq.Name)
	}
	if gotReq.ModelOverride != "qwen3.5:cloud" {
		t.Fatalf("expected model override to be forwarded, got %q", gotReq.ModelOverride)
	}
	if gotReq.ForceConfigure {
		t.Fatal("expected enter action not to force configure")
	}
	if len(gotReq.ExtraArgs) != 2 || gotReq.ExtraArgs[0] != "--sandbox" || gotReq.ExtraArgs[1] != "workspace-write" {
		t.Fatalf("unexpected extra args: %v", gotReq.ExtraArgs)
	}
}

func TestRunLauncherAction_IntegrationIgnoresInvocationOverrideOnChange(t *testing.T) {
	setCmdTestHome(t, t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	var gotReq launch.IntegrationLaunchRequest
	continueLoop, err := runLauncherAction(cmd, launch.LauncherInvocation{
		ModelOverride: "qwen3.5:cloud",
		ExtraArgs:     []string{"--sandbox", "workspace-write"},
	}, tui.TUIAction{Kind: tui.TUIActionLaunchIntegration, Integration: "claude", ForceConfigure: true}, launcherDeps{
		resolveRunModel:          unexpectedRunModelResolution(t),
		resolveRequestedRunModel: unexpectedRequestedRunModelResolution(t),
		launchIntegration: func(ctx context.Context, req launch.IntegrationLaunchRequest) error {
			gotReq = req
			return nil
		},
		runModel: unexpectedModelLaunch(t),
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !continueLoop {
		t.Fatal("expected menu loop to continue after configure")
	}
	if gotReq.ModelOverride != "" {
		t.Fatalf("expected change action to ignore model override, got %q", gotReq.ModelOverride)
	}
	if len(gotReq.ExtraArgs) != 0 {
		t.Fatalf("expected change action to ignore extra args, got %v", gotReq.ExtraArgs)
	}
	if !gotReq.ForceConfigure {
		t.Fatal("expected change action to force configure")
	}
}
