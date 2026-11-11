package launch

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/proxy"
	"github.com/spf13/cobra"
)

func TestClaudeRestoreCheckpointsInterruptedProgress(t *testing.T) {
	for _, failureAt := range []int{0, 1, 2} {
		t.Run([]string{"completed before journal removal", "before file write", "after file write"}[failureAt], func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			original := map[string]any{"inferenceProvider": "original-provider", "inferenceGatewayBaseUrl": "https://original.example", "userOwned": "keep"}
			if err := writeClaudeDesktopJSON(paths.profile, original); err != nil {
				t.Fatal(err)
			}
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			statePath, _ := claudeRestorePath()
			state, beforeJournal, err := claudeReadState(statePath)
			if err != nil {
				t.Fatal(err)
			}
			beforeFiles := make(map[string][]byte)
			for path := range state.Files {
				beforeFiles[path], err = os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
			}
			calls := 0
			failure := errors.New("checkpoint failed")
			err = claudeRestoreFiles(t.Context(), state.Files, func() error {
				calls++
				if calls == failureAt {
					return failure
				}
				return claudeSaveState(statePath, state)
			})
			if failureAt != 0 && !errors.Is(err, failure) || failureAt == 0 && err != nil {
				t.Fatalf("restore = %v", err)
			}
			if failureAt == 1 {
				for path, before := range beforeFiles {
					after, _ := os.ReadFile(path)
					if !bytes.Equal(before, after) {
						t.Fatalf("file changed before its checkpoint: %s", path)
					}
				}
				after, _ := os.ReadFile(statePath)
				if !bytes.Equal(beforeJournal, after) {
					t.Fatal("failed first checkpoint changed the journal")
				}
			}
			// Keep the disk journal as it stood at the interruption boundary.
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatalf("reconnect after interrupted restore: %v", err)
			}
			if err := restoreClaudeDesktopTargets(targets); err != nil {
				t.Fatal(err)
			}
			if got := claudeDesktopReadJSON(t, paths.profile); !reflect.DeepEqual(got, original) {
				t.Fatalf("original settings lost: %#v", got)
			}
		})
	}
}

func claudeInterruptedRestartFixture(t *testing.T, recordedQuit bool) (claudeDesktopPaths, claudeDesktopTargets) {
	t.Helper()
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceProvider": "original-provider", "inferenceGatewayBaseUrl": "https://original.example", "userOwned": "before"}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	snapshot, err := snapshotClaudeProfileForRestart(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if recordedQuit {
		if err := snapshot.recordAfterQuit(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	delete(profile, "inferenceGatewayBaseUrl")
	profile["userOwned"] = "later"
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	return paths, targets
}

func TestClaudeInterruptedRestartRecoveryRequiresExplicitConsent(t *testing.T) {
	for _, recordedQuit := range []bool{false, true} {
		stage := "before quit recording"
		if recordedQuit {
			stage = "after quit recording"
		}
		t.Run(stage, func(t *testing.T) {
			for _, action := range []string{"accept", "decline", "headless", "edit during confirmation"} {
				t.Run(action, func(t *testing.T) {
					paths, _ := claudeInterruptedRestartFixture(t, recordedQuit)
					beforeProfile, _ := os.ReadFile(paths.profile)
					statePath, _ := claudeRestorePath()
					beforeJournal, _ := os.ReadFile(statePath)
					withInteractiveSession(t, action != "headless")
					restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
					defer restorePolicy()
					oldConfirm := DefaultConfirmPrompt
					prompts := 0
					DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
						prompts++
						if options.Default != ConfirmDefaultNo {
							t.Fatal("recovery must keep current changes by default")
						}
						if !strings.Contains(prompt, "replace those changes") {
							t.Fatalf("missing overwrite warning: %s", prompt)
						}
						if action == "edit during confirmation" {
							profile := claudeDesktopReadJSON(t, paths.profile)
							profile["inferenceGatewayBaseUrl"] = "https://later-user-edit.example"
							if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
								t.Fatal(err)
							}
						}
						return action != "decline", nil
					}
					t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
					withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { t.Fatal("stopped Claude quit"); return nil }, func() error { t.Fatal("restore opened Claude"); return nil })
					err := (&ClaudeDesktop{}).Restore()
					if action == "accept" {
						if err != nil {
							t.Fatal(err)
						}
						profile := claudeDesktopReadJSON(t, paths.profile)
						if profile["inferenceProvider"] != "original-provider" || profile["inferenceGatewayBaseUrl"] != "https://original.example" || profile["userOwned"] != "later" {
							t.Fatalf("wrong recovered settings: %#v", profile)
						}
						if _, err := os.Stat(statePath); !os.IsNotExist(err) {
							t.Fatalf("recovery journal survived: %v", err)
						}
					} else {
						if err == nil {
							t.Fatal("recovery unexpectedly succeeded")
						}
						if action == "decline" && !errors.Is(err, ErrCancelled) {
							t.Fatalf("decline = %v", err)
						}
						if action == "edit during confirmation" {
							if !strings.Contains(err.Error(), "changed while confirming") || claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"] != "https://later-user-edit.example" {
								t.Fatalf("new edit was not protected: %v", err)
							}
						} else if after, _ := os.ReadFile(paths.profile); !bytes.Equal(beforeProfile, after) {
							t.Fatal("unconfirmed recovery changed the profile")
						}
						if after, _ := os.ReadFile(statePath); !bytes.Equal(beforeJournal, after) {
							t.Fatal("unconfirmed recovery changed the checkpoint")
						}
					}
					wantPrompts := 1
					if action == "headless" {
						wantPrompts = 0
					}
					if prompts != wantPrompts {
						t.Fatalf("prompts = %d, want %d", prompts, wantPrompts)
					}
				})
			}
		})
	}
}

func TestClaudeShutdownLeavesAmbiguousRecoveryPending(t *testing.T) {
	for _, running := range []bool{false, true} {
		t.Run(map[bool]string{false: "stopped", true: "running"}[running], func(t *testing.T) {
			paths, _ := claudeInterruptedRestartFixture(t, false)
			before, _ := os.ReadFile(paths.profile)
			statePath, _ := claudeRestorePath()
			beforeJournal, _ := os.ReadFile(statePath)
			withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error { t.Fatal("ambiguous recovery quit Claude"); return nil }, func() error { t.Fatal("shutdown opened Claude"); return nil })
			var recovery *claudeRestartRecovery
			if err := (&ClaudeDesktop{}).RestoreForShutdown(t.Context()); !errors.As(err, &recovery) {
				t.Fatalf("shutdown recovery error = %v", err)
			}
			if after, _ := os.ReadFile(paths.profile); !bytes.Equal(before, after) {
				t.Fatal("shutdown changed ambiguous values")
			}
			if after, _ := os.ReadFile(statePath); !bytes.Equal(beforeJournal, after) {
				t.Fatal("shutdown discarded pending recovery")
			}
		})
	}
}

func TestClaudeKnownInterruptedRestartRestoresWithoutPrompt(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceGatewayBaseUrl": "https://original.example"}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := snapshotClaudeProfileForRestart(t.Context()); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["inferenceGatewayBaseUrl"] = "https://original.example"
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatalf("known shutdown write required a decision: %v", err)
	}
	if claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"] != "https://original.example" {
		t.Fatal("original value lost")
	}
}

func TestClaudeQuitFailureKeepsDurableRecoveryIntent(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceGatewayBaseUrl": "https://original.example"}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	statePath, _ := claudeRestorePath()
	running := true
	failure := errors.New("quit failed after shutdown started")
	withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error {
		state, _, err := claudeReadState(statePath)
		if err != nil || !state.Files[paths.profile].Values["inferenceGatewayBaseUrl"].Restarting {
			t.Fatalf("quit started before its durable checkpoint: %v", err)
		}
		profile := claudeDesktopReadJSON(t, paths.profile)
		delete(profile, "inferenceGatewayBaseUrl")
		if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
			t.Fatal(err)
		}
		running = false
		return failure
	}, func() error { t.Fatal("shutdown opened Claude"); return nil })
	if err := (&ClaudeDesktop{}).RestoreForShutdown(t.Context()); !errors.Is(err, failure) {
		t.Fatalf("quit failure = %v", err)
	}
	withInteractiveSession(t, true)
	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
	if err := (&ClaudeDesktop{}).Restore(); err != nil {
		t.Fatalf("confirmed recovery after failed quit: %v", err)
	}
	if claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"] != "https://original.example" {
		t.Fatal("quit failure lost the original value")
	}
}

func TestClaudeConfigureDiscoveryHonorsParentDeadline(t *testing.T) {
	for _, stage := range []string{"probe", "catalog"} {
		t.Run(stage, func(t *testing.T) {
			paths, _ := claudeRestoreFixture(t)
			claudeDesktopProbeGateway = func(ctx context.Context, _ string) error {
				if stage == "probe" {
					<-ctx.Done()
					return ctx.Err()
				}
				return nil
			}
			claudeDesktopFetchCatalog = func(ctx context.Context, _ *http.Client, _, _ string) ([]proxy.ClaudeDesktopCatalogModel, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			}
			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
			defer cancel()
			started := time.Now()
			if err := (&ClaudeDesktop{}).configureAutodiscoveryWithAutoModeContext(ctx, false); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("configure deadline = %v", err)
			}
			if elapsed := time.Since(started); elapsed > time.Second {
				t.Fatalf("discovery exceeded parent deadline: %v", elapsed)
			}
			if _, err := os.Stat(paths.profile); !os.IsNotExist(err) {
				t.Fatalf("canceled discovery wrote a profile: %v", err)
			}
		})
	}
}

func TestClaudeLaunchHonorsCallerCancellation(t *testing.T) {
	for _, action := range []string{"restore", "configure"} {
		for _, stage := range []string{"before launch", "during operation"} {
			t.Run(action+"/"+stage, func(t *testing.T) {
				paths, targets := claudeRestoreFixture(t)
				if action == "restore" {
					if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
						t.Fatal(err)
					}
				}
				statePath, _ := claudeRestorePath()
				beforeProfile, _ := os.ReadFile(paths.profile)
				beforeJournal, _ := os.ReadFile(statePath)
				oldStat := claudeDesktopStat
				claudeDesktopStat = func(string) (os.FileInfo, error) { return nil, nil }
				t.Cleanup(func() { claudeDesktopStat = oldStat })
				withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { t.Fatal("canceled request quit Claude"); return nil }, func() error { t.Fatal("canceled request opened Claude"); return nil })
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				operationReached := false
				if stage == "before launch" {
					cancel()
				} else if action == "restore" {
					claudeDesktopIsRunning = func(context.Context) (bool, error) {
						operationReached = true
						cancel()
						return false, nil
					}
				} else {
					claudeDesktopProbeGateway = func(ctx context.Context, _ string) error {
						operationReached = true
						cancel()
						return ctx.Err()
					}
				}
				err := LaunchIntegration(ctx, IntegrationLaunchRequest{Name: "claude-desktop", Restore: action == "restore", ConfigureOnly: action == "configure"})
				if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrCancelled) {
					t.Fatalf("canceled request = %v", err)
				}
				if stage == "during operation" && !operationReached {
					t.Fatal("request did not reach the operation before cancellation")
				}
				for path, before := range map[string][]byte{paths.profile: beforeProfile, statePath: beforeJournal} {
					after, err := os.ReadFile(path)
					if !bytes.Equal(before, after) || before == nil && !os.IsNotExist(err) {
						t.Fatalf("canceled request changed %s", path)
					}
				}
			})
		}
	}
}

func TestClaudeLaunchCanceledDuringModelResolution(t *testing.T) {
	_, targets := claudeRestoreFixture(t)
	withInteractiveSession(t, true)
	withLauncherHooks(t)
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", true, nil); err != nil {
		t.Fatal(err)
	}
	if err := config.SaveIntegration(claudeDesktopIntegrationName, []string{claudeDesktopModelLabel}); err != nil {
		t.Fatal(err)
	}
	if err := config.MarkIntegrationOnboarded(claudeDesktopIntegrationName); err != nil {
		t.Fatal(err)
	}
	oldStat := claudeDesktopStat
	claudeDesktopStat = func(string) (os.FileInfo, error) { return nil, nil }
	t.Cleanup(func() { claudeDesktopStat = oldStat })
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		cancel()
		http.Error(w, "canceled", http.StatusInternalServerError)
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { t.Fatal("canceled launch quit Claude"); return nil }, func() error { t.Fatal("canceled launch opened Claude"); return nil })
	err := LaunchIntegration(ctx, IntegrationLaunchRequest{Name: claudeDesktopIntegrationName})
	if ctx.Err() == nil {
		t.Fatal("model resolution did not cancel the caller context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled launch = %v", err)
	}
}

func TestClaudeRunHonorsCallerCancellation(t *testing.T) {
	for _, stage := range []string{"gateway", "process check", "confirmation", "quit", "open", "reopen"} {
		t.Run(stage, func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			before, _ := os.ReadFile(paths.profile)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			running := stage == "confirmation" || stage == "quit" || stage == "reopen"
			withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error {
				if stage != "quit" && stage != "reopen" {
					t.Fatal("canceled run quit Claude")
				}
				if stage == "quit" {
					cancel()
					return ctx.Err()
				}
				running = false
				return nil
			}, func() error {
				if stage != "open" && stage != "reopen" {
					t.Fatal("canceled run opened Claude")
				}
				cancel()
				return nil
			})
			if stage == "gateway" {
				claudeDesktopProbeGateway = func(ctx context.Context, _ string) error {
					cancel()
					return ctx.Err()
				}
			}
			if stage == "process check" {
				claudeDesktopIsRunning = func(context.Context) (bool, error) {
					cancel()
					return false, nil
				}
			}
			restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{})
			defer restorePolicy()
			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
				if stage == "confirmation" {
					cancel()
				}
				return true, nil
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
			if err := (&ClaudeDesktop{}).run(ctx, nil); !errors.Is(err, context.Canceled) || !errors.Is(err, ErrCancelled) {
				t.Fatalf("canceled run = %v", err)
			}
			// Reapply commits before reopening; cancellation during open leaves
			// that completed profile change in place.
			if after, _ := os.ReadFile(paths.profile); stage != "reopen" && !bytes.Equal(before, after) {
				t.Fatal("canceled run rewrote the profile")
			}
		})
	}
}

func TestClaudeExitWaitPreservesContextErrors(t *testing.T) {
	oldRunning := claudeDesktopIsRunning
	claudeDesktopIsRunning = func(context.Context) (bool, error) { return true, nil }
	t.Cleanup(func() { claudeDesktopIsRunning = oldRunning })
	for _, deadline := range []bool{false, true} {
		t.Run(map[bool]string{false: "canceled", true: "deadline"}[deadline], func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			want := context.Canceled
			if deadline {
				cancel()
				ctx, cancel = context.WithTimeout(t.Context(), 10*time.Millisecond)
				want = context.DeadlineExceeded
			} else {
				timer := time.AfterFunc(10*time.Millisecond, cancel)
				defer timer.Stop()
			}
			defer cancel()
			err := waitForClaudeDesktopExit(ctx)
			if !errors.Is(err, want) {
				t.Fatalf("exit wait = %v, want %v", err, want)
			}
			if !deadline && !errors.Is(err, ErrCancelled) {
				t.Fatal("exit wait lost launcher cancellation identity")
			}
		})
	}
}

func TestClaudeRestoreCommandWorksWithoutLaunchSupport(t *testing.T) {
	for _, platform := range []string{"darwin", "windows"} {
		t.Run(platform, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			t.Setenv("LOCALAPPDATA", t.TempDir())
			withClaudeDesktopPlatform(t, platform)
			targets, err := claudeDesktopTargetPaths()
			if err != nil {
				t.Fatal(err)
			}
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { t.Fatal("stopped app quit"); return nil }, func() error { t.Fatal("restore opened Claude"); return nil })
			if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: claudeDesktopIntegrationName, Restore: true}); err != nil {
				t.Fatal(err)
			}
			for _, target := range targets.thirdPartyProfiles {
				if _, err := os.Stat(target.profile); !os.IsNotExist(err) {
					t.Fatalf("managed profile survived restore: %v", err)
				}
			}
		})
	}
}

func TestClaudeConfirmationHonorsCallerCancellation(t *testing.T) {
	for _, action := range []string{"launch", "restore", "recovery"} {
		t.Run(action, func(t *testing.T) {
			if action == "recovery" {
				claudeInterruptedRestartFixture(t, false)
			} else {
				_, targets := claudeRestoreFixture(t)
				if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", true, nil); err != nil {
					t.Fatal(err)
				}
			}
			withInteractiveSession(t, true)
			withLauncherHooks(t)
			oldStat := claudeDesktopStat
			claudeDesktopStat = func(string) (os.FileInfo, error) { return nil, nil }
			t.Cleanup(func() { claudeDesktopStat = oldStat })
			withClaudeDesktopProcessHooks(t, func() bool { return true }, func() error { t.Error("canceled confirmation quit Claude"); return nil }, func() error { t.Error("canceled confirmation opened Claude"); return nil })
			restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{yes: action == "recovery"})
			defer restorePolicy()
			oldConfirm := DefaultConfirmPromptContext
			started := make(chan struct{})
			DefaultConfirmPromptContext = func(ctx context.Context, _ string, _ ConfirmOptions) (bool, error) {
				close(started)
				<-ctx.Done()
				return false, ctx.Err()
			}
			t.Cleanup(func() { DefaultConfirmPromptContext = oldConfirm })
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			done := make(chan error, 1)
			go func() {
				done <- LaunchIntegration(ctx, IntegrationLaunchRequest{Name: claudeDesktopIntegrationName, Restore: action != "launch"})
			}()
			select {
			case <-started:
			case err := <-done:
				t.Fatalf("request returned before confirmation: %v", err)
			case <-time.After(time.Second):
				t.Fatal("confirmation did not start")
			}
			cancel()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrCancelled) {
					t.Fatalf("canceled confirmation = %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("confirmation ignored caller cancellation")
			}
		})
	}
}

func TestClaudeRestartPreservesPreexistingEditsFromShutdownWrites(t *testing.T) {
	for _, stage := range []string{"restart", "after snapshot", "after quit recording", "deleted field", "newer edit", "original value edit"} {
		t.Run(stage, func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"deploymentDisplayName": "original label"}); err != nil {
				t.Fatal(err)
			}
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			profile := claudeDesktopReadJSON(t, paths.profile)
			var want any = "user label"
			profile["deploymentDisplayName"] = want
			if stage == "deleted field" {
				delete(profile, "deploymentDisplayName")
				want = nil
			} else if stage == "newer edit" {
				want = "later label"
			} else if stage == "original value edit" {
				want = "original label"
			}
			if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
				t.Fatal(err)
			}
			quit := func() error {
				stale := claudeDesktopReadJSON(t, paths.profile)
				stale["deploymentDisplayName"] = claudeDesktopProfileName
				if stage == "newer edit" || stage == "original value edit" {
					stale["deploymentDisplayName"] = want
				}
				return writeClaudeDesktopJSON(paths.profile, stale)
			}
			configure := func(ctx context.Context) error {
				return configureClaudeDesktopTargetsContext(ctx, targets, ClaudeDesktopGatewayURL(), "ollama", false, nil)
			}
			var err error
			if stage == "after snapshot" || stage == "after quit recording" {
				snapshot, snapshotErr := snapshotClaudeProfileForRestart(t.Context())
				if snapshotErr != nil {
					t.Fatal(snapshotErr)
				}
				if err := quit(); err != nil {
					t.Fatal(err)
				}
				if stage == "after quit recording" {
					if err := snapshot.recordAfterQuit(t.Context()); err != nil {
						t.Fatal(err)
					}
				}
				err = configure(t.Context())
			} else {
				running := true
				withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error { running = false; return quit() }, func() error { return nil })
				err = restartClaudeDesktopContext(t.Context(), configure)
			}
			if err != nil && !strings.Contains(err.Error(), "changed after setup") {
				t.Fatal(err)
			}
			if got := claudeDesktopReadJSON(t, paths.profile)["deploymentDisplayName"]; got != want {
				t.Fatalf("user edit lost: got %v, want %v", got, want)
			}
			statePath, _ := claudeRestorePath()
			state, _, readErr := claudeReadState(statePath)
			if readErr != nil {
				t.Fatal(readErr)
			}
			value := state.Files[paths.profile].Values["deploymentDisplayName"]
			if string(value.Before) != `"original label"` || claudeValueOwned(claudeField(claudeDesktopReadJSON(t, paths.profile), "deploymentDisplayName"), value) {
				t.Fatal("user edit was reclassified as owned or original checkpoint changed")
			}
		})
	}
}

func TestClaudeLaunchCancellationWhileWaitingForProfileLock(t *testing.T) {
	for _, restore := range []bool{false, true} {
		t.Run(map[bool]string{false: "configure", true: "restore"}[restore], func(t *testing.T) {
			_, targets := claudeRestoreFixture(t)
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			oldStat := claudeDesktopStat
			claudeDesktopStat = func(string) (os.FileInfo, error) { return nil, nil }
			t.Cleanup(func() { claudeDesktopStat = oldStat })
			unlock, err := claudeProfileLock(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			defer unlock()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			timer := time.AfterFunc(20*time.Millisecond, cancel)
			defer timer.Stop()
			err = LaunchIntegration(ctx, IntegrationLaunchRequest{Name: claudeDesktopIntegrationName, Restore: restore, ConfigureOnly: !restore})
			if !errors.Is(err, context.Canceled) || !errors.Is(err, ErrCancelled) {
				t.Fatalf("lock-wait cancellation = %v", err)
			}
		})
	}
}

func TestClaudeCancellationDoesNotHideOperationalErrors(t *testing.T) {
	for _, err := range []error{context.Canceled, fmt.Errorf("setup failed: %w", context.Canceled), errors.Join(context.Canceled)} {
		if !claudeDesktopCancellationOnly(err) {
			t.Fatalf("pure cancellation was not recognized: %v", err)
		}
	}
	for _, err := range []error{nil, context.DeadlineExceeded, os.ErrPermission, errors.Join(context.Canceled, os.ErrPermission), fmt.Errorf("rollback failed: %w", errors.Join(context.Canceled, os.ErrPermission))} {
		if claudeDesktopCancellationOnly(err) {
			t.Fatalf("operational failure would be hidden as cancellation: %v", err)
		}
	}
}

func TestClaudeCancellationRetainsOperationalFailures(t *testing.T) {
	for _, stage := range []string{"quit", "reapply", "open"} {
		t.Run(stage, func(t *testing.T) {
			_, targets := claudeRestoreFixture(t)
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			failure := errors.New("operation failed")
			running := true
			withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error {
				if stage == "quit" {
					cancel()
					return failure
				}
				running = false
				return nil
			}, func() error {
				cancel()
				return failure
			})
			err := restartClaudeDesktopContext(ctx, func(context.Context) error {
				if stage == "reapply" {
					cancel()
					return failure
				}
				return nil
			})
			if !errors.Is(err, context.Canceled) || !errors.Is(err, failure) || errors.Is(err, ErrCancelled) {
				t.Fatalf("mixed failure was lost or suppressed: %v", err)
			}
		})
	}
}

func TestClaudeCLIKeepsOperationalFailuresVisible(t *testing.T) {
	for _, stage := range []string{"cancel", "quit failure", "open failure", "nested failure"} {
		t.Run(stage, func(t *testing.T) {
			_, targets := claudeRestoreFixture(t)
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			failure := errors.New("operation failed")
			running := true
			withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error {
				if stage == "cancel" || stage == "quit failure" {
					cancel()
					if stage == "quit failure" {
						return failure
					}
					return ctx.Err()
				}
				running = false
				return nil
			}, func() error {
				cancel()
				if stage == "nested failure" {
					return errors.Join(ErrCancelled, failure)
				}
				return failure
			})
			command := LaunchCmd(func(*cobra.Command, []string) error { t.Fatal("restore checked server"); return nil }, func(*cobra.Command) { t.Fatal("restore opened launcher") })
			command.SilenceErrors, command.SilenceUsage = true, true
			command.SetArgs([]string{claudeDesktopIntegrationName, "--restore", "--yes"})
			err := command.ExecuteContext(ctx)
			if stage == "cancel" {
				if err != nil {
					t.Fatalf("pure cancellation was not quiet: %v", err)
				}
			} else if !errors.Is(err, failure) {
				t.Fatalf("CLI hid the operational failure: %v", err)
			}
		})
	}
}

func TestClaudeCommandContextCancellation(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_CLAUDE_COMMAND_CANCEL") == "1" {
		fmt.Fprintln(os.Stdout, "ready")
		time.Sleep(time.Minute)
		return
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestClaudeCommandContextCancellation$")
	command.Env = append(os.Environ(), "OLLAMA_TEST_CLAUDE_COMMAND_CANCEL=1")
	output, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- runClaudeDesktopCommand(ctx, command) }()
	ready := make(chan error, 1)
	go func() {
		line, err := bufio.NewReader(output).ReadString('\n')
		if err == nil && line != "ready\n" {
			err = fmt.Errorf("unexpected child output %q", line)
		}
		ready <- err
	}()
	select {
	case err := <-ready:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("command helper did not start")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("killed command lost cancellation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("command did not stop after cancellation")
	}
}
