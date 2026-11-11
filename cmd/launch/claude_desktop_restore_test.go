package launch

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/internal/proxy"
)

func claudeRestoreFixture(t *testing.T) (claudeDesktopPaths, claudeDesktopTargets) {
	t.Helper()
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		t.Fatal(err)
	}
	return paths, targets
}

func configureClaudeDesktopTargets(targets claudeDesktopTargets, baseURL, apiKey string, autoMode bool, models []proxy.ClaudeDesktopCatalogModel) error {
	return configureClaudeDesktopTargetsContext(context.Background(), targets, baseURL, apiKey, autoMode, models)
}

func withClaudeProfileChanges(targets claudeDesktopTargets, change func() error) error {
	return withClaudeProfileChangesContext(context.Background(), targets, change)
}

func restoreClaudeDesktopProfile() error {
	return restoreClaudeDesktopProfileContext(context.Background())
}

func TestClaudeRestorePreservesOriginalValuesAndLaterUnrelatedEdits(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	originalProfile := map[string]any{"inferenceProvider": "bedrock", "inferenceGatewayApiKey": "synthetic-old-key", "autoModeEnabled": false, "inferenceModels": []any{map[string]any{"name": "original"}}, "userOwned": "before"}
	originalMeta := map[string]any{"appliedId": "custom", "entries": []any{map[string]any{"id": "custom", "name": "Existing"}}}
	originalNormal := map[string]any{"deploymentMode": "1p", "mcpServers": map[string]any{"keep": map[string]any{"command": "synthetic"}}}
	for path, value := range map[string]any{paths.profile: originalProfile, paths.meta: originalMeta, paths.normalConfig: originalNormal} {
		if err := writeClaudeDesktopJSON(path, value); err != nil {
			t.Fatal(err)
		}
	}
	history := filepath.Join(filepath.Dir(paths.normalConfig), "history-sentinel")
	if err := os.WriteFile(history, []byte("history remains byte identical"), 0o600); err != nil {
		t.Fatal(err)
	}
	configure := func(label string) error {
		return configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, []proxy.ClaudeDesktopCatalogModel{{ID: "claude-sonnet-5", DisplayName: label}})
	}
	if err := configure("first"); err != nil {
		t.Fatal(err)
	}
	// A second connect must retain the first checkpoint, including nested JSON.
	if err := configure("second"); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["userOwned"] = "later"
	profile["newPreference"] = true
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatal(err)
	}
	want := originalProfile
	want["userOwned"] = "later"
	want["newPreference"] = true
	if got := claudeDesktopReadJSON(t, paths.profile); !reflect.DeepEqual(got, want) {
		t.Fatalf("restored profile = %#v, want %#v", got, want)
	}
	if got := claudeDesktopReadJSON(t, paths.meta); !reflect.DeepEqual(got, originalMeta) {
		t.Fatalf("metadata changed: %#v", got)
	}
	if got := claudeDesktopReadJSON(t, paths.normalConfig); !reflect.DeepEqual(got, originalNormal) {
		t.Fatalf("normal profile changed: %#v", got)
	}
	if _, err := os.Stat(paths.desktopConfig); !os.IsNotExist(err) {
		t.Fatalf("previously absent config survived restore: %v", err)
	}
	if data, err := os.ReadFile(history); err != nil || string(data) != "history remains byte identical" {
		t.Fatalf("history changed: %q %v", data, err)
	}
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatalf("repeat restore: %v", err)
	}
}

func TestClaudeRestorePreservesEditedOwnedSettings(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["autoModeEnabled"] = true
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("expected edited setting to remain pending")
	}
	if got := claudeDesktopReadJSON(t, paths.profile)["autoModeEnabled"]; got != true {
		t.Fatalf("user edit replaced: %v", got)
	}
	statePath, _ := claudeRestorePath()
	if _, err := os.Stat(statePath); err != nil {
		t.Fatalf("recovery checkpoint missing: %v", err)
	}
}

func TestClaudeRestorePreservesChangedProvider(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["inferenceGatewayBaseUrl"] = "https://user.example"
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(paths.normalConfig)
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("changed provider was not detected")
	}
	after, _ := os.ReadFile(paths.normalConfig)
	if string(after) != string(before) || claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"] != "https://user.example" {
		t.Fatal("restore changed the edited connection")
	}
}

func TestClaudeLegacyConnectionRestoresUsualProfile(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceGatewayBaseUrl": "http://127.0.0.1:11435", "inferenceProvider": "gateway", "unrelated": "keep"}); err != nil {
		t.Fatal(err)
	}
	if err := writeClaudeDesktopJSON(paths.meta, map[string]any{"appliedId": claudeDesktopProfileID, "entries": []any{map[string]any{"id": claudeDesktopProfileID, "name": "Ollama"}}}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	if got := claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"]; got != ClaudeDesktopGatewayURL() {
		t.Fatalf("migration URL = %v", got)
	}
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceGatewayBaseUrl"] != nil || profile["unrelated"] != "keep" {
		t.Fatalf("legacy restore = %#v", profile)
	}
}

func TestClaudeProfileSetupRollsBackPartialWrite(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	original := map[string]any{"keep": "original"}
	if err := writeClaudeDesktopJSON(paths.profile, original); err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("later file write failed")
	err := withClaudeProfileChanges(targets, func() error {
		if err := writeClaudeDesktopGatewayProfile(paths.profile, ClaudeDesktopGatewayURL(), "ollama", true, false, nil); err != nil {
			return err
		}
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("setup error = %v", err)
	}
	if got := claudeDesktopReadJSON(t, paths.profile); !reflect.DeepEqual(got, original) {
		t.Fatalf("partial profile survived: %#v", got)
	}
	statePath, _ := claudeRestorePath()
	if _, err := os.Stat(statePath); !os.IsNotExist(err) {
		t.Fatalf("failed setup retained state: %v", err)
	}
}

func TestClaudeRestoreRejectsUnexpectedJournalKeys(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	state := &claudeRestoreState{Version: 1, Files: map[string]*claudeOwnedFile{paths.normalConfig: {Existed: true, Values: map[string]claudeOwnedValue{"mcpServers": {Before: []byte(`{}`), Applied: []byte(`{}`)}}}}}
	if err := validateClaudeRestoreState(targets, state); err == nil {
		t.Fatal("journal could replace unrelated settings")
	}
}

func TestClaudePartialRollbackRestoresAbsentInFlightValue(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"autoModeEnabled": true}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	snapshot, err := snapshotClaudeProfileForRestart(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// Claude removes this owned value while quitting. The original value is
	// still true, but this attempt must roll back to the post-quit absence.
	profile := claudeDesktopReadJSON(t, paths.profile)
	delete(profile, "autoModeEnabled")
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := snapshot.recordAfterQuit(context.Background()); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("later profile update failed")
	err = withClaudeProfileChanges(targets, func() error {
		profile["autoModeEnabled"] = false
		if err := claudeWriteOwned(paths.profile, profile, "autoModeEnabled"); err != nil {
			return err
		}
		if err := writeClaudeDesktopDeploymentMode(paths.normalConfig, "3p"); err != nil {
			return err
		}
		// An independent user edit prevents full rollback and leaves recovery
		// pending after the profile has already been rolled back.
		if err := writeClaudeDesktopJSON(paths.normalConfig, map[string]any{"deploymentMode": "1p"}); err != nil {
			return err
		}
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("partial rollback error = %v", err)
	}
	if _, exists := claudeDesktopReadJSON(t, paths.profile)["autoModeEnabled"]; exists {
		t.Fatal("attempt rollback did not restore the missing post-quit value")
	}
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("restore should preserve the independent deployment-mode edit")
	}
	if got := claudeDesktopReadJSON(t, paths.profile)["autoModeEnabled"]; got != true {
		t.Fatalf("original autoModeEnabled was not recovered after partial rollback: %v", got)
	}
	if got := readClaudeDesktopDeploymentMode(paths.normalConfig); got != "1p" {
		t.Fatalf("later user edit was not preserved: %v", got)
	}
}

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
