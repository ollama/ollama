package launch

import (
	"bytes"
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/internal/proxy"
)

func requireClaudeSuccess(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func claudeRestoreFixture(t *testing.T) (claudeDesktopPaths, claudeDesktopTargets) {
	t.Helper()
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { return nil })
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
		requireClaudeSuccess(t, writeClaudeDesktopJSON(path, value))
	}
	configure := func(label string) error {
		return configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, []proxy.ClaudeDesktopCatalogModel{{ID: "claude-sonnet-5", DisplayName: label}})
	}
	requireClaudeSuccess(t, configure("first"))
	// A second connect must retain the first checkpoint, including nested JSON.
	requireClaudeSuccess(t, configure("second"))
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["userOwned"] = "later"
	profile["newPreference"] = true
	requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
	requireClaudeSuccess(t, restoreClaudeDesktopTargets(targets))
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
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatalf("repeat restore: %v", err)
	}
}

func TestClaudeRestorePreservesEditedOwnedSettings(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	requireClaudeSuccess(t, configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil))
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["autoModeEnabled"] = true
	requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
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
	requireClaudeSuccess(t, configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil))
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["inferenceGatewayBaseUrl"] = "https://user.example"
	requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
	before, _ := os.ReadFile(paths.normalConfig)
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("changed provider was not detected")
	}
	after, _ := os.ReadFile(paths.normalConfig)
	if string(after) != string(before) || claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"] != "https://user.example" {
		t.Fatal("restore changed the edited connection")
	}
}

func TestClaudeFailedChangeRetainsRecoverableCheckpoint(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	original := map[string]any{"autoModeEnabled": true, "keep": "original"}
	requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, original))
	failure := errors.New("later file failed")
	err := withClaudeProfileChanges(targets, func() error {
		if err := writeClaudeDesktopGatewayProfile(paths.profile, ClaudeDesktopGatewayURL(), "ollama", true, false, nil); err != nil {
			return err
		}
		return failure
	})
	if !errors.Is(err, failure) || !strings.Contains(err.Error(), "--restore") {
		t.Fatalf("setup error = %v", err)
	}
	statePath, _ := claudeRestorePath()
	journal, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}
	// Simulate settings being restored before the progress checkpoint commits.
	requireClaudeSuccess(t, restoreClaudeDesktopTargets(targets))
	requireClaudeSuccess(t, os.WriteFile(statePath, journal, 0o600))
	requireClaudeSuccess(t, restoreClaudeDesktopTargets(targets))
	if got := claudeDesktopReadJSON(t, paths.profile); !reflect.DeepEqual(got, original) {
		t.Fatalf("original settings lost: %#v", got)
	}
}

func TestClaudeWriteAheadRecoversAbsentAndNullValues(t *testing.T) {
	for _, original := range []map[string]any{{}, {"autoModeEnabled": nil}, {"autoModeEnabled": true}} {
		t.Run(string(claudeField(original, "autoModeEnabled")), func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, original))
			requireClaudeSuccess(t, configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil))
			statePath, _ := claudeRestorePath()
			state, _, err := claudeReadState(statePath)
			if err != nil {
				t.Fatal(err)
			}
			value := state.Files[paths.profile].Values["autoModeEnabled"]
			value.Pending = &claudePendingValue{Value: value.Applied}
			value.Applied = []byte(`true`)
			state.Files[paths.profile].Values["autoModeEnabled"] = value
			requireClaudeSuccess(t, claudeSaveState(statePath, state))
			snapshot, err := snapshotClaudeProfileForRestart(t.Context())
			requireClaudeSuccess(t, err)
			requireClaudeSuccess(t, snapshot.recordAfterQuit(t.Context()))
			requireClaudeSuccess(t, restoreClaudeDesktopTargets(targets))
			if got := claudeDesktopReadJSON(t, paths.profile); !reflect.DeepEqual(got, original) {
				t.Fatalf("restore = %#v, want %#v", got, original)
			}
		})
	}
}

func TestClaudeRestoreRetryPreservesLaterProfiles(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	requireClaudeSuccess(t, configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil))
	meta := claudeDesktopReadJSON(t, paths.meta)
	meta["entries"] = append(meta["entries"].([]any), map[string]any{"id": "later", "name": "Keep"})
	requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.meta, meta))
	statePath, _ := claudeRestorePath()
	journal, err := os.ReadFile(statePath)
	requireClaudeSuccess(t, err)
	requireClaudeSuccess(t, restoreClaudeDesktopTargets(targets))
	// Crash after restoring metadata but before removing the checkpoint.
	requireClaudeSuccess(t, os.WriteFile(statePath, journal, 0o600))
	requireClaudeSuccess(t, restoreClaudeDesktopTargets(targets))
	want := map[string]any{"entries": []any{map[string]any{"id": "later", "name": "Keep"}}}
	if got := claudeDesktopReadJSON(t, paths.meta); !reflect.DeepEqual(got, want) {
		t.Fatalf("later profile changed: %#v", got)
	}
}

func TestClaudeRestartRecoveryPreservesPreQuitEdits(t *testing.T) {
	for _, live := range []bool{false, true} {
		t.Run(map[bool]string{false: "interrupted", true: "live"}[live], func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			requireClaudeSuccess(t, configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil))
			profile := claudeDesktopReadJSON(t, paths.profile)
			profile["deploymentDisplayName"] = "user edit"
			requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
			snapshot, err := snapshotClaudeProfileForRestart(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			profile["deploymentDisplayName"] = claudeDesktopProfileName
			requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
			if live {
				requireClaudeSuccess(t, snapshot.recordAfterQuit(t.Context()))
			}
			if err := restoreClaudeDesktopTargets(targets); err == nil {
				t.Fatal("expected user edit to remain pending")
			}
			if got := claudeDesktopReadJSON(t, paths.profile)["deploymentDisplayName"]; got != "user edit" {
				t.Fatalf("edit replaced: %v", got)
			}
		})
	}
}

func TestClaudeInterruptedRestartRequiresExplicitConsent(t *testing.T) {
	for _, action := range []string{"accept", "decline", "headless", "changed"} {
		t.Run(action, func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceGatewayBaseUrl": "https://original.example"}))
			requireClaudeSuccess(t, configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil))
			if _, err := snapshotClaudeProfileForRestart(t.Context()); err != nil {
				t.Fatal(err)
			}
			profile := claudeDesktopReadJSON(t, paths.profile)
			delete(profile, "inferenceGatewayBaseUrl")
			requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
			before, _ := os.ReadFile(paths.profile)
			withInteractiveSession(t, action != "headless")
			restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
			defer restorePolicy()
			oldPrompt := DefaultConfirmPrompt
			t.Cleanup(func() { DefaultConfirmPrompt = oldPrompt })
			prompts := 0
			DefaultConfirmPrompt = func(_ string, options ConfirmOptions) (bool, error) {
				prompts++
				if options.Default != ConfirmDefaultNo {
					t.Fatal("unsafe recovery default")
				}
				if action == "changed" {
					profile["inferenceGatewayBaseUrl"] = "https://later.example"
					requireClaudeSuccess(t, writeClaudeDesktopJSON(paths.profile, profile))
				}
				return action != "decline", nil
			}
			err := (&ClaudeDesktop{}).Restore()
			if action == "accept" {
				if err != nil {
					t.Fatal(err)
				}
				if got := claudeDesktopReadJSON(t, paths.profile)["inferenceGatewayBaseUrl"]; got != "https://original.example" {
					t.Fatalf("original URL lost: %v", got)
				}
			} else {
				if err == nil {
					t.Fatal("unconfirmed recovery succeeded")
				}
				if action == "changed" {
					if !strings.Contains(err.Error(), "changed while confirming") {
						t.Fatal(err)
					}
				} else if after, _ := os.ReadFile(paths.profile); !bytes.Equal(before, after) {
					t.Fatal("unconfirmed recovery wrote settings")
				}
			}
			if action == "headless" && prompts != 0 || action != "headless" && prompts != 1 {
				t.Fatalf("prompts = %d", prompts)
			}
		})
	}
}

func TestClaudeRestoreRejectsUnexpectedJournalTargets(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	for _, field := range []string{"mcpServers", "entries"} {
		state := &claudeRestoreState{Version: 2, Files: map[string]*claudeOwnedFile{paths.normalConfig: {Values: map[string]claudeOwnedValue{field: {Before: []byte(`{}`)}}}}}
		if err := validateClaudeRestoreState(targets, state); err == nil {
			t.Fatal("journal could change unrelated settings")
		}
	}
}
