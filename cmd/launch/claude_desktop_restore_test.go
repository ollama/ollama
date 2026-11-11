package launch

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/internal/proxy"
)

func TestClaudeProfileLockRespectsShutdownDeadline(t *testing.T) {
	if path := os.Getenv("OLLAMA_TEST_CLAUDE_HELD_LOCK"); path != "" {
		unlock, err := fileutil.Lock(path)
		if err != nil {
			t.Fatal(err)
		}
		defer unlock()
		fmt.Fprintln(os.Stdout, "locked")
		_, _ = io.Copy(io.Discard, os.Stdin)
		return
	}
	for _, running := range []bool{false, true} {
		t.Run(fmt.Sprintf("running_%t", running), func(t *testing.T) {
			paths, targets := claudeRestoreFixture(t)
			if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
				t.Fatal(err)
			}
			before, _ := os.ReadFile(paths.profile)
			path, _ := claudeRestorePath()
			child := exec.Command(os.Args[0], "-test.run=^TestClaudeProfileLockRespectsShutdownDeadline$")
			child.Env = append(os.Environ(), "OLLAMA_TEST_CLAUDE_HELD_LOCK="+path+".lock")
			child.Stderr = os.Stderr
			input, err := child.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			output, err := child.StdoutPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := child.Start(); err != nil {
				t.Fatal(err)
			}
			defer func() { _ = input.Close(); _ = child.Wait() }()
			if line, err := bufio.NewReader(output).ReadString('\n'); err != nil || line != "locked\n" {
				t.Fatalf("lock helper: %q, %v", line, err)
			}
			quit := false
			withClaudeDesktopProcessHooks(t,
				func() bool { return running },
				func() error { quit = true; running = false; return nil },
				func() error { t.Error("Claude opened while waiting for a lock"); return nil },
			)
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- (&ClaudeDesktop{}).RestoreForShutdown(ctx) }()
			select {
			case err := <-done:
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("shutdown lock error = %v", err)
				}
			case <-time.After(time.Second):
				_ = input.Close()
				t.Errorf("shutdown ignored its lock deadline: %v", <-done)
			}
			after, _ := os.ReadFile(paths.profile)
			if quit || string(after) != string(before) {
				t.Fatal("waiting for a lock changed Claude settings or process state")
			}
		})
	}
}

func TestClaudeRestoreRejectsCorruptLegacyMetadata(t *testing.T) {
	for _, metadata := range []string{"{", "null"} {
		for _, running := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s_running_%t", metadata, running), func(t *testing.T) {
				paths, _ := claudeRestoreFixture(t)
				if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceProvider": "gateway", "inferenceGatewayBaseUrl": "http://127.0.0.1:11435"}); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(paths.meta, []byte(metadata), 0o600); err != nil {
					t.Fatal(err)
				}
				before, _ := os.ReadFile(paths.profile)
				withClaudeDesktopProcessHooks(t,
					func() bool { return running },
					func() error { t.Error("Claude quit before metadata validation"); running = false; return nil },
					func() error { t.Error("Claude opened after failed restore"); return nil },
				)
				restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
				defer restoreConfirm()
				if err := (&ClaudeDesktop{}).Restore(); err == nil {
					t.Fatal("corrupt legacy metadata was reported as restored")
				}
				after, _ := os.ReadFile(paths.profile)
				if string(after) != string(before) {
					t.Fatal("failed restore changed the legacy profile")
				}
			})
		}
	}
}

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

func TestClaudeLegacyRecoveryAfterInterruptedConnect(t *testing.T) {
	const interruptionHome = "OLLAMA_TEST_CLAUDE_RECOVERY_INTERRUPTION_HOME"
	if home := os.Getenv(interruptionHome); home != "" {
		setTestHome(t, home)
		withClaudeDesktopPlatform(t, "darwin")
		targets, err := claudeDesktopTargetPaths()
		if err != nil {
			t.Fatal(err)
		}
		err = withClaudeProfileChanges(targets, func() error {
			if err := writeClaudeDesktopGatewayProfile(targets.thirdPartyProfiles[0].profile, ClaudeDesktopGatewayURL(), "ollama", true, false, nil); err != nil {
				return err
			}
			// A process exit releases the lock but skips the transaction's rollback.
			os.Exit(77)
			return nil
		})
		t.Fatalf("interrupted helper returned: %v", err)
	}

	paths, targets := claudeRestoreFixture(t)
	for _, path := range []string{paths.normalConfig, paths.desktopConfig} {
		if err := writeClaudeDesktopJSON(path, map[string]any{"deploymentMode": "3p"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"inferenceProvider": "gateway", "inferenceGatewayBaseUrl": "http://127.0.0.1:11435", "inferenceGatewayApiKey": "synthetic-legacy-key", "userOwned": "keep"}); err != nil {
		t.Fatal(err)
	}
	if err := writeClaudeDesktopJSON(paths.meta, map[string]any{"appliedId": claudeDesktopProfileID, "entries": []any{map[string]any{"id": claudeDesktopProfileID, "name": "Ollama"}, map[string]any{"id": "custom", "name": "Keep"}}}); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestClaudeLegacyRecoveryAfterInterruptedConnect$")
	cmd.Env = append(os.Environ(), interruptionHome+"="+os.Getenv("HOME"))
	output, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 77 {
		t.Fatalf("interrupted helper: %v\n%s", err, output)
	}
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{paths.normalConfig, paths.desktopConfig} {
		if got := readClaudeDesktopDeploymentMode(path); got != "1p" {
			t.Fatalf("deploymentMode after recovery = %q, want 1p", got)
		}
	}
	meta := claudeDesktopReadJSON(t, paths.meta)
	if meta["appliedId"] != nil || !reflect.DeepEqual(meta["entries"], []any{map[string]any{"id": "custom", "name": "Keep"}}) {
		t.Fatalf("metadata after recovery = %#v", meta)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceProvider"] != nil || profile["inferenceGatewayBaseUrl"] != nil || profile["inferenceGatewayApiKey"] != "synthetic-legacy-key" || profile["userOwned"] != "keep" {
		t.Fatalf("profile after recovery = %#v", profile)
	}
	statePath, _ := claudeRestorePath()
	if _, err := os.Stat(statePath); !os.IsNotExist(err) {
		t.Fatalf("completed recovery retained journal: %v", err)
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

func TestClaudeProfilePreflightRejectsMalformedMetadataBeforeWrites(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"keep": true}); err != nil {
		t.Fatal(err)
	}
	if err := writeClaudeDesktopJSON(paths.meta, map[string]any{"entries": "invalid"}); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(paths.profile)
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err == nil {
		t.Fatal("malformed metadata accepted")
	}
	after, _ := os.ReadFile(paths.profile)
	if string(before) != string(after) {
		t.Fatal("profile changed before preflight completed")
	}
}

func TestClaudeRestoreRejectsUnexpectedJournalKeys(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	state := &claudeRestoreState{Version: 1, Files: map[string]*claudeOwnedFile{paths.normalConfig: {Existed: true, Values: map[string]claudeOwnedValue{"mcpServers": {Before: []byte(`{}`), Applied: []byte(`{}`)}}}}}
	if err := validateClaudeRestoreState(targets, state); err == nil {
		t.Fatal("journal could replace unrelated settings")
	}
}

func TestClaudeRestorePreservesNullMetadataEntries(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.meta, map[string]any{"entries": nil}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatal(err)
	}
	meta := claudeDesktopReadJSON(t, paths.meta)
	if value, exists := meta["entries"]; !exists || value != nil {
		t.Fatalf("metadata null was not restored: %#v", meta)
	}
}

func TestClaudeRestoreRejectsDamagedMetadataBeforeAnyWrites(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	meta := claudeDesktopReadJSON(t, paths.meta)
	meta["entries"] = "user-edited-invalid-shape"
	if err := writeClaudeDesktopJSON(paths.meta, meta); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(paths.normalConfig)
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("damaged metadata was accepted")
	}
	after, _ := os.ReadFile(paths.normalConfig)
	if string(before) != string(after) || claudeDesktopReadJSON(t, paths.meta)["entries"] != "user-edited-invalid-shape" {
		t.Fatal("restore wrote files before completing metadata validation")
	}
}

func TestClaudePartialRollbackKeepsOriginalRestoreOwnership(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	original := map[string]any{"deploymentDisplayName": "original label", "inferenceGatewayApiKey": "synthetic-original", "userOwned": "keep"}
	if err := writeClaudeDesktopJSON(paths.profile, original); err != nil {
		t.Fatal(err)
	}
	if err := withClaudeProfileChanges(targets, func() error {
		desired := claudeDesktopReadJSON(t, paths.profile)
		desired["deploymentDisplayName"] = "first label"
		desired["inferenceGatewayApiKey"] = "synthetic-first"
		return claudeWriteOwned(paths.profile, desired, "deploymentDisplayName", "inferenceGatewayApiKey")
	}); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("later profile update failed")
	err := withClaudeProfileChanges(targets, func() error {
		desired := claudeDesktopReadJSON(t, paths.profile)
		desired["deploymentDisplayName"] = "second label"
		desired["inferenceGatewayApiKey"] = "synthetic-second"
		if err := claudeWriteOwned(paths.profile, desired, "deploymentDisplayName", "inferenceGatewayApiKey"); err != nil {
			return err
		}
		desired["deploymentDisplayName"] = "third label"
		if err := claudeWriteOwned(paths.profile, desired, "deploymentDisplayName"); err != nil {
			return err
		}
		desired["inferenceGatewayApiKey"] = "synthetic-user-edit"
		if err := writeClaudeDesktopJSON(paths.profile, desired); err != nil {
			return err
		}
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("partial failure lost original error: %v", err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["deploymentDisplayName"] != "first label" || profile["inferenceGatewayApiKey"] != "synthetic-user-edit" {
		t.Fatalf("incorrect partial rollback: %#v", profile)
	}
	// A later successful update to another key must retain pending recovery facts.
	if err := withClaudeProfileChanges(targets, func() error {
		desired := claudeDesktopReadJSON(t, paths.profile)
		desired["chatTabEnabled"] = true
		return claudeWriteOwned(paths.profile, desired, "chatTabEnabled")
	}); err != nil {
		t.Fatal(err)
	}
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("disconnect should preserve the later key edit and retain its checkpoint")
	}
	profile = claudeDesktopReadJSON(t, paths.profile)
	if profile["deploymentDisplayName"] != "original label" || profile["inferenceGatewayApiKey"] != "synthetic-user-edit" || profile["chatTabEnabled"] != nil || profile["userOwned"] != "keep" {
		t.Fatalf("disconnect lost restore ownership or later edit: %#v", profile)
	}
	statePath, _ := claudeRestorePath()
	state, _, err := claudeReadState(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if file := state.Files[paths.profile]; file == nil || len(file.Values) != 1 || file.Values["inferenceGatewayApiKey"].Before == nil {
		t.Fatalf("wrong pending recovery keys: %+v", file)
	}
}

func TestClaudeRestartPreservesEditsAndOriginalRestoreState(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	original := map[string]any{"inferenceProvider": "original-provider", "inferenceGatewayBaseUrl": "http://original.example", "chatTabEnabled": false, "userOwned": "original"}
	if err := writeClaudeDesktopJSON(paths.profile, original); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["autoModeEnabled"] = true
	profile["userOwned"] = "later edit"
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	snapshot, err := snapshotClaudeProfileForRestart(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// Claude's quit write changes owned fields and adds an unrelated setting.
	profile["chatTabEnabled"] = false
	delete(profile, "inferenceGatewayBaseUrl")
	profile["quitAdded"] = "keep"
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := snapshot.recordAfterQuit(context.Background()); err != nil {
		t.Fatal(err)
	}
	// Recovery must also work if the process stops before reapplying the profile.
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("expected the later Auto mode edit to be preserved")
	}
	profile = claudeDesktopReadJSON(t, paths.profile)
	for _, key := range []string{"inferenceProvider", "inferenceGatewayBaseUrl", "chatTabEnabled"} {
		if profile[key] != original[key] {
			t.Fatalf("original %s was lost: %v", key, profile[key])
		}
	}
	if profile["autoModeEnabled"] != true || profile["userOwned"] != "later edit" || profile["quitAdded"] != "keep" {
		t.Fatalf("restart or restore lost unrelated edits: %#v", profile)
	}
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

func TestClaudeReconnectPreservesEditMatchingOriginalValue(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{"autoModeEnabled": true}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["autoModeEnabled"] = true
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	// A confirmed restart must not reclassify an existing user edit as owned.
	snapshot, err := snapshotClaudeProfileForRestart(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := snapshot.recordAfterQuit(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err == nil {
		t.Fatal("reconnect overwrote a later user edit matching the original value")
	}
	if claudeDesktopReadJSON(t, paths.profile)["autoModeEnabled"] != true {
		t.Fatal("reconnect lost the later user edit")
	}
	// Restore may recognize an original value without replacing the edit.
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatal(err)
	}
	if claudeDesktopReadJSON(t, paths.profile)["autoModeEnabled"] != true {
		t.Fatal("restore lost the original value")
	}
}

func TestClaudeLegacyPartialRestoreReconnectPreservesNewEdit(t *testing.T) {
	paths, targets := claudeRestoreFixture(t)
	if err := writeClaudeDesktopJSON(paths.profile, map[string]any{
		"inferenceProvider":       "gateway",
		"inferenceGatewayBaseUrl": "http://127.0.0.1:11435",
		"autoModeEnabled":         true,
	}); err != nil {
		t.Fatal(err)
	}
	if err := writeClaudeDesktopJSON(paths.meta, map[string]any{
		"appliedId": claudeDesktopProfileID,
		"entries":   []any{map[string]any{"id": claudeDesktopProfileID, "name": "Ollama"}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}

	// Force a partial restore. The edit remains tracked, while successfully
	// restored fields are removed from the legacy journal.
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["autoModeEnabled"] = true
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := restoreClaudeDesktopTargets(targets); err == nil {
		t.Fatal("expected partial restore")
	}

	// This field is now untracked, so a reconnect must checkpoint its new value.
	profile = claudeDesktopReadJSON(t, paths.profile)
	profile["deploymentDisplayName"] = "User label"
	// The user has also returned the one still-tracked field to the value Ollama
	// applied, so reconnect is otherwise allowed.
	profile["autoModeEnabled"] = false
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if err := configureClaudeDesktopTargets(targets, ClaudeDesktopGatewayURL(), "ollama", false, nil); err != nil {
		t.Fatal(err)
	}

	// Restore should return the newly checkpointed label.
	if err := restoreClaudeDesktopTargets(targets); err != nil {
		t.Fatal(err)
	}
	if got := claudeDesktopReadJSON(t, paths.profile)["deploymentDisplayName"]; got != "User label" {
		t.Fatalf("deploymentDisplayName = %v, want preserved user edit", got)
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
