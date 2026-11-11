package launch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

var (
	claudeProfileGate        = make(chan struct{}, 1)
	claudeProfileTransaction *claudeProfileChanges
)

var claudeGatewayProfileKeys = []string{"inferenceProvider", "inferenceGatewayBaseUrl", "inferenceGatewayApiKey", "inferenceGatewayAuthScheme", "deploymentDisplayName", "chatTabEnabled", "inferenceModels", "disableDeploymentModeChooser", "coworkEgressAllowedHosts", "disableEssentialTelemetry", "disableNonessentialTelemetry", "autoModeEnabled"}

type claudeOwnedValue struct {
	Before  json.RawMessage     `json:"before,omitempty"`
	Applied json.RawMessage     `json:"applied,omitempty"`
	Pending *claudePendingValue `json:"pending,omitempty"`
}

// The wrapper distinguishes an absent value from no pending write.
type claudePendingValue struct {
	Value  json.RawMessage `json:"value,omitempty"`
	Quit   bool            `json:"quit,omitempty"`
	Edited bool            `json:"edited,omitempty"`
}

type claudeOwnedFile struct {
	Existed bool                        `json:"existed"`
	Values  map[string]claudeOwnedValue `json:"values"`
}

type claudeRestoreState struct {
	Version int                         `json:"version"`
	Files   map[string]*claudeOwnedFile `json:"files"`
}

type claudeProfileChanges struct {
	state *claudeRestoreState
	path  string
}

type claudeRestartProfile struct {
	state []byte
}

type claudeRestartRecovery struct {
	fingerprint string
}

type claudeRestartConsentKey struct{}

func (*claudeRestartRecovery) Error() string {
	return "Claude settings changed during an interrupted restart; run 'ollama launch claude-desktop --restore' in an interactive terminal to recover"
}

func withClaudeRestoreState(ctx context.Context, targets claudeDesktopTargets, change func(*claudeRestoreState, string, []byte) error) error {
	unlock, err := claudeProfileLock(ctx)
	if err != nil {
		return err
	}
	defer unlock()
	path, err := claudeRestorePath()
	if err != nil {
		return err
	}
	state, data, err := claudeReadState(path)
	if err != nil {
		return err
	}
	if data == nil {
		state, err = legacyClaudeRestoreState(targets)
		if err != nil {
			return err
		}
	}
	if err := validateClaudeRestoreState(targets, state); err != nil {
		return err
	}
	return change(state, path, data)
}

// Approval is limited to the journal and settings the user actually saw.
func prepareClaudeRestartRecovery(ctx context.Context, targets claudeDesktopTargets) (context.Context, error) {
	err := withClaudeRestoreState(ctx, targets, func(state *claudeRestoreState, path string, data []byte) error {
		return reconcileClaudeQuit(ctx, state, path, data, false, false)
	})
	if err == nil {
		return ctx, nil
	}
	var recovery *claudeRestartRecovery
	if !errors.As(err, &recovery) || !isInteractiveSession() {
		return nil, err
	}
	restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{})
	confirmed, err := ConfirmPromptWithOptions("An interrupted restart left Claude settings changed. Restore the saved settings and replace those changes?", ConfirmOptions{
		YesLabel: "Restore saved settings", NoLabel: "Keep current changes", Default: ConfirmDefaultNo,
	})
	restorePolicy()
	if err != nil {
		return nil, err
	}
	if !confirmed {
		return nil, ErrCancelled
	}
	return context.WithValue(ctx, claudeRestartConsentKey{}, recovery.fingerprint), nil
}

func snapshotClaudeProfileForRestart(ctx context.Context) (*claudeRestartProfile, error) {
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return nil, err
	}
	var snapshot *claudeRestartProfile
	err = withClaudeRestoreState(ctx, targets, func(state *claudeRestoreState, path string, data []byte) error {
		if len(state.Files) == 0 {
			return nil
		}
		// Reuse an interrupted quit's checkpoint rather than replacing saved edits.
		if hasClaudePendingQuit(state) {
			if err := reconcileClaudeQuit(ctx, state, path, data, false, false); err != nil {
				return err
			}
		} else {
			for filePath, file := range state.Files {
				cfg, err := readClaudeRestoreConfig(filePath)
				if err != nil {
					return err
				}
				for key, value := range file.Values {
					current := claudeField(cfg, key)
					value.Pending = &claudePendingValue{Value: current, Quit: true, Edited: !claudeValueOwned(current, value)}
					file.Values[key] = value
				}
			}
			if err := claudeSaveState(path, state); err != nil {
				return err
			}
			data, err = os.ReadFile(path)
			if err != nil {
				return err
			}
		}
		snapshot = &claudeRestartProfile{state: data}
		return nil
	})
	return snapshot, err
}

func (snapshot *claudeRestartProfile) recordAfterQuit(ctx context.Context) error {
	if snapshot == nil {
		return nil
	}
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return err
	}
	return withClaudeRestoreState(ctx, targets, func(state *claudeRestoreState, path string, data []byte) error {
		if !bytes.Equal(data, snapshot.state) {
			return errors.New("Claude settings changed while restarting; try again")
		}
		return reconcileClaudeQuit(ctx, state, path, data, true, true)
	})
}

func hasClaudePendingQuit(state *claudeRestoreState) bool {
	for _, file := range state.Files {
		for _, value := range file.Values {
			if value.Pending != nil && value.Pending.Quit {
				return true
			}
		}
	}
	return false
}

// Undo shutdown writes before configuring or restoring. After an interruption,
// unknown values need explicit approval; native and headless calls retain them.
func reconcileClaudeQuit(ctx context.Context, state *claudeRestoreState, path string, data []byte, live, apply bool) error {
	current := make(map[string]map[string]any)
	observed := make(map[string]map[string]claudePendingValue)
	unknown := false
	for filePath, file := range state.Files {
		for key, value := range file.Values {
			prior := value.Pending
			if prior == nil || !prior.Quit {
				continue
			}
			if current[filePath] == nil {
				cfg, err := readClaudeRestoreConfig(filePath)
				if err != nil {
					return err
				}
				current[filePath] = cfg
				observed[filePath] = make(map[string]claudePendingValue)
			}
			raw := claudeField(current[filePath], key)
			observed[filePath][key] = claudePendingValue{Value: raw}
			unknown = unknown || !prior.Edited && !claudeValueRestorable(key, raw, value)
		}
	}
	if len(current) == 0 {
		return nil
	}
	values, _ := json.Marshal(observed)
	fingerprint := string(data) + string(values)
	if !live {
		consent, _ := ctx.Value(claudeRestartConsentKey{}).(string)
		if consent != "" && consent != fingerprint {
			return errors.New("Claude settings changed while confirming recovery; try again")
		}
		if unknown && consent == "" {
			return &claudeRestartRecovery{fingerprint: fingerprint}
		}
	}
	if !apply {
		return nil
	}
	for filePath, cfg := range current {
		file := state.Files[filePath]
		for key, value := range file.Values {
			prior := value.Pending
			if prior == nil || !prior.Quit {
				continue
			}
			// Restore a pre-quit user edit only when Claude replaced it with our value.
			if !prior.Edited || claudeJSONEqual(claudeField(cfg, key), value.Applied) {
				claudePutField(cfg, key, prior.Value)
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := writeClaudeDesktopJSON(filePath, cfg); err != nil {
			return err
		}
		for key, value := range file.Values {
			if value.Pending != nil && value.Pending.Quit {
				if !value.Pending.Edited {
					value.Applied = value.Pending.Value
				}
				value.Pending = nil
				file.Values[key] = value
			}
		}
	}
	return claudeSaveState(path, state)
}

func recoverClaudePendingQuit(ctx context.Context, state *claudeRestoreState, path string, data []byte) error {
	if !hasClaudePendingQuit(state) {
		return nil
	}
	if err := reconcileClaudeQuit(ctx, state, path, data, false, false); err != nil {
		return err
	}
	running, err := claudeDesktopIsRunning(ctx)
	if err != nil {
		return err
	}
	if running {
		return fmt.Errorf("quit Claude Desktop before recovering settings; %s", claudeDesktopRestoreMessage)
	}
	return reconcileClaudeQuit(ctx, state, path, data, false, true)
}

func claudeRestorePath() (string, error) {
	home, err := claudeDesktopUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "backup", "claude-desktop", "restore.json"), nil
}

func claudeProfileLock(ctx context.Context) (func(), error) {
	select {
	case claudeProfileGate <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	path, err := claudeRestorePath()
	if err == nil {
		err = os.MkdirAll(filepath.Dir(path), 0o700)
	}
	if err != nil {
		<-claudeProfileGate
		return nil, err
	}
	unlock, err := fileutil.LockContext(ctx, path+".lock")
	if err != nil {
		<-claudeProfileGate
		return nil, err
	}
	return func() { unlock(); <-claudeProfileGate }, nil
}

func claudeReadState(path string) (*claudeRestoreState, []byte, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &claudeRestoreState{Version: 2, Files: make(map[string]*claudeOwnedFile)}, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	var state claudeRestoreState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, nil, fmt.Errorf("read Claude restore state: %w", err)
	}
	if state.Version != 2 || state.Files == nil {
		return nil, nil, errors.New("unsupported Claude restore state; leaving settings unchanged")
	}
	return &state, data, nil
}

func claudeSaveState(path string, state *claudeRestoreState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, append(data, '\n'), "claude-desktop")
}

func claudeValueOwned(current json.RawMessage, value claudeOwnedValue) bool {
	return claudeJSONEqual(current, value.Applied) || value.Pending != nil && !value.Pending.Edited && claudeJSONEqual(current, value.Pending.Value)
}

func claudeValueRestorable(key string, current json.RawMessage, value claudeOwnedValue) bool {
	// Other profiles can keep entries present after our original entry is gone.
	return claudeValueOwned(current, value) || claudeJSONEqual(current, value.Before) || key == "entries" && value.Before == nil && bytes.Equal(current, []byte("[]"))
}

func claudeJSONEqual(a, b json.RawMessage) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	var compactA, compactB bytes.Buffer
	return json.Compact(&compactA, a) == nil && json.Compact(&compactB, b) == nil && bytes.Equal(compactA.Bytes(), compactB.Bytes())
}

func readClaudeRestoreConfig(path string) (map[string]any, error) {
	cfg, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return nil, err
	}
	if filepath.Base(path) == "_meta.json" && cfg["entries"] != nil {
		if _, ok := cfg["entries"].([]any); !ok {
			return nil, errors.New("Claude profile entries must be an array; leaving settings unchanged")
		}
	}
	return cfg, nil
}

func validateClaudeRestoreState(targets claudeDesktopTargets, state *claudeRestoreState) error {
	allowed := make(map[string][]string)
	for _, path := range targets.normalConfigs {
		allowed[path] = []string{"deploymentMode"}
	}
	for _, target := range targets.thirdPartyProfiles {
		allowed[target.profile] = claudeGatewayProfileKeys
		allowed[target.meta] = []string{"appliedId", "entries"}
		allowed[target.desktopConfig] = []string{"deploymentMode"}
	}
	for path, file := range state.Files {
		keys, ok := allowed[path]
		if !ok || file == nil || file.Values == nil {
			return errors.New("Claude restore state refers to an unexpected file; leaving settings unchanged")
		}
		for key, value := range file.Values {
			if !slices.Contains(keys, key) {
				return fmt.Errorf("unexpected Claude restore setting %s; leaving settings unchanged", key)
			}
			values := []json.RawMessage{value.Before, value.Applied}
			if value.Pending != nil {
				if value.Pending.Edited && !value.Pending.Quit {
					return errors.New("invalid Claude restart state; leaving settings unchanged")
				}
				values = append(values, value.Pending.Value)
			}
			if key != "entries" {
				continue
			}
			for _, raw := range values {
				if raw == nil || bytes.Equal(raw, []byte("null")) {
					continue
				}
				var entries []map[string]any
				if err := json.Unmarshal(raw, &entries); err != nil {
					return errors.New("invalid Claude profile entries in restore state; leaving settings unchanged")
				}
				for _, entry := range entries {
					if entry["id"] != claudeDesktopProfileID {
						return errors.New("Claude restore state refers to another profile; leaving settings unchanged")
					}
				}
			}
		}
	}
	return nil
}

func claudeWriteOwned(path string, desired map[string]any, keys ...string) error {
	tx := claudeProfileTransaction
	if tx == nil {
		return writeClaudeDesktopJSON(path, desired)
	}
	current, err := readClaudeRestoreConfig(path)
	if err != nil {
		return err
	}
	file := tx.state.Files[path]
	if file == nil {
		_, err := os.Stat(path)
		file = &claudeOwnedFile{Existed: err == nil, Values: make(map[string]claudeOwnedValue)}
		tx.state.Files[path] = file
	}
	for _, key := range keys {
		before := claudeField(current, key)
		value, tracked := file.Values[key]
		if tracked && !claudeValueOwned(before, value) {
			return fmt.Errorf("Claude setting %s in %s changed after setup; leaving that change intact", key, path)
		}
		if !tracked {
			value.Before = before
		}
		value.Pending = &claudePendingValue{Value: before}
		value.Applied = claudeField(desired, key)
		file.Values[key] = value
		claudePutField(current, key, value.Applied)
	}
	// Persist both possible values before touching a settings file.
	if err := claudeSaveState(tx.path, tx.state); err != nil {
		return err
	}
	if err := writeClaudeDesktopJSON(path, current); err != nil {
		return err
	}
	for _, key := range keys {
		value := file.Values[key]
		value.Pending = nil
		file.Values[key] = value
	}
	return claudeSaveState(tx.path, tx.state)
}

func withClaudeProfileChangesContext(ctx context.Context, targets claudeDesktopTargets, change func() error) error {
	return withClaudeRestoreState(ctx, targets, func(state *claudeRestoreState, path string, data []byte) error {
		if err := recoverClaudePendingQuit(ctx, state, path, data); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		claudeProfileTransaction = &claudeProfileChanges{state: state, path: path}
		defer func() { claudeProfileTransaction = nil }()
		if err := change(); err != nil {
			return fmt.Errorf("%w; %s", err, claudeDesktopRestoreMessage)
		}
		return ctx.Err()
	})
}

func restoreClaudeOwnedProfile(ctx context.Context, targets claudeDesktopTargets) error {
	return withClaudeRestoreState(ctx, targets, func(state *claudeRestoreState, path string, data []byte) error {
		if len(state.Files) == 0 {
			return nil
		}
		if err := recoverClaudePendingQuit(ctx, state, path, data); err != nil {
			return err
		}
		configs := make(map[string]map[string]any)
		for filePath, file := range state.Files {
			cfg, err := readClaudeRestoreConfig(filePath)
			if err != nil {
				return err
			}
			for _, key := range []string{"appliedId", "inferenceProvider", "inferenceGatewayBaseUrl"} {
				if value, ok := file.Values[key]; ok && !claudeValueRestorable(key, claudeField(cfg, key), value) {
					return fmt.Errorf("Claude routing changed at %s; preserving the current profile", filePath)
				}
			}
			configs[filePath] = cfg
		}
		if err := claudeSaveState(path, state); err != nil {
			return err
		}
		// Keep the checkpoint until every file is restored. Retrying a partial
		// restore is safe because original values are always recognized.
		var failures []error
		for filePath, file := range state.Files {
			cfg := configs[filePath]
			changed := false
			for key, value := range file.Values {
				if !claudeValueRestorable(key, claudeField(cfg, key), value) {
					failures = append(failures, fmt.Errorf("preserved edited Claude setting %s in %s", key, filePath))
					continue
				}
				claudePutField(cfg, key, value.Before)
				changed = true
			}
			if !changed {
				continue
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			var err error
			if !file.Existed && len(cfg) == 0 {
				err = os.Remove(filePath)
				if os.IsNotExist(err) {
					err = nil
				}
			} else {
				err = writeClaudeDesktopJSON(filePath, cfg)
			}
			if err != nil {
				return err
			}
		}
		if err := errors.Join(failures...); err != nil {
			return err
		}
		err := os.Remove(path)
		if os.IsNotExist(err) {
			return nil
		}
		return err
	})
}

func claudeField(cfg map[string]any, key string) json.RawMessage {
	if key == "entries" {
		if value, exists := cfg[key]; exists && value == nil {
			return json.RawMessage(`null`)
		}
		var entries []any
		for _, entry := range claudeDesktopAnySlice(cfg[key]) {
			m, _ := entry.(map[string]any)
			if m["id"] == claudeDesktopProfileID {
				entries = append(entries, entry)
			}
		}
		if len(entries) == 0 {
			if _, exists := cfg[key]; exists {
				return json.RawMessage(`[]`)
			}
			return nil
		}
		data, _ := json.Marshal(entries)
		return data
	}
	value, exists := cfg[key]
	if !exists {
		return nil
	}
	data, _ := json.Marshal(value)
	return data
}

func claudePutField(cfg map[string]any, key string, raw json.RawMessage) {
	if key == "entries" {
		var entries []any
		for _, entry := range claudeDesktopAnySlice(cfg[key]) {
			m, _ := entry.(map[string]any)
			if m["id"] != claudeDesktopProfileID {
				entries = append(entries, entry)
			}
		}
		var original []any
		if raw != nil {
			_ = json.Unmarshal(raw, &original)
		}
		entries = append(entries, original...)
		if len(entries) == 0 && raw == nil {
			delete(cfg, key)
		} else if len(entries) == 0 && claudeJSONEqual(raw, json.RawMessage(`null`)) {
			cfg[key] = nil
		} else {
			if entries == nil {
				entries = []any{}
			}
			cfg[key] = entries
		}
		return
	}
	if raw == nil {
		delete(cfg, key)
		return
	}
	var value any
	_ = json.Unmarshal(raw, &value)
	cfg[key] = value
}

// Connections made before checkpoints existed can only restore the known
// legacy defaults. Require both the local gateway and our applied profile ID.
func legacyClaudeRestoreState(targets claudeDesktopTargets) (*claudeRestoreState, error) {
	state := &claudeRestoreState{Version: 2, Files: make(map[string]*claudeOwnedFile)}
	track := func(path string, before map[string]json.RawMessage) error {
		cfg, err := readClaudeRestoreConfig(path)
		if err != nil {
			return err
		}
		file := &claudeOwnedFile{Existed: true, Values: make(map[string]claudeOwnedValue)}
		for key, value := range before {
			file.Values[key] = claudeOwnedValue{Before: value, Applied: claudeField(cfg, key)}
		}
		state.Files[path] = file
		return nil
	}
	for _, target := range targets.thirdPartyProfiles {
		cfg, err := readClaudeDesktopJSONAllowMissing(target.profile)
		if err != nil {
			return nil, err
		}
		endpoint, _ := cfg["inferenceGatewayBaseUrl"].(string)
		if cfg["inferenceProvider"] != "gateway" || (endpoint != "http://127.0.0.1:11435" && !claudeDesktopGatewayURLMatches(endpoint)) {
			continue
		}
		metadata, err := readClaudeDesktopJSONAllowMissing(target.meta)
		if err != nil {
			return nil, err
		}
		if metadata["appliedId"] != claudeDesktopProfileID {
			continue
		}
		before := make(map[string]json.RawMessage)
		for _, key := range claudeGatewayProfileKeys {
			if key != "inferenceGatewayApiKey" && key != "chatTabEnabled" {
				before[key] = nil
			}
		}
		before["disableDeploymentModeChooser"] = json.RawMessage(`false`)
		defaults := map[string]map[string]json.RawMessage{
			target.profile:       before,
			target.meta:          {"appliedId": nil, "entries": nil},
			target.desktopConfig: {"deploymentMode": json.RawMessage(`"1p"`)},
		}
		normalRoot := strings.TrimSuffix(filepath.Dir(target.desktopConfig), "-3p")
		for _, normal := range targets.normalConfigs {
			if filepath.Dir(normal) == normalRoot {
				defaults[normal] = map[string]json.RawMessage{"deploymentMode": json.RawMessage(`"1p"`)}
			}
		}
		for path, before := range defaults {
			if err := track(path, before); err != nil {
				return nil, err
			}
		}
	}
	return state, nil
}
