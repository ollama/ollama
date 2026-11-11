package launch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

var (
	claudeProfileGate        = make(chan struct{}, 1)
	claudeProfileTransaction *claudeProfileChanges
)

var claudeGatewayProfileKeys = []string{"inferenceProvider", "inferenceGatewayBaseUrl", "inferenceGatewayApiKey", "inferenceGatewayAuthScheme", "deploymentDisplayName", "chatTabEnabled", "inferenceModels", "disableDeploymentModeChooser", "coworkEgressAllowedHosts", "disableEssentialTelemetry", "disableNonessentialTelemetry", "autoModeEnabled"}

type claudeOwnedValue struct {
	Before   json.RawMessage `json:"before,omitempty"`
	Applied  json.RawMessage `json:"applied,omitempty"`
	Previous json.RawMessage `json:"previous,omitempty"`
	// PreviousSet distinguishes an absent rollback target from no pending change.
	PreviousSet bool `json:"previousSet,omitempty"`
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
	state   *claudeRestoreState
	path    string
	attempt map[string]*claudeOwnedFile
}

type claudeRestartProfile struct {
	path  string
	state []byte
	files map[string]*claudeOwnedFile
}

func snapshotClaudeProfileForRestart(ctx context.Context) (*claudeRestartProfile, error) {
	unlock, err := claudeProfileLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return nil, err
	}
	path, err := claudeRestorePath()
	if err != nil {
		return nil, err
	}
	state, data, err := claudeReadState(path)
	if err != nil {
		return nil, err
	}
	if data == nil {
		state, err = legacyClaudeRestoreState(targets)
		if err != nil || len(state.Files) == 0 {
			return nil, err
		}
		if err := claudeSaveState(path, state); err != nil {
			return nil, err
		}
		data, err = os.ReadFile(path)
		if err != nil {
			return nil, err
		}
	}
	if err := validateClaudeRestoreState(targets, state); err != nil {
		return nil, err
	}
	snapshot := &claudeRestartProfile{path: path, state: data, files: make(map[string]*claudeOwnedFile)}
	for path, file := range state.Files {
		current, err := readClaudeDesktopJSONAllowMissing(path)
		if err != nil {
			return nil, err
		}
		if _, tracked := file.Values["entries"]; tracked && current["entries"] != nil {
			if _, ok := current["entries"].([]any); !ok {
				return nil, errors.New("Claude profile entries changed after setup; leaving settings unchanged")
			}
		}
		owned := &claudeOwnedFile{Existed: true, Values: make(map[string]claudeOwnedValue)}
		for key, value := range file.Values {
			before := claudeField(current, key)
			if claudeValueOwned(before, value) {
				owned.Values[key] = claudeOwnedValue{Before: before}
			}
		}
		if len(owned.Values) > 0 {
			snapshot.files[path] = owned
		}
	}
	return snapshot, nil
}

// Accept shutdown writes only for fields owned before this confirmed restart.
// Retain the original restore values and leave existing user edits unowned.
func (snapshot *claudeRestartProfile) recordAfterQuit(ctx context.Context) error {
	if snapshot == nil {
		return nil
	}
	unlock, err := claudeProfileLock(ctx)
	if err != nil {
		return err
	}
	defer unlock()
	state, data, err := claudeReadState(snapshot.path)
	if err != nil {
		return err
	}
	if !bytes.Equal(data, snapshot.state) {
		return errors.New("Claude settings changed while restarting; try again")
	}
	for path, file := range snapshot.files {
		current, err := readClaudeDesktopJSONAllowMissing(path)
		if err != nil {
			return err
		}
		if _, tracked := file.Values["entries"]; tracked && current["entries"] != nil {
			if _, ok := current["entries"].([]any); !ok {
				return errors.New("Claude profile entries changed while restarting; leaving settings unchanged")
			}
		}
		for key, value := range file.Values {
			owned := state.Files[path].Values[key]
			owned.Previous = value.Before
			owned.PreviousSet = true
			owned.Applied = claudeField(current, key)
			state.Files[path].Values[key] = owned
		}
	}
	return claudeSaveState(snapshot.path, state)
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
	adopted := false
	defer func() {
		if !adopted {
			<-claudeProfileGate
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path, err := claudeRestorePath()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	unlock, err := fileutil.LockContext(ctx, path+".lock")
	if err != nil {
		return nil, err
	}
	adopted = true
	return func() { unlock(); <-claudeProfileGate }, nil
}

func claudeReadState(path string) (*claudeRestoreState, []byte, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &claudeRestoreState{Version: 1, Files: make(map[string]*claudeOwnedFile)}, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	var state claudeRestoreState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, nil, fmt.Errorf("read Claude restore state: %w", err)
	}
	if state.Version != 1 || state.Files == nil {
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

func claudeValueOwned(current json.RawMessage, value claudeOwnedValue) bool {
	return claudeJSONEqual(current, value.Applied) || (value.PreviousSet || value.Previous != nil) && claudeJSONEqual(current, value.Previous)
}

func claudeValueRestorable(current json.RawMessage, value claudeOwnedValue) bool {
	return claudeValueOwned(current, value) || claudeJSONEqual(current, value.Before)
}

func claudeJSONEqual(a, b json.RawMessage) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	var compactA, compactB bytes.Buffer
	if a == nil {
		return true
	}
	return json.Compact(&compactA, a) == nil && json.Compact(&compactB, b) == nil && bytes.Equal(compactA.Bytes(), compactB.Bytes())
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
			found := false
			for _, allowedKey := range keys {
				found = found || key == allowedKey
			}
			if !found {
				return fmt.Errorf("unexpected Claude restore setting %s; leaving settings unchanged", key)
			}
			for _, raw := range []json.RawMessage{value.Before, value.Applied, value.Previous} {
				if raw != nil && !json.Valid(raw) {
					return errors.New("invalid Claude restore value; leaving settings unchanged")
				}
				if key == "entries" && raw != nil && !claudeJSONEqual(raw, json.RawMessage(`null`)) {
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
	}
	return nil
}

func claudeWriteOwned(path string, desired map[string]any, keys ...string) error {
	current, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return err
	}
	tx := claudeProfileTransaction
	if tx == nil {
		return writeClaudeDesktopJSON(path, desired)
	}
	file := tx.state.Files[path]
	if file == nil {
		_, statErr := os.Stat(path)
		file = &claudeOwnedFile{Existed: statErr == nil, Values: make(map[string]claudeOwnedValue)}
		tx.state.Files[path] = file
	}
	attempt := tx.attempt[path]
	if attempt == nil {
		_, statErr := os.Stat(path)
		attempt = &claudeOwnedFile{Existed: statErr == nil, Values: make(map[string]claudeOwnedValue)}
		tx.attempt[path] = attempt
	}
	for _, key := range keys {
		before, after := claudeField(current, key), claudeField(desired, key)
		value, tracked := file.Values[key]
		if tracked && !claudeValueOwned(before, value) {
			return fmt.Errorf("Claude setting %s in %s changed after setup; leaving that change intact", key, path)
		}
		if !tracked {
			value.Before = before
		}
		if _, exists := attempt.Values[key]; !exists {
			// Retain the first rollback target until the entire change commits.
			// Recovery can then recognize keys rolled back before a later failure.
			value.Previous = before
			value.PreviousSet = true
			attempt.Values[key] = claudeOwnedValue{Before: before, Applied: after}
		} else {
			v := attempt.Values[key]
			v.Applied = after
			attempt.Values[key] = v
		}
		value.Applied = after
		file.Values[key] = value
		claudePutField(current, key, after)
	}
	if err := claudeSaveState(tx.path, tx.state); err != nil {
		return err
	}
	return writeClaudeDesktopJSON(path, current)
}

func claudeRestoreFiles(ctx context.Context, files map[string]*claudeOwnedFile) error {
	var failures []error
	for path, file := range files {
		if err := ctx.Err(); err != nil {
			return errors.Join(append(failures, err)...)
		}
		current, err := readClaudeDesktopJSONAllowMissing(path)
		if err != nil {
			failures = append(failures, err)
			continue
		}
		changed := false
		restored := make([]string, 0, len(file.Values))
		for key, value := range file.Values {
			if !claudeValueRestorable(claudeField(current, key), value) {
				failures = append(failures, fmt.Errorf("preserved edited Claude setting %s in %s", key, path))
				continue
			}
			claudePutField(current, key, value.Before)
			restored = append(restored, key)
			changed = true
		}
		if changed {
			if !file.Existed && len(current) == 0 {
				err = os.Remove(path)
				if os.IsNotExist(err) {
					err = nil
				}
			} else {
				err = writeClaudeDesktopJSON(path, current)
			}
			if err != nil {
				failures = append(failures, err)
			} else {
				for _, key := range restored {
					delete(file.Values, key)
				}
			}
		}
	}
	return errors.Join(failures...)
}

func withClaudeProfileChangesContext(ctx context.Context, targets claudeDesktopTargets, change func() error) error {
	unlock, err := claudeProfileLock(ctx)
	if err != nil {
		return err
	}
	defer unlock()
	path, err := claudeRestorePath()
	if err != nil {
		return err
	}
	state, prior, err := claudeReadState(path)
	if err != nil {
		return err
	}
	if err := validateClaudeRestoreState(targets, state); err != nil {
		return err
	}
	// Parse every file before the first write so a damaged later file leaves
	// the existing connection untouched.
	paths := append([]string(nil), targets.normalConfigs...)
	for _, target := range targets.thirdPartyProfiles {
		paths = append(paths, target.profile, target.meta, target.desktopConfig)
	}
	for _, file := range paths {
		cfg, err := readClaudeDesktopJSONAllowMissing(file)
		if err != nil {
			return err
		}
		for _, target := range targets.thirdPartyProfiles {
			if file == target.meta && cfg["entries"] != nil {
				if _, ok := cfg["entries"].([]any); !ok {
					return errors.New("Claude profile entries must be an array; leaving settings unchanged")
				}
			}
		}
	}
	if prior == nil {
		legacy, err := legacyClaudeRestoreState(targets)
		if err != nil {
			return err
		}
		if len(legacy.Files) > 0 {
			// Legacy metadata and mode files are already managed. Checkpoint all
			// of them before the first write so an interrupted connect can restore.
			state = legacy
			if err := claudeSaveState(path, state); err != nil {
				return err
			}
			prior, err = os.ReadFile(path)
			if err != nil {
				return err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	tx := &claudeProfileChanges{state: state, path: path, attempt: make(map[string]*claudeOwnedFile)}
	claudeProfileTransaction = tx
	err = change()
	claudeProfileTransaction = nil
	if err == nil {
		err = ctx.Err()
	}
	if err == nil {
		if len(tx.attempt) == 0 {
			return nil
		}
		for file, attempt := range tx.attempt {
			for key := range attempt.Values {
				value := state.Files[file].Values[key]
				value.Previous = nil
				value.PreviousSet = false
				state.Files[file].Values[key] = value
			}
		}
		if err = claudeSaveState(path, state); err == nil {
			return nil
		}
	}
	rollbackErr := claudeRestoreFiles(context.Background(), tx.attempt)
	if rollbackErr == nil {
		if prior == nil {
			rollbackErr = os.Remove(path)
			if os.IsNotExist(rollbackErr) {
				rollbackErr = nil
			}
		} else {
			rollbackErr = fileutil.WriteWithBackup(path, prior, "claude-desktop")
		}
	}
	return errors.Join(err, rollbackErr)
}

func legacyClaudeRestoreState(targets claudeDesktopTargets) (*claudeRestoreState, error) {
	state := &claudeRestoreState{Version: 1, Files: make(map[string]*claudeOwnedFile)}
	track := func(path, key string, before json.RawMessage) error {
		cfg, err := readClaudeDesktopJSONAllowMissing(path)
		if err != nil {
			return err
		}
		if key == "entries" && cfg[key] != nil {
			if _, ok := cfg[key].([]any); !ok {
				return errors.New("Claude profile entries must be an array; leaving settings unchanged")
			}
		}
		file := state.Files[path]
		if file == nil {
			file = &claudeOwnedFile{Existed: true, Values: make(map[string]claudeOwnedValue)}
			state.Files[path] = file
		}
		file.Values[key] = claudeOwnedValue{Before: before, Applied: claudeField(cfg, key)}
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
		metadata, err := readClaudeDesktopJSON(target.meta)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read Claude profile metadata: %w", err)
		}
		if metadata["appliedId"] != claudeDesktopProfileID {
			continue
		}
		for _, key := range claudeGatewayProfileKeys {
			if key == "inferenceGatewayApiKey" || key == "chatTabEnabled" {
				continue
			}
			var before json.RawMessage
			if key == "disableDeploymentModeChooser" {
				before = json.RawMessage(`false`)
			}
			if err := track(target.profile, key, before); err != nil {
				return nil, err
			}
		}
		if err := track(target.meta, "appliedId", nil); err != nil {
			return nil, err
		}
		if err := track(target.meta, "entries", nil); err != nil {
			return nil, err
		}
		if err := track(target.desktopConfig, "deploymentMode", json.RawMessage(`"1p"`)); err != nil {
			return nil, err
		}
		normalRoot := strings.TrimSuffix(filepath.Dir(target.desktopConfig), "-3p")
		for _, normal := range targets.normalConfigs {
			if filepath.Dir(normal) == normalRoot {
				if err := track(normal, "deploymentMode", json.RawMessage(`"1p"`)); err != nil {
					return nil, err
				}
			}
		}
	}
	return state, nil
}

func restoreClaudeOwnedProfile(ctx context.Context, targets claudeDesktopTargets) error {
	path, err := claudeRestorePath()
	if err != nil {
		return err
	}
	unlock, err := claudeProfileLock(ctx)
	if err != nil {
		return err
	}
	defer unlock()
	state, data, err := claudeReadState(path)
	if err != nil {
		return err
	}
	if data == nil {
		state, err = legacyClaudeRestoreState(targets)
		if err != nil {
			return err
		}
		if len(state.Files) == 0 {
			return nil
		}
		if err := claudeSaveState(path, state); err != nil {
			return err
		}
	}
	if err := validateClaudeRestoreState(targets, state); err != nil {
		return err
	}
	for file, values := range state.Files {
		current, err := readClaudeDesktopJSONAllowMissing(file)
		if err != nil {
			return err
		}
		if _, tracked := values.Values["entries"]; tracked && current["entries"] != nil {
			if _, ok := current["entries"].([]any); !ok {
				return errors.New("Claude profile entries changed after setup; leaving settings unchanged")
			}
		}
		for _, key := range []string{"appliedId", "inferenceProvider", "inferenceGatewayBaseUrl"} {
			if value, ok := values.Values[key]; ok && !claudeValueRestorable(claudeField(current, key), value) {
				return fmt.Errorf("Claude routing changed at %s; preserving the current profile", file)
			}
		}
	}
	err = claudeRestoreFiles(ctx, state.Files)
	if err != nil {
		return errors.Join(err, claudeSaveState(path, state))
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
