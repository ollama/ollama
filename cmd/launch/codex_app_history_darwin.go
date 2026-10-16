//go:build darwin

package launch

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/openai"
	"golang.org/x/sys/unix"
)

func restoreCodexAppProfileFromDesktop() error {
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}
	if err := prepareCodexAppHistoryForDisconnect(filepath.Dir(configPath), restoreCodexAppProfile); err != nil {
		return fmt.Errorf("prepare ChatGPT conversations before disconnecting: %w", err)
	}
	return nil
}

type codexAppHistoryEdit struct {
	path      string
	temporary string
	info      fs.FileInfo
	digest    [sha256.Size]byte
}

var errCodexAppHistoryGrowth = errors.New("cannot convert compaction without changing saved history offsets")

// The caller stops the app before changing history or restoring its connection.
func prepareCodexAppHistoryForDisconnect(home string, restore func() error) error {
	if _, err := os.Stat(home); errors.Is(err, os.ErrNotExist) {
		return restore()
	} else if err != nil {
		return err
	}
	unlock, err := lockCodexAppHistory(home)
	if err != nil {
		return err
	}
	defer unlock()

	var edits []codexAppHistoryEdit
	defer func() {
		for _, edit := range edits {
			_ = os.Remove(edit.temporary)
		}
	}()
	err = walkCodexAppHistory(home, func(path string, info fs.FileInfo) error {
		original, data, err := readCodexAppHistory(path)
		if err != nil {
			return err
		}
		rewritten, changed, err := portableCodexAppHistory(data)
		if err != nil {
			return fmt.Errorf("conversation %s: %w", path, err)
		}
		if !changed {
			return nil
		}
		if strings.HasSuffix(path, ".zst") {
			encoder, err := zstd.NewWriter(nil)
			if err != nil {
				return err
			}
			rewritten = encoder.EncodeAll(rewritten, nil)
			encoder.Close()
		}
		relative, err := filepath.Rel(home, path)
		if err != nil {
			return err
		}
		digest := sha256.Sum256(original)
		backup := filepath.Join(fileutil.BackupDir(), "chatgpt-history", relative+fmt.Sprintf(".%x.bak", digest))
		if err := backupCodexAppHistory(backup, original); err != nil {
			return fmt.Errorf("back up conversation %s: %w", path, err)
		}
		temporary, err := stageCodexAppHistory(path, rewritten, info.Mode().Perm())
		if err != nil {
			return err
		}
		edits = append(edits, codexAppHistoryEdit{path: path, temporary: temporary, info: info, digest: digest})
		return nil
	})
	if errors.Is(err, errCodexAppHistoryGrowth) {
		return migrateCodexAppGrowingHistory(home, restore)
	}
	if err != nil {
		return err
	}
	// Validate every conversation before publishing any changes. A failed publish
	// leaves the old connection in place; already converted history is safe to retry.
	for _, edit := range edits {
		if err := edit.publish(); err != nil {
			return err
		}
	}
	return restore()
}

func walkCodexAppHistory(home string, visit func(string, fs.FileInfo) error) error {
	for _, directory := range []string{"sessions", "archived_sessions"} {
		root := filepath.Join(home, directory)
		err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if errors.Is(err, os.ErrNotExist) && path == root {
				return nil
			}
			if err != nil {
				return err
			}
			if entry.Type()&os.ModeSymlink != 0 {
				return fmt.Errorf("cannot safely update linked conversation path %s", path)
			}
			if entry.IsDir() || !strings.HasPrefix(entry.Name(), "rollout-") || (!strings.HasSuffix(path, ".jsonl") && !strings.HasSuffix(path, ".jsonl.zst")) {
				return nil
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() {
				return fmt.Errorf("conversation is not a regular file: %s", path)
			}
			return visit(path, info)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func readCodexAppHistory(path string) (original, data []byte, err error) {
	original, err = os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	if !strings.HasSuffix(path, ".zst") {
		return original, original, nil
	}
	decoder, err := zstd.NewReader(nil)
	if err != nil {
		return nil, nil, err
	}
	defer decoder.Close()
	data, err = decoder.DecodeAll(original, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("read compressed conversation %s: %w", path, err)
	}
	return original, data, nil
}

func (e codexAppHistoryEdit) publish() error {
	if err := e.validate(); err != nil {
		return err
	}
	return os.Rename(e.temporary, e.path)
}

func (e codexAppHistoryEdit) validate() error {
	info, err := os.Lstat(e.path)
	if err != nil {
		return err
	}
	current, err := os.ReadFile(e.path)
	if err != nil {
		return err
	}
	if !os.SameFile(e.info, info) || sha256.Sum256(current) != e.digest {
		return fmt.Errorf("conversation changed during disconnect; close other ChatGPT or Codex sessions and retry: %s", e.path)
	}
	return nil
}

func backupCodexAppHistory(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	temporary, err := stageCodexAppHistory(path, data, 0o600)
	if err != nil {
		return err
	}
	defer os.Remove(temporary)
	err = os.Link(temporary, path)
	if errors.Is(err, os.ErrExist) {
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return errors.New("conversation backup is not a regular file")
		}
		existing, err := os.ReadFile(path)
		if err == nil && !bytes.Equal(existing, data) {
			return errors.New("existing conversation backup does not match")
		}
		return err
	}
	return err
}

func stageCodexAppHistory(path string, data []byte, mode fs.FileMode) (string, error) {
	f, err := os.CreateTemp(filepath.Dir(path), ".ollama-history-*")
	if err != nil {
		return "", err
	}
	_, writeErr := f.Write(data)
	modeErr := f.Chmod(mode)
	syncErr := f.Sync()
	closeErr := f.Close()
	if err := errors.Join(writeErr, modeErr, syncErr, closeErr); err != nil {
		_ = os.Remove(f.Name())
		return "", err
	}
	return f.Name(), nil
}

func tryLockCodexAppHistoryFile(file *os.File) error {
	return unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB)
}

func lockCodexAppHistory(home string) (func(), error) {
	directory := filepath.Join(home, "thread-writer-locks")
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return nil, err
	}
	info, err := os.Lstat(directory)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, errors.New("unsupported ChatGPT history lock directory")
	}
	lockPath := filepath.Join(directory, ".coordination.lock")
	if info, err := os.Lstat(lockPath); err == nil && !info.Mode().IsRegular() {
		return nil, errors.New("unsupported ChatGPT history coordination lock")
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	coordinator, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := tryLockCodexAppHistoryFile(coordinator); err != nil {
		coordinator.Close()
		return nil, errors.New("ChatGPT history is busy; close other ChatGPT or Codex sessions and retry")
	}
	entries, err := os.ReadDir(directory)
	if err == nil {
		for _, entry := range entries {
			if entry.Name() == ".coordination.lock" || !strings.HasSuffix(entry.Name(), ".lock") {
				continue
			}
			if !entry.Type().IsRegular() {
				err = errors.New("unsupported history writer lock")
				break
			}
			var f *os.File
			f, err = os.OpenFile(filepath.Join(directory, entry.Name()), os.O_RDWR, 0)
			if err == nil {
				err = tryLockCodexAppHistoryFile(f)
				f.Close()
			}
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		coordinator.Close()
		return nil, errors.New("ChatGPT history is busy; close other ChatGPT or Codex sessions and retry")
	}
	return func() { coordinator.Close() }, nil
}

func portableCodexAppHistory(data []byte) ([]byte, bool, error) {
	return rewriteCodexAppHistory(data, true)
}

func rewriteCodexAppHistory(data []byte, preserveOffsets bool) ([]byte, bool, error) {
	if !bytes.Contains(data, []byte("ollama_compaction")) && !bytes.Contains(data, []byte("encrypted_content")) {
		return data, false, nil
	}
	var output bytes.Buffer
	changed := false
	var validationErr error
	for line := range bytes.SplitAfterSeq(data, []byte{'\n'}) {
		body := bytes.TrimSuffix(line, []byte{'\n'})
		if len(bytes.TrimSpace(body)) == 0 {
			output.Write(line)
			continue
		}
		var record map[string]json.RawMessage
		if err := json.Unmarshal(body, &record); err != nil {
			validationErr = errors.New("invalid saved conversation JSON")
			output.Write(line)
			continue
		}
		var kind string
		if err := json.Unmarshal(record["type"], &kind); err != nil {
			validationErr = errors.New("unsupported saved conversation record")
			output.Write(line)
			continue
		}
		var replacement json.RawMessage
		var err error
		switch kind {
		case "response_item":
			var items []json.RawMessage
			items, err = portableCodexAppHistoryItem(record["payload"])
			if len(items) > 1 {
				err = errors.New("cannot expand a standalone compaction record without changing history positions")
			} else if len(items) == 1 {
				replacement = items[0]
			}
		case "compacted":
			replacement, err = portableCodexAppCheckpoint(record["payload"])
		}
		if err != nil {
			return nil, false, err
		}
		if replacement == nil {
			output.Write(line)
			continue
		}
		record["payload"] = replacement
		encoded, err := marshalCodexAppHistory(record)
		if err != nil {
			return nil, false, err
		}
		// In-place conversion preserves fork offsets by keeping record lengths.
		// The replacement-file migration can grow records and remaps the offsets.
		if preserveOffsets && len(encoded) > len(body) {
			return nil, false, errCodexAppHistoryGrowth
		}
		output.Write(encoded)
		if len(encoded) < len(body) {
			output.Write(bytes.Repeat([]byte{' '}, len(body)-len(encoded)))
		}
		output.Write(line[len(body):])
		changed = true
	}
	if changed {
		// Do not rewrite a partially readable conversation.
		if validationErr != nil {
			return nil, false, validationErr
		}
		first, _, _ := bytes.Cut(data, []byte{'\n'})
		var header struct {
			Type    string `json:"type"`
			Payload struct {
				HistoryMode string `json:"history_mode"`
			} `json:"payload"`
		}
		if json.Unmarshal(first, &header) != nil || header.Type != "session_meta" {
			return nil, false, errors.New("unsupported saved conversation header")
		}
		switch header.Payload.HistoryMode {
		case "", "legacy", "paginated":
		default:
			return nil, false, errors.New("unsupported saved conversation history mode")
		}
	}
	return output.Bytes(), changed, nil
}

func portableCodexAppCheckpoint(raw json.RawMessage) (json.RawMessage, error) {
	var checkpoint map[string]json.RawMessage
	if err := json.Unmarshal(raw, &checkpoint); err != nil {
		return nil, err
	}
	changed := false
	for _, name := range []string{"replacement_history", "guardian_history"} {
		if len(checkpoint[name]) == 0 || bytes.Equal(checkpoint[name], []byte("null")) {
			continue
		}
		var items, metadata []json.RawMessage
		if err := json.Unmarshal(checkpoint[name], &items); err != nil {
			return nil, fmt.Errorf("unsupported %s format", name)
		}
		hasMetadata := name == "replacement_history" && len(checkpoint["replacement_history_metadata"]) > 0 && string(checkpoint["replacement_history_metadata"]) != "null"
		if hasMetadata {
			if err := json.Unmarshal(checkpoint["replacement_history_metadata"], &metadata); err != nil || len(metadata) != len(items) {
				return nil, errors.New("invalid compaction history metadata")
			}
		}
		var result, resultMetadata []json.RawMessage
		arrayChanged := false
		for i, item := range items {
			expanded, err := portableCodexAppHistoryItem(item)
			if err != nil {
				return nil, err
			}
			if expanded == nil {
				expanded = []json.RawMessage{item}
			} else {
				arrayChanged = true
			}
			result = append(result, expanded...)
			if hasMetadata {
				for range expanded {
					resultMetadata = append(resultMetadata, metadata[i])
				}
			}
		}
		if arrayChanged {
			changed = true
			checkpoint[name], _ = marshalCodexAppHistory(result)
			if hasMetadata {
				checkpoint["replacement_history_metadata"], _ = marshalCodexAppHistory(resultMetadata)
			}
		}
	}
	if !changed {
		return nil, nil
	}
	return marshalCodexAppHistory(checkpoint)
}

func portableCodexAppHistoryItem(raw json.RawMessage) ([]json.RawMessage, error) {
	var item map[string]json.RawMessage
	if err := json.Unmarshal(raw, &item); err != nil {
		return nil, err
	}
	var kind, id, encrypted string
	_ = json.Unmarshal(item["type"], &kind)
	_ = json.Unmarshal(item["id"], &id)
	_ = json.Unmarshal(item["encrypted_content"], &encrypted)
	missingID := len(item["id"]) == 0 || bytes.Equal(item["id"], []byte("null"))
	if kind == "reasoning" && (codexAppOllamaReasoningID(id) || missingID) && encrypted != "" {
		var summary []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(item["summary"], &summary); err != nil {
			return nil, nil //nolint:nilerr // Leave unrecognized reasoning unchanged.
		}
		var text strings.Builder
		for _, part := range summary {
			if part.Type != "summary_text" {
				return nil, nil
			}
			text.WriteString(part.Text)
		}
		if text.String() != encrypted {
			return nil, nil
		}
		if missingID {
			// Older clients omitted reasoning IDs on disk. Matching plaintext is
			// Ollama's wire shape, but does not prove which provider wrote it.
			// Keep the original and stop before restoring an incompatible connection.
			return nil, errors.New("cannot safely convert saved reasoning without an ID: its provider is unknown")
		}
		delete(item, "encrypted_content")
		encoded, err := marshalCodexAppHistory(item)
		return []json.RawMessage{encoded}, err
	}
	if kind != "compaction" && kind != "context_compaction" {
		return nil, nil
	}
	var marker struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(encrypted), &marker); err != nil {
		if strings.Contains(encrypted, openai.OllamaCompactionPayloadType) {
			return nil, errors.New("invalid Ollama compaction payload")
		}
		return nil, nil
	}
	if marker.Type != openai.OllamaCompactionPayloadType {
		return nil, nil
	}
	// Use Ollama's existing versioned decoder and retained-tool conversion.
	request, _ := marshalCodexAppHistory(map[string]any{"model": "history", "input": []any{
		map[string]any{"type": "compaction", "encrypted_content": encrypted},
	}})
	expanded, _, err := openai.ExpandResponsesCompactionInput(request)
	if err != nil {
		return nil, errors.New("unsupported Ollama compaction payload; conversation was not changed")
	}
	var response struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(expanded, &response); err != nil || len(response.Input) < 2 {
		return nil, errors.New("invalid expanded Ollama compaction")
	}
	var summary struct {
		Summary string `json:"summary"`
	}
	if err := json.Unmarshal([]byte(encrypted), &summary); err != nil {
		return nil, err
	}
	message, _ := marshalCodexAppHistory(map[string]any{"type": "message", "role": "assistant", "content": []any{
		map[string]string{"type": "output_text", "text": summary.Summary},
	}})
	result := []json.RawMessage{message}
	for _, raw := range response.Input[2:] {
		var retained map[string]json.RawMessage
		if err := json.Unmarshal(raw, &retained); err != nil {
			return nil, err
		}
		if string(retained["type"]) == `"message"` {
			var text string
			if json.Unmarshal(retained["content"], &text) == nil {
				contentType := "input_text"
				if string(retained["role"]) == `"assistant"` {
					contentType = "output_text"
				}
				retained["content"], _ = marshalCodexAppHistory([]any{map[string]string{"type": contentType, "text": text}})
				raw, err = marshalCodexAppHistory(retained)
				if err != nil {
					return nil, err
				}
			}
		}
		result = append(result, raw)
	}
	if metadata := item["guardian_metadata"]; len(metadata) > 0 {
		for i, raw := range result {
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				return nil, err
			}
			fields["guardian_metadata"] = metadata
			result[i], err = marshalCodexAppHistory(fields)
			if err != nil {
				return nil, err
			}
		}
	}
	return result, nil
}

func codexAppOllamaReasoningID(id string) bool {
	suffix, ok := strings.CutPrefix(strings.TrimSpace(id), "rs_")
	if !ok {
		return false
	}
	suffix = strings.TrimPrefix(suffix, "resp_")
	if len(suffix) == 0 || len(suffix) > 6 {
		return false
	}
	for _, char := range suffix {
		if char < '0' || char > '9' {
			return false
		}
	}
	return true
}

func marshalCodexAppHistory(value any) (json.RawMessage, error) {
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(output.Bytes(), []byte{'\n'}), nil
}
