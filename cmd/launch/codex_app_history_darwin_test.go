//go:build darwin

package launch

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func historyJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	data, err := marshalCodexAppHistory(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func historyFixture(t *testing.T, retained bool) []byte {
	t.Helper()
	payload := map[string]any{"type": "ollama_compaction", "version": 1, "summary": "The project code is amber-42.", "retained": []any{}}
	if retained {
		payload["retained"] = []any{
			map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
				"id": "call_note", "function": map[string]any{"name": "read_note", "arguments": map[string]any{}},
			}}},
			map[string]any{"role": "tool", "tool_call_id": "call_note", "content": "The note is tulip-73."},
		}
	}
	compaction := map[string]any{"type": "compaction", "id": "cmp_ollama", "encrypted_content": string(historyJSON(t, payload))}
	reasoning := map[string]any{"type": "reasoning", "id": "rs_resp_42", "summary": []any{
		map[string]any{"type": "summary_text", "text": "Remember the note."},
	}, "encrypted_content": "Remember the note."}
	items := []any{
		map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "Keep my project code."}}},
		compaction,
	}
	rows := []any{
		map[string]any{"type": "session_meta", "payload": map[string]any{"id": "same-thread", "history_mode": "paginated"}},
		map[string]any{"type": "response_item", "ordinal": 1, "payload": reasoning},
		map[string]any{"type": "compacted", "ordinal": 2, "payload": map[string]any{
			"message": "", "replacement_history": items, "guardian_history": []any{reasoning},
			"replacement_history_metadata": []any{map[string]bool{"client_authored": true}, map[string]bool{"client_authored": false}},
			"window_id":                    "original-window", "retained_context": map[string]any{"keep": "unchanged"},
		}},
		map[string]any{"type": "response_item", "ordinal": 3, "payload": map[string]any{
			"type": "message", "role": "assistant", "content": []any{map[string]string{"type": "output_text", "text": "Original answer."}},
		}},
	}
	var data []byte
	for _, row := range rows {
		data = append(data, historyJSON(t, row)...)
		data = append(data, '\n')
	}
	return data
}

func writeHistoryFixture(t *testing.T, home, relative string, data []byte) string {
	t.Helper()
	path := filepath.Join(home, relative)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCodexAppPortableHistory(t *testing.T) {
	original := historyFixture(t, true)
	converted, changed, err := portableCodexAppHistory(original)
	if err != nil || !changed {
		t.Fatalf("convert: changed=%v, err=%v", changed, err)
	}
	before, after := bytes.Split(original, []byte{'\n'}), bytes.Split(converted, []byte{'\n'})
	if len(before) != len(after) {
		t.Fatal("record count changed")
	}
	for i := range before {
		if len(before[i]) != len(after[i]) {
			t.Fatalf("record %d changed its byte length", i)
		}
	}
	for _, index := range []int{0, 3} {
		if !bytes.Equal(before[index], after[index]) {
			t.Fatalf("ordinary record %d changed", index)
		}
	}
	if bytes.Contains(converted, []byte("encrypted_content")) || bytes.Contains(converted, []byte("ollama_compaction")) {
		t.Fatal("incompatible state remains")
	}
	var checkpoint struct {
		Payload struct {
			Items           []map[string]json.RawMessage `json:"replacement_history"`
			Metadata        []json.RawMessage            `json:"replacement_history_metadata"`
			GuardianHistory []map[string]json.RawMessage `json:"guardian_history"`
			RetainedContext json.RawMessage              `json:"retained_context"`
			WindowID        string                       `json:"window_id"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(after[2], &checkpoint); err != nil {
		t.Fatal(err)
	}
	items := checkpoint.Payload.Items
	if len(items) != 4 || len(checkpoint.Payload.Metadata) != len(items) || checkpoint.Payload.WindowID != "original-window" {
		t.Fatalf("invalid checkpoint: %+v", checkpoint)
	}
	if !bytes.Contains(items[1]["content"], []byte("amber-42")) || string(items[2]["call_id"]) != `"call_note"` || string(items[3]["call_id"]) != `"call_note"` || !bytes.Contains(items[3]["output"], []byte("tulip-73")) {
		t.Fatalf("summary or retained tool pair lost: %s", after[2])
	}
	wantMetadata := `[{"client_authored":true},{"client_authored":false},{"client_authored":false},{"client_authored":false}]`
	if got := string(historyJSON(t, checkpoint.Payload.Metadata)); got != wantMetadata {
		t.Fatalf("checkpoint metadata = %s, want %s", got, wantMetadata)
	}
	if string(checkpoint.Payload.RetainedContext) != `{"keep":"unchanged"}` {
		t.Fatalf("retained context changed: %s", checkpoint.Payload.RetainedContext)
	}
	var reasoning struct {
		Payload map[string]json.RawMessage `json:"payload"`
	}
	if err := json.Unmarshal(after[1], &reasoning); err != nil {
		t.Fatal(err)
	}
	if len(checkpoint.Payload.GuardianHistory) != 1 {
		t.Fatalf("guardian history changed: %v", checkpoint.Payload.GuardianHistory)
	}
	wantReasoning := `{"id":"rs_resp_42","summary":[{"text":"Remember the note.","type":"summary_text"}],"type":"reasoning"}`
	for _, item := range []map[string]json.RawMessage{reasoning.Payload, checkpoint.Payload.GuardianHistory[0]} {
		if got := string(historyJSON(t, item)); got != wantReasoning {
			t.Fatalf("readable thinking = %s, want %s", got, wantReasoning)
		}
	}
	again, changed, err := portableCodexAppHistory(converted)
	if err != nil || changed || !bytes.Equal(again, converted) {
		t.Fatalf("second cleanup changed history: %v, %v", changed, err)
	}
}

func TestCodexAppPortableHistoryLeavesOtherProvidersUntouched(t *testing.T) {
	for _, item := range []string{
		`{"type":"reasoning","id":"rs_native_openai","encrypted_content":"opaque","summary":[]}`,
		`{"type":"reasoning","id":"rs_42","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"A readable summary, with unrelated encrypted state."}]}`,
		`{"type":"reasoning","id":"rs_42","encrypted_content":"opaque","summary":"unrecognized format"}`,
		`{"type":"compaction","encrypted_content":"opaque"}`,
		`{"type":"compaction","encrypted_content":"{\"type\":\"another_provider\",\"version\":1}"}`,
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"ollama_compaction rs_42 are literal text"}]}`,
	} {
		original := []byte(`{"type":"response_item","payload":` + item + "}\n")
		got, changed, err := portableCodexAppHistory(original)
		if err != nil || changed || !bytes.Equal(got, original) {
			t.Fatalf("unrelated history changed: %s, %v", original, err)
		}
	}
}

func TestCodexAppHistoryDisconnectIgnoresUnrelatedArchive(t *testing.T) {
	setTestHome(t, t.TempDir())
	home := t.TempDir()
	ollama := writeHistoryFixture(t, home, "sessions/rollout-ollama.jsonl", historyFixture(t, false))
	original := []byte("{\"type\":\"session_meta\",\"payload\":{\"history_mode\":\"legacy\"}}\n" +
		"{\"type\":\"response_item\",\"payload\":{\"type\":\"reasoning\",\"id\":\"rs_native_openai\",\"summary\":[],\"encrypted_content\":\"opaque\"}}\n" +
		"{\"type\":")
	archive := writeHistoryFixture(t, home, "archived_sessions/rollout-openai.jsonl", original)
	restored := false
	if err := prepareCodexAppHistoryForDisconnect(home, func() error { restored = true; return nil }); err != nil {
		t.Fatal(err)
	}
	if !restored {
		t.Fatal("connection was not restored")
	}
	if got, err := os.ReadFile(archive); err != nil || !bytes.Equal(got, original) {
		t.Fatalf("unrelated archive changed: %v", err)
	}
	if got, err := os.ReadFile(ollama); err != nil || bytes.Contains(got, []byte("ollama_compaction")) {
		t.Fatalf("Ollama history was not converted: %v", err)
	}
}

func TestCodexAppHistoryBackupAndRetry(t *testing.T) {
	setTestHome(t, t.TempDir())
	home := t.TempDir()
	original := historyFixture(t, true)
	for _, compressed := range []bool{false, true} {
		relative := filepath.Join("sessions", "2026", "rollout-active.jsonl")
		data := original
		if compressed {
			relative = filepath.Join("archived_sessions", "rollout-archived.jsonl.zst")
			encoder, err := zstd.NewWriter(nil)
			if err != nil {
				t.Fatal(err)
			}
			data = encoder.EncodeAll(original, nil)
			encoder.Close()
		}
		path := writeHistoryFixture(t, home, relative, data)
		if err := prepareCodexAppHistoryForDisconnect(home, func() error { return nil }); err != nil {
			t.Fatal(err)
		}
		backup := filepath.Join(fileutil.BackupDir(), "chatgpt-history", relative+fmt.Sprintf(".%x.bak", sha256.Sum256(data)))
		if saved, err := os.ReadFile(backup); err != nil || !bytes.Equal(saved, data) {
			t.Fatalf("original backup missing: %v", err)
		}
		info, err := os.Stat(backup)
		if err != nil || info.Mode().Perm() != 0o600 {
			t.Fatalf("backup permissions: %v, %v", info, err)
		}
		cleaned, err := os.ReadFile(path)
		if err != nil || bytes.Equal(cleaned, data) {
			t.Fatalf("history was not converted: %v", err)
		}
		if compressed {
			decoder, err := zstd.NewReader(nil)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := decoder.DecodeAll(cleaned, nil)
			decoder.Close()
			if err != nil || len(decoded) != len(original) || bytes.Contains(decoded, []byte("ollama_compaction")) {
				t.Fatalf("compressed history is invalid: %v", err)
			}
		}
		if err := prepareCodexAppHistoryForDisconnect(home, func() error { return nil }); err != nil {
			t.Fatal(err)
		}
		if got, _ := os.ReadFile(path); !bytes.Equal(got, cleaned) {
			t.Fatal("retry rewrote history")
		}
	}
}

func TestCodexAppHistoryRejectsUnsupportedStateBeforePublishing(t *testing.T) {
	for name, invalid := range map[string][]byte{
		"future version": bytes.ReplaceAll(historyFixture(t, false), []byte(`\"version\":1`), []byte(`\"version\":2`)),
		"truncated JSON": append(historyFixture(t, false), []byte(`{"type":`)...),
		"future storage": bytes.ReplaceAll(historyFixture(t, false), []byte(`"history_mode":"paginated"`), []byte(`"history_mode":"future"`)),
	} {
		t.Run(name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			home := t.TempDir()
			original := historyFixture(t, false)
			good := writeHistoryFixture(t, home, "sessions/rollout-a.jsonl", original)
			bad := writeHistoryFixture(t, home, "sessions/rollout-b.jsonl", invalid)
			if err := prepareCodexAppHistoryForDisconnect(home, func() error { return nil }); err == nil {
				t.Fatal("expected unsupported history error")
			}
			for path, want := range map[string][]byte{good: original, bad: invalid} {
				if got, _ := os.ReadFile(path); !bytes.Equal(got, want) {
					t.Fatalf("modified %s after validation failed", path)
				}
			}
		})
	}
}

func TestCodexAppHistoryBackupFailureAndRestoreRetry(t *testing.T) {
	setTestHome(t, t.TempDir())
	home := t.TempDir()
	original := historyFixture(t, true)
	path := writeHistoryFixture(t, home, "sessions/rollout-test.jsonl", original)
	backupRoot := filepath.Join(fileutil.BackupDir(), "chatgpt-history")
	writeHistoryFixture(t, filepath.Dir(backupRoot), filepath.Base(backupRoot), []byte("blocked"))
	restoreCalls := 0
	restore := func() error { restoreCalls++; return errors.New("restore failed") }
	if err := prepareCodexAppHistoryForDisconnect(home, restore); err == nil || restoreCalls != 0 {
		t.Fatalf("restored connection despite failed backup: %v", err)
	}
	if got, _ := os.ReadFile(path); !bytes.Equal(got, original) {
		t.Fatal("changed history without a backup")
	}
	if err := os.Remove(backupRoot); err != nil {
		t.Fatal(err)
	}
	if err := prepareCodexAppHistoryForDisconnect(home, restore); err == nil || restoreCalls != 1 {
		t.Fatalf("expected connection restore failure: %v", err)
	}
	converted, _ := os.ReadFile(path)
	if bytes.Contains(converted, []byte("ollama_compaction")) {
		t.Fatal("cleanup was not completed before restoring the connection")
	}
	if err := prepareCodexAppHistoryForDisconnect(home, func() error { return nil }); err != nil {
		t.Fatalf("could not retry after connection restore failure: %v", err)
	}
	if got, _ := os.ReadFile(path); !bytes.Equal(got, converted) {
		t.Fatal("retry changed converted history")
	}
}

func TestCodexAppHistoryKeepsGuardianMetadata(t *testing.T) {
	item := json.RawMessage(`{"type":"compaction","guardian_metadata":{"client_authored":false,"origin_turn_id":"original-turn"},"encrypted_content":"{\"type\":\"ollama_compaction\",\"version\":1,\"summary\":\"Keep the note.\",\"retained\":[{\"role\":\"user\",\"content\":\"The note is tulip-73.\"}]}"}`)
	items, err := portableCodexAppHistoryItem(item)
	if err != nil || len(items) != 2 {
		t.Fatalf("convert guardian checkpoint: %v", err)
	}
	for _, raw := range items {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatal(err)
		}
		if string(fields["guardian_metadata"]) != `{"client_authored":false,"origin_turn_id":"original-turn"}` {
			t.Fatalf("guardian provenance changed: %s", raw)
		}
	}
}

func TestCodexAppHistoryRefusesActiveWriterAndConcurrentChange(t *testing.T) {
	setTestHome(t, t.TempDir())
	home := t.TempDir()
	writeHistoryFixture(t, home, "thread-writer-locks/thread.lock", nil)
	writer, err := os.OpenFile(filepath.Join(home, "thread-writer-locks/thread.lock"), os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	if err := tryLockCodexAppHistoryFile(writer); err != nil {
		t.Fatal(err)
	}
	if err := prepareCodexAppHistoryForDisconnect(home, func() error { return nil }); err == nil || !strings.Contains(err.Error(), "busy") {
		t.Fatalf("expected active writer rejection: %v", err)
	}
	path := writeHistoryFixture(t, home, "sessions/rollout-test.jsonl", []byte("original"))
	info, _ := os.Stat(path)
	staged, err := stageCodexAppHistory(path, []byte("converted"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(staged)
	edit := codexAppHistoryEdit{path: path, temporary: staged, info: info, digest: sha256.Sum256([]byte("original"))}
	if err := os.WriteFile(path, []byte("concurrent change"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := edit.publish(); err == nil {
		t.Fatal("overwrote concurrent change")
	}
}

func TestCodexAppHistoryRefusesLinkedConversations(t *testing.T) {
	setTestHome(t, t.TempDir())
	home := t.TempDir()
	original := historyFixture(t, false)
	target := writeHistoryFixture(t, t.TempDir(), "rollout-outside.jsonl", original)
	if err := os.Mkdir(filepath.Join(home, "sessions"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(home, "sessions", "rollout-linked.jsonl")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	err := prepareCodexAppHistoryForDisconnect(home, func() error { t.Fatal("restored connection after unsafe cleanup"); return nil })
	if err == nil {
		t.Fatal("followed a conversation symlink")
	}
	if got, _ := os.ReadFile(target); !bytes.Equal(got, original) {
		t.Fatal("changed conversation outside the scanned directory")
	}
}

func TestCodexAppHistoryRejectsOffsetChangingExpansion(t *testing.T) {
	var retained []any
	for range 20 {
		retained = append(retained, map[string]string{"role": "user", "content": "short"})
	}
	payload := historyJSON(t, map[string]any{"type": "ollama_compaction", "version": 1, "summary": "Summary.", "retained": retained})
	checkpoint := historyJSON(t, map[string]any{"type": "compacted", "payload": map[string]any{
		"replacement_history": []any{map[string]string{"type": "compaction", "encrypted_content": string(payload)}},
	}})
	data := append([]byte("{\"type\":\"session_meta\",\"payload\":{}}\n"), checkpoint...)
	if _, _, err := portableCodexAppHistory(data); err == nil || !strings.Contains(err.Error(), "offsets") {
		t.Fatalf("expected offset protection, got %v", err)
	}
}

func TestCodexAppDisconnectPreparesHistoryBeforeRestoringConnection(t *testing.T) {
	for _, invalid := range []bool{false, true} {
		t.Run(fmt.Sprint("invalid=", invalid), func(t *testing.T) {
			root := t.TempDir()
			setTestHome(t, root)
			withCodexAppPlatform(t, "darwin")
			home := filepath.Join(root, ".codex")
			configPath := writeHistoryFixture(t, home, "config.toml", []byte("model = \"gpt-5.6-sol\"\n"))
			app := &CodexApp{}
			if err := app.ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
				t.Fatal(err)
			}
			connected, _ := os.ReadFile(configPath)
			original := historyFixture(t, false)
			if invalid {
				original = bytes.ReplaceAll(original, []byte(`\"version\":1`), []byte(`\"version\":2`))
			}
			path := writeHistoryFixture(t, home, "sessions/rollout-test.jsonl", original)
			running, quit, reopened := true, false, false
			withCodexAppProcessHooks(t, func() bool { return running }, func() error {
				if got, _ := os.ReadFile(path); !bytes.Equal(got, original) {
					t.Fatal("history changed before the app stopped")
				}
				running, quit = false, true
				return nil
			}, func() error {
				reopened = true
				return nil
			})
			if err := app.RestoreFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
				t.Fatalf("unexpected cancelled disconnect: %v", err)
			}
			err := app.RestoreFromDesktop(true)
			if !quit || !reopened {
				t.Fatal("app lifecycle was not preserved")
			}
			config, _ := os.ReadFile(configPath)
			history, _ := os.ReadFile(path)
			if invalid {
				if err == nil || !bytes.Equal(config, connected) || !bytes.Equal(history, original) {
					t.Fatalf("failed cleanup changed connection or history: %v", err)
				}
			} else if err != nil || bytes.Contains(config, []byte("openai_base_url")) || bytes.Contains(history, []byte("ollama_compaction")) {
				t.Fatalf("disconnect failed: %v", err)
			}
		})
	}
}
