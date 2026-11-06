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

func TestCodexAppHistoryLegacyReasoningWithoutID(t *testing.T) {
	for _, id := range []string{"", `,"id":null`} {
		item := json.RawMessage(`{"type":"reasoning","summary":[{"type":"summary_text","text":"Remember the note."}],"encrypted_content":"Remember the note."` + id + `}`)
		for _, location := range []string{"response_item", "replacement_history", "guardian_history"} {
			t.Run(location+id, func(t *testing.T) {
				row := map[string]any{"type": "response_item", "payload": item}
				if location != "response_item" {
					row = map[string]any{"type": "compacted", "payload": map[string]any{location: []any{item}}}
				}
				original := append([]byte("{\"type\":\"session_meta\",\"payload\":{\"id\":\"legacy-thread\",\"history_mode\":\"legacy\"}}\n"), historyJSON(t, row)...)
				untouched := bytes.Clone(original)
				got, changed, err := portableCodexAppHistory(original)
				if err == nil || !strings.Contains(err.Error(), "provider is unknown") || changed || got != nil {
					t.Fatalf("ambiguous reasoning was accepted: changed=%v, err=%v", changed, err)
				}
				if !bytes.Equal(original, untouched) {
					t.Fatal("modified ambiguous history")
				}
			})
		}
	}
}

func TestCodexAppHistoryLegacyReasoningPreservesOtherState(t *testing.T) {
	for _, item := range []string{
		`{"type":"reasoning","summary":[{"type":"summary_text","text":"Native summary"}],"encrypted_content":"opaque-native-state"}`,
		`{"type":"reasoning","summary":[],"encrypted_content":"opaque-native-state"}`,
		`{"type":"reasoning","summary":[{"type":"summary_text","text":"same"}],"encrypted_content":"different"}`,
		`{"type":"reasoning","summary":"unrecognized","encrypted_content":"unrecognized"}`,
		`{"type":"reasoning","summary":[{"type":"text","text":"same"}],"encrypted_content":"same"}`,
		`{"type":"reasoning","id":"rs_native_openai","summary":[{"type":"summary_text","text":"same"}],"encrypted_content":"same"}`,
		`{"type":"reasoning","id":"another_provider","summary":[{"type":"summary_text","text":"same"}],"encrypted_content":"same"}`,
		`{"type":"reasoning","id":42,"summary":[{"type":"summary_text","text":"same"}],"encrypted_content":"same"}`,
		`{"type":"reasoning","summary":[],"encrypted_content":""}`,
		`{"type":"reasoning","summary":[{"type":"summary_text","text":"same"}]}`,
		`{"type":"message","role":"user","content":"encrypted_content is literal text"}`,
	} {
		original := []byte(`{"type":"response_item","payload":` + item + "}\n")
		got, changed, err := portableCodexAppHistory(original)
		if err != nil || changed || !bytes.Equal(got, original) {
			t.Fatalf("unrelated history changed: %s; changed=%v, err=%v", item, changed, err)
		}
	}
}

func expandedHistoryFixture(t *testing.T) []byte {
	t.Helper()
	payload := historyJSON(t, map[string]any{
		"type": "ollama_compaction", "version": 1, "summary": "Summary.",
		"retained": []any{
			map[string]string{"role": "user", "content": "a"},
			map[string]string{"role": "user", "content": "b"},
		},
	})
	checkpoint := historyJSON(t, map[string]any{"replacement_history": []any{
		map[string]string{"type": "compaction", "encrypted_content": string(payload)},
	}})
	expanded, err := portableCodexAppCheckpoint(checkpoint)
	if err != nil || len(expanded) <= len(checkpoint) {
		t.Fatalf("expected a valid, growing checkpoint: %d -> %d bytes, err=%v", len(checkpoint), len(expanded), err)
	}
	t.Logf("checkpoint payload grows from %d to %d bytes with two retained one-character user messages", len(checkpoint), len(expanded))
	row := historyJSON(t, map[string]any{"type": "compacted", "payload": checkpoint})
	data := append([]byte("{\"type\":\"session_meta\",\"payload\":{\"id\":\"growth-thread\",\"history_mode\":\"paginated\"}}\n"), row...)
	return append(data, '\n')
}

func TestCodexAppDisconnectWithExpandedHistory(t *testing.T) {
	for _, test := range []struct {
		name            string
		relative        string
		legacyReasoning bool
	}{
		{"active expansion", "sessions/rollout-z-growth.jsonl", false},
		{"archived expansion", "archived_sessions/rollout-z-growth.jsonl", false},
		{"compressed archived expansion", "archived_sessions/rollout-z-growth.jsonl.zst", false},
		{"legacy reasoning", "sessions/rollout-z-legacy.jsonl", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			setTestHome(t, root)
			withCodexAppPlatform(t, "darwin")
			home := filepath.Join(root, ".codex")
			configPath := writeHistoryFixture(t, home, "config.toml", []byte("model = \"gpt-5.6-sol\"\n"))
			app := &CodexApp{}
			if err := app.ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
				t.Fatal(err)
			}
			connected, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			var original []byte
			wantError := "saved history offsets"
			if test.legacyReasoning {
				original = []byte("{\"type\":\"session_meta\",\"payload\":{\"history_mode\":\"legacy\"}}\n" +
					"{\"type\":\"response_item\",\"payload\":{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"same\"}],\"encrypted_content\":\"same\"}}\n")
				wantError = "provider is unknown"
			} else {
				original = expandedHistoryFixture(t)
			}
			if strings.HasSuffix(test.relative, ".zst") {
				encoder, err := zstd.NewWriter(nil)
				if err != nil {
					t.Fatal(err)
				}
				original = encoder.EncodeAll(original, nil)
				encoder.Close()
			}
			path := writeHistoryFixture(t, home, test.relative, original)
			peerOriginal := historyFixture(t, true)
			peer := writeHistoryFixture(t, home, "sessions/rollout-a-convertible.jsonl", peerOriginal)
			auth := []byte(`{"OPENAI_API_KEY":"synthetic-test-key"}`)
			authPath := writeHistoryFixture(t, home, "auth.json", auth)
			statePath := codexAppRestoreStatePath()
			state, err := os.ReadFile(statePath)
			if err != nil {
				t.Fatal(err)
			}
			running, quit, reopened := true, 0, 0
			withCodexAppProcessHooks(t, func() bool { return running }, func() error {
				running = false
				quit++
				return nil
			}, func() error {
				running = true
				reopened++
				return nil
			})
			assertUnchanged := func() {
				t.Helper()
				for file, want := range map[string][]byte{configPath: connected, path: original, peer: peerOriginal, authPath: auth, statePath: state} {
					if got, err := os.ReadFile(file); err != nil || !bytes.Equal(got, want) {
						t.Fatalf("unsafe cleanup changed %s: %v", file, err)
					}
				}
			}
			if err := app.RestoreFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
				t.Fatalf("cancelled disconnect: %v", err)
			}
			assertUnchanged()
			if quit != 0 || reopened != 0 {
				t.Fatal("cancellation changed the app lifecycle")
			}
			for attempt := 1; attempt <= 2; attempt++ {
				err := app.RestoreFromDesktop(true)
				if err == nil || !strings.Contains(err.Error(), wantError) {
					t.Fatalf("attempt %d: expected cleanup rejection, got %v", attempt, err)
				}
				assertUnchanged()
				if quit != attempt || reopened != attempt {
					t.Fatal("failed disconnect did not reopen the existing connection")
				}
				backup := filepath.Join(fileutil.BackupDir(), "chatgpt-history", "sessions", fmt.Sprintf("rollout-a-convertible.jsonl.%x.bak", sha256.Sum256(peerOriginal)))
				if saved, err := os.ReadFile(backup); err != nil || !bytes.Equal(saved, peerOriginal) {
					t.Fatalf("staged peer's backup was not preserved: %v", err)
				}
				for _, directory := range []string{"sessions", "archived_sessions"} {
					staged, err := filepath.Glob(filepath.Join(home, directory, ".ollama-history-*"))
					if err != nil || len(staged) != 0 {
						t.Fatalf("temporary edits remain: %v, %v", staged, err)
					}
				}
				t.Logf("attempt %d blocked at the desktop Disconnect boundary; configuration and both conversations unchanged", attempt)
			}
		})
	}
}
