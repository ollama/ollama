//go:build darwin

package launch

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
)

const historyMigrationSchema = `
CREATE TABLE _sqlx_migrations (version INTEGER PRIMARY KEY, success BOOLEAN);
INSERT INTO _sqlx_migrations VALUES (6, 1);
CREATE TABLE thread_history_projection_state (thread_id TEXT PRIMARY KEY, next_rollout_byte_offset INTEGER, next_rollout_ordinal INTEGER);
CREATE TABLE thread_turns (thread_id TEXT, turn_id TEXT, rollout_ordinal INTEGER, status TEXT, error_json TEXT, started_at INTEGER, completed_at INTEGER, duration_ms INTEGER, first_user_item_id TEXT, final_agent_item_id TEXT, rollout_byte_offset INTEGER, rollout_end_ordinal INTEGER, rollout_end_byte_offset INTEGER, PRIMARY KEY(thread_id, turn_id));
CREATE TABLE thread_items (thread_id TEXT, turn_id TEXT, item_id TEXT, rollout_ordinal INTEGER, created_at_ms INTEGER, item_json TEXT, item_type TEXT, updated_at_ordinal INTEGER, PRIMARY KEY(thread_id, turn_id, item_id));
CREATE TABLE thread_realtime_items (thread_id TEXT, item_id TEXT, rollout_ordinal INTEGER, created_at_ms INTEGER, item_type TEXT, item_json TEXT, PRIMARY KEY(thread_id, item_id));
`

type historyMigrationFixture struct {
	home                   string
	state, history         *sql.DB
	ids, rolloutIDs, paths []string
	originals              [][]byte
}

func newHistoryMigrationFixture(t *testing.T, compressed bool) *historyMigrationFixture {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	setTestHome(t, root)
	t.Setenv("CODEX_SQLITE_HOME", "")
	oldCheck := codexAppCheckHistoryMigrationVersion
	codexAppCheckHistoryMigrationVersion = func() error { return nil }
	t.Cleanup(func() { codexAppCheckHistoryMigrationVersion = oldCheck })
	f := &historyMigrationFixture{home: filepath.Join(root, ".codex")}
	f.ids = []string{"019a1234-5678-7000-8000-000000000001", "019a1234-5678-7000-8000-000000000002", "019a1234-5678-7000-8000-000000000003"}
	// The parent already went through Revert: its storage ID differs from its
	// stable chat ID. Both an active fork and an archived fork share that storage.
	f.rolloutIDs = []string{"019a1234-5678-7000-8000-000000000004", f.ids[1], f.ids[2]}
	writeHistoryFixture(t, f.home, "config.toml", []byte("model = \"synthetic\"\n"))
	f.state = openHistoryMigrationTestDB(t, filepath.Join(f.home, "state_5.sqlite"), `CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, history_mode TEXT, archived BOOLEAN, title TEXT);`)
	f.history = openHistoryMigrationTestDB(t, filepath.Join(f.home, "thread_history_1.sqlite"), historyMigrationSchema)
	var parentEnd int64
	for i, id := range f.ids {
		payload := map[string]any{"id": id, "session_id": id, "history_mode": "paginated", "timestamp": "2026-10-14T01:02:03.000Z", "unknown_metadata": "keep"}
		start := int64(0)
		if i > 0 {
			start = 3
			payload["history_base"] = map[string]any{"thread_id": f.rolloutIDs[0], "end_ordinal_exclusive": start, "end_byte_offset": parentEnd, "unknown_position": true}
			payload["forked_from_id"] = f.ids[0]
			payload["forked_from_ordinal_exclusive"] = start
		}
		header := append(historyJSON(t, map[string]any{"type": "session_meta", "ordinal": start, "timestamp": "2026-10-14T01:02:03Z", "payload": payload}), '\n')
		data := bytes.Clone(header)
		if i == 0 {
			compaction := map[string]any{"type": "ollama_compaction", "version": 1, "summary": "Summary.", "retained": []any{map[string]string{"role": "user", "content": "a"}, map[string]string{"role": "user", "content": "b"}}}
			data = append(data, historyJSON(t, map[string]any{"type": "compacted", "ordinal": 1, "payload": map[string]any{"message": "", "replacement_history": []any{map[string]string{"type": "compaction", "encrypted_content": string(historyJSON(t, compaction))}}}})...)
			data = append(data, '\n')
		}
		endOrdinal := start + 1
		if i == 0 {
			endOrdinal = 2
		}
		data = append(data, historyJSON(t, map[string]any{"type": "event_msg", "ordinal": endOrdinal, "payload": map[string]string{"type": "task_complete", "turn_id": "turn-stays-the-same"}})...)
		data = append(data, '\n')
		if i == 0 {
			parentEnd = int64(len(data))
		}
		directory := "sessions"
		if i == 2 {
			directory = "archived_sessions"
		}
		name := "rollout-2026-10-14T01-02-03-" + id
		if f.rolloutIDs[i] != id {
			name += "_" + f.rolloutIDs[i]
		}
		name += ".jsonl"
		stored := data
		if i == 2 && compressed {
			encoder, err := zstd.NewWriter(nil)
			if err != nil {
				t.Fatal(err)
			}
			stored = encoder.EncodeAll(data, nil)
			encoder.Close()
			name += ".zst"
		}
		path := writeHistoryFixture(t, f.home, filepath.Join(directory, name), stored)
		f.paths = append(f.paths, path)
		f.originals = append(f.originals, bytes.Clone(stored))
		// Archived compressed histories may retain the logical .jsonl path in SQLite.
		selectedPath := strings.TrimSuffix(path, ".zst")
		historyMigrationExec(t, f.state, "INSERT INTO threads VALUES (?, ?, 'paginated', ?, 'Keep my title')", id, selectedPath, i == 2)
		historyMigrationExec(t, f.history, "INSERT INTO thread_history_projection_state VALUES (?, ?, ?)", f.rolloutIDs[i], len(data), endOrdinal+1)
		historyMigrationExec(t, f.history, "INSERT INTO thread_turns (thread_id, turn_id, rollout_ordinal, rollout_byte_offset, rollout_end_ordinal, rollout_end_byte_offset, status) VALUES (?, 'turn-stays-the-same', ?, ?, ?, ?, 'completed')", f.rolloutIDs[i], start+1, len(header), endOrdinal+1, len(data))
		historyMigrationExec(t, f.history, "INSERT INTO thread_items (thread_id, turn_id, item_id, rollout_ordinal, item_json) VALUES (?, 'turn-stays-the-same', 'same-item', ?, ?)", f.rolloutIDs[i], start+1, `{"type":"userMessage","id":"same-item","content":[]}`)
		historyMigrationExec(t, f.history, "INSERT INTO thread_realtime_items (thread_id, item_id, rollout_ordinal, item_json) VALUES (?, 'same-realtime-item', ?, ?)", f.rolloutIDs[i], start+1, `{"type":"realtime_session_started"}`)
	}
	return f
}

func openHistoryMigrationTestDB(t *testing.T, path, schema string) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	historyMigrationExec(t, db, schema)
	return db
}

func historyMigrationExec(t *testing.T, db *sql.DB, query string, args ...any) {
	t.Helper()
	if _, err := db.Exec(query, args...); err != nil {
		t.Fatal(err)
	}
}

func (f *historyMigrationFixture) selected(t *testing.T) []string {
	t.Helper()
	var paths []string
	for _, id := range f.ids {
		var path string
		if err := f.state.QueryRow("SELECT rollout_path FROM threads WHERE id = ?", id).Scan(&path); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	return paths
}

func (f *historyMigrationFixture) assertOriginals(t *testing.T) {
	t.Helper()
	for i, path := range f.paths {
		data, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(data, f.originals[i]) {
			t.Fatalf("original changed: %s: %v", path, err)
		}
		var count int
		if err := f.history.QueryRow("SELECT count(*) FROM thread_items WHERE thread_id = ?", f.rolloutIDs[i]).Scan(&count); err != nil || count != 1 {
			t.Fatalf("original index changed: %v", err)
		}
	}
}

func TestCodexAppHistoryMigrationPreservesGraphAndIndexes(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("compressed=%v", compressed), func(t *testing.T) {
			f := newHistoryMigrationFixture(t, compressed)
			restored := 0
			if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { restored++; return nil }); err != nil {
				t.Fatal(err)
			}
			if restored != 1 {
				t.Fatal("connection was not restored")
			}
			f.assertOriginals(t)
			paths := f.selected(t)
			byPath, _, err := readCodexAppHistoryRollouts(f.home)
			if err != nil {
				t.Fatal(err)
			}
			parent := byPath[paths[0]]
			for i, path := range paths {
				rollout := byPath[path]
				if path == f.paths[i] || rollout == nil || rollout.threadID != f.ids[i] || rollout.id == f.rolloutIDs[i] {
					t.Fatal("chat identity or selected path incorrect")
				}
				if bytes.Contains(rollout.data, []byte("ollama_compaction")) {
					t.Fatal("incompatible compaction remains")
				}
				if i > 0 && (rollout.base.ThreadID != parent.id || rollout.base.Offset != int64(len(parent.data)) || rollout.base.Ordinal != 3) {
					t.Fatal("fork boundary was not remapped")
				}
				var offset, ordinal int64
				if err := f.history.QueryRow("SELECT next_rollout_byte_offset, next_rollout_ordinal FROM thread_history_projection_state WHERE thread_id = ?", rollout.id).Scan(&offset, &ordinal); err != nil || offset != int64(len(rollout.data)) {
					t.Fatalf("projection checkpoint not remapped: %d, %v", offset, err)
				}
				first, _, _ := bytes.Cut(rollout.data, []byte{'\n'})
				var start, end int64
				if err := f.history.QueryRow("SELECT rollout_byte_offset, rollout_end_byte_offset FROM thread_turns WHERE thread_id = ? AND turn_id = 'turn-stays-the-same'", rollout.id).Scan(&start, &end); err != nil || start != int64(len(first)+1) || end != int64(len(rollout.data)) {
					t.Fatalf("turn boundaries changed incorrectly: %d %d %v", start, end, err)
				}
				var originalItem, copiedItem string
				if err := f.history.QueryRow("SELECT item_json FROM thread_items WHERE thread_id = ?", f.rolloutIDs[i]).Scan(&originalItem); err != nil {
					t.Fatal(err)
				}
				if err := f.history.QueryRow("SELECT item_json FROM thread_items WHERE thread_id = ?", rollout.id).Scan(&copiedItem); err != nil || copiedItem != originalItem {
					t.Fatal("visible saved message changed")
				}
				var title string
				var archived bool
				if err := f.state.QueryRow("SELECT title, archived FROM threads WHERE id = ?", f.ids[i]).Scan(&title, &archived); err != nil || title != "Keep my title" || archived != (i == 2) {
					t.Fatal("chat metadata changed")
				}
			}
			if int64(len(parent.data)) <= int64(len(f.originals[0])) {
				t.Fatal("fixture did not exercise history growth")
			}
			if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { restored++; return nil }); err != nil {
				t.Fatal(err)
			}
			if fmt.Sprint(f.selected(t)) != fmt.Sprint(paths) || restored != 2 {
				t.Fatal("retry replaced history again")
			}
			f.assertOriginals(t)
		})
	}
}

func TestCodexAppHistoryMigrationFailureAndRetry(t *testing.T) {
	f := newHistoryMigrationFixture(t, true)
	before := f.selected(t)
	historyMigrationExec(t, f.state, `CREATE TRIGGER fail_cutover BEFORE UPDATE OF rollout_path ON threads WHEN OLD.id = '`+f.ids[1]+`' BEGIN SELECT RAISE(ABORT, 'injected cutover failure'); END;`)
	restored := false
	if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { restored = true; return nil }); err == nil || !strings.Contains(err.Error(), "injected cutover") {
		t.Fatalf("expected transaction failure, got %v", err)
	}
	if restored || fmt.Sprint(before) != fmt.Sprint(f.selected(t)) {
		t.Fatal("partial selection or premature restore")
	}
	f.assertOriginals(t)
	filesBefore, _, err := readCodexAppHistoryRollouts(f.home)
	if err != nil {
		t.Fatal(err)
	}
	historyMigrationExec(t, f.state, "DROP TRIGGER fail_cutover")
	if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { return errors.New("injected settings failure") }); err == nil || !strings.Contains(err.Error(), "injected settings") {
		t.Fatal(err)
	}
	selected := f.selected(t)
	if fmt.Sprint(before) == fmt.Sprint(selected) {
		t.Fatal("expected complete history cutover before settings restore")
	}
	if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { restored = true; return nil }); err != nil {
		t.Fatal(err)
	}
	filesAfter, _, err := readCodexAppHistoryRollouts(f.home)
	if err != nil || len(filesAfter) != len(filesBefore) || !restored || fmt.Sprint(selected) != fmt.Sprint(f.selected(t)) {
		t.Fatalf("retry did not reuse prepared histories: %v", err)
	}
	f.assertOriginals(t)
}

func TestCodexAppHistoryMigrationRejectsUnsafeInput(t *testing.T) {
	for _, scenario := range []string{"unknown schema", "invalid index offset", "missing ancestor", "bad fork boundary", "old backend", "concurrent edit"} {
		t.Run(scenario, func(t *testing.T) {
			f := newHistoryMigrationFixture(t, false)
			before := f.selected(t)
			switch scenario {
			case "unknown schema":
				historyMigrationExec(t, f.history, "ALTER TABLE thread_items ADD COLUMN new_offset INTEGER")
			case "invalid index offset":
				historyMigrationExec(t, f.history, "UPDATE thread_history_projection_state SET next_rollout_byte_offset = 1")
			case "missing ancestor", "bad fork boundary":
				data := f.originals[1]
				if scenario == "missing ancestor" {
					data = bytes.ReplaceAll(data, []byte(f.rolloutIDs[0]), []byte("019a1234-5678-7000-8000-000000000099"))
				} else {
					var first map[string]any
					header, rest, _ := bytes.Cut(data, []byte{'\n'})
					if err := json.Unmarshal(header, &first); err != nil {
						t.Fatal(err)
					}
					first["payload"].(map[string]any)["history_base"].(map[string]any)["end_byte_offset"] = 1
					data = append(append(historyJSON(t, first), '\n'), rest...)
				}
				if err := os.WriteFile(f.paths[1], data, 0o600); err != nil {
					t.Fatal(err)
				}
				f.originals[1] = data
			case "old backend":
				codexAppCheckHistoryMigrationVersion = func() error { return errors.New("unsupported backend") }
			case "concurrent edit":
				codexAppCheckHistoryMigrationVersion = func() error {
					f.originals[0] = append(f.originals[0], '\n')
					return os.WriteFile(f.paths[0], f.originals[0], 0o600)
				}
			}
			restored := false
			if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { restored = true; return nil }); err == nil {
				t.Fatal("unsafe migration accepted")
			}
			if restored || fmt.Sprint(before) != fmt.Sprint(f.selected(t)) {
				t.Fatal("unsafe migration changed selected histories or connection")
			}
			f.assertOriginals(t)
			files, _, err := readCodexAppHistoryRollouts(f.home)
			if err != nil || len(files) != 3 {
				t.Fatalf("published files before validation: %v", err)
			}
		})
	}
}

func TestCodexAppHistorySQLiteHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_SQLITE_HOME", filepath.Join(home, "environment"))
	writeHistoryFixture(t, home, "config.toml", []byte("sqlite_home = 'configured'\n"))
	got, err := codexAppHistorySQLiteHome(home)
	if err != nil || got != filepath.Join(home, "configured") {
		t.Fatalf("config precedence: %q %v", got, err)
	}
}

func TestCodexAppHistoryMigrationNewerIndexColumns(t *testing.T) {
	f := newHistoryMigrationFixture(t, false)
	historyMigrationExec(t, f.history, `UPDATE _sqlx_migrations SET version = 8;
	ALTER TABLE thread_items ADD COLUMN started_at_ms INTEGER;
	ALTER TABLE thread_items ADD COLUMN completed_at_ms INTEGER;
	ALTER TABLE thread_turns ADD COLUMN root_turn_id TEXT;
	UPDATE thread_items SET started_at_ms = 123, completed_at_ms = 456;
	UPDATE thread_turns SET root_turn_id = 'original-root-turn';`)
	if err := prepareCodexAppHistoryForDisconnect(f.home, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	rollouts, _, err := readCodexAppHistoryRollouts(f.home)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range f.selected(t) {
		id := rollouts[path].id
		var start, end int64
		var rootTurn string
		if err := f.history.QueryRow("SELECT started_at_ms, completed_at_ms FROM thread_items WHERE thread_id = ?", id).Scan(&start, &end); err != nil || start != 123 || end != 456 {
			t.Fatalf("item lifecycle lost: %v", err)
		}
		if err := f.history.QueryRow("SELECT root_turn_id FROM thread_turns WHERE thread_id = ?", id).Scan(&rootTurn); err != nil || rootTurn != "original-root-turn" {
			t.Fatalf("root turn lost: %v", err)
		}
	}
}

func TestCodexAppDisconnectMigratesGrowingHistory(t *testing.T) {
	f := newHistoryMigrationFixture(t, true)
	withCodexAppPlatform(t, "darwin")
	app := &CodexApp{}
	configPath := filepath.Join(f.home, "config.toml")
	originalConfig, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatal(err)
	}
	auth := []byte(`{"OPENAI_API_KEY":"synthetic-test-key"}`)
	authPath := writeHistoryFixture(t, f.home, "auth.json", auth)
	before := f.selected(t)
	running, quit, opened := true, 0, 0
	withCodexAppProcessHooks(t, func() bool { return running }, func() error { running = false; quit++; return nil }, func() error { running = true; opened++; return nil })
	if err := app.RestoreFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatal(err)
	}
	if quit != 0 || opened != 0 || fmt.Sprint(before) != fmt.Sprint(f.selected(t)) {
		t.Fatal("cancel changed history or app lifecycle")
	}
	if err := app.RestoreFromDesktop(true); err != nil {
		t.Fatal(err)
	}
	if quit != 1 || opened != 1 || !running || fmt.Sprint(before) == fmt.Sprint(f.selected(t)) {
		t.Fatal("disconnect did not migrate and reopen")
	}
	if data, err := os.ReadFile(configPath); err != nil || !bytes.Equal(data, originalConfig) {
		t.Fatalf("original configuration not restored: got %q, want %q, error %v", data, originalConfig, err)
	}
	if data, err := os.ReadFile(authPath); err != nil || !bytes.Equal(data, auth) {
		t.Fatalf("credentials changed: %v", err)
	}
	f.assertOriginals(t)
}
