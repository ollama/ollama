//go:build darwin

package launch

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"golang.org/x/mod/semver"
)

type codexAppHistoryPosition struct {
	ThreadID string `json:"thread_id"`
	Ordinal  int64  `json:"end_ordinal_exclusive"`
	Offset   int64  `json:"end_byte_offset"`
}

type codexAppHistoryBoundary struct {
	offset  int64
	ordinal int64
}

type codexAppHistoryRollout struct {
	codexAppHistoryEdit
	id, threadID, mode, timestamp string
	base                          *codexAppHistoryPosition
	original, data, replacement   []byte
	newID, newPath                string
	boundaries                    map[int64]codexAppHistoryBoundary
	canonical, visiting, visited  bool
}

type codexAppHistoryRoot struct {
	ID, Path, Mode string
	Archived       bool
	rollout        *codexAppHistoryRollout
}

// Growing records cannot be replaced in place: descendants and SQLite contain
// byte offsets into the old file. Use the same stable-chat/new-rollout convention
// as ChatGPT's Revert. Files and projections are immutable additions; the only
// cutover is one transaction updating the selected paths in state_5.sqlite.
// The caller holds the history coordinator lock throughout this operation.
func migrateCodexAppGrowingHistory(home string, restore func() error) error {
	home, err := filepath.EvalSymlinks(home)
	if err != nil {
		return err
	}
	sqliteHome, err := codexAppHistorySQLiteHome(home)
	if err != nil {
		return err
	}
	state, err := openCodexAppHistoryDB(filepath.Join(sqliteHome, "state_5.sqlite"))
	if err != nil {
		return fmt.Errorf("%w; cannot open the ChatGPT history index: %v", errCodexAppHistoryGrowth, err)
	}
	defer state.Close()
	stateTx, err := state.Begin()
	if err != nil {
		return err
	}
	defer stateTx.Rollback()
	roots, err := readCodexAppHistoryRoots(stateTx)
	if err != nil {
		return err
	}
	byPath, byID, err := readCodexAppHistoryRollouts(home)
	if err != nil {
		return err
	}
	selected := make(map[string]bool)
	for i := range roots {
		root := &roots[i]
		selected[root.ID] = true
		selectedPath := strings.TrimSuffix(root.Path, ".zst")
		if _, err := os.Lstat(selectedPath); errors.Is(err, os.ErrNotExist) {
			selectedPath += ".zst"
		}
		path, err := filepath.EvalSymlinks(selectedPath)
		if err != nil {
			return fmt.Errorf("cannot locate saved conversation %s: %w", root.ID, err)
		}
		root.rollout = byPath[path]
		if root.rollout == nil || root.rollout.threadID != root.ID || root.rollout.mode != root.Mode {
			return errors.New("ChatGPT's selected history does not match its index; reopen ChatGPT before retrying")
		}
		if err := root.rollout.convert(byID); err != nil {
			return fmt.Errorf("conversation %s: %w", root.ID, err)
		}
	}
	// Files belonging to an indexed chat but not selected or referenced are old
	// immutable rollouts (including our previous copies). Never convert them again.
	for _, rollout := range byPath {
		if !rollout.visited && !selected[rollout.threadID] {
			if err := rollout.convert(byID); err != nil {
				return err
			}
			if rollout.newID != "" && rollout.mode == "paginated" {
				return errors.New("ChatGPT must index this conversation before its history can grow; open ChatGPT and retry disconnecting")
			}
		}
	}
	var copies, legacy []*codexAppHistoryRollout
	for _, rollout := range byPath {
		if rollout.newID == "" {
			continue
		}
		if rollout.mode == "paginated" {
			copies = append(copies, rollout)
		} else {
			legacy = append(legacy, rollout)
		}
	}
	if len(copies) == 0 && len(legacy) == 0 {
		_ = stateTx.Rollback()
		return restore()
	}
	slices.SortFunc(copies, func(a, b *codexAppHistoryRollout) int { return strings.Compare(a.path, b.path) })
	if err := codexAppCheckHistoryMigrationVersion(); err != nil {
		return err
	}
	history, err := openCodexAppHistoryDB(filepath.Join(sqliteHome, "thread_history_1.sqlite"))
	if err != nil {
		return err
	}
	defer history.Close()
	historyTx, err := history.Begin()
	if err != nil {
		return err
	}
	defer historyTx.Rollback()
	columns, err := codexAppHistoryIndexColumns(historyTx)
	if err != nil {
		return err
	}
	for _, rollout := range copies {
		if existing := byID[rollout.newID]; existing != nil && existing.visited {
			return errors.New("replacement history is already in use; its existing index was not changed")
		}
		if err := copyCodexAppHistoryIndex(historyTx, columns, rollout); err != nil {
			return err
		}
	}
	// All inputs and all index offsets have been checked before publishing files.
	for _, rollout := range byPath {
		if rollout.visited {
			if err := rollout.validate(); err != nil {
				return err
			}
		}
	}
	for _, rollout := range append(slices.Clone(copies), legacy...) {
		relative, err := filepath.Rel(home, rollout.path)
		if err != nil {
			return err
		}
		backup := filepath.Join(fileutil.BackupDir(), "chatgpt-history", relative+fmt.Sprintf(".%x.bak", rollout.digest))
		if err := backupCodexAppHistory(backup, rollout.original); err != nil {
			return err
		}
		data := rollout.replacement
		if strings.HasSuffix(rollout.path, ".zst") {
			encoder, err := zstd.NewWriter(nil)
			if err != nil {
				return err
			}
			data = encoder.EncodeAll(data, nil)
			encoder.Close()
		}
		if rollout.mode == "paginated" {
			if err := backupCodexAppHistory(rollout.newPath, data); err != nil {
				return fmt.Errorf("stage replacement history: %w", err)
			}
		} else {
			temporary, err := stageCodexAppHistory(rollout.path, data, rollout.info.Mode().Perm())
			if err != nil {
				return err
			}
			defer os.Remove(temporary)
			rollout.temporary = temporary
			if err := rollout.publish(); err != nil {
				return err
			}
		}
		if err := syncCodexAppHistoryDirectory(filepath.Dir(rollout.path)); err != nil {
			return err
		}
	}
	// A local manifest retains the old selections for recovery. Existing files
	// and index rows remain intact even after a successful disconnect.
	var switches []map[string]string
	for _, root := range roots {
		if root.rollout.newID != "" && root.Mode == "paginated" {
			switches = append(switches, map[string]string{"thread_id": root.ID, "old_path": root.Path, "new_path": root.rollout.newPath})
		}
	}
	manifest, err := json.Marshal(switches)
	if err != nil {
		return err
	}
	manifestPath := filepath.Join(fileutil.BackupDir(), "chatgpt-history", fmt.Sprintf("migration-%x.json", sha256.Sum256(manifest)))
	if err := backupCodexAppHistory(manifestPath, manifest); err != nil {
		return err
	}
	if err := syncCodexAppHistoryDirectory(filepath.Dir(manifestPath)); err != nil {
		return err
	}
	if err := historyTx.Commit(); err != nil {
		return err
	}
	for _, root := range roots {
		if root.rollout.newID == "" || root.Mode != "paginated" {
			continue
		}
		result, err := stateTx.Exec("UPDATE threads SET rollout_path = ? WHERE id = ? AND rollout_path = ? AND history_mode = ? AND archived = ?", root.rollout.newPath, root.ID, root.Path, root.Mode, root.Archived)
		if err != nil {
			return err
		}
		if count, err := result.RowsAffected(); err != nil || count != 1 {
			return errors.New("ChatGPT's selected history changed during disconnect; close other sessions and retry")
		}
	}
	if err := stateTx.Commit(); err != nil {
		return err
	}
	return restore()
}

func codexAppHistorySQLiteHome(home string) (string, error) {
	data, err := os.ReadFile(filepath.Join(home, "config.toml"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	config, err := codexParseConfig(string(data))
	if err != nil {
		return "", err
	}
	if config.Exists("sqlite_home") {
		path, ok := config.RootStringOK("sqlite_home")
		if !ok || path == "" {
			return "", errors.New("unsupported ChatGPT sqlite_home setting")
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(home, path)
		}
		return path, nil
	}
	if path := strings.TrimSpace(os.Getenv("CODEX_SQLITE_HOME")); path != "" {
		if !filepath.IsAbs(path) {
			return "", errors.New("cannot resolve a relative CODEX_SQLITE_HOME while ChatGPT is stopped")
		}
		return path, nil
	}
	return home, nil
}

func openCodexAppHistoryDB(path string) (*sql.DB, error) {
	for _, suffix := range []string{"", "-wal", "-shm", "-journal"} {
		info, err := os.Lstat(path + suffix)
		if suffix != "" && errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, errors.New("unsupported ChatGPT history database path")
		}
	}
	u := url.URL{Scheme: "file", Path: path, RawQuery: "mode=rw&_busy_timeout=1000&_txlock=immediate&_synchronous=FULL"}
	db, err := sql.Open("sqlite3", u.String())
	if err == nil {
		db.SetMaxOpenConns(1)
	}
	return db, err
}

func readCodexAppHistoryRoots(tx *sql.Tx) ([]codexAppHistoryRoot, error) {
	rows, err := tx.Query("SELECT id, rollout_path, history_mode, archived FROM threads ORDER BY id")
	if err != nil {
		return nil, fmt.Errorf("unsupported ChatGPT history index: %w", err)
	}
	defer rows.Close()
	var roots []codexAppHistoryRoot
	for rows.Next() {
		var root codexAppHistoryRoot
		if err := rows.Scan(&root.ID, &root.Path, &root.Mode, &root.Archived); err != nil {
			return nil, err
		}
		if !filepath.IsAbs(root.Path) {
			return nil, errors.New("unsupported relative ChatGPT history path")
		}
		roots = append(roots, root)
	}
	return roots, rows.Err()
}

func readCodexAppHistoryRollouts(home string) (map[string]*codexAppHistoryRollout, map[string]*codexAppHistoryRollout, error) {
	byPath, byID := make(map[string]*codexAppHistoryRollout), make(map[string]*codexAppHistoryRollout)
	err := walkCodexAppHistory(home, func(path string, info fs.FileInfo) error {
		// The backend prefers the plain representation when both are present.
		if strings.HasSuffix(path, ".zst") {
			if _, err := os.Lstat(strings.TrimSuffix(path, ".zst")); err == nil {
				return nil
			} else if !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
		original, data, err := readCodexAppHistory(path)
		if err != nil {
			return err
		}
		var header struct {
			Type    string `json:"type"`
			Payload struct {
				ID        string                   `json:"id"`
				Mode      string                   `json:"history_mode"`
				Timestamp string                   `json:"timestamp"`
				Base      *codexAppHistoryPosition `json:"history_base"`
			} `json:"payload"`
		}
		first, _, _ := bytes.Cut(data, []byte{'\n'})
		if err := json.Unmarshal(first, &header); err != nil || header.Type != "session_meta" {
			return fmt.Errorf("unsupported saved conversation header: %s", path)
		}
		mode := header.Payload.Mode
		if mode == "" {
			mode = "legacy"
		}
		id, canonicalName := codexAppHistoryRolloutID(filepath.Base(path), header.Payload.ID)
		canonical, err := filepath.EvalSymlinks(path)
		if err != nil {
			return err
		}
		rollout := &codexAppHistoryRollout{codexAppHistoryEdit: codexAppHistoryEdit{path: canonical, info: info, digest: sha256.Sum256(original)}, id: id, threadID: header.Payload.ID, mode: mode, timestamp: header.Payload.Timestamp, base: header.Payload.Base, original: original, data: data, canonical: canonicalName}
		if prior := byID[id]; prior != nil {
			return fmt.Errorf("ambiguous ChatGPT rollout ID: %s", id)
		}
		byPath[canonical], byID[id] = rollout, rollout
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return byPath, byID, nil
}

func codexAppHistoryRolloutID(name, threadID string) (string, bool) {
	stem := strings.TrimSuffix(strings.TrimSuffix(name, ".zst"), ".jsonl")
	if len(stem) < 64 || !strings.HasPrefix(stem, "rollout-") || stem[27] != '-' {
		return threadID, false
	}
	if _, err := time.Parse("2006-01-02T15-04-05", stem[8:27]); err != nil {
		return threadID, false
	}
	stable, physical, found := strings.Cut(stem[28:], "_")
	if !found {
		physical = stable
	}
	if _, err := uuid.Parse(stable); err != nil || stable != threadID {
		return threadID, false
	}
	if _, err := uuid.Parse(physical); err != nil {
		return threadID, false
	}
	return physical, true
}

func (r *codexAppHistoryRollout) convert(byID map[string]*codexAppHistoryRollout) error {
	if r.visited {
		return nil
	}
	if r.visiting {
		return errors.New("cyclic ChatGPT history references")
	}
	r.visiting = true
	data, changed, err := rewriteCodexAppHistory(r.data, r.mode != "paginated")
	if err != nil {
		return err
	}
	if r.base != nil {
		parent := byID[r.base.ThreadID]
		if parent == nil || parent.mode != "paginated" || r.mode != "paginated" {
			return errors.New("missing ChatGPT history ancestor")
		}
		if err := parent.convert(byID); err != nil {
			return err
		}
		boundary, ok := parent.boundaries[r.base.Offset]
		if !ok || boundary.ordinal != r.base.Ordinal {
			return errors.New("invalid ChatGPT fork history boundary")
		}
		if parent.newID != "" {
			first, rest, _ := bytes.Cut(data, []byte{'\n'})
			var record, payload, base map[string]json.RawMessage
			if json.Unmarshal(first, &record) != nil || json.Unmarshal(record["payload"], &payload) != nil || json.Unmarshal(payload["history_base"], &base) != nil {
				return errors.New("unsupported ChatGPT fork header")
			}
			base["thread_id"], _ = json.Marshal(parent.newID)
			base["end_byte_offset"], _ = json.Marshal(boundary.offset)
			payload["history_base"], _ = marshalCodexAppHistory(base)
			record["payload"], _ = marshalCodexAppHistory(payload)
			header, err := marshalCodexAppHistory(record)
			if err != nil {
				return err
			}
			data = append(append(header, '\n'), rest...)
			changed = true
		}
	}
	if r.mode == "paginated" {
		r.boundaries, err = codexAppHistoryBoundaries(r.data, data, r.base)
		if err != nil {
			return err
		}
	}
	if changed {
		r.replacement = data
		if r.mode == "paginated" {
			// Deterministic UUIDv7 storage IDs make retries reuse prepared files.
			// Keep the session's timestamp; derive the random bits from its exact
			// source storage ID and replacement bytes, with a migration namespace.
			stamp, err := time.Parse(time.RFC3339Nano, r.timestamp)
			if err != nil {
				return errors.New("unsupported ChatGPT session timestamp")
			}
			sum := sha256.Sum256(append([]byte("ollama-history-v1\x00"+r.id+"\x00"), data...))
			id := uuid.UUID(sum[:16])
			millis := stamp.UnixMilli()
			for i := 5; i >= 0; i-- {
				id[i] = byte(millis)
				millis >>= 8
			}
			id[6], id[8] = (id[6]&0x0f)|0x70, (id[8]&0x3f)|0x80
			r.newID = id.String()
			name := filepath.Base(r.path)
			if !r.canonical {
				return errors.New("unsupported ChatGPT rollout filename")
			}
			r.newPath = filepath.Join(filepath.Dir(r.path), name[:64]+"_"+r.newID+".jsonl")
			if strings.HasSuffix(r.path, ".zst") {
				r.newPath += ".zst"
			}
		} else {
			r.newID = r.id
		}
	}
	r.visiting, r.visited = false, true
	return nil
}

func codexAppHistoryBoundaries(before, after []byte, base *codexAppHistoryPosition) (map[int64]codexAppHistoryBoundary, error) {
	if !bytes.HasSuffix(before, []byte{'\n'}) || !bytes.HasSuffix(after, []byte{'\n'}) {
		return nil, errors.New("incomplete ChatGPT history record; reopen ChatGPT before retrying")
	}
	oldLines, newLines := bytes.SplitAfter(before, []byte{'\n'}), bytes.SplitAfter(after, []byte{'\n'})
	if len(oldLines) != len(newLines) {
		return nil, errors.New("history migration changed saved ordinals")
	}
	var oldOffset, newOffset, next int64
	if base != nil {
		next = base.Ordinal
	}
	boundaries := map[int64]codexAppHistoryBoundary{0: {ordinal: next}}
	for i, line := range oldLines {
		if len(bytes.TrimSpace(line)) > 0 {
			var record struct {
				Ordinal *int64 `json:"ordinal"`
			}
			if json.Unmarshal(line, &record) != nil || record.Ordinal == nil || *record.Ordinal != next || next < 0 || next == int64(^uint64(0)>>1) {
				return nil, errors.New("unsupported ChatGPT history ordinals")
			}
			next++
		}
		oldOffset += int64(len(line))
		newOffset += int64(len(newLines[i]))
		boundaries[oldOffset] = codexAppHistoryBoundary{offset: newOffset, ordinal: next}
	}
	return boundaries, nil
}

// These records are UI projections of event_msg/realtime_item, neither of which
// the converter changes. Clone their values and remap only physical offsets.
var codexAppHistoryIndexSchema = map[string]string{
	"thread_turns":                    "thread_id turn_id rollout_ordinal status error_json started_at completed_at duration_ms first_user_item_id final_agent_item_id rollout_byte_offset rollout_end_ordinal rollout_end_byte_offset",
	"thread_items":                    "thread_id turn_id item_id rollout_ordinal created_at_ms item_json item_type updated_at_ordinal",
	"thread_history_projection_state": "thread_id next_rollout_byte_offset next_rollout_ordinal",
	"thread_realtime_items":           "thread_id item_id rollout_ordinal created_at_ms item_type item_json",
}

func codexAppHistoryIndexColumns(tx *sql.Tx) (map[string][]string, error) {
	var version, failures int
	if err := tx.QueryRow("SELECT max(version), sum(NOT success) FROM _sqlx_migrations").Scan(&version, &failures); err != nil || version < 6 || version > 8 || failures != 0 {
		return nil, errors.New("unsupported ChatGPT history index version")
	}
	var tableCount int
	if err := tx.QueryRow("SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").Scan(&tableCount); err != nil || tableCount != len(codexAppHistoryIndexSchema)+1 {
		return nil, errors.New("unsupported ChatGPT history index tables")
	}
	result := make(map[string][]string)
	for table, required := range codexAppHistoryIndexSchema {
		allowed := strings.Fields(required)
		if table == "thread_turns" && version >= 8 {
			allowed = append(allowed, "root_turn_id")
		}
		if table == "thread_items" && version >= 7 {
			allowed = append(allowed, "started_at_ms", "completed_at_ms")
		}
		rows, err := tx.Query("SELECT name FROM pragma_table_info(?) ORDER BY cid", table)
		if err != nil {
			return nil, err
		}
		var columns []string
		for rows.Next() {
			var column string
			if err := rows.Scan(&column); err != nil {
				rows.Close()
				return nil, err
			}
			columns = append(columns, column)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return nil, err
		}
		if len(columns) != len(allowed) {
			return nil, fmt.Errorf("unsupported ChatGPT history table %s", table)
		}
		for _, column := range columns {
			if !slices.Contains(allowed, column) {
				return nil, fmt.Errorf("unsupported ChatGPT history column %s", column)
			}
		}
		result[table] = columns
	}
	return result, nil
}

func copyCodexAppHistoryIndex(tx *sql.Tx, schema map[string][]string, rollout *codexAppHistoryRollout) error {
	for _, table := range []string{"thread_history_projection_state", "thread_turns", "thread_items", "thread_realtime_items"} {
		columns := schema[table]
		rows, err := tx.Query("SELECT "+strings.Join(columns, ",")+" FROM "+table+" WHERE thread_id = ?", rollout.id)
		if err != nil {
			return err
		}
		var records [][]any
		for rows.Next() {
			values, pointers := make([]any, len(columns)), make([]any, len(columns))
			for i := range values {
				pointers[i] = &values[i]
			}
			if err := rows.Scan(pointers...); err != nil {
				rows.Close()
				return err
			}
			for i, column := range columns {
				if column == "thread_id" {
					values[i] = rollout.newID
				}
				if column == "next_rollout_byte_offset" || column == "rollout_byte_offset" || column == "rollout_end_byte_offset" {
					if values[i] == nil {
						continue
					}
					offset, ok := values[i].(int64)
					boundary, found := rollout.boundaries[offset]
					if !ok || !found {
						rows.Close()
						return errors.New("invalid byte offset in ChatGPT history index")
					}
					values[i] = boundary.offset
				}
			}
			records = append(records, values)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return err
		}
		// A previous attempt may have committed these additions before the path
		// cutover failed. Replacing only our deterministic IDs is retry-safe.
		if _, err := tx.Exec("DELETE FROM "+table+" WHERE thread_id = ?", rollout.newID); err != nil {
			return err
		}
		for _, record := range records {
			query := "INSERT INTO " + table + " (" + strings.Join(columns, ",") + ") VALUES (" + strings.TrimSuffix(strings.Repeat("?,", len(columns)), ",") + ")"
			if _, err := tx.Exec(query, record...); err != nil {
				return err
			}
		}
	}
	return nil
}

func syncCodexAppHistoryDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

var codexAppCheckHistoryMigrationVersion = func() error {
	app := codexAppAppPath()
	if app == "" {
		return errors.New("cannot verify ChatGPT's history format without its installed backend")
	}
	executable := filepath.Join(app, "Contents", "Resources", "codex")
	scratch, err := os.MkdirTemp("", "ollama-history-version-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(scratch)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, executable, "--version")
	command.Dir = scratch
	command.Env = []string{"HOME=" + scratch, "CODEX_HOME=" + scratch, "CODEX_SQLITE_HOME=" + scratch, "PATH=" + os.Getenv("PATH")}
	output, err := command.Output()
	if err != nil {
		return fmt.Errorf("check ChatGPT history compatibility: %w", err)
	}
	fields := strings.Fields(string(output))
	if len(fields) != 2 || fields[0] != "codex-cli" || !semver.IsValid("v"+fields[1]) || semver.Compare("v"+fields[1], "v0.155.0-alpha.16.3") < 0 {
		return errors.New("update ChatGPT before disconnecting conversations whose saved history needs to grow")
	}
	return nil
}
