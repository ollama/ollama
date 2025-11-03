//go:build windows || darwin

package store

import (
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestConfigMigration(t *testing.T) {
	tmpDir := t.TempDir()
	// Create a legacy config.json
	legacyConfig := legacyData{
		ID:           "test-device-id-12345",
		FirstTimeRun: true, // In old system, true meant "has completed first run"
	}

	configData, err := json.MarshalIndent(legacyConfig, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	configPath := filepath.Join(tmpDir, "config.json")
	if err := os.WriteFile(configPath, configData, 0o644); err != nil {
		t.Fatal(err)
	}

	// Override the legacy config path for testing
	oldLegacyConfigPath := legacyConfigPath
	legacyConfigPath = configPath
	defer func() { legacyConfigPath = oldLegacyConfigPath }()

	// Create store with database in same directory
	s := Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}
	defer s.Close()

	// First access should trigger migration
	id, err := s.ID()
	if err != nil {
		t.Fatalf("failed to get ID: %v", err)
	}

	if id != "test-device-id-12345" {
		t.Errorf("expected migrated ID 'test-device-id-12345', got '%s'", id)
	}

	// Check HasCompletedFirstRun
	hasCompleted, err := s.HasCompletedFirstRun()
	if err != nil {
		t.Fatalf("failed to get has completed first run: %v", err)
	}

	if !hasCompleted {
		t.Error("expected has completed first run to be true after migration")
	}

	// Verify migration is marked as complete
	migrated, err := s.db.isConfigMigrated()
	if err != nil {
		t.Fatalf("failed to check migration status: %v", err)
	}

	if !migrated {
		t.Error("expected config to be marked as migrated")
	}

	// Create a new store instance to verify migration doesn't run again
	s2 := Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}
	defer s2.Close()

	// Delete the config file to ensure we're not reading from it
	os.Remove(configPath)

	// Verify data is still there
	id2, err := s2.ID()
	if err != nil {
		t.Fatalf("failed to get ID from second store: %v", err)
	}

	if id2 != "test-device-id-12345" {
		t.Errorf("expected persisted ID 'test-device-id-12345', got '%s'", id2)
	}
}

func TestNoConfigToMigrate(t *testing.T) {
	tmpDir := t.TempDir()
	// Override the legacy config path for testing
	oldLegacyConfigPath := legacyConfigPath
	legacyConfigPath = filepath.Join(tmpDir, "config.json")
	defer func() { legacyConfigPath = oldLegacyConfigPath }()

	// Create store without any config.json
	s := Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}
	defer s.Close()

	// Should generate a new ID
	id, err := s.ID()
	if err != nil {
		t.Fatalf("failed to get ID: %v", err)
	}

	if id == "" {
		t.Error("expected auto-generated ID, got empty string")
	}

	// HasCompletedFirstRun should be false (default)
	hasCompleted, err := s.HasCompletedFirstRun()
	if err != nil {
		t.Fatalf("failed to get has completed first run: %v", err)
	}

	if hasCompleted {
		t.Error("expected has completed first run to be false by default")
	}

	// Migration should still be marked as complete
	migrated, err := s.db.isConfigMigrated()
	if err != nil {
		t.Fatalf("failed to check migration status: %v", err)
	}

	if !migrated {
		t.Error("expected config to be marked as migrated even with no config.json")
	}
}

const (
	v1Schema = `
	CREATE TABLE IF NOT EXISTS settings (
		id INTEGER PRIMARY KEY CHECK (id = 1),
		device_id TEXT NOT NULL DEFAULT '',
		has_completed_first_run BOOLEAN NOT NULL DEFAULT 0,
		expose BOOLEAN NOT NULL DEFAULT 0,
		browser BOOLEAN NOT NULL DEFAULT 0,
		models TEXT NOT NULL DEFAULT '',
		remote TEXT NOT NULL DEFAULT '',
		agent BOOLEAN NOT NULL DEFAULT 0,
		tools BOOLEAN NOT NULL DEFAULT 0,
		working_dir TEXT NOT NULL DEFAULT '',
		window_width INTEGER NOT NULL DEFAULT 0,
		window_height INTEGER NOT NULL DEFAULT 0,
		config_migrated BOOLEAN NOT NULL DEFAULT 0,
		schema_version INTEGER NOT NULL DEFAULT 1
	);

	-- Insert default settings row if it doesn't exist
	INSERT OR IGNORE INTO settings (id) VALUES (1);

	CREATE TABLE IF NOT EXISTS chats (
		id TEXT PRIMARY KEY,
		title TEXT NOT NULL DEFAULT '',
		created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS messages (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		chat_id TEXT NOT NULL,
		role TEXT NOT NULL,
		content TEXT NOT NULL DEFAULT '',
		thinking TEXT NOT NULL DEFAULT '',
		stream BOOLEAN NOT NULL DEFAULT 0,
		model_name TEXT,
		model_cloud BOOLEAN,
		model_ollama_host BOOLEAN,
		created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
		thinking_time_start TIMESTAMP,
		thinking_time_end TIMESTAMP,
		FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);

	CREATE TABLE IF NOT EXISTS tool_calls (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		message_id INTEGER NOT NULL,
		type TEXT NOT NULL,
		function_name TEXT NOT NULL,
		function_arguments TEXT NOT NULL,
		function_result TEXT,
		FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);
	`
)

func TestMigrationFromEpoc(t *testing.T) {
	tmpDir := t.TempDir()
	s := Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}
	defer s.Close()
	// Open database connection
	conn, err := sql.Open("sqlite3", s.DBPath+"?_foreign_keys=on&_journal_mode=WAL")
	if err != nil {
		t.Fatal(err)
	}
	// Test the connection
	if err := conn.Ping(); err != nil {
		conn.Close()
		t.Fatal(err)
	}
	s.db = &database{conn: conn}
	t.Logf("DB created: %s", s.DBPath)
	_, err = s.db.conn.Exec(v1Schema)
	if err != nil {
		t.Fatal(err)
	}
	version, err := s.db.getSchemaVersion()
	if err != nil {
		t.Fatalf("failed to get schema version: %v", err)
	}
	if version != 1 {
		t.Fatalf("expected: %d\n got: %d", 1, version)
	}

	t.Logf("v1 schema created")
	if err := s.db.migrate(); err != nil {
		t.Fatal(err)
	}
	t.Logf("migrations completed")
	version, err = s.db.getSchemaVersion()
	if err != nil {
		t.Fatalf("failed to get schema version: %v", err)
	}
	if version != currentSchemaVersion {
		t.Fatalf("expected: %d\n got: %d", currentSchemaVersion, version)
	}
}
