//go:build windows || darwin

package store

import (
	"path/filepath"
	"testing"
)

func TestSchemaVersioning(t *testing.T) {
	tmpDir := t.TempDir()
	// Override legacy config path to avoid migration logs
	oldLegacyConfigPath := legacyConfigPath
	legacyConfigPath = filepath.Join(tmpDir, "config.json")
	defer func() { legacyConfigPath = oldLegacyConfigPath }()

	t.Run("new database has correct schema version", func(t *testing.T) {
		dbPath := filepath.Join(tmpDir, "new_db.sqlite")
		db, err := newDatabase(dbPath)
		if err != nil {
			t.Fatalf("failed to create database: %v", err)
		}
		defer db.Close()

		// Check schema version
		version, err := db.getSchemaVersion()
		if err != nil {
			t.Fatalf("failed to get schema version: %v", err)
		}

		if version != currentSchemaVersion {
			t.Errorf("expected schema version %d, got %d", currentSchemaVersion, version)
		}
	})

	t.Run("can update schema version", func(t *testing.T) {
		dbPath := filepath.Join(tmpDir, "update_db.sqlite")
		db, err := newDatabase(dbPath)
		if err != nil {
			t.Fatalf("failed to create database: %v", err)
		}
		defer db.Close()

		// Set a different version
		testVersion := 42
		if err := db.setSchemaVersion(testVersion); err != nil {
			t.Fatalf("failed to set schema version: %v", err)
		}

		// Verify it was updated
		version, err := db.getSchemaVersion()
		if err != nil {
			t.Fatalf("failed to get schema version: %v", err)
		}

		if version != testVersion {
			t.Errorf("expected schema version %d, got %d", testVersion, version)
		}
	})
}
