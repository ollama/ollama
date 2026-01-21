package integrations

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAtomicWriteJSON(t *testing.T) {
	tmpDir := t.TempDir()

	t.Run("creates file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "new.json")
		data := map[string]string{"key": "value"}

		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}

		var result map[string]string
		if err := json.Unmarshal(content, &result); err != nil {
			t.Fatal(err)
		}
		if result["key"] != "value" {
			t.Errorf("expected value, got %s", result["key"])
		}
	})

	t.Run("creates backup in /tmp/ollama-backups", func(t *testing.T) {
		path := filepath.Join(tmpDir, "backup.json")

		// Write initial file
		os.WriteFile(path, []byte(`{"original": true}`), 0o644)

		// Update with atomicWriteJSON
		if err := atomicWriteJSON(path, map[string]bool{"updated": true}); err != nil {
			t.Fatal(err)
		}

		// Check backup exists in /tmp/ollama-backups/ with original content
		entries, err := os.ReadDir(getBackupDir())
		if err != nil {
			t.Fatal("backup directory not created")
		}

		var foundBackup bool
		for _, entry := range entries {
			if filepath.Ext(entry.Name()) != ".json" {
				// Look for backup.json.<timestamp>
				name := entry.Name()
				if len(name) > len("backup.json.") && name[:len("backup.json.")] == "backup.json." {
					backupPath := filepath.Join(getBackupDir(), name)
					backup, err := os.ReadFile(backupPath)
					if err == nil {
						var backupData map[string]bool
						json.Unmarshal(backup, &backupData)
						if backupData["original"] {
							foundBackup = true
							// Clean up after test
							os.Remove(backupPath)
							break
						}
					}
				}
			}
		}

		if !foundBackup {
			t.Error("backup file not created in /tmp/ollama-backups")
		}

		// Check new file has updated content
		current, _ := os.ReadFile(path)
		var currentData map[string]bool
		json.Unmarshal(current, &currentData)
		if !currentData["updated"] {
			t.Error("file doesn't contain updated data")
		}
	})

	t.Run("no backup for new file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "nobak.json")

		if err := atomicWriteJSON(path, map[string]string{"new": "file"}); err != nil {
			t.Fatal(err)
		}

		// Check no backup was created for this specific file
		entries, _ := os.ReadDir(getBackupDir())
		for _, entry := range entries {
			if len(entry.Name()) > len("nobak.json.") && entry.Name()[:len("nobak.json.")] == "nobak.json." {
				t.Error("backup should not exist for new file")
			}
		}
	})

	t.Run("valid JSON output", func(t *testing.T) {
		path := filepath.Join(tmpDir, "valid.json")
		data := map[string]any{
			"string": "hello",
			"number": 42,
			"nested": map[string]string{"a": "b"},
		}

		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		content, _ := os.ReadFile(path)
		var parsed map[string]any
		if err := json.Unmarshal(content, &parsed); err != nil {
			t.Errorf("output is not valid JSON: %v", err)
		}
	})

	t.Run("no backup when content unchanged", func(t *testing.T) {
		path := filepath.Join(tmpDir, "unchanged.json")

		data := map[string]string{"key": "value"}

		// First write
		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		// Count backups before
		entries1, _ := os.ReadDir(getBackupDir())
		countBefore := 0
		for _, e := range entries1 {
			if len(e.Name()) > len("unchanged.json.") && e.Name()[:len("unchanged.json.")] == "unchanged.json." {
				countBefore++
			}
		}

		// Second write with same content
		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		// Count backups after - should be same (no new backup created)
		entries2, _ := os.ReadDir(getBackupDir())
		countAfter := 0
		for _, e := range entries2 {
			if len(e.Name()) > len("unchanged.json.") && e.Name()[:len("unchanged.json.")] == "unchanged.json." {
				countAfter++
			}
		}

		if countAfter != countBefore {
			t.Errorf("backup was created when content unchanged (before=%d, after=%d)", countBefore, countAfter)
		}
	})

	t.Run("backup filename contains unix timestamp", func(t *testing.T) {
		path := filepath.Join(tmpDir, "timestamped.json")

		os.WriteFile(path, []byte(`{"v": 1}`), 0o644)
		if err := atomicWriteJSON(path, map[string]int{"v": 2}); err != nil {
			t.Fatal(err)
		}

		entries, _ := os.ReadDir(getBackupDir())
		var found bool
		for _, entry := range entries {
			name := entry.Name()
			if len(name) > len("timestamped.json.") && name[:len("timestamped.json.")] == "timestamped.json." {
				// Extract timestamp part and verify it's numeric
				timestamp := name[len("timestamped.json."):]
				for _, c := range timestamp {
					if c < '0' || c > '9' {
						t.Errorf("backup filename timestamp contains non-numeric character: %s", name)
					}
				}
				found = true
				os.Remove(filepath.Join(getBackupDir(), name))
				break
			}
		}
		if !found {
			t.Error("backup file with timestamp not found")
		}
	})
}
