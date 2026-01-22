package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestAtomicWrite(t *testing.T) {
	tmpDir := t.TempDir()

	t.Run("creates file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "new.json")
		data := mustMarshal(t, map[string]string{"key": "value"})

		if err := atomicWrite(path, data); err != nil {
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

		os.WriteFile(path, []byte(`{"original": true}`), 0o644)

		data := mustMarshal(t, map[string]bool{"updated": true})
		if err := atomicWrite(path, data); err != nil {
			t.Fatal(err)
		}

		entries, err := os.ReadDir(backupDir())
		if err != nil {
			t.Fatal("backup directory not created")
		}

		var foundBackup bool
		for _, entry := range entries {
			if filepath.Ext(entry.Name()) != ".json" {
				name := entry.Name()
				if len(name) > len("backup.json.") && name[:len("backup.json.")] == "backup.json." {
					backupPath := filepath.Join(backupDir(), name)
					backup, err := os.ReadFile(backupPath)
					if err == nil {
						var backupData map[string]bool
						json.Unmarshal(backup, &backupData)
						if backupData["original"] {
							foundBackup = true
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

		current, _ := os.ReadFile(path)
		var currentData map[string]bool
		json.Unmarshal(current, &currentData)
		if !currentData["updated"] {
			t.Error("file doesn't contain updated data")
		}
	})

	t.Run("no backup for new file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "nobak.json")

		data := mustMarshal(t, map[string]string{"new": "file"})
		if err := atomicWrite(path, data); err != nil {
			t.Fatal(err)
		}

		entries, _ := os.ReadDir(backupDir())
		for _, entry := range entries {
			if len(entry.Name()) > len("nobak.json.") && entry.Name()[:len("nobak.json.")] == "nobak.json." {
				t.Error("backup should not exist for new file")
			}
		}
	})

	t.Run("no backup when content unchanged", func(t *testing.T) {
		path := filepath.Join(tmpDir, "unchanged.json")

		data := mustMarshal(t, map[string]string{"key": "value"})

		if err := atomicWrite(path, data); err != nil {
			t.Fatal(err)
		}

		entries1, _ := os.ReadDir(backupDir())
		countBefore := 0
		for _, e := range entries1 {
			if len(e.Name()) > len("unchanged.json.") && e.Name()[:len("unchanged.json.")] == "unchanged.json." {
				countBefore++
			}
		}

		if err := atomicWrite(path, data); err != nil {
			t.Fatal(err)
		}

		entries2, _ := os.ReadDir(backupDir())
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
		data := mustMarshal(t, map[string]int{"v": 2})
		if err := atomicWrite(path, data); err != nil {
			t.Fatal(err)
		}

		entries, _ := os.ReadDir(backupDir())
		var found bool
		for _, entry := range entries {
			name := entry.Name()
			if len(name) > len("timestamped.json.") && name[:len("timestamped.json.")] == "timestamped.json." {
				timestamp := name[len("timestamped.json."):]
				for _, c := range timestamp {
					if c < '0' || c > '9' {
						t.Errorf("backup filename timestamp contains non-numeric character: %s", name)
					}
				}
				found = true
				os.Remove(filepath.Join(backupDir(), name))
				break
			}
		}
		if !found {
			t.Error("backup file with timestamp not found")
		}
	})
}

// Edge case tests for files.go

// TestAtomicWrite_FailsIfBackupFails documents critical behavior: if backup fails, we must not proceed.
// User could lose their config with no way to recover.
func TestAtomicWrite_FailsIfBackupFails(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission tests unreliable on Windows")
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "config.json")

	// Create original file
	originalContent := []byte(`{"original": true}`)
	os.WriteFile(path, originalContent, 0o644)

	// Make backup directory read-only to force backup failure
	backupDir := backupDir()
	os.MkdirAll(backupDir, 0o755)
	os.Chmod(backupDir, 0o444) // Read-only
	defer os.Chmod(backupDir, 0o755)

	newContent := []byte(`{"updated": true}`)
	err := atomicWrite(path, newContent)

	// Should fail because backup couldn't be created
	if err == nil {
		t.Error("expected error when backup fails, got nil")
	}

	// Original file should be preserved
	current, _ := os.ReadFile(path)
	if string(current) != string(originalContent) {
		t.Errorf("original file was modified despite backup failure: got %s", string(current))
	}
}

// TestAtomicWrite_PermissionDenied verifies clear error when target file has wrong permissions.
// Common issue when config owned by root or wrong perms.
func TestAtomicWrite_PermissionDenied(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission tests unreliable on Windows")
	}

	tmpDir := t.TempDir()

	// Create a read-only directory
	readOnlyDir := filepath.Join(tmpDir, "readonly")
	os.MkdirAll(readOnlyDir, 0o755)
	os.Chmod(readOnlyDir, 0o444)
	defer os.Chmod(readOnlyDir, 0o755)

	path := filepath.Join(readOnlyDir, "config.json")
	err := atomicWrite(path, []byte(`{"test": true}`))

	if err == nil {
		t.Error("expected permission error, got nil")
	}
}

// TestAtomicWrite_DirectoryDoesNotExist verifies behavior when target directory doesn't exist.
// atomicWrite doesn't create directories - caller is responsible.
func TestAtomicWrite_DirectoryDoesNotExist(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "nonexistent", "subdir", "config.json")

	err := atomicWrite(path, []byte(`{"test": true}`))

	// Should fail because directory doesn't exist
	if err == nil {
		t.Error("expected error for nonexistent directory, got nil")
	}
}

// TestAtomicWrite_SymlinkTarget documents behavior when target is a symlink.
// Documents what happens if user symlinks their config file.
func TestAtomicWrite_SymlinkTarget(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink tests may require admin on Windows")
	}

	tmpDir := t.TempDir()
	realFile := filepath.Join(tmpDir, "real.json")
	symlink := filepath.Join(tmpDir, "link.json")

	// Create real file and symlink
	os.WriteFile(realFile, []byte(`{"v": 1}`), 0o644)
	os.Symlink(realFile, symlink)

	// Write through symlink
	err := atomicWrite(symlink, []byte(`{"v": 2}`))
	if err != nil {
		t.Fatalf("atomicWrite through symlink failed: %v", err)
	}

	// The real file should be updated (symlink followed for temp file creation)
	content, _ := os.ReadFile(symlink)
	if string(content) != `{"v": 2}` {
		t.Errorf("symlink target not updated correctly: got %s", string(content))
	}
}

// TestBackupToTmp_SpecialCharsInFilename verifies backup works with special characters.
// User may have config files with unusual names.
func TestBackupToTmp_SpecialCharsInFilename(t *testing.T) {
	tmpDir := t.TempDir()

	// File with spaces and special chars
	path := filepath.Join(tmpDir, "my config (backup).json")
	os.WriteFile(path, []byte(`{"test": true}`), 0o644)

	backupPath, err := backupToTmp(path)
	if err != nil {
		t.Fatalf("backupToTmp with special chars failed: %v", err)
	}

	// Verify backup exists and has correct content
	content, err := os.ReadFile(backupPath)
	if err != nil {
		t.Fatalf("could not read backup: %v", err)
	}
	if string(content) != `{"test": true}` {
		t.Errorf("backup content mismatch: got %s", string(content))
	}

	os.Remove(backupPath)
}

// TestCopyFile_PreservesPermissions verifies that copyFile preserves file permissions.
func TestCopyFile_PreservesPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission preservation tests unreliable on Windows")
	}

	tmpDir := t.TempDir()
	src := filepath.Join(tmpDir, "src.json")
	dst := filepath.Join(tmpDir, "dst.json")

	// Create source with specific permissions
	os.WriteFile(src, []byte(`{"test": true}`), 0o600)

	err := copyFile(src, dst)
	if err != nil {
		t.Fatalf("copyFile failed: %v", err)
	}

	srcInfo, _ := os.Stat(src)
	dstInfo, _ := os.Stat(dst)

	if srcInfo.Mode().Perm() != dstInfo.Mode().Perm() {
		t.Errorf("permissions not preserved: src=%v, dst=%v", srcInfo.Mode().Perm(), dstInfo.Mode().Perm())
	}
}

// TestCopyFile_SourceNotFound verifies clear error when source doesn't exist.
func TestCopyFile_SourceNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	src := filepath.Join(tmpDir, "nonexistent.json")
	dst := filepath.Join(tmpDir, "dst.json")

	err := copyFile(src, dst)
	if err == nil {
		t.Error("expected error for nonexistent source, got nil")
	}
}
