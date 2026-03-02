package config

import (
	"encoding/json"
	"fmt"
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

func TestWriteWithBackup(t *testing.T) {
	tmpDir := t.TempDir()

	t.Run("creates file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "new.json")
		data := mustMarshal(t, map[string]string{"key": "value"})

		if err := writeWithBackup(path, data); err != nil {
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
		if err := writeWithBackup(path, data); err != nil {
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
		if err := writeWithBackup(path, data); err != nil {
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

		if err := writeWithBackup(path, data); err != nil {
			t.Fatal(err)
		}

		entries1, _ := os.ReadDir(backupDir())
		countBefore := 0
		for _, e := range entries1 {
			if len(e.Name()) > len("unchanged.json.") && e.Name()[:len("unchanged.json.")] == "unchanged.json." {
				countBefore++
			}
		}

		if err := writeWithBackup(path, data); err != nil {
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
		if err := writeWithBackup(path, data); err != nil {
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

// TestWriteWithBackup_FailsIfBackupFails documents critical behavior: if backup fails, we must not proceed.
// User could lose their config with no way to recover.
func TestWriteWithBackup_FailsIfBackupFails(t *testing.T) {
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
	err := writeWithBackup(path, newContent)

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

// TestWriteWithBackup_PermissionDenied verifies clear error when target file has wrong permissions.
// Common issue when config owned by root or wrong perms.
func TestWriteWithBackup_PermissionDenied(t *testing.T) {
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
	err := writeWithBackup(path, []byte(`{"test": true}`))

	if err == nil {
		t.Error("expected permission error, got nil")
	}
}

// TestWriteWithBackup_DirectoryDoesNotExist verifies behavior when target directory doesn't exist.
// writeWithBackup doesn't create directories - caller is responsible.
func TestWriteWithBackup_DirectoryDoesNotExist(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "nonexistent", "subdir", "config.json")

	err := writeWithBackup(path, []byte(`{"test": true}`))

	// Should fail because directory doesn't exist
	if err == nil {
		t.Error("expected error for nonexistent directory, got nil")
	}
}

// TestWriteWithBackup_SymlinkTarget documents behavior when target is a symlink.
// Documents what happens if user symlinks their config file.
func TestWriteWithBackup_SymlinkTarget(t *testing.T) {
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
	err := writeWithBackup(symlink, []byte(`{"v": 2}`))
	if err != nil {
		t.Fatalf("writeWithBackup through symlink failed: %v", err)
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

// TestWriteWithBackup_TargetIsDirectory verifies error when path points to a directory.
func TestWriteWithBackup_TargetIsDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	dirPath := filepath.Join(tmpDir, "actualdir")
	os.MkdirAll(dirPath, 0o755)

	err := writeWithBackup(dirPath, []byte(`{"test": true}`))
	if err == nil {
		t.Error("expected error when target is a directory, got nil")
	}
}

// TestWriteWithBackup_EmptyData verifies writing zero bytes works correctly.
func TestWriteWithBackup_EmptyData(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "empty.json")

	err := writeWithBackup(path, []byte{})
	if err != nil {
		t.Fatalf("writeWithBackup with empty data failed: %v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("could not read file: %v", err)
	}
	if len(content) != 0 {
		t.Errorf("expected empty file, got %d bytes", len(content))
	}
}

// TestWriteWithBackup_FileUnreadableButDirWritable verifies behavior when existing file
// cannot be read (for backup comparison) but directory is writable.
func TestWriteWithBackup_FileUnreadableButDirWritable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission tests unreliable on Windows")
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "unreadable.json")

	// Create file and make it unreadable
	os.WriteFile(path, []byte(`{"original": true}`), 0o644)
	os.Chmod(path, 0o000)
	defer os.Chmod(path, 0o644)

	// Should fail because we can't read the file to compare/backup
	err := writeWithBackup(path, []byte(`{"updated": true}`))
	if err == nil {
		t.Error("expected error when file is unreadable, got nil")
	}
}

// TestWriteWithBackup_RapidSuccessiveWrites verifies backup works with multiple writes
// within the same second (timestamp collision scenario).
func TestWriteWithBackup_RapidSuccessiveWrites(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "rapid.json")

	// Create initial file
	os.WriteFile(path, []byte(`{"v": 0}`), 0o644)

	// Rapid successive writes
	for i := 1; i <= 3; i++ {
		data := []byte(fmt.Sprintf(`{"v": %d}`, i))
		if err := writeWithBackup(path, data); err != nil {
			t.Fatalf("write %d failed: %v", i, err)
		}
	}

	// Verify final content
	content, _ := os.ReadFile(path)
	if string(content) != `{"v": 3}` {
		t.Errorf("expected final content {\"v\": 3}, got %s", string(content))
	}

	// Verify at least one backup exists
	entries, _ := os.ReadDir(backupDir())
	var backupCount int
	for _, e := range entries {
		if len(e.Name()) > len("rapid.json.") && e.Name()[:len("rapid.json.")] == "rapid.json." {
			backupCount++
		}
	}
	if backupCount == 0 {
		t.Error("expected at least one backup file from rapid writes")
	}
}

// TestWriteWithBackup_BackupDirIsFile verifies error when backup directory path is a file.
func TestWriteWithBackup_BackupDirIsFile(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test modifies system temp directory")
	}

	// Create a file at the backup directory path
	backupPath := backupDir()
	// Clean up any existing directory first
	os.RemoveAll(backupPath)
	// Create a file instead of directory
	os.WriteFile(backupPath, []byte("not a directory"), 0o644)
	defer func() {
		os.Remove(backupPath)
		os.MkdirAll(backupPath, 0o755)
	}()

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.json")
	os.WriteFile(path, []byte(`{"original": true}`), 0o644)

	err := writeWithBackup(path, []byte(`{"updated": true}`))
	if err == nil {
		t.Error("expected error when backup dir is a file, got nil")
	}
}

// TestWriteWithBackup_NoOrphanTempFiles verifies temp files are cleaned up on failure.
func TestWriteWithBackup_NoOrphanTempFiles(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission tests unreliable on Windows")
	}

	tmpDir := t.TempDir()

	// Count existing temp files
	countTempFiles := func() int {
		entries, _ := os.ReadDir(tmpDir)
		count := 0
		for _, e := range entries {
			if len(e.Name()) > 4 && e.Name()[:4] == ".tmp" {
				count++
			}
		}
		return count
	}

	before := countTempFiles()

	// Create a file, then make directory read-only to cause rename failure
	path := filepath.Join(tmpDir, "orphan.json")
	os.WriteFile(path, []byte(`{"v": 1}`), 0o644)

	// Make a subdirectory and try to write there after making parent read-only
	subDir := filepath.Join(tmpDir, "subdir")
	os.MkdirAll(subDir, 0o755)
	subPath := filepath.Join(subDir, "config.json")
	os.WriteFile(subPath, []byte(`{"v": 1}`), 0o644)

	// Make subdir read-only after creating temp file would succeed but rename would fail
	// This is tricky to test - the temp file is created in the same dir, so if we can't
	// rename, we also couldn't create. Let's just verify normal failure cleanup works.

	// Force a failure by making the target a directory
	badPath := filepath.Join(tmpDir, "isdir")
	os.MkdirAll(badPath, 0o755)

	_ = writeWithBackup(badPath, []byte(`{"test": true}`))

	after := countTempFiles()
	if after > before {
		t.Errorf("orphan temp files left behind: before=%d, after=%d", before, after)
	}
}
