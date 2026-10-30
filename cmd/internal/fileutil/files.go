// Package fileutil provides small shared helpers for reading JSON files
// and writing config files with backup-on-overwrite semantics.
package fileutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Keep a bounded number of backups per file so config backups do not grow
// without limit. We keep the 5 most recent backups and do not pin the oldest.
const maxBackupsPerFile = 5

// ReadJSON reads a JSON object file into a generic map.
func ReadJSON(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func copyFile(src, dst string) error {
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, info.Mode().Perm())
}

// BackupDir returns the shared backup root used before overwriting files.
func BackupDir() string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".ollama", "backup")
	}
	return filepath.Join(os.TempDir(), "ollama-backup")
}

func writeBackupCopy(srcPath string, integration string) (string, error) {
	dir := BackupDir()
	name := filepath.Base(srcPath)
	if integration != "" {
		dir = filepath.Join(dir, integration)
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}

	backupPath := filepath.Join(dir, fmt.Sprintf("%s.%d", name, time.Now().Unix()))
	if err := copyFile(srcPath, backupPath); err != nil {
		return "", err
	}
	pruneOldBackups(dir, name, maxBackupsPerFile)
	return backupPath, nil
}

// WriteWithBackup writes data to path via temp file + rename, backing up any
// existing file first. Callers may optionally pass one integration name to
// store backups under BackupDir()/.../<integration>/.
func WriteWithBackup(path string, data []byte, integration ...string) error {
	backupIntegration := ""
	if len(integration) > 0 {
		backupIntegration = integration[0]
	}

	var backupPath string
	// backup must be created before any writes to the target file
	if existingContent, err := os.ReadFile(path); err == nil {
		if bytes.Equal(existingContent, data) {
			return nil
		}
		backupPath, err = writeBackupCopy(path, backupIntegration)
		if err != nil {
			return fmt.Errorf("backup failed: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("read existing file: %w", err)
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp failed: %w", err)
	}
	tmpPath := tmp.Name()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("write failed: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("sync failed: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close failed: %w", err)
	}

	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		if backupPath != "" {
			_ = copyFile(backupPath, path)
		}
		return fmt.Errorf("rename failed: %w", err)
	}

	return nil
}

func pruneOldBackups(dir, name string, keep int) {
	if keep < 1 {
		return
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}

	type backupEntry struct {
		name      string
		timestamp int64
	}

	prefix := name + "."
	backups := make([]backupEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), prefix) {
			continue
		}

		timestamp, err := strconv.ParseInt(strings.TrimPrefix(entry.Name(), prefix), 10, 64)
		if err != nil {
			continue
		}

		backups = append(backups, backupEntry{
			name:      entry.Name(),
			timestamp: timestamp,
		})
	}

	if len(backups) <= keep {
		return
	}

	sort.Slice(backups, func(i, j int) bool {
		if backups[i].timestamp != backups[j].timestamp {
			return backups[i].timestamp > backups[j].timestamp
		}
		return backups[i].name > backups[j].name
	})

	for _, backup := range backups[keep:] {
		_ = os.Remove(filepath.Join(dir, backup.name))
	}
}
