package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

func readJSONFile(path string) (map[string]any, error) {
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

func backupDir() string {
	return filepath.Join(os.TempDir(), "ollama-backups")
}

func backupToTmp(srcPath string) (string, error) {
	dir := backupDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}

	backupPath := filepath.Join(dir, fmt.Sprintf("%s.%d", filepath.Base(srcPath), time.Now().Unix()))
	if err := copyFile(srcPath, backupPath); err != nil {
		return "", err
	}
	return backupPath, nil
}

// writeWithBackup writes data to path via temp file + rename, backing up any existing file first
func writeWithBackup(path string, data []byte) error {
	var backupPath string
	// backup must be created before any writes to the target file
	if existingContent, err := os.ReadFile(path); err == nil {
		if !bytes.Equal(existingContent, data) {
			backupPath, err = backupToTmp(path)
			if err != nil {
				return fmt.Errorf("backup failed: %w", err)
			}
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
