package integrations

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
)

func copyFile(src, dst string) error {
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	// Preserve source file permissions (important for files containing API keys)
	return os.WriteFile(dst, data, info.Mode().Perm())
}

func getBackupDir() string {
	return filepath.Join(os.TempDir(), "ollama-backups")
}

func backupToTmp(srcPath string) (string, error) {
	backupDir := getBackupDir()
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		return "", err
	}

	backupPath := filepath.Join(backupDir, fmt.Sprintf("%s.%d", filepath.Base(srcPath), time.Now().Unix()))
	if err := copyFile(srcPath, backupPath); err != nil {
		return "", err
	}
	return backupPath, nil
}

func atomicWriteJSON(path string, data any) error {
	content, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal failed: %w", err)
	}

	var check any
	if err := json.Unmarshal(content, &check); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}

	var backupPath string
	if existingContent, err := os.ReadFile(path); err == nil {
		if !bytes.Equal(existingContent, content) {
			backupPath, err = backupToTmp(path)
			if err != nil {
				return fmt.Errorf("backup failed: %w", err)
			}
		}
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("create temp failed: %w", err)
	}
	tmpPath := tmp.Name()

	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write failed: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close failed: %w", err)
	}

	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		if backupPath != "" {
			copyFile(backupPath, path)
		}
		return fmt.Errorf("rename failed: %w", err)
	}

	return nil
}

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

func getModelInfo(model string) *api.ShowResponse {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil
	}
	resp, err := client.Show(context.Background(), &api.ShowRequest{Model: model})
	if err != nil {
		return nil
	}
	return resp
}

func getModelContextLength(model string) int {
	const defaultCtx = 64000 // default context is set to 64k to support coding agents
	resp := getModelInfo(model)
	if resp == nil || resp.ModelInfo == nil {
		return defaultCtx
	}
	arch, ok := resp.ModelInfo["general.architecture"].(string)
	if !ok {
		return defaultCtx
	}
	// currently being capped at 128k
	if v, ok := resp.ModelInfo[fmt.Sprintf("%s.context_length", arch)].(float64); ok {
		return min(int(v), 128000)
	}
	return defaultCtx
}

func modelSupportsImages(model string) bool {
	resp := getModelInfo(model)
	if resp == nil {
		return false
	}
	return slices.Contains(resp.Capabilities, "vision")
}
