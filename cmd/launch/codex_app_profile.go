package launch

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"
)

// seedCodexAppOllamaProfileSkills mirrors personal skills into the isolated
// ChatGPT profile. System and plugin-provided skills are managed independently
// by ChatGPT, so copying those directories would mix profile installation and
// account state.
func seedCodexAppOllamaProfileSkills(profileConfigPath string) error {
	regularConfigPath, err := codexConfigPath()
	if err != nil {
		return err
	}
	sourceRoot := filepath.Join(filepath.Dir(regularConfigPath), "skills")
	destinationRoot := filepath.Join(filepath.Dir(profileConfigPath), "skills")

	entries, err := os.ReadDir(sourceRoot)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := os.MkdirAll(destinationRoot, 0o700); err != nil {
		return err
	}
	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == ".system" {
			continue
		}
		source := filepath.Join(sourceRoot, entry.Name())
		if _, err := os.Stat(filepath.Join(source, "SKILL.md")); err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			return err
		}
		if err := mergeCodexAppSkillDirectory(source, filepath.Join(destinationRoot, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}

func mergeCodexAppSkillDirectory(source, destination string) error {
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		if info.Mode()&os.ModeSymlink != 0 {
			link, err := os.Readlink(path)
			if err != nil {
				return err
			}
			if err := os.Remove(target); err != nil && !errors.Is(err, fs.ErrNotExist) {
				return err
			}
			return os.Symlink(link, target)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
}

type codexAppRequestCursor struct {
	mu    sync.Mutex
	start time.Time
	files map[string]int64
	count uint64
}

var codexAppRequests codexAppRequestCursor

func codexAppOllamaProfileSessionStartPath() (string, error) {
	root, err := codexAppOllamaProfileRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, codexAppSessionStartFilename), nil
}

func resetCodexAppOllamaProfileRequestCount() error {
	path, err := codexAppOllamaProfileSessionStartPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	start := time.Now().UTC()
	if err := os.WriteFile(path, []byte(start.Format(time.RFC3339Nano)+"\n"), 0o600); err != nil {
		return err
	}
	codexAppRequests.mu.Lock()
	codexAppRequests.start = start
	codexAppRequests.files = make(map[string]int64)
	codexAppRequests.count = 0
	codexAppRequests.mu.Unlock()
	return nil
}

func codexAppOllamaProfileRequestCount() uint64 {
	startPath, err := codexAppOllamaProfileSessionStartPath()
	if err != nil {
		return 0
	}
	data, err := os.ReadFile(startPath)
	if err != nil {
		return 0
	}
	start, err := time.Parse(time.RFC3339Nano, string(bytes.TrimSpace(data)))
	if err != nil {
		return 0
	}
	codexHome, err := codexAppOllamaProfileCodexHome()
	if err != nil {
		return 0
	}
	return codexAppRequests.scan(filepath.Join(codexHome, "sessions"), start)
}

func (c *codexAppRequestCursor) scan(root string, start time.Time) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.scanLocked(root, start, true)
}

func (c *codexAppRequestCursor) scanLocked(root string, start time.Time, retryOnTruncate bool) uint64 {
	if !c.start.Equal(start) || c.files == nil {
		c.start = start
		c.files = make(map[string]int64)
		c.count = 0
	}

	paths := make([]string, 0)
	_ = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || filepath.Ext(path) != ".jsonl" {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	slices.Sort(paths)
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil || info.ModTime().Before(start) {
			continue
		}
		offset := c.files[path]
		if info.Size() < offset {
			c.start = start
			c.files = make(map[string]int64)
			c.count = 0
			if retryOnTruncate {
				return c.scanLocked(root, start, false)
			}
			return 0
		}
		file, err := os.Open(path)
		if err != nil {
			continue
		}
		if _, err := file.Seek(offset, io.SeekStart); err != nil {
			_ = file.Close()
			continue
		}
		reader := bufio.NewReader(file)
		for {
			line, readErr := reader.ReadBytes('\n')
			if readErr != nil {
				break
			}
			offset += int64(len(line))
			if codexAppLineIsUserRequest(line, start) {
				c.count++
			}
		}
		_ = file.Close()
		c.files[path] = offset
	}
	return c.count
}

func codexAppLineIsUserRequest(line []byte, start time.Time) bool {
	if !bytes.Contains(line, []byte(`"role":"user"`)) || !bytes.Contains(line, []byte(`"user.text"`)) {
		return false
	}
	var event struct {
		Timestamp time.Time `json:"timestamp"`
		Type      string    `json:"type"`
		Payload   struct {
			Type     string `json:"type"`
			Role     string `json:"role"`
			Metadata struct {
				ContentItemKinds []string `json:"content_item_kinds"`
			} `json:"internal_chat_message_metadata_passthrough"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(line, &event); err != nil {
		return false
	}
	return !event.Timestamp.Before(start) &&
		event.Type == "response_item" &&
		event.Payload.Type == "message" &&
		event.Payload.Role == "user" &&
		slices.Contains(event.Payload.Metadata.ContentItemKinds, "user.text")
}
