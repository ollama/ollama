package launch

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

type codexAppRequestCursor struct {
	mu        sync.Mutex
	start     time.Time
	filterKey string
	files     map[string]int64
	models    map[string]string
	count     uint64
}

var codexAppRequests codexAppRequestCursor

func resetCodexAppRequestCountAt(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	start := time.Now().UTC()
	if err := os.WriteFile(path, []byte(start.Format(time.RFC3339Nano)+"\n"), 0o600); err != nil {
		return err
	}
	codexAppRequests.mu.Lock()
	codexAppRequests.start = start
	codexAppRequests.filterKey = ""
	codexAppRequests.files = make(map[string]int64)
	codexAppRequests.models = make(map[string]string)
	codexAppRequests.count = 0
	codexAppRequests.mu.Unlock()
	return nil
}

func resetCodexAppRegularProfileRequestCount() error {
	path, err := codexAppRegularProfileSessionStartPath()
	if err != nil {
		return err
	}
	return resetCodexAppRequestCountAt(path)
}

func codexAppRegularProfileRequestCount() uint64 {
	startPath, err := codexAppRegularProfileSessionStartPath()
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
	configPath, err := codexConfigPath()
	if err != nil {
		return 0
	}
	// The regular profile contains native Codex and Ollama turns. Filter its
	// user-prompt count by the Ollama-only routing catalog; the proxy's raw API
	// counter would overcount prompts that need multiple tool-loop requests.
	return codexAppRequests.scan(
		filepath.Join(filepath.Dir(configPath), "sessions"),
		start,
		codexAppRegularProfileRoutingModels(configPath),
	)
}

func codexAppRegularProfileSessionStartPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "launch", "chatgpt-session-start"), nil
}

func codexAppRegularProfileRoutingModels(configPath string) map[string]struct{} {
	models := make(map[string]struct{})
	data, err := os.ReadFile(codexAppRoutingCatalogPathForConfig(configPath))
	if err != nil {
		return models
	}
	var catalog struct {
		Models []struct {
			Slug string `json:"slug"`
		} `json:"models"`
	}
	if json.Unmarshal(data, &catalog) != nil {
		return models
	}
	for _, model := range catalog.Models {
		if key := codexAppCatalogModelKey(strings.TrimSpace(model.Slug)); key != "" {
			models[key] = struct{}{}
		}
	}
	return models
}

func (c *codexAppRequestCursor) scan(root string, start time.Time, allowedModels map[string]struct{}) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.scanLocked(root, start, allowedModels, true)
}

func (c *codexAppRequestCursor) scanLocked(root string, start time.Time, allowedModels map[string]struct{}, retryOnTruncate bool) uint64 {
	filterKey := codexAppRequestModelFilterKey(allowedModels)
	if !c.start.Equal(start) || c.filterKey != filterKey || c.files == nil || c.models == nil {
		c.start = start
		c.filterKey = filterKey
		c.files = make(map[string]int64)
		c.models = make(map[string]string)
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
			c.filterKey = filterKey
			c.files = make(map[string]int64)
			c.models = make(map[string]string)
			c.count = 0
			if retryOnTruncate {
				return c.scanLocked(root, start, allowedModels, false)
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
		model := c.models[path]
		for {
			line, readErr := reader.ReadBytes('\n')
			if readErr != nil {
				break
			}
			offset += int64(len(line))
			if turnModel, ok := codexAppLineTurnModel(line); ok {
				model = codexAppCatalogModelKey(turnModel)
			}
			if codexAppLineIsUserRequest(line, start) && codexAppRequestModelAllowed(model, allowedModels) {
				c.count++
			}
		}
		_ = file.Close()
		c.files[path] = offset
		c.models[path] = model
	}
	return c.count
}

func codexAppRequestModelFilterKey(models map[string]struct{}) string {
	if models == nil {
		return "*"
	}
	keys := make([]string, 0, len(models))
	for model := range models {
		keys = append(keys, model)
	}
	slices.Sort(keys)
	return strings.Join(keys, "\x00")
}

func codexAppRequestModelAllowed(model string, allowedModels map[string]struct{}) bool {
	if allowedModels == nil {
		return true
	}
	_, ok := allowedModels[model]
	return ok
}

func codexAppLineTurnModel(line []byte) (string, bool) {
	if !bytes.Contains(line, []byte(`"type":"turn_context"`)) || !bytes.Contains(line, []byte(`"model"`)) {
		return "", false
	}
	var event struct {
		Type    string `json:"type"`
		Payload struct {
			Model string `json:"model"`
		} `json:"payload"`
	}
	if json.Unmarshal(line, &event) != nil || event.Type != "turn_context" {
		return "", false
	}
	model := strings.TrimSpace(event.Payload.Model)
	return model, model != ""
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
