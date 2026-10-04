//go:build windows || darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/server"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/updater"
	"github.com/ollama/ollama/app/version"
	"github.com/ollama/ollama/envconfig"
)

// settingsPane names a pane in the settings window. The default pane is the
// one people viewed last.
type settingsPane string

const (
	settingsPaneDefault settingsPane = ""
	settingsPaneAccount settingsPane = "account"
	settingsPaneApps    settingsPane = "apps"
)

// contextLengths are the context lengths offered in Settings. Zero lets the
// server choose a context length from available memory.
var contextLengths = []int{0, 4096, 8192, 16384, 32768, 65536, 131072, 262144}

// settingsState is what the settings window shows for app and server
// settings. The JSON form is read by the native settings window.
type settingsState struct {
	Version     string `json:"version"`
	UpdateReady bool   `json:"updateReady"`
	AutoUpdate  bool   `json:"autoUpdate"`

	Expose bool `json:"expose"`

	ModelsPath          string `json:"modelsPath"`
	DefaultModelsPath   string `json:"defaultModelsPath"`
	ModelsPathIsDefault bool   `json:"modelsPathIsDefault"`

	// ContextLength is zero when the server picks it from available memory;
	// DefaultContextLength is that choice, or zero when it is not known yet.
	ContextLength        int   `json:"contextLength"`
	DefaultContextLength int   `json:"defaultContextLength,omitempty"`
	ContextLengths       []int `json:"contextLengths"`

	CloudEnabled bool `json:"cloudEnabled"`
	// CloudDisabledByEnv is set when OLLAMA_NO_CLOUD turns cloud off, so the
	// setting cannot be changed from the app.
	CloudDisabledByEnv bool `json:"cloudDisabledByEnv"`

	// ChatCount is how many chats earlier versions of the app saved, which
	// can be exported.
	ChatCount int `json:"chatCount"`
}

// updateController is the part of the updater that settings changes drive.
type updateController interface {
	CancelOngoingDownload()
	TriggerImmediateCheck()
}

// settingsController applies changes made in the settings window. Every
// native settings window uses it so settings behave the same on each
// platform. Its methods may block on disk, network, or a server restart, so
// call them off the UI thread.
type settingsController struct {
	store         *store.Store
	restartServer func()
	updater       updateController
	// notifyUpdate shows that a downloaded update is ready to install.
	notifyUpdate func()

	// updateReady and defaultContextLength are replaced in tests.
	updateReady          func() bool
	defaultContextLength func(context.Context) int

	// mu serializes read-modify-write cycles of the stored settings.
	mu sync.Mutex

	// contextLengthMu guards serverContextLength, the server's default
	// context length, which is kept until the server restarts.
	contextLengthMu     sync.Mutex
	serverContextLength int
}

// State returns the current app and server settings.
func (c *settingsController) State(ctx context.Context) (settingsState, error) {
	stored, err := c.store.Settings()
	if err != nil {
		return settingsState{}, err
	}
	cloudDisabled, cloudSource, err := c.store.CloudStatus()
	if err != nil {
		return settingsState{}, err
	}
	chatCount, err := c.store.ChatCount()
	if err != nil {
		return settingsState{}, err
	}

	updateReady := c.updateReady
	if updateReady == nil {
		updateReady = func() bool { return updater.IsUpdatePending() || updater.UpdateDownloaded }
	}

	return settingsState{
		Version:              version.Version,
		UpdateReady:          updateReady(),
		AutoUpdate:           stored.AutoUpdateEnabled,
		Expose:               stored.Expose,
		ModelsPath:           stored.Models,
		DefaultModelsPath:    defaultModelsPath(),
		ModelsPathIsDefault:  isDefaultModelsPath(stored.Models),
		ContextLength:        stored.ContextLength,
		DefaultContextLength: c.defaultServerContextLength(ctx),
		ContextLengths:       contextLengths,
		CloudEnabled:         !cloudDisabled,
		CloudDisabledByEnv:   cloudSource == "env" || cloudSource == "both",
		ChatCount:            chatCount,
	}, nil
}

// ExportChats saves every chat to dir as Markdown files and returns how many
// it saved.
func (c *settingsController) ExportChats(dir string) (int, error) {
	return exportChats(c.store, dir)
}

// SetAutoUpdate turns automatic update downloads on or off.
func (c *settingsController) SetAutoUpdate(enabled bool) error {
	changed := false
	err := c.update(func(s *store.Settings) bool {
		changed = s.AutoUpdateEnabled != enabled
		s.AutoUpdateEnabled = enabled
		return false
	})
	if err != nil || !changed || c.updater == nil {
		return err
	}

	if !enabled {
		c.updater.CancelOngoingDownload()
		return nil
	}
	updateReady := c.updateReady
	if updateReady == nil {
		updateReady = func() bool { return updater.IsUpdatePending() || updater.UpdateDownloaded }
	}
	if updateReady() {
		if c.notifyUpdate != nil {
			c.notifyUpdate()
		}
		return nil
	}
	c.updater.TriggerImmediateCheck()
	return nil
}

// SetExpose makes the server listen on all network interfaces, or only on
// this computer.
func (c *settingsController) SetExpose(enabled bool) error {
	return c.update(func(s *store.Settings) bool {
		if s.Expose == enabled {
			return false
		}
		s.Expose = enabled
		return true
	})
}

// SetModelsPath changes where the server stores models. An empty path
// restores the default location.
func (c *settingsController) SetModelsPath(path string) error {
	if path != "" {
		path = filepath.Clean(path)
		info, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf("the folder %q can't be used: %w", path, err)
		}
		if !info.IsDir() {
			return fmt.Errorf("%q is not a folder", path)
		}
		if isDefaultModelsPath(path) {
			path = ""
		}
	}
	return c.update(func(s *store.Settings) bool {
		current := s.Models
		if isDefaultModelsPath(current) {
			current = ""
		}
		if current == path {
			return false
		}
		s.Models = path
		return true
	})
}

// SetContextLength changes the server's default context length. Zero lets
// the server choose from available memory.
func (c *settingsController) SetContextLength(length int) error {
	if !slices.Contains(contextLengths, length) {
		return fmt.Errorf("unsupported context length %d", length)
	}
	return c.update(func(s *store.Settings) bool {
		if s.ContextLength == length {
			return false
		}
		s.ContextLength = length
		return true
	})
}

// SetCloudEnabled turns cloud models and web search on or off.
func (c *settingsController) SetCloudEnabled(enabled bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	disabled, source, err := c.store.CloudStatus()
	if err != nil {
		return err
	}
	if source == "env" || source == "both" {
		return errors.New("OLLAMA_NO_CLOUD is set, so cloud can't be turned on from Settings")
	}
	if disabled != enabled {
		return nil
	}
	if err := c.store.SetCloudEnabled(enabled); err != nil {
		return err
	}
	c.restart()
	return nil
}

// update applies change to the stored settings and restarts the server when
// change reports that the server must read the new settings.
func (c *settingsController) update(change func(*store.Settings) bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	s, err := c.store.Settings()
	if err != nil {
		return err
	}
	restart := change(&s)
	if err := c.store.SetSettings(s); err != nil {
		return err
	}
	if restart {
		c.restart()
	}
	return nil
}

func (c *settingsController) restart() {
	if c.restartServer != nil {
		c.restartServer()
	}
	// The new server picks its own default context length.
	c.contextLengthMu.Lock()
	c.serverContextLength = 0
	c.contextLengthMu.Unlock()
}

// defaultServerContextLength returns the context length the server chose
// from available memory, or zero until the server reports one.
func (c *settingsController) defaultServerContextLength(ctx context.Context) int {
	c.contextLengthMu.Lock()
	defer c.contextLengthMu.Unlock()
	if c.serverContextLength == 0 {
		lookup := c.defaultContextLength
		if lookup == nil {
			lookup = serverDefaultContextLength
		}
		c.serverContextLength = lookup(ctx)
	}
	return c.serverContextLength
}

func defaultModelsPath() string {
	if dir := os.Getenv("OLLAMA_MODELS"); dir != "" {
		return dir
	}
	return envconfig.Models()
}

// isDefaultModelsPath reports whether path is the default models folder.
// Earlier versions of the app saved the default folder's full path.
func isDefaultModelsPath(path string) bool {
	if path == "" {
		return true
	}
	path, defaultPath := filepath.Clean(path), filepath.Clean(defaultModelsPath())
	if path == defaultPath {
		return true
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return false
	}
	resolvedDefault, err := filepath.EvalSymlinks(defaultPath)
	return err == nil && resolved == resolvedDefault
}

// serverDefaultContextLength reports the context length the running server
// chose from available memory, or zero when it has not reported one yet.
func serverDefaultContextLength(ctx context.Context) int {
	ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	// The server logs its choice before it serves requests. Until it serves,
	// the log may still be the one from the server it replaced, which could
	// have stopped before finding the GPU.
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return 0
	}
	if _, err := client.Version(ctx); err != nil {
		return 0
	}
	info, err := server.GetInferenceInfo(ctx)
	if err != nil {
		slog.Debug("default context length unavailable", "error", err)
		return 0
	}
	return info.DefaultContextLength
}
