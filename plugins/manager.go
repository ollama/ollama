package plugins

import (
	"fmt"
	"sync"
)

// Plugin interface
type Plugin interface {
	Name() string
	Version() string
	Description() string
	Initialize(config map[string]interface{}) error
	OnChatMessage(msg *Message) error
	Shutdown() error
}

// Message represents a chat message
type Message struct {
	Role    string
	Content string
}

// Manager manages plugins
type Manager struct {
	plugins map[string]*LoadedPlugin
	mu      sync.RWMutex
}

// LoadedPlugin represents a loaded plugin
type LoadedPlugin struct {
	Plugin  Plugin
	Enabled bool
}

// NewManager creates a new plugin manager
func NewManager() *Manager {
	return &Manager{
		plugins: make(map[string]*LoadedPlugin),
	}
}

// LoadPlugin loads a plugin
func (pm *Manager) LoadPlugin(plugin Plugin, config map[string]interface{}) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if err := plugin.Initialize(config); err != nil {
		return fmt.Errorf("failed to initialize plugin: %w", err)
	}

	pm.plugins[plugin.Name()] = &LoadedPlugin{
		Plugin:  plugin,
		Enabled: true,
	}

	return nil
}

// UnloadPlugin unloads a plugin
func (pm *Manager) UnloadPlugin(name string) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	loaded, ok := pm.plugins[name]
	if !ok {
		return fmt.Errorf("plugin not found: %s", name)
	}

	if err := loaded.Plugin.Shutdown(); err != nil {
		return err
	}

	delete(pm.plugins, name)
	return nil
}

// OnChatMessage notifies all plugins of a chat message
func (pm *Manager) OnChatMessage(msg *Message) error {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	for _, loaded := range pm.plugins {
		if !loaded.Enabled {
			continue
		}

		if err := loaded.Plugin.OnChatMessage(msg); err != nil {
			// Log error but continue to other plugins
			fmt.Printf("Plugin %s error: %v\n", loaded.Plugin.Name(), err)
		}
	}

	return nil
}

// ListPlugins returns all loaded plugins
func (pm *Manager) ListPlugins() []Plugin {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	plugins := make([]Plugin, 0, len(pm.plugins))
	for _, loaded := range pm.plugins {
		plugins = append(plugins, loaded.Plugin)
	}

	return plugins
}
