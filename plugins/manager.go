package plugins

import (
	"fmt"
	"sync"
	"time"
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
	plugins       map[string]*LoadedPlugin
	mu            sync.RWMutex
	watcherActive bool
	stopWatcher   chan bool
}

// LoadedPlugin represents a loaded plugin
type LoadedPlugin struct {
	Plugin    Plugin
	Enabled   bool
	LoadedAt  time.Time
	Config    map[string]interface{}
	Version   string
}

// NewManager creates a new plugin manager
func NewManager() *Manager {
	return &Manager{
		plugins:     make(map[string]*LoadedPlugin),
		stopWatcher: make(chan bool),
	}
}

// StartHotReload starts watching for plugin changes and reloads them automatically
func (pm *Manager) StartHotReload(pluginDir string, interval time.Duration) {
	pm.mu.Lock()
	if pm.watcherActive {
		pm.mu.Unlock()
		return
	}
	pm.watcherActive = true
	pm.mu.Unlock()

	go pm.watchPlugins(pluginDir, interval)
}

// StopHotReload stops the hot reload watcher
func (pm *Manager) StopHotReload() {
	pm.mu.Lock()
	if !pm.watcherActive {
		pm.mu.Unlock()
		return
	}
	pm.watcherActive = false
	pm.mu.Unlock()

	pm.stopWatcher <- true
}

// watchPlugins watches the plugin directory for changes
func (pm *Manager) watchPlugins(pluginDir string, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-pm.stopWatcher:
			return
		case <-ticker.C:
			// Check for plugin updates
			pm.checkPluginUpdates(pluginDir)
		}
	}
}

// checkPluginUpdates checks if any plugins need to be reloaded
func (pm *Manager) checkPluginUpdates(pluginDir string) {
	// In production, this would:
	// 1. Scan plugin directory for .so/.dll files
	// 2. Compare modification times
	// 3. Reload plugins that have changed
	// 4. Handle version conflicts

	// For now, this is a placeholder
	fmt.Println("Checking for plugin updates...")
}

// ReloadPlugin reloads a specific plugin
func (pm *Manager) ReloadPlugin(name string, newPlugin Plugin, config map[string]interface{}) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Get existing plugin
	loaded, ok := pm.plugins[name]
	if !ok {
		return fmt.Errorf("plugin not found: %s", name)
	}

	// Shutdown old plugin
	if err := loaded.Plugin.Shutdown(); err != nil {
		return fmt.Errorf("failed to shutdown old plugin: %w", err)
	}

	// Initialize new plugin
	if err := newPlugin.Initialize(config); err != nil {
		return fmt.Errorf("failed to initialize new plugin: %w", err)
	}

	// Replace plugin
	pm.plugins[name] = &LoadedPlugin{
		Plugin:   newPlugin,
		Enabled:  true,
		LoadedAt: time.Now(),
		Config:   config,
		Version:  newPlugin.Version(),
	}

	fmt.Printf("Plugin %s reloaded to version %s\n", name, newPlugin.Version())
	return nil
}

// LoadPlugin loads a plugin
func (pm *Manager) LoadPlugin(plugin Plugin, config map[string]interface{}) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if err := plugin.Initialize(config); err != nil {
		return fmt.Errorf("failed to initialize plugin: %w", err)
	}

	pm.plugins[plugin.Name()] = &LoadedPlugin{
		Plugin:   plugin,
		Enabled:  true,
		LoadedAt: time.Now(),
		Config:   config,
		Version:  plugin.Version(),
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
