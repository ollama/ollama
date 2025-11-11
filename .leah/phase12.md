# PHASE 12: PLUGIN SİSTEMİ (Genişletilebilir Mimari)

## 📋 HEDEFLER
1. ✅ Plugin loader sistemi
2. ✅ Plugin API
3. ✅ Hooks & Events
4. ✅ Plugin marketplace (community)
5. ✅ Hot reload (restart olmadan plugin yükleme)
6. ✅ Plugin security & sandboxing

## 🏗️ MİMARİ

### Plugin Interface
```go
type Plugin interface {
    Name() string
    Version() string
    Description() string

    Initialize(config map[string]interface{}) error
    OnChatMessage(msg *Message) error
    OnToolCall(tool *ToolCall) (*ToolResult, error)
    OnBeforeResponse(resp *Response) (*Response, error)
    OnAfterResponse(resp *Response) error

    Shutdown() error
}
```

### Plugin Structure
```
plugins/
├── my-plugin/
│   ├── plugin.yaml      # Metadata
│   ├── plugin.so        # Compiled Go plugin
│   ├── main.go          # Plugin source
│   └── README.md
```

### Plugin Manifest (plugin.yaml)
```yaml
name: my-awesome-plugin
version: 1.0.0
author: John Doe
description: Does awesome things

permissions:
  - read_messages
  - write_files
  - network_access

hooks:
  - on_chat_message
  - on_tool_call

config:
  api_key:
    type: string
    required: true
  enabled:
    type: boolean
    default: true
```

## 📁 DOSYALAR

### 1. Plugin Manager
**Dosya:** `/home/user/ollama/plugins/manager.go` (YENİ)

```go
package plugins

import (
    "plugin"
    "sync"
)

type PluginManager struct {
    plugins map[string]*LoadedPlugin
    mu      sync.RWMutex
    hooks   *HookRegistry
}

type LoadedPlugin struct {
    Plugin   Plugin
    Metadata *PluginMetadata
    Enabled  bool
}

type PluginMetadata struct {
    Name        string   `yaml:"name"`
    Version     string   `yaml:"version"`
    Author      string   `yaml:"author"`
    Description string   `yaml:"description"`
    Permissions []string `yaml:"permissions"`
    Hooks       []string `yaml:"hooks"`
}

func NewPluginManager() *PluginManager {
    return &PluginManager{
        plugins: make(map[string]*LoadedPlugin),
        hooks:   NewHookRegistry(),
    }
}

func (pm *PluginManager) LoadPlugin(path string) error {
    pm.mu.Lock()
    defer pm.mu.Unlock()

    // Load metadata
    metadataPath := filepath.Join(path, "plugin.yaml")
    metadata, err := pm.loadMetadata(metadataPath)
    if err != nil {
        return err
    }

    // Load compiled plugin (.so file)
    pluginPath := filepath.Join(path, "plugin.so")
    p, err := plugin.Open(pluginPath)
    if err != nil {
        return err
    }

    // Lookup "Plugin" symbol
    symPlugin, err := p.Lookup("Plugin")
    if err != nil {
        return err
    }

    // Type assert to Plugin interface
    pluginInstance, ok := symPlugin.(Plugin)
    if !ok {
        return errors.New("invalid plugin: does not implement Plugin interface")
    }

    // Initialize plugin
    if err := pluginInstance.Initialize(metadata.Config); err != nil {
        return err
    }

    // Register plugin
    pm.plugins[metadata.Name] = &LoadedPlugin{
        Plugin:   pluginInstance,
        Metadata: metadata,
        Enabled:  true,
    }

    // Register hooks
    for _, hook := range metadata.Hooks {
        pm.hooks.Register(hook, metadata.Name, pluginInstance)
    }

    log.Printf("Plugin loaded: %s v%s", metadata.Name, metadata.Version)

    return nil
}

func (pm *PluginManager) UnloadPlugin(name string) error {
    pm.mu.Lock()
    defer pm.mu.Unlock()

    loaded, ok := pm.plugins[name]
    if !ok {
        return errors.New("plugin not found")
    }

    // Shutdown plugin
    if err := loaded.Plugin.Shutdown(); err != nil {
        return err
    }

    // Unregister hooks
    pm.hooks.UnregisterPlugin(name)

    // Remove from loaded plugins
    delete(pm.plugins, name)

    log.Printf("Plugin unloaded: %s", name)

    return nil
}

func (pm *PluginManager) CallHook(hookName string, data interface{}) error {
    return pm.hooks.Call(hookName, data)
}
```

### 2. Hook Registry
**Dosya:** `/home/user/ollama/plugins/hooks.go` (YENİ)

```go
package plugins

type HookRegistry struct {
    hooks map[string][]HookHandler
    mu    sync.RWMutex
}

type HookHandler struct {
    PluginName string
    Plugin     Plugin
}

func NewHookRegistry() *HookRegistry {
    return &HookRegistry{
        hooks: make(map[string][]HookHandler),
    }
}

func (hr *HookRegistry) Register(hookName, pluginName string, plugin Plugin) {
    hr.mu.Lock()
    defer hr.mu.Unlock()

    hr.hooks[hookName] = append(hr.hooks[hookName], HookHandler{
        PluginName: pluginName,
        Plugin:     plugin,
    })
}

func (hr *HookRegistry) Call(hookName string, data interface{}) error {
    hr.mu.RLock()
    handlers := hr.hooks[hookName]
    hr.mu.RUnlock()

    for _, handler := range handlers {
        switch hookName {
        case "on_chat_message":
            if msg, ok := data.(*Message); ok {
                if err := handler.Plugin.OnChatMessage(msg); err != nil {
                    log.Printf("Plugin %s hook error: %v", handler.PluginName, err)
                }
            }
        case "on_tool_call":
            if tool, ok := data.(*ToolCall); ok {
                result, err := handler.Plugin.OnToolCall(tool)
                if err != nil {
                    log.Printf("Plugin %s hook error: %v", handler.PluginName, err)
                }
                // Handle result...
                _ = result
            }
        }
    }

    return nil
}
```

### 3. Example Plugin
**Dosya:** `/home/user/ollama/plugins/examples/hello-world/main.go` (YENİ)

```go
package main

import (
    "log"
)

type HelloWorldPlugin struct {
    config map[string]interface{}
}

func (p *HelloWorldPlugin) Name() string {
    return "hello-world"
}

func (p *HelloWorldPlugin) Version() string {
    return "1.0.0"
}

func (p *HelloWorldPlugin) Description() string {
    return "A simple hello world plugin"
}

func (p *HelloWorldPlugin) Initialize(config map[string]interface{}) error {
    p.config = config
    log.Println("HelloWorldPlugin initialized")
    return nil
}

func (p *HelloWorldPlugin) OnChatMessage(msg *Message) error {
    log.Printf("HelloWorldPlugin: Received message from %s: %s", msg.Role, msg.Content)
    return nil
}

func (p *HelloWorldPlugin) OnToolCall(tool *ToolCall) (*ToolResult, error) {
    return nil, nil
}

func (p *HelloWorldPlugin) OnBeforeResponse(resp *Response) (*Response, error) {
    // Modify response before sending to user
    return resp, nil
}

func (p *HelloWorldPlugin) OnAfterResponse(resp *Response) error {
    return nil
}

func (p *HelloWorldPlugin) Shutdown() error {
    log.Println("HelloWorldPlugin shutting down")
    return nil
}

// Export plugin instance
var Plugin HelloWorldPlugin
```

### 4. Plugin UI
**Dosya:** `/home/user/ollama/app/ui/app/src/components/PluginManager.tsx` (YENİ)

```typescript
export function PluginManager() {
  const { data: plugins } = usePlugins();
  const installPlugin = useInstallPlugin();
  const togglePlugin = useTogglePlugin();
  const uninstallPlugin = useUninstallPlugin();

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Plugins</h2>
        <button
          onClick={() => {/* Open marketplace */}}
          className="px-4 py-2 bg-indigo-600 text-white rounded-md"
        >
          Browse Marketplace
        </button>
      </div>

      {/* Installed Plugins */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {plugins?.map(plugin => (
          <div key={plugin.name} className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="font-semibold">{plugin.name}</h3>
                <p className="text-sm text-gray-500">v{plugin.version}</p>
              </div>
              <Switch
                checked={plugin.enabled}
                onChange={() => togglePlugin.mutate(plugin.name)}
              />
            </div>

            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              {plugin.description}
            </p>

            <div className="flex gap-2 text-xs text-gray-500">
              <span>By {plugin.author}</span>
              <span>•</span>
              <span>{plugin.downloads} downloads</span>
            </div>

            <div className="mt-3 flex gap-2">
              <button className="text-sm text-indigo-600">Configure</button>
              <button
                onClick={() => uninstallPlugin.mutate(plugin.name)}
                className="text-sm text-red-600"
              >
                Uninstall
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Plugin Marketplace Preview */}
      <div>
        <h3 className="text-xl font-bold mb-4">Available Plugins</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Community plugins from registry */}
        </div>
      </div>
    </div>
  );
}
```

### 5. Plugin Marketplace API
**Dosya:** `/home/user/ollama/plugins/marketplace.go` (YENİ)

```go
type MarketplaceClient struct {
    baseURL string
}

func (mc *MarketplaceClient) ListPlugins(category string) ([]*PluginListing, error) {
    url := fmt.Sprintf("%s/plugins?category=%s", mc.baseURL, category)

    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var plugins []*PluginListing
    if err := json.NewDecoder(resp.Body).Decode(&plugins); err != nil {
        return nil, err
    }

    return plugins, nil
}

func (mc *MarketplaceClient) InstallPlugin(name, version string) error {
    // Download plugin package
    url := fmt.Sprintf("%s/plugins/%s/%s/download", mc.baseURL, name, version)

    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    // Save to plugins directory
    pluginDir := filepath.Join("plugins", name)
    if err := os.MkdirAll(pluginDir, 0755); err != nil {
        return err
    }

    // Extract tarball
    return extractPlugin(resp.Body, pluginDir)
}

type PluginListing struct {
    Name        string   `json:"name"`
    Version     string   `json:"version"`
    Author      string   `json:"author"`
    Description string   `json:"description"`
    Category    string   `json:"category"`
    Downloads   int      `json:"downloads"`
    Rating      float64  `json:"rating"`
    Tags        []string `json:"tags"`
}
```

## 📊 PLUGIN ÖRNEKLERİ

### Community Plugin Ideas
1. **Translator Plugin** - Multi-language translation
2. **Code Formatter** - Auto-format code blocks
3. **Database Query** - SQL helper & visualization
4. **Git Integration** - Commit, push, PR creation
5. **Diagram Generator** - Mermaid/PlantUML diagrams
6. **Math Solver** - Advanced math with visualization
7. **PDF Reader** - Enhanced PDF parsing
8. **Calendar Integration** - Google Calendar, Outlook
9. **Email Plugin** - Send/read emails
10. **Slack Integration** - Send messages, read channels

## ✅ BAŞARI KRİTERLERİ
1. ✅ Plugin loader çalışıyor
2. ✅ Plugins hot-reload destekliyor
3. ✅ Hook system çalışıyor
4. ✅ Example plugin çalışıyor
5. ✅ Marketplace entegrasyonu
6. ✅ Security sandboxing aktif

---

## 🎉 TÜM PHASE'LER TAMAMLANDI!

Bu 12 phase'i sırayla implement ederseniz, tam özellikli, modern, performanslı ve genişletilebilir bir AI chat uygulaması elde edeceksiniz.

### Toplam Özellikler:
- ✅ Multi-API support (OpenAI, Anthropic, Google, Groq, custom)
- ✅ Context management & auto-summarization
- ✅ Cost tracking & analytics
- ✅ Rules & todo systems (.leah)
- ✅ Modern UI/UX (glassmorphism, animations)
- ✅ Multi-model chat & comparison
- ✅ Prompt templates library
- ✅ RAG system (document upload)
- ✅ Performance monitoring
- ✅ Model benchmarking
- ✅ Workspace integration (file operations)
- ✅ Agent system (dual-model)
- ✅ Voice input/output
- ✅ Image generation
- ✅ Web scraping
- ✅ Code execution
- ✅ Plugin system

**Toplam Kod Satırı Tahmini:** ~50,000 lines
**Tahmini Geliştirme Süresi:** 3-6 ay (tek developer)
**Performans:** Production-ready, scalable

Good luck! 🚀
