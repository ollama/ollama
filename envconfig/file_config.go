package envconfig

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/BurntSushi/toml"
	"golang.org/x/exp/slog"
)

// Config represents the TOML configuration structure
type Config struct {
	Server struct {
		Host     string   `toml:"host"`
		Origins  []string `toml:"origins"`
	} `toml:"server"`

	Models struct {
		Path            string `toml:"path"`
		KeepAlive       string `toml:"keep_alive"`
		MaxLoadedModels int    `toml:"max_loaded_models"`
		MaxQueue        int    `toml:"max_queue"`
	} `toml:"models"`

	Performance struct {
		NumParallel     int    `toml:"num_parallel"`
		GpuOverhead     int64  `toml:"gpu_overhead"`
		FlashAttention  bool   `toml:"flash_attention"`
		KvCacheType     string `toml:"kv_cache_type"`
	} `toml:"performance"`

	Logging struct {
		Debug bool `toml:"debug"`
	} `toml:"logging"`
}

var (
	configOnce sync.Once
	config     *Config
	configPath string
)

// GetConfigPaths returns the list of possible config file paths for the current OS
func GetConfigPaths() []string {
	var paths []string

	switch runtime.GOOS {
	case "windows":
		if appData := os.Getenv("APPDATA"); appData != "" {
			paths = append(paths, filepath.Join(appData, "ollama", "config.toml"))
		}
		if userProfile := os.Getenv("USERPROFILE"); userProfile != "" {
			paths = append(paths, filepath.Join(userProfile, ".ollama", "config.toml"))
		}
	case "darwin":
		home, err := os.UserHomeDir()
		if err == nil {
			paths = append(paths,
				filepath.Join(home, "Library", "Application Support", "ollama", "config.toml"),
				filepath.Join(home, ".config", "ollama", "config.toml"),
				filepath.Join(home, ".ollama", "config.toml"),
			)
		}
	default: // Linux and others
		if xdgConfig := os.Getenv("XDG_CONFIG_HOME"); xdgConfig != "" {
			paths = append(paths, filepath.Join(xdgConfig, "ollama", "config.toml"))
		}
		home, err := os.UserHomeDir()
		if err == nil {
			paths = append(paths,
				filepath.Join(home, ".config", "ollama", "config.toml"),
				filepath.Join(home, ".ollama", "config.toml"),
			)
		}
		paths = append(paths, "/etc/ollama/config.toml")
	}

	return paths
}

// loadConfig loads the first available configuration file
func loadConfig() (*Config, string, error) {
	paths := GetConfigPaths()
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			var cfg Config
			if _, err := toml.DecodeFile(path, &cfg); err != nil {
				return nil, "", fmt.Errorf("error parsing config file %s: %w", path, err)
			}
			return &cfg, path, nil
		}
	}
	return nil, "", nil
}

// GetConfigValue returns the value for a given environment variable key from the config file
func GetConfigValue(key string) string {
	configOnce.Do(func() {
		var err error
		config, configPath, err = loadConfig()
		if err != nil {
			slog.Warn("failed to load config file", "error", err)
		} else if config != nil {
			slog.Debug("loaded config file", "path", configPath)
		}
	})

	if config == nil {
		return ""
	}

	// Map environment variables to config values
	switch key {
	case "OLLAMA_HOST":
		return config.Server.Host
	case "OLLAMA_ORIGINS":
		if len(config.Server.Origins) > 0 {
			return strings.Join(config.Server.Origins, ",")
		}
	case "OLLAMA_MODELS":
		return config.Models.Path
	case "OLLAMA_KEEP_ALIVE":
		return config.Models.KeepAlive
	case "OLLAMA_MAX_LOADED_MODELS":
		if config.Models.MaxLoadedModels > 0 {
			return fmt.Sprintf("%d", config.Models.MaxLoadedModels)
		}
	case "OLLAMA_MAX_QUEUE":
		if config.Models.MaxQueue > 0 {
			return fmt.Sprintf("%d", config.Models.MaxQueue)
		}
	case "OLLAMA_NUM_PARALLEL":
		if config.Performance.NumParallel > 0 {
			return fmt.Sprintf("%d", config.Performance.NumParallel)
		}
	case "OLLAMA_GPU_OVERHEAD":
		if config.Performance.GpuOverhead > 0 {
			return fmt.Sprintf("%d", config.Performance.GpuOverhead)
		}
	case "OLLAMA_FLASH_ATTENTION":
		return fmt.Sprintf("%t", config.Performance.FlashAttention)
	case "OLLAMA_KV_CACHE_TYPE":
		return config.Performance.KvCacheType
	case "OLLAMA_DEBUG":
		return fmt.Sprintf("%t", config.Logging.Debug)
	}

	return ""
}

// GenerateExampleConfig returns a commented example TOML configuration
func GenerateExampleConfig() string {
	return `# Ollama Configuration File
# This is an example configuration file. Uncomment and modify values as needed.

[server]
# Network binding address (default: "127.0.0.1:11434")
host = "127.0.0.1:11434"
# Allowed CORS origins (comma-separated)
origins = ["http://localhost:3000"]

[models]
# Custom models directory path
path = "/path/to/models"
# How long to keep models loaded in memory (default: "5m")
keep_alive = "5m"
# Maximum number of models to keep loaded (default: 0 = unlimited)
max_loaded_models = 3
# Maximum number of queued requests (default: 512)
max_queue = 512

[performance]
# Number of parallel model requests (default: 0 = auto)
num_parallel = 4
# GPU memory overhead in bytes (default: 0)
gpu_overhead = 0
# Enable flash attention (default: false)
flash_attention = false
# KV cache type: "f16" or "q4_0" (default: "f16")
kv_cache_type = "f16"

[logging]
# Enable debug logging (default: false)
debug = false
`
} 