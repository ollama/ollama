package envconfig

import (
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Host returns the scheme and host. Host can be configured via the OLLAMA_HOST environment variable.
// Default is scheme "http" and host "127.0.0.1:11434"
func Host() *url.URL {
	defaultPort := "11434"

	s := strings.TrimSpace(Var("OLLAMA_HOST"))
	scheme, hostport, ok := strings.Cut(s, "://")
	switch {
	case !ok:
		scheme, hostport = "http", s
		if s == "ollama.com" {
			scheme, hostport = "https", "ollama.com:443"
		}
	case scheme == "http":
		defaultPort = "80"
	case scheme == "https":
		defaultPort = "443"
	}

	hostport, path, _ := strings.Cut(hostport, "/")
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if n, err := strconv.ParseInt(port, 10, 32); err != nil || n > 65535 || n < 0 {
		slog.Warn("invalid port, using default", "port", port, "default", defaultPort)
		port = defaultPort
	}

	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, port),
		Path:   path,
	}
}

// AllowedOrigins returns a list of allowed origins. AllowedOrigins can be configured via the OLLAMA_ORIGINS environment variable.
func AllowedOrigins() (origins []string) {
	if s := Var("OLLAMA_ORIGINS"); s != "" {
		origins = strings.Split(s, ",")
	}

	for _, origin := range []string{"localhost", "127.0.0.1", "0.0.0.0"} {
		origins = append(origins,
			fmt.Sprintf("http://%s", origin),
			fmt.Sprintf("https://%s", origin),
			fmt.Sprintf("http://%s", net.JoinHostPort(origin, "*")),
			fmt.Sprintf("https://%s", net.JoinHostPort(origin, "*")),
		)
	}

	origins = append(origins,
		"app://*",
		"file://*",
		"tauri://*",
		"vscode-webview://*",
		"vscode-file://*",
	)

	return origins
}

// Models returns the path to the models directory. Models directory can be configured via the OLLAMA_MODELS environment variable.
// Default is $HOME/.ollama/models
func Models() string {
	if s := Var("OLLAMA_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".ollama", "models")
}

// KeepAlive returns the duration that models stay loaded in memory. KeepAlive can be configured via the OLLAMA_KEEP_ALIVE environment variable.
// Negative values are treated as infinite. Zero is treated as no keep alive.
// Default is 5 minutes.
func KeepAlive() (keepAlive time.Duration) {
	keepAlive = 5 * time.Minute
	if s := Var("OLLAMA_KEEP_ALIVE"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			keepAlive = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			keepAlive = time.Duration(n) * time.Second
		}
	}

	if keepAlive < 0 {
		return time.Duration(math.MaxInt64)
	}

	return keepAlive
}

// LoadTimeout returns the duration for stall detection during model loads. LoadTimeout can be configured via the OLLAMA_LOAD_TIMEOUT environment variable.
// Zero or Negative values are treated as infinite.
// Default is 5 minutes.
func LoadTimeout() (loadTimeout time.Duration) {
	loadTimeout = 5 * time.Minute
	if s := Var("OLLAMA_LOAD_TIMEOUT"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			loadTimeout = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			loadTimeout = time.Duration(n) * time.Second
		}
	}

	if loadTimeout <= 0 {
		return time.Duration(math.MaxInt64)
	}

	return loadTimeout
}

func Remotes() []string {
	var r []string
	raw := strings.TrimSpace(Var("OLLAMA_REMOTES"))
	if raw == "" {
		r = []string{"ollama.com"}
	} else {
		r = strings.Split(raw, ",")
	}
	return r
}

func BoolWithDefault(k string) func(defaultValue bool) bool {
	return func(defaultValue bool) bool {
		if s := Var(k); s != "" {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return true
			}

			return b
		}

		return defaultValue
	}
}

func Bool(k string) func() bool {
	withDefault := BoolWithDefault(k)
	return func() bool {
		return withDefault(false)
	}
}

// LogLevel returns the log level for the application.
// Values are 0 or false INFO (Default), 1 or true DEBUG, 2 TRACE
func LogLevel() slog.Level {
	level := slog.LevelInfo
	if s := Var("OLLAMA_DEBUG"); s != "" {
		if b, _ := strconv.ParseBool(s); b {
			level = slog.LevelDebug
		} else if i, _ := strconv.ParseInt(s, 10, 64); i != 0 {
			level = slog.Level(i * -4)
		}
	}

	return level
}

var (
	// FlashAttention enables the experimental flash attention feature.
	FlashAttention = BoolWithDefault("OLLAMA_FLASH_ATTENTION")
	// KvCacheType is the quantization type for the K/V cache.
	KvCacheType = String("OLLAMA_KV_CACHE_TYPE")
	// NoHistory disables readline history.
	NoHistory = Bool("OLLAMA_NOHISTORY")
	// NoPrune disables pruning of model blobs on startup.
	NoPrune = Bool("OLLAMA_NOPRUNE")
	// SchedSpread allows scheduling models across all GPUs.
	SchedSpread = Bool("OLLAMA_SCHED_SPREAD")
	// MultiUserCache optimizes prompt caching for multi-user scenarios
	MultiUserCache = Bool("OLLAMA_MULTIUSER_CACHE")
	// Enable the new Ollama engine
	NewEngine = Bool("OLLAMA_NEW_ENGINE")
	// ContextLength sets the default context length
	ContextLength = Uint("OLLAMA_CONTEXT_LENGTH", 4096)
	// Auth enables authentication between the Ollama client and server
	UseAuth = Bool("OLLAMA_AUTH")
	// Enable Vulkan backend
	EnableVulkan = Bool("OLLAMA_VULKAN")
)

func String(s string) func() string {
	return func() string {
		return Var(s)
	}
}

var (
	LLMLibrary = String("OLLAMA_LLM_LIBRARY")

	CudaVisibleDevices    = String("CUDA_VISIBLE_DEVICES")
	HipVisibleDevices     = String("HIP_VISIBLE_DEVICES")
	RocrVisibleDevices    = String("ROCR_VISIBLE_DEVICES")
	VkVisibleDevices      = String("GGML_VK_VISIBLE_DEVICES")
	GpuDeviceOrdinal      = String("GPU_DEVICE_ORDINAL")
	HsaOverrideGfxVersion = String("HSA_OVERRIDE_GFX_VERSION")
)

func Uint(key string, defaultValue uint) func() uint {
	return func() uint {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return uint(n)
			}
		}

		return defaultValue
	}
}

var (
	// NumParallel sets the number of parallel model requests. NumParallel can be configured via the OLLAMA_NUM_PARALLEL environment variable.
	NumParallel = Uint("OLLAMA_NUM_PARALLEL", 1)
	// MaxRunners sets the maximum number of loaded models. MaxRunners can be configured via the OLLAMA_MAX_LOADED_MODELS environment variable.
	MaxRunners = Uint("OLLAMA_MAX_LOADED_MODELS", 0)
	// MaxQueue sets the maximum number of queued requests. MaxQueue can be configured via the OLLAMA_MAX_QUEUE environment variable.
	MaxQueue = Uint("OLLAMA_MAX_QUEUE", 512)
)

func Uint64(key string, defaultValue uint64) func() uint64 {
	return func() uint64 {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return n
			}
		}

		return defaultValue
	}
}

// Set aside VRAM per GPU
var GpuOverhead = Uint64("OLLAMA_GPU_OVERHEAD", 0)

type item struct {
	enable              bool
	name, usage         string
	value, defaultValue any
}

func (i item) IsZero() bool {
	return (i.value == i.defaultValue) || (i.defaultValue == nil && reflect.ValueOf(i.value).IsZero())
}

func (i item) LogValue() slog.Value {
	return slog.GroupValue(slog.Any(i.name, i.value))
}

type slice []item

func (s slice) LogValue() slog.Value {
	attrs := make([]slog.Attr, 0, 2*len(s))
	for _, e := range s {
		attrs = append(attrs, e.LogValue().Group()...)
	}
	return slog.GroupValue(attrs...)
}

var all = slice{
	{true, "OLLAMA_DEBUG", "Show additional debug information (e.g. OLLAMA_DEBUG=1). Verbosity increase with value", LogLevel(), nil},
	{true, "OLLAMA_FLASH_ATTENTION", "Enable flash attention", FlashAttention(false), nil},
	{true, "OLLAMA_KV_CACHE_TYPE", "Quantization type for the K/V cache", KvCacheType(), nil},
	{true, "OLLAMA_GPU_OVERHEAD", "Reserve a portion of VRAM per GPU (bytes)", GpuOverhead(), 0},
	{true, "OLLAMA_HOST", "IP Address for the ollama server", Host(), "127.0.0.1:11434"},
	{true, "OLLAMA_KEEP_ALIVE", "The duration that models stay loaded in memory", KeepAlive(), 5 * time.Minute},
	{true, "OLLAMA_LLM_LIBRARY", "Set LLM library to bypass autodetection", LLMLibrary(), nil},
	{true, "OLLAMA_LOAD_TIMEOUT", "How long to allow model loads to stall before giving up", LoadTimeout(), 5 * time.Minute},
	{true, "OLLAMA_MAX_LOADED_MODELS", "Maximum number of loaded models per GPU", MaxRunners(), 0},
	{true, "OLLAMA_MAX_QUEUE", "Maximum number of queued requests", MaxQueue(), 512},
	{true, "OLLAMA_MODELS", "The path to the models directory", Models(), filepath.Join(os.Getenv("HOME"), ".ollama", "models")},
	{true, "OLLAMA_NOHISTORY", "Do not preserve readline history", NoHistory(), false},
	{true, "OLLAMA_NOPRUNE", "Do not prune model blobs on startup", NoPrune(), false},
	{true, "OLLAMA_NUM_PARALLEL", "Maximum number of parallel requests", NumParallel(), 1},
	{true, "OLLAMA_ORIGINS", "A comma separated list of allowed origins", AllowedOrigins(), nil},
	{true, "OLLAMA_SCHED_SPREAD", "Always schedule model across all GPUs", SchedSpread(), false},
	{true, "OLLAMA_MULTIUSER_CACHE", "Optimize prompt caching for multi-user scenarios", MultiUserCache(), false},
	{true, "OLLAMA_CONTEXT_LENGTH", "Context length to use unless otherwise specified", ContextLength(), 4096},
	{true, "OLLAMA_NEW_ENGINE", "Enable the new Ollama engine", NewEngine(), false},
	{true, "OLLAMA_REMOTES", "Allowed hosts for remote models", Remotes(), []string{"ollama.com"}},
	{runtime.GOOS != "windows", "HTTP_PROXY", "HTTP proxy", String("http_proxy")(), nil},
	{runtime.GOOS != "windows", "HTTPS_PROXY", "HTTPS proxy", String("https_proxy")(), nil},
	{runtime.GOOS != "windows", "NO_PROXY", "No proxy", String("no_proxy")(), nil},
	{runtime.GOOS != "darwin", "CUDA_VISIBLE_DEVICES", "Set which NVIDIA devices are visible", CudaVisibleDevices(), nil},
	{runtime.GOOS != "darwin", "HIP_VISIBLE_DEVICES", "Set which AMD devices are visible by numeric ID", HipVisibleDevices(), nil},
	{runtime.GOOS != "darwin", "ROCR_VISIBLE_DEVICES", "Set which AMD devices are visible by UUID or numeric ID", RocrVisibleDevices(), nil},
	{runtime.GOOS != "darwin", "GGML_VK_VISIBLE_DEVICES", "Set which Vulkan devices are visible by numeric ID", VkVisibleDevices(), nil},
	{runtime.GOOS != "darwin", "GPU_DEVICE_ORDINAL", "Set which AMD devices are visible by numeric ID", GpuDeviceOrdinal(), nil},
	{runtime.GOOS != "darwin", "HSA_OVERRIDE_GFX_VERSION", "Override the gfx used for all detected AMD GPUs", HsaOverrideGfxVersion(), nil},
}

func Enabled() slice {
	enabled := make(slice, 0, len(all))
	for _, i := range all {
		if i.enable {
			enabled = append(enabled, i)
		}
	}
	return enabled
}

func Lookup(s ...string) []item {
	enabled := Enabled()
	filtered := make([]item, 0, len(s))
	for _, k := range s {
		if i := slices.IndexFunc(enabled, func(i item) bool { return i.name == k }); i != -1 {
			filtered = append(filtered, enabled[i])
		}
	}
	return filtered
}

// Usage returns enabled environment variables and their usage descriptions.
// If a variable has a default value, it is included in the description.
func Usage(s ...string) map[string]string {
	enabled := Enabled()
	m := make(map[string]string, len(s))
	for _, k := range s {
		if i := slices.IndexFunc(enabled, func(i item) bool { return i.name == k }); i != -1 {
			m[k] = enabled[i].usage
			if enabled[i].defaultValue != nil {
				m[k] += fmt.Sprintf(" (default: %v)", enabled[i].defaultValue)
			}
		}
	}
	return m
}

// Var returns an environment variable stripped of leading and trailing quotes or spaces
func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}
