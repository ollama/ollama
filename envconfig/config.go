package envconfig

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

var (
	// Set via OLLAMA_ORIGINS in the environment
	AllowOrigins []string
	// Set via OLLAMA_DEBUG in the environment
	Debug bool
	// Experimental flash attention
	FlashAttention bool
	// Set via OLLAMA_KEEP_ALIVE in the environment
	KeepAlive string
	// Set via OLLAMA_LLM_LIBRARY in the environment
	LLMLibrary string
	// Set via OLLAMA_MAX_LOADED_MODELS in the environment
	MaxRunners int
	// Set via OLLAMA_MAX_QUEUE in the environment
	MaxQueuedRequests int
	// Set via OLLAMA_MAX_VRAM in the environment
	MaxVRAM uint64
	// Set via OLLAMA_NOHISTORY in the environment
	NoHistory bool
	// Set via OLLAMA_NOPRUNE in the environment
	NoPrune bool
	// Set via OLLAMA_NUM_PARALLEL in the environment
	NumParallel int
	// Set via OLLAMA_RUNNERS_DIR in the environment
	RunnersDir string
	// Set via OLLAMA_TMPDIR in the environment
	TmpDir string
)

type EnvVar struct {
	Name        string
	Value       any
	Description string
}

func AsMap() map[string]EnvVar {
	return map[string]EnvVar{
		"OLLAMA_DEBUG":             {"OLLAMA_DEBUG", Debug, "Show additional debug information (e.g. OLLAMA_DEBUG=1)"},
		"OLLAMA_FLASH_ATTENTION":   {"OLLAMA_FLASH_ATTENTION", FlashAttention, "Enabled flash attention"},
		"OLLAMA_HOST":              {"OLLAMA_HOST", "", "IP Address for the ollama server (default 127.0.0.1:11434)"},
		"OLLAMA_KEEP_ALIVE":        {"OLLAMA_KEEP_ALIVE", KeepAlive, "The duration that models stay loaded in memory (default \"5m\")"},
		"OLLAMA_LLM_LIBRARY":       {"OLLAMA_ORIGINS", LLMLibrary, ""},
		"OLLAMA_MAX_LOADED_MODELS": {"OLLAMA_MAX_LOADED_MODELS", MaxRunners, "Maximum number of loaded models (default 1)"},
		"OLLAMA_MAX_QUEUE":         {"OLLAMA_MAX_QUEUE", MaxQueuedRequests, "Maximum number of queued requests"},
		"OLLAMA_MAX_VRAM":          {"OLLAMA_MAX_VRAM", MaxVRAM, ""},
		"OLLAMA_MODELS":            {"OLLAMA_MODELS", "", "The path to the models directory"},
		"OLLAMA_NOHISTORY":         {"OLLAMA_NOHISTORY", NoHistory, "Do not preserve readline history"},
		"OLLAMA_NOPRUNE":           {"OLLAMA_NOPRUNE", NoPrune, "Do not prune model blobs on startup"},
		"OLLAMA_NUM_PARALLEL":      {"OLLAMA_NUM_PARALLEL", NumParallel, "Maximum number of parallel requests (default 1)"},
		"OLLAMA_ORIGINS":           {"OLLAMA_ORIGINS", AllowOrigins, "A comma separated list of allowed origins"},
		"OLLAMA_RUNNERS_DIR":       {"OLLAMA_RUNNERS_DIR", RunnersDir, ""},
		"OLLAMA_TMPDIR":            {"OLLAMA_TMPDIR", TmpDir, "Location for temporary files"},
	}
}

func Values() map[string]string {
	vals := make(map[string]string)
	for k, v := range AsMap() {
		vals[k] = fmt.Sprintf("%v", v.Value)
	}
	return vals
}

var defaultAllowOrigins = []string{
	"localhost",
	"127.0.0.1",
	"0.0.0.0",
}

// Clean quotes and spaces from the value
func clean(key string) string {
	return strings.Trim(os.Getenv(key), "\"' ")
}

func init() {
	// default values
	NumParallel = 1
	MaxRunners = 1
	MaxQueuedRequests = 512

	LoadConfig()
}

func LoadConfig() {
	if debug := clean("OLLAMA_DEBUG"); debug != "" {
		d, err := strconv.ParseBool(debug)
		if err == nil {
			Debug = d
		} else {
			Debug = true
		}
	}

	if fa := clean("OLLAMA_FLASH_ATTENTION"); fa != "" {
		d, err := strconv.ParseBool(fa)
		if err == nil {
			FlashAttention = d
		}
	}

	RunnersDir = clean("OLLAMA_RUNNERS_DIR")
	if runtime.GOOS == "windows" && RunnersDir == "" {
		// On Windows we do not carry the payloads inside the main executable
		appExe, err := os.Executable()
		if err != nil {
			slog.Error("failed to lookup executable path", "error", err)
		}

		cwd, err := os.Getwd()
		if err != nil {
			slog.Error("failed to lookup working directory", "error", err)
		}

		var paths []string
		for _, root := range []string{filepath.Dir(appExe), cwd} {
			paths = append(paths,
				filepath.Join(root),
				filepath.Join(root, "windows-"+runtime.GOARCH),
				filepath.Join(root, "dist", "windows-"+runtime.GOARCH),
			)
		}

		// Try a few variations to improve developer experience when building from source in the local tree
		for _, p := range paths {
			candidate := filepath.Join(p, "ollama_runners")
			_, err := os.Stat(candidate)
			if err == nil {
				RunnersDir = candidate
				break
			}
		}
		if RunnersDir == "" {
			slog.Error("unable to locate llm runner directory.  Set OLLAMA_RUNNERS_DIR to the location of 'ollama_runners'")
		}
	}

	TmpDir = clean("OLLAMA_TMPDIR")

	userLimit := clean("OLLAMA_MAX_VRAM")
	if userLimit != "" {
		avail, err := strconv.ParseUint(userLimit, 10, 64)
		if err != nil {
			slog.Error("invalid setting, ignoring", "OLLAMA_MAX_VRAM", userLimit, "error", err)
		} else {
			MaxVRAM = avail
		}
	}

	LLMLibrary = clean("OLLAMA_LLM_LIBRARY")

	if onp := clean("OLLAMA_NUM_PARALLEL"); onp != "" {
		val, err := strconv.Atoi(onp)
		if err != nil || val <= 0 {
			slog.Error("invalid setting must be greater than zero", "OLLAMA_NUM_PARALLEL", onp, "error", err)
		} else {
			NumParallel = val
		}
	}

	if nohistory := clean("OLLAMA_NOHISTORY"); nohistory != "" {
		NoHistory = true
	}

	if noprune := clean("OLLAMA_NOPRUNE"); noprune != "" {
		NoPrune = true
	}

	if origins := clean("OLLAMA_ORIGINS"); origins != "" {
		AllowOrigins = strings.Split(origins, ",")
	}
	for _, allowOrigin := range defaultAllowOrigins {
		AllowOrigins = append(AllowOrigins,
			fmt.Sprintf("http://%s", allowOrigin),
			fmt.Sprintf("https://%s", allowOrigin),
			fmt.Sprintf("http://%s:*", allowOrigin),
			fmt.Sprintf("https://%s:*", allowOrigin),
		)
	}

	maxRunners := clean("OLLAMA_MAX_LOADED_MODELS")
	if maxRunners != "" {
		m, err := strconv.Atoi(maxRunners)
		if err != nil {
			slog.Error("invalid setting", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "error", err)
		} else {
			MaxRunners = m
		}
	}

	if onp := os.Getenv("OLLAMA_MAX_QUEUE"); onp != "" {
		p, err := strconv.Atoi(onp)
		if err != nil || p <= 0 {
			slog.Error("invalid setting", "OLLAMA_MAX_QUEUE", onp, "error", err)
		} else {
			MaxQueuedRequests = p
		}
	}

	KeepAlive = clean("OLLAMA_KEEP_ALIVE")
}
