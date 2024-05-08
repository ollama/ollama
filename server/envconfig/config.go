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
	// Set via OLLAMA_LLM_LIBRARY in the environment
	LLMLibrary string
	// Set via OLLAMA_MAX_LOADED_MODELS in the environment
	MaxRunners int
	// Set via OLLAMA_MAX_QUEUE in the environment
	MaxQueuedRequests int
	// Set via OLLAMA_MAX_VRAM in the environment
	MaxVRAM uint64
	// Set via OLLAMA_NOPRUNE in the environment
	NoPrune bool
	// Set via OLLAMA_NUM_PARALLEL in the environment
	NumParallel int
	// Set via OLLAMA_RUNNERS_DIR in the environment
	RunnersDir string
	// Set via OLLAMA_TMPDIR in the environment
	TmpDir string
)

func AsMap() map[string]string {
	return map[string]string{
		"OLLAMA_ORIGINS":           fmt.Sprintf("%v", AllowOrigins),
		"OLLAMA_DEBUG":             fmt.Sprintf("%v", Debug),
		"OLLAMA_LLM_LIBRARY":       fmt.Sprintf("%v", LLMLibrary),
		"OLLAMA_MAX_LOADED_MODELS": fmt.Sprintf("%v", MaxRunners),
		"OLLAMA_MAX_QUEUE":         fmt.Sprintf("%v", MaxQueuedRequests),
		"OLLAMA_MAX_VRAM":          fmt.Sprintf("%v", MaxVRAM),
		"OLLAMA_NOPRUNE":           fmt.Sprintf("%v", NoPrune),
		"OLLAMA_NUM_PARALLEL":      fmt.Sprintf("%v", NumParallel),
		"OLLAMA_RUNNERS_DIR":       fmt.Sprintf("%v", RunnersDir),
		"OLLAMA_TMPDIR":            fmt.Sprintf("%v", TmpDir),
	}
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
}
