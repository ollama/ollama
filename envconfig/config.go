package envconfig

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

type OllamaHost struct {
	Scheme string
	Host   string
	Port   string
}

func (o OllamaHost) String() string {
	return fmt.Sprintf("%s://%s:%s", o.Scheme, o.Host, o.Port)
}

var ErrInvalidHostPort = errors.New("invalid port specified in OLLAMA_HOST")

var (
	// Set via OLLAMA_ORIGINS in the environment
	AllowOrigins []string
	// Set via OLLAMA_DEBUG in the environment
	Debug bool
	// Experimental flash attention
	FlashAttention bool
	// Set via OLLAMA_HOST in the environment
	Host *OllamaHost
	// Set via OLLAMA_KEEP_ALIVE in the environment
	KeepAlive string
	// Set via OLLAMA_LLM_LIBRARY in the environment
	LLMLibrary string
	// Set via OLLAMA_MAX_LOADED_MODELS in the environment
	MaxRunners int
	// Set via OLLAMA_MAX_QUEUE in the environment
	MaxQueuedRequests int
	// Set via OLLAMA_MODELS in the environment
	ModelsDir string
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
	// Set via OLLAMA_SCHED_SPREAD in the environment
	SchedSpread bool
	// Set via OLLAMA_TMPDIR in the environment
	TmpDir string
	// Set via OLLAMA_INTEL_GPU in the environment
	IntelGpu bool

	// Set via CUDA_VISIBLE_DEVICES in the environment
	CudaVisibleDevices string
	// Set via HIP_VISIBLE_DEVICES in the environment
	HipVisibleDevices string
	// Set via ROCR_VISIBLE_DEVICES in the environment
	RocrVisibleDevices string
	// Set via GPU_DEVICE_ORDINAL in the environment
	GpuDeviceOrdinal string
	// Set via HSA_OVERRIDE_GFX_VERSION in the environment
	HsaOverrideGfxVersion string
)

type EnvVar struct {
	Name        string
	Value       any
	Description string
}

func AsMap() map[string]EnvVar {
	ret := map[string]EnvVar{
		"OLLAMA_DEBUG":             {"OLLAMA_DEBUG", Debug, "Show additional debug information (e.g. OLLAMA_DEBUG=1)"},
		"OLLAMA_FLASH_ATTENTION":   {"OLLAMA_FLASH_ATTENTION", FlashAttention, "Enabled flash attention"},
		"OLLAMA_HOST":              {"OLLAMA_HOST", Host, "IP Address for the ollama server (default 127.0.0.1:11434)"},
		"OLLAMA_KEEP_ALIVE":        {"OLLAMA_KEEP_ALIVE", KeepAlive, "The duration that models stay loaded in memory (default \"5m\")"},
		"OLLAMA_LLM_LIBRARY":       {"OLLAMA_LLM_LIBRARY", LLMLibrary, "Set LLM library to bypass autodetection"},
		"OLLAMA_MAX_LOADED_MODELS": {"OLLAMA_MAX_LOADED_MODELS", MaxRunners, "Maximum number of loaded models per GPU (default 4)"},
		"OLLAMA_MAX_QUEUE":         {"OLLAMA_MAX_QUEUE", MaxQueuedRequests, "Maximum number of queued requests"},
		"OLLAMA_MAX_VRAM":          {"OLLAMA_MAX_VRAM", MaxVRAM, "Maximum VRAM"},
		"OLLAMA_MODELS":            {"OLLAMA_MODELS", ModelsDir, "The path to the models directory"},
		"OLLAMA_NOHISTORY":         {"OLLAMA_NOHISTORY", NoHistory, "Do not preserve readline history"},
		"OLLAMA_NOPRUNE":           {"OLLAMA_NOPRUNE", NoPrune, "Do not prune model blobs on startup"},
		"OLLAMA_NUM_PARALLEL":      {"OLLAMA_NUM_PARALLEL", NumParallel, "Maximum number of parallel requests"},
		"OLLAMA_ORIGINS":           {"OLLAMA_ORIGINS", AllowOrigins, "A comma separated list of allowed origins"},
		"OLLAMA_RUNNERS_DIR":       {"OLLAMA_RUNNERS_DIR", RunnersDir, "Location for runners"},
		"OLLAMA_SCHED_SPREAD":      {"OLLAMA_SCHED_SPREAD", SchedSpread, "Always schedule model across all GPUs"},
		"OLLAMA_TMPDIR":            {"OLLAMA_TMPDIR", TmpDir, "Location for temporary files"},
	}
	if runtime.GOOS != "darwin" {
		ret["CUDA_VISIBLE_DEVICES"] = EnvVar{"CUDA_VISIBLE_DEVICES", CudaVisibleDevices, "Set which NVIDIA devices are visible"}
		ret["HIP_VISIBLE_DEVICES"] = EnvVar{"HIP_VISIBLE_DEVICES", HipVisibleDevices, "Set which AMD devices are visible"}
		ret["ROCR_VISIBLE_DEVICES"] = EnvVar{"ROCR_VISIBLE_DEVICES", RocrVisibleDevices, "Set which AMD devices are visible"}
		ret["GPU_DEVICE_ORDINAL"] = EnvVar{"GPU_DEVICE_ORDINAL", GpuDeviceOrdinal, "Set which AMD devices are visible"}
		ret["HSA_OVERRIDE_GFX_VERSION"] = EnvVar{"HSA_OVERRIDE_GFX_VERSION", HsaOverrideGfxVersion, "Override the gfx used for all detected AMD GPUs"}
		ret["OLLAMA_INTEL_GPU"] = EnvVar{"OLLAMA_INTEL_GPU", IntelGpu, "Enable experimental Intel GPU detection"}
	}
	return ret
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
	NumParallel = 0
	MaxRunners = 4
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
				root,
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
		if err != nil {
			slog.Error("invalid setting, ignoring", "OLLAMA_NUM_PARALLEL", onp, "error", err)
		} else {
			NumParallel = val
		}
	}

	if nohistory := clean("OLLAMA_NOHISTORY"); nohistory != "" {
		NoHistory = true
	}

	if spread := clean("OLLAMA_SCHED_SPREAD"); spread != "" {
		s, err := strconv.ParseBool(spread)
		if err == nil {
			SchedSpread = s
		} else {
			SchedSpread = true
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
			fmt.Sprintf("http://%s", net.JoinHostPort(allowOrigin, "*")),
			fmt.Sprintf("https://%s", net.JoinHostPort(allowOrigin, "*")),
		)
	}

	AllowOrigins = append(AllowOrigins,
		"app://*",
		"file://*",
		"tauri://*",
	)

	maxRunners := clean("OLLAMA_MAX_LOADED_MODELS")
	if maxRunners != "" {
		m, err := strconv.Atoi(maxRunners)
		if err != nil {
			slog.Error("invalid setting, ignoring", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "error", err)
		} else {
			MaxRunners = m
		}
	}

	if onp := os.Getenv("OLLAMA_MAX_QUEUE"); onp != "" {
		p, err := strconv.Atoi(onp)
		if err != nil || p <= 0 {
			slog.Error("invalid setting, ignoring", "OLLAMA_MAX_QUEUE", onp, "error", err)
		} else {
			MaxQueuedRequests = p
		}
	}

	KeepAlive = clean("OLLAMA_KEEP_ALIVE")

	var err error
	ModelsDir, err = getModelsDir()
	if err != nil {
		slog.Error("invalid setting", "OLLAMA_MODELS", ModelsDir, "error", err)
	}

	Host, err = getOllamaHost()
	if err != nil {
		slog.Error("invalid setting", "OLLAMA_HOST", Host, "error", err, "using default port", Host.Port)
	}

	if set, err := strconv.ParseBool(clean("OLLAMA_INTEL_GPU")); err == nil {
		IntelGpu = set
	}

	CudaVisibleDevices = clean("CUDA_VISIBLE_DEVICES")
	HipVisibleDevices = clean("HIP_VISIBLE_DEVICES")
	RocrVisibleDevices = clean("ROCR_VISIBLE_DEVICES")
	GpuDeviceOrdinal = clean("GPU_DEVICE_ORDINAL")
	HsaOverrideGfxVersion = clean("HSA_OVERRIDE_GFX_VERSION")
}

func getModelsDir() (string, error) {
	if models, exists := os.LookupEnv("OLLAMA_MODELS"); exists {
		return models, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "models"), nil
}

func getOllamaHost() (*OllamaHost, error) {
	defaultPort := "11434"

	hostVar := os.Getenv("OLLAMA_HOST")
	hostVar = strings.TrimSpace(strings.Trim(strings.TrimSpace(hostVar), "\"'"))

	scheme, hostport, ok := strings.Cut(hostVar, "://")
	switch {
	case !ok:
		scheme, hostport = "http", hostVar
	case scheme == "http":
		defaultPort = "80"
	case scheme == "https":
		defaultPort = "443"
	}

	// trim trailing slashes
	hostport = strings.TrimRight(hostport, "/")

	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if portNum, err := strconv.ParseInt(port, 10, 32); err != nil || portNum > 65535 || portNum < 0 {
		return &OllamaHost{
			Scheme: scheme,
			Host:   host,
			Port:   defaultPort,
		}, ErrInvalidHostPort
	}

	return &OllamaHost{
		Scheme: scheme,
		Host:   host,
		Port:   port,
	}, nil
}
