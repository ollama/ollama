package llm

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"runtime"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/gpu"
)

type LLM interface {
	Predict(context.Context, PredictOpts, func(PredictResult)) error
	Embedding(context.Context, string) ([]float64, error)
	Encode(context.Context, string) ([]int, error)
	Decode(context.Context, []int) (string, error)
	Close()
}

func New(model string, adapters, projectors []string, opts api.Options) (LLM, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := DecodeGGML(f)
	if err != nil {
		return nil, err
	}

	if opts.NumCtx > int(ggml.NumCtx()) {
		slog.Warn(fmt.Sprintf("requested context length is greater than model's max context length (%d > %d), using %d instead", opts.NumCtx, ggml.NumCtx(), ggml.NumCtx()))
		opts.NumCtx = int(ggml.NumCtx())
	}

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	vram, _ := gpu.CheckVRAM()
	size := ggml.Size

	// fp16 k,v matrices require = n_ctx * n_layer * n_embd / n_head * n_head_kv * 2 bytes each * 2 key and value
	kv := 2 * 2 * int64(opts.NumCtx) * int64(ggml.NumLayers()) * int64(ggml.NumEmbed()) * int64(ggml.NumHeadKv()) / int64(ggml.NumHead())

	// this amount is the overhead + tensors in memory
	// TODO: get this from the llama.cpp's graph calculations instead of
	// estimating it's 1/6 * kv_cache_size * num_gqa
	graph := int64(ggml.NumGQA()) * kv / 6

	info := gpu.GetGPUInfo()
	switch runtime.GOOS {
	case "darwin":
		if opts.NumGPU == 0 {
			break
		}

		if size+kv+graph > vram {
			slog.Info("not enough vram available, falling back to CPU only")
			info.Library = "cpu"
			info.Variant = gpu.GetCPUVariant()
			opts.NumGPU = 0
			break
		}

		// TODO: implement layer splitting on macOS
		opts.NumGPU = 999
	default:
		if info.Library == "cpu" {
			slog.Info("GPU not available, falling back to CPU")
			opts.NumGPU = 0
			break
		}

		// don't use GPU at all if no layers are loaded
		if opts.NumGPU == 0 {
			info.Library = "cpu"
			info.Variant = gpu.GetCPUVariant()
			break
		}

		// user-defined GPU count
		if opts.NumGPU != -1 {
			break
		}

		// the "main" GPU needs the most memory and determines the limit
		// of how many layers can be loaded. It needs to fit:
		// 1. the full compute graph allocation for all devices (graph)
		// 2. the proportional kv cache for all devices (kv * % layers)
		// 3. the proportional model (size * % layers / # devices)
		// This estimates the number of layers
		maxlayers := int64(ggml.NumLayers()) + 1
		devices := int64(info.DeviceCount)
		avg := vram / devices
		layers := maxlayers * (avg - graph) / (kv + size/devices)
		if layers > maxlayers {
			layers = maxlayers
		}

		// 1 + 2 must fit on the main gpu
		min := graph + kv*layers/maxlayers
		if layers <= 0 || min > avg {
			slog.Info("not enough vram available, falling back to CPU only")
			info.Library = "cpu"
			info.Variant = gpu.GetCPUVariant()
			opts.NumGPU = 0
			break
		}

		opts.NumGPU = int(layers)
	}

	opts.RopeFrequencyBase = 0.0
	opts.RopeFrequencyScale = 0.0
	return newLlmServer(info, model, adapters, projectors, opts)
}

// Give any native cgo implementations an opportunity to initialize
func Init() error {
	return nativeInit()
}

func newLlmServer(gpuInfo gpu.GpuInfo, model string, adapters, projectors []string, opts api.Options) (LLM, error) {
	dynLibs := getDynLibs(gpuInfo)

	// Check to see if the user has requested a specific library instead of auto-detecting
	demandLib := os.Getenv("OLLAMA_LLM_LIBRARY")
	if demandLib != "" {
		libPath := availableDynLibs[demandLib]
		if libPath == "" {
			slog.Info(fmt.Sprintf("Invalid OLLAMA_LLM_LIBRARY %s - not found", demandLib))
		} else {
			slog.Info(fmt.Sprintf("Loading OLLAMA_LLM_LIBRARY=%s", demandLib))
			dynLibs = []string{libPath}
		}
	}

	// We stage into a temp directory, and if we've been idle for a while, it may have been reaped
	_, err := os.Stat(dynLibs[0])
	if err != nil {
		slog.Info(fmt.Sprintf("%s has disappeared, reloading libraries", dynLibs[0]))
		err = nativeInit()
		if err != nil {
			return nil, err
		}
	}

	err2 := fmt.Errorf("unable to locate suitable llm library")
	for _, dynLib := range dynLibs {
		srv, err := newDynExtServer(dynLib, model, adapters, projectors, opts)
		if err == nil {
			return srv, nil
		}
		slog.Warn(fmt.Sprintf("Failed to load dynamic library %s  %s", dynLib, err))
		err2 = err
	}

	return nil, err2
}
