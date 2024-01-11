package llm

import (
	"context"
	"log"
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

var AvailableShims = map[string]string{}

func New(workDir, model string, adapters, projectors []string, opts api.Options) (LLM, error) {
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

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	vram, _ := gpu.CheckVRAM()
	size := ggml.Size

	// fp16 k,v matrices require = n_ctx * n_layer * n_embd / n_head * n_head_kv * 2 bytes each * 2 key and value
	kv := 2 * 2 * int64(opts.NumCtx) * int64(ggml.NumLayers()) * int64(ggml.NumEmbed()) * int64(ggml.NumHeadKv()) / int64(ggml.NumHead())

	// this amount is the overhead + tensors in memory
	// TODO: get this from the llama.cpp's graph calcluations instead of
	// estimating it's 1/6 * kv_cache_size * num_gqa
	graph := int64(ggml.NumGQA()) * kv / 6

	info := gpu.GetGPUInfo()
	library := info.Library
	switch runtime.GOOS {
	case "darwin":
		if opts.NumGPU == 0 {
			break
		}

		if size+kv+graph > vram {
			log.Println("not enough vram available, falling back to CPU only")
			opts.NumGPU = 0
			break
		}

		opts.NumGPU = 1
	default:
		if library == "cpu" || library == "default" {
			log.Println("GPU not available, falling back to CPU")
			opts.NumGPU = 0
			break
		}

		// don't use GPU at all if no layers are loaded
		if opts.NumGPU == 0 {
			library = "cpu"
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
			log.Printf("not enough vram available, falling back to CPU only")
			library = "cpu"
			opts.NumGPU = 0
			break
		}

		opts.NumGPU = int(layers)
	}

	opts.RopeFrequencyBase = 0.0
	opts.RopeFrequencyScale = 0.0
	return newLlmServer(library, model, adapters, projectors, opts)
}

// Give any native cgo implementations an opportunity to initialize
func Init(workdir string) error {
	return nativeInit(workdir)
}

func newLlmServer(library, model string, adapters, projectors []string, opts api.Options) (extServer, error) {
	if _, libPresent := AvailableShims[library]; libPresent && library != "default" {
		srv, err := newDynamicShimExtServer(AvailableShims[library], model, adapters, projectors, opts)
		if err == nil {
			return srv, nil
		}
		log.Printf("Failed to load dynamic library %s - falling back to CPU mode %s", library, err)
		// TODO - update some state to indicate we were unable to load the GPU library for future "info" ux
	}

	return newDefaultExtServer(model, adapters, projectors, opts)
}
