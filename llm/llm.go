package llm

import (
	"context"
	"fmt"
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

	fmt.Println("size", ggml.Size)
	fmt.Println("filetype", ggml.FileType())
	fmt.Println("architecture", ggml.ModelFamily())
	fmt.Println("type", ggml.ModelType())
	fmt.Println("name", ggml.Name())
	fmt.Println("embd", ggml.NumEmbed())
	fmt.Println("head", ggml.NumHead())
	fmt.Println("head_kv", ggml.NumHeadKv())
	fmt.Println("gqa", ggml.NumGQA())

	available, _ := gpu.CheckVRAM()

	// For now assume filesize = model size
	// TODO: use actual model size
	requiredModel := ggml.Size

	// fp16 k,v matrices require = n_ctx * n_layer * n_embd / n_head * n_head_kv * 2 bytes each * 2 key and value
	requiredKv := 2 * 2 * int64(opts.NumCtx) * int64(ggml.NumLayers()) * int64(ggml.NumEmbed()) * int64(ggml.NumHeadKv()) / int64(ggml.NumHead())

	// this amount is the overhead + tensors in memory
	// TODO: get this from the llama.cpp's graph calcluations instead of
	// estimating it's 1/6 * kv_cache_size * num_gqa
	requiredAlloc := int64(ggml.NumGQA()) * requiredKv / 6

	requiredTotal := requiredModel + requiredKv + requiredAlloc

	log.Println("system memory bytes:", available)
	log.Println("required model bytes:", requiredModel)
	log.Println("required kv bytes:", requiredKv)
	log.Println("required alloc bytes:", requiredAlloc)
	log.Println("required total bytes:", requiredTotal)

	info := gpu.GetGPUInfo()
	library := info.Library

	if opts.NumGPU == -1 {
		// default to offloading all layers
		opts.NumGPU = int(ggml.NumLayers()) + 1
	}

	// decide how many layers to put on the GPU
	if opts.NumGPU > 0 {
		switch runtime.GOOS {
		case "darwin":
			if requiredTotal > available {
				log.Println("not enough vram available, falling back to CPU only")
				opts.NumGPU = 0
			}
		default:
			if library == "cpu" || library == "default" {
				opts.NumGPU = 0
				break
			}

			// no offloading required
			if requiredTotal <= available {
				break
			}

			// requiredAlloc is always loaded for the CUDA runner, so don't load it if it won't fit
			if requiredAlloc > available {
				log.Printf("not enough vram available, falling back to CPU only")
				library = "cpu"
				opts.NumGPU = 0
				break
			}

			available -= requiredAlloc

			// fill remaining vram with layers
			log.Println("splitting", available, "of available memory bytes into layers")
			bytesPerLayer := int64((requiredModel + requiredKv) / int64(ggml.NumLayers()))
			log.Println("bytes per layer:", bytesPerLayer)
			layers := available / bytesPerLayer
			log.Println("total required with split:", requiredAlloc+(layers*bytesPerLayer))
			if layers < int64(opts.NumGPU) {
				opts.NumGPU = int(layers)
			}
		}
	}

	opts.NumGQA = 0
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
