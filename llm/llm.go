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

			// alloc buffer and kv cache is allocated as a fixed amount on the main gpu
			// TODO: find the largest GPU and only reserve memory there
			avgAvailable := available / int64(info.DeviceCount)
			if requiredAlloc > avgAvailable {
				log.Printf("not enough vram available, falling back to CPU only")
				library = "cpu"
				opts.NumGPU = 0
				break
			}

			// we don't know which GPU will be used, so estimate
			// the scratch buffer space on all of them
			// TODO: allocate less layers to the GPU with the scratch buffer
			// and more to the others (based on their available memory)
			available -= requiredAlloc * int64(info.DeviceCount)

			// no offloading required
			if requiredModel+requiredKv <= available {
				break
			}

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
	gpuInfo := gpu.GetGPUInfo()
	return newLlmServer(gpuInfo, model, adapters, projectors, opts)
}

// Give any native cgo implementations an opportunity to initialize
func Init(workdir string) error {
	return nativeInit(workdir)
}

func newLlmServer(gpuInfo gpu.GpuInfo, model string, adapters, projectors []string, opts api.Options) (LLM, error) {
	dynLibs := getDynLibs(gpuInfo)

	// Check to see if the user has requested a specific library instead of auto-detecting
	demandLib := os.Getenv("OLLAMA_LLM_LIBRARY")
	if demandLib != "" {
		libPath := availableDynLibs[demandLib]
		if libPath == "" {
			log.Printf("Invalid OLLAMA_LLM_LIBRARY %s - not found", demandLib)
		} else {
			log.Printf("Loading OLLAMA_LLM_LIBRARY=%s", demandLib)
			dynLibs = []string{libPath}
		}
	}

	err2 := fmt.Errorf("unable to locate suitable llm library")
	for _, dynLib := range dynLibs {
		srv, err := newDynExtServer(dynLib, model, adapters, projectors, opts)
		if err == nil {
			return srv, nil
		}
		log.Printf("Failed to load dynamic library %s  %s", dynLib, err)
		err2 = err
	}

	return nil, err2
}
