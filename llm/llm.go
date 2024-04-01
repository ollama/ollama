package llm

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
)

type LLM interface {
	Predict(context.Context, PredictOpts, func(PredictResult)) error
	Embedding(context.Context, string) ([]float64, error)
	Encode(context.Context, string) ([]int, error)
	Decode(context.Context, []int) (string, error)
	Close()
}

var cpuOnlyFamilies = []string{
	"mamba",
}

func New(model string, adapters, projectors []string, opts *api.Options) (LLM, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, _, err := DecodeGGML(f)
	if err != nil {
		return nil, err
	}

	if opts.NumCtx > int(ggml.KV().ContextLength()) {
		slog.Warn("requested context length is greater than model max context length", "requested", opts.NumCtx, "model", ggml.KV().ContextLength())
		opts.NumCtx = int(ggml.KV().ContextLength())
	}

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	availableMemory, _ := gpu.CheckVRAM()
	info := gpu.GetGPUInfo()

	usedMemory := info.MinimumMemory
	for _, projector := range projectors {
		usedMemory += projectorMemoryRequirements(projector)

		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	// fp16 k,v = (1 (k) + 1 (v)) * sizeof(float16) * n_ctx * n_layer * n_embd / n_head * n_head_kv
	kv := 2 * 2 * int64(opts.NumCtx) * int64(ggml.KV().BlockCount()) * int64(ggml.KV().EmbeddingLength()) / int64(ggml.KV().HeadCount()) * int64(ggml.KV().HeadCountKV())

	// this amount is the overhead + tensors in memory
	// TODO: get this from the llama.cpp's graph calculations instead of
	// estimating it's 1/6 * kv_cache_size * num_gqa
	graph := int64(ggml.KV().GQA()) * kv / 6
	usedMemory += graph

	if usedMemory > availableMemory || slices.Contains(cpuOnlyFamilies, ggml.KV().Architecture()) {
		info.Library = "cpu"
	}

	requiredMemory := usedMemory

	var layers int
	for i := 0; i < int(ggml.KV().BlockCount()); i++ {
		layerMemory := ggml.LayerSize(fmt.Sprintf("blk.%d.", i)) + kv/int64(ggml.KV().BlockCount())
		requiredMemory += layerMemory

		if availableMemory > usedMemory+layerMemory && (opts.NumGPU < 0 || layers < opts.NumGPU) {
			usedMemory += layerMemory
			layers++
		}
	}

	memOutputLayer := ggml.LayerSize("output.")
	requiredMemory += memOutputLayer

	// only offload output layer if all repeating layers are offloaded
	if layers >= int(ggml.KV().BlockCount()) && availableMemory > usedMemory+memOutputLayer {
		usedMemory += memOutputLayer
		layers++
	}

	slog.Info(
		"offload to gpu",
		"layers", layers,
		"required", format.HumanBytes2(requiredMemory),
		"used", format.HumanBytes2(usedMemory),
		"available", format.HumanBytes2(availableMemory),
		"kv", format.HumanBytes2(kv),
		"graph", format.HumanBytes2(graph),
	)

	if opts.NumGPU < 0 && info.Library != "cpu" {
		opts.NumGPU = layers
	}

	return newLlmServer(info, model, adapters, projectors, opts)
}

func projectorMemoryRequirements(filename string) int64 {
	file, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer file.Close()

	ggml, _, err := DecodeGGML(file)
	if err != nil {
		return 0
	}

	prefixes := make(map[string]struct{})
	for _, layer := range ggml.Tensors() {
		parts := strings.Split(layer.Name, ".")
		prefixes[strings.Join(parts[:2], ".")] = struct{}{}
	}

	var ask int64
	for prefix := range prefixes {
		ask += ggml.LayerSize(prefix)
	}

	return ask
}

// Give any native cgo implementations an opportunity to initialize
func Init() error {
	return nativeInit()
}

func newLlmServer(gpuInfo gpu.GpuInfo, model string, adapters, projectors []string, opts *api.Options) (LLM, error) {
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
