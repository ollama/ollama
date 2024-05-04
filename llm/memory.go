package llm

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
	"github.com/ollama/ollama/server/envconfig"
)

// This algorithm looks for a complete fit to determine if we need to unload other models
func PredictServerFit(allGpus gpu.GpuInfoList, ggml *GGML, adapters, projectors []string, opts api.Options) (bool, uint64) {
	var estimatedVRAM uint64
	if opts.NumCtx > int(ggml.KV().ContextLength()) {
		slog.Warn("requested context length is greater than model max context length", "requested", opts.NumCtx, "model", ggml.KV().ContextLength())
		opts.NumCtx = int(ggml.KV().ContextLength())
	}

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	// Split up the GPUs by type and try them
	for _, gpus := range allGpus.ByLibrary() {
		var layerCount int
		layerCount, estimatedVRAM = EstimateGPULayers(gpus, ggml, projectors, opts)
		if opts.NumGPU < 0 {
			if layerCount > 0 && layerCount >= int(ggml.KV().BlockCount()+1) {
				return true, estimatedVRAM
			}
		} else {
			if layerCount > 0 && layerCount >= opts.NumGPU {
				return true, estimatedVRAM
			}
		}
	}
	return false, estimatedVRAM
}

// Given a model and one or more GPU targets, predict how many layers and bytes we can load
// The GPUs provided must all be the same Library
func EstimateGPULayers(gpus []gpu.GpuInfo, ggml *GGML, projectors []string, opts api.Options) (int, uint64) {
	if gpus[0].Library == "cpu" {
		return 0, 0
	}
	var memoryAvailable uint64
	for _, info := range gpus {
		memoryAvailable += info.FreeMemory
	}
	if envconfig.MaxVRAM > 0 {
		memoryAvailable = envconfig.MaxVRAM
	}

	slog.Debug("evaluating", "library", gpus[0].Library, "gpu_count", len(gpus), "available", format.HumanBytes2(memoryAvailable))

	// TODO - this is probably wrong, first GPU vs secondaries will have different overheads
	memoryMinimum := gpus[0].MinimumMemory

	for _, projector := range projectors {
		memoryMinimum += projectorMemoryRequirements(projector)

		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	// fp16 k,v = (1 (k) + 1 (v)) * sizeof(float16) * n_ctx * n_layer * n_embd / n_head * n_head_kv
	var kv uint64 = 2 * 2 * uint64(opts.NumCtx) * ggml.KV().BlockCount() * ggml.KV().EmbeddingLength() / ggml.KV().HeadCount() * ggml.KV().HeadCountKV()

	graphPartialOffload, graphFullOffload := ggml.GraphSize(uint64(opts.NumCtx), uint64(min(opts.NumCtx, opts.NumBatch)))
	if graphPartialOffload == 0 {
		graphPartialOffload = ggml.KV().GQA() * kv / 6
	}

	if graphFullOffload == 0 {
		graphFullOffload = graphPartialOffload
	}

	graphFullOffload *= uint64(len(gpus))
	graphPartialOffload *= uint64(len(gpus))

	// on metal there's no partial offload overhead
	if gpus[0].Library == "metal" {
		graphPartialOffload = graphFullOffload
	}

	// memoryRequiredTotal represents the memory required for full GPU offloading (all layers)
	memoryRequiredTotal := memoryMinimum + graphFullOffload

	// memoryRequiredPartial represents the memory required for partial GPU offloading (n > 0, n < layers)
	memoryRequiredPartial := memoryMinimum + graphPartialOffload

	if memoryRequiredPartial > memoryAvailable {
		slog.Debug("insufficient VRAM to load any model layers")
		return 0, 0
	}

	layers := ggml.Tensors().Layers()

	var memoryLayerOutput uint64
	if layer, ok := layers["output_norm"]; ok {
		memoryLayerOutput += layer.size()
	}

	if layer, ok := layers["output"]; ok {
		memoryLayerOutput += layer.size()
	} else if layer, ok := layers["token_embd"]; ok {
		memoryLayerOutput += layer.size()
	}

	if gpus[0].Library == "metal" && opts.UseMMap {
		// memory is preallocated for output tensors
		memoryRequiredTotal += memoryLayerOutput
		memoryRequiredPartial += memoryLayerOutput
	}

	var layerCount int
	for i := 0; i < int(ggml.KV().BlockCount()); i++ {
		memoryLayer := layers[fmt.Sprintf("blk.%d", i)].size()

		// KV is proportional to the number of layers
		memoryLayer += kv / ggml.KV().BlockCount()

		memoryRequiredTotal += memoryLayer
		if memoryAvailable > memoryRequiredPartial+memoryLayer {
			memoryRequiredPartial += memoryLayer
			layerCount++
		}
	}

	if gpus[0].Library != "metal" || !opts.UseMMap {
		// memory was not preallocated for output tensors
		memoryRequiredTotal += memoryLayerOutput
	}

	if memoryAvailable > memoryRequiredTotal {
		layerCount = int(ggml.KV().BlockCount()) + 1
		memoryRequiredPartial = memoryRequiredTotal
	}

	memoryWeights := memoryRequiredTotal - memoryMinimum - graphFullOffload - kv

	slog.Info(
		"offload to gpu",
		slog.Group(
			"layers",
			// actual number of layers offloaded
			"real", opts.NumGPU,
			// estimated number of layers that can be offloaded
			"estimate", layerCount,
		),
		slog.Group(
			"memory",
			// memory available for offloading
			"available", format.HumanBytes2(memoryAvailable),
			slog.Group(
				"required",
				// memory required for full offloading
				"full", format.HumanBytes2(memoryRequiredTotal),
				// memory required to offload layers.estimate layers
				"partial", format.HumanBytes2(memoryRequiredPartial),
				// memory of KV cache
				"kv", format.HumanBytes2(kv),
			),
			slog.Group(
				"weights",
				// memory of the weights
				"total", format.HumanBytes2(memoryWeights),
				// memory of repeating layers
				"repeating", format.HumanBytes2(memoryWeights-memoryLayerOutput),
				// memory of non-repeating layers
				"nonrepeating", format.HumanBytes2(memoryLayerOutput),
			),
			slog.Group(
				"graph",
				// memory of graph when fully offloaded
				"full", format.HumanBytes2(graphFullOffload),
				// memory of graph when not fully offloaded
				"partial", format.HumanBytes2(graphPartialOffload),
			),
		),
	)
	return layerCount, uint64(memoryRequiredPartial)
}
