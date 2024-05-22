package llm

import (
	"fmt"
	"log/slog"

	"github.com/uppercaveman/ollama-server/api"
	"github.com/uppercaveman/ollama-server/format"
	"github.com/uppercaveman/ollama-server/gpu"
	"github.com/uppercaveman/ollama-server/server/envconfig"
)

// This algorithm looks for a complete fit to determine if we need to unload other models
func PredictServerFit(allGpus gpu.GpuInfoList, ggml *GGML, adapters, projectors []string, opts api.Options) (bool, uint64) {
	// Split up the GPUs by type and try them
	var estimatedVRAM uint64
	for _, gpus := range allGpus.ByLibrary() {
		var layerCount int
		layerCount, estimatedVRAM, _ = EstimateGPULayers(gpus, ggml, projectors, opts)
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

// Given a model and one or more GPU targets, predict how many layers and bytes we can load, and the total size
// The GPUs provided must all be the same Library
func EstimateGPULayers(gpus []gpu.GpuInfo, ggml *GGML, projectors []string, opts api.Options) (int, uint64, uint64) {
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

	layers := ggml.Tensors().Layers()
	// add one layer worth of memory as a buffer
	if blk0, ok := layers["blk.0"]; ok {
		memoryMinimum += blk0.size()
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
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			memoryLayer := blk.size()

			// KV is proportional to the number of layers
			memoryLayer += kv / ggml.KV().BlockCount()

			memoryRequiredTotal += memoryLayer
			if (opts.NumGPU >= 0 && layerCount+1 <= opts.NumGPU) || (opts.NumGPU < 0 && memoryAvailable > memoryRequiredPartial+memoryLayer) {
				memoryRequiredPartial += memoryLayer
				layerCount++
			}
		}
	}

	if gpus[0].Library != "metal" || !opts.UseMMap {
		// memory was not preallocated for output tensors
		memoryRequiredTotal += memoryLayerOutput
	}

	if (opts.NumGPU >= 0 && layerCount+1 <= opts.NumGPU) || (opts.NumGPU < 0 && memoryAvailable > memoryRequiredTotal) {
		layerCount = int(ggml.KV().BlockCount()) + 1
		memoryRequiredPartial = memoryRequiredTotal
	}

	memoryWeights := memoryRequiredTotal - memoryMinimum - graphFullOffload - kv

	slog.Info(
		"offload to gpu",
		slog.Group(
			"layers",
			// requested number of layers to offload
			"requested", opts.NumGPU,
			// estimated number of layers that can be offloaded
			"real", layerCount,
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
	if gpus[0].Library == "cpu" {
		return 0, 0, memoryRequiredTotal
	}
	if memoryRequiredPartial > memoryAvailable {
		slog.Debug("insufficient VRAM to load any model layers")
		return 0, 0, memoryRequiredTotal
	}

	return layerCount, memoryRequiredPartial, memoryRequiredTotal
}
