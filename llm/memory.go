package llm

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
)

// This algorithm looks for a complete fit to determine if we need to unload other models
func PredictServerFit(allGpus gpu.GpuInfoList, ggml *GGML, adapters, projectors []string, opts api.Options) (bool, uint64) {
	// Split up the GPUs by type and try them
	var estimatedVRAM uint64
	for _, gpus := range allGpus.ByLibrary() {
		var layerCount int
		estimate := EstimateGPULayers(gpus, ggml, projectors, opts)
		layerCount, estimatedVRAM = estimate.Layers, estimate.VRAMSize
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

type MemoryEstimate struct {
	// How many layers we predict we can load
	Layers int

	// The size of the graph which occupies the main GPU
	Graph uint64

	// How much VRAM will be allocated given the number of layers we predict
	VRAMSize uint64

	// The total size of the model if loaded into VRAM.  If all layers are loaded, VRAMSize == TotalSize
	TotalSize uint64

	// For multi-GPU scenarios, this provides the tensor split parameter
	TensorSplit string

	// For multi-GPU scenarios, this is the size in bytes per GPU
	GPUSizes []uint64
}

// Given a model and one or more GPU targets, predict how many layers and bytes we can load, and the total size
// The GPUs provided must all be the same Library
func EstimateGPULayers(gpus []gpu.GpuInfo, ggml *GGML, projectors []string, opts api.Options) MemoryEstimate {
	// Graph size for a partial offload, applies to all GPUs
	var graphPartialOffload uint64

	// Graph size when all layers are offloaded, applies to all GPUs
	var graphFullOffload uint64

	// Final graph offload once we know full or partial
	var graphOffload uint64

	// Projectors loaded into GPU0 only
	var projectorSize uint64

	// Conditional output size on GPU 0
	var memoryLayerOutput uint64
	var includeOutput bool

	// One extra layer as a pad for each GPU
	var layerBuffer uint64

	// The sizes of the main layers
	var layerSizes []uint64

	// The sum of all the layer sizes (just for logging)
	var memoryWeights uint64

	// True if all the layers are loaded
	var fullyLoaded bool

	// Overflow that didn't fit into the GPU
	var overflow uint64

	availableList := make([]string, len(gpus))
	for i, gpu := range gpus {
		availableList[i] = format.HumanBytes2(gpu.FreeMemory)
	}
	slog.Debug("evaluating", "library", gpus[0].Library, "gpu_count", len(gpus), "available", availableList)

	for _, projector := range projectors {
		projectorSize += projectorMemoryRequirements(projector)

		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	layers := ggml.Tensors().Layers()
	// add one layer worth of memory as a buffer
	if blk0, ok := layers["blk.0"]; ok {
		layerBuffer = blk0.size()
	}

	// fp16 k,v = (1 (k) + 1 (v)) * sizeof(float16) * n_ctx * n_layer * n_embd / n_head * n_head_kv
	var kv uint64 = 2 * 2 * uint64(opts.NumCtx) * ggml.KV().BlockCount() * ggml.KV().EmbeddingLength() / ggml.KV().HeadCount() * ggml.KV().HeadCountKV()

	graphPartialOffload, graphFullOffload = ggml.GraphSize(uint64(opts.NumCtx), uint64(min(opts.NumCtx, opts.NumBatch)))
	if graphPartialOffload == 0 {
		graphPartialOffload = ggml.KV().GQA() * kv / 6
	}
	if graphFullOffload == 0 {
		graphFullOffload = graphPartialOffload
	}

	// on metal there's no partial offload overhead
	if gpus[0].Library == "metal" {
		graphPartialOffload = graphFullOffload
	}

	if layer, ok := layers["output_norm"]; ok {
		memoryLayerOutput += layer.size()
	}
	if layer, ok := layers["output"]; ok {
		memoryLayerOutput += layer.size()
	} else if layer, ok := layers["token_embd"]; ok {
		memoryLayerOutput += layer.size()
	}

	if gpus[0].Library == "metal" && opts.UseMMap {
		includeOutput = true
	} else if gpus[0].Library != "metal" || !opts.UseMMap {
		includeOutput = true
	}

	gpuZeroOverhead := projectorSize
	if includeOutput {
		gpuZeroOverhead += memoryLayerOutput
	}

	// Reduce set of GPUs to only those that have sufficient space to fit overhead and at least one layer
	var layerCount int
	layerCounts := make([]int, len(gpus))
	gpuAllocations := make([]uint64, len(gpus))
	type gs struct {
		i int
		g *gpu.GpuInfo
	}
	gpusWithSpace := []gs{}
	for i := range gpus {
		var gzo uint64
		if len(gpusWithSpace) == 0 {
			gzo = gpuZeroOverhead
		}
		// Only include GPUs that can fit the graph, gpu minimum, the layer buffer and at least more layer
		if gpus[i].FreeMemory < gzo+max(graphPartialOffload, graphFullOffload)+gpus[i].MinimumMemory+2*layerBuffer {
			slog.Debug("gpu has too little memory to allocate any layers", "gpu", gpus[i])
			continue
		}
		gpusWithSpace = append(gpusWithSpace, gs{i, &gpus[i]})
		gpuAllocations[i] += gpus[i].MinimumMemory + layerBuffer // We hold off on graph until we know partial vs. full
	}

	var gpuZeroID int
	if len(gpusWithSpace) > 0 {
		gpuZeroID = gpusWithSpace[0].i
		gpuAllocations[gpuZeroID] += gpuZeroOverhead
	}

	layerSizes = make([]uint64, int(ggml.KV().BlockCount()))
	for i := range int(ggml.KV().BlockCount()) {
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			memoryLayer := blk.size()

			// KV is proportional to the number of layers
			memoryLayer += kv / ggml.KV().BlockCount()
			layerSizes[i] = memoryLayer
			memoryWeights += memoryLayer
		}
	}

	// For all the layers, find where they can fit on the GPU(s)
	for i := range layerSizes {
		if layerSizes[i] == 0 {
			continue
		}
		if opts.NumGPU >= 0 && layerCount >= opts.NumGPU {
			// Stop allocating on GPU(s) once we hit the users target NumGPU
			continue
		}

		// distribute the layers across the GPU(s) that have space
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[i%j]
			used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
			if g.g.FreeMemory > used+layerSizes[i] {
				gpuAllocations[g.i] += layerSizes[i]
				layerCounts[g.i]++
				layerCount++
				break
			} else {
				gpusWithSpace = append(gpusWithSpace[:i%j], gpusWithSpace[i%j+1:]...)
			}
		}

	}
	if layerCount >= int(ggml.KV().BlockCount()) {
		fullyLoaded = true
	} else {
		for i := layerCount; i < int(ggml.KV().BlockCount()); i++ {
			overflow += layerSizes[i]
		}
	}
	// Find where the output fits
	if includeOutput && memoryLayerOutput > 0 && (opts.NumGPU < 0 || layerCount < opts.NumGPU) {
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[layerCount%j]
			used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
			if g.g.FreeMemory > used+memoryLayerOutput {
				gpuAllocations[g.i] += memoryLayerOutput
				layerCounts[g.i]++
				layerCount++
				break
			}
		}
		if layerCount < int(ggml.KV().BlockCount())+1 {
			fullyLoaded = false
			overflow += memoryLayerOutput
		}
	}

	// Add the applicable (full or partial) graph allocations
	for i := range gpus {
		if layerCounts[i] <= 0 {
			continue
		}
		if fullyLoaded {
			gpuAllocations[i] += graphFullOffload
		} else {
			gpuAllocations[i] += graphPartialOffload
		}
	}
	if fullyLoaded {
		graphOffload = graphFullOffload
	} else {
		graphOffload = graphPartialOffload
	}

	// Summaries for the log
	var memoryRequiredPartial, memoryRequiredTotal uint64
	for i := range gpuAllocations {
		memoryRequiredPartial += gpuAllocations[i]

	}
	memoryRequiredTotal = memoryRequiredPartial + overflow

	tensorSplit := ""
	if len(gpus) > 1 {
		splits := make([]string, len(gpus))
		for i, count := range layerCounts {
			splits[i] = strconv.Itoa(count)
		}
		tensorSplit = strings.Join(splits, ",")
	}
	allocationsList := []string{}
	for _, a := range gpuAllocations {
		allocationsList = append(allocationsList, format.HumanBytes2(a))
	}

	slog.Info(
		"offload to gpu",
		slog.Group(
			"layers",
			// requested number of layers to offload
			"requested", opts.NumGPU,
			// The number of layers the model has (including output)
			"model", int(ggml.KV().BlockCount())+1,
			// estimated number of layers that can be offloaded
			"offload", layerCount,
			// multi-gpu split for tesnors
			"split", tensorSplit,
		),
		slog.Group(
			"memory",
			// memory available by GPU for offloading
			"available", availableList,
			slog.Group(
				"required",
				// memory required for full offloading
				"full", format.HumanBytes2(memoryRequiredTotal),
				// memory required to offload layers.estimate layers
				"partial", format.HumanBytes2(memoryRequiredPartial),
				// memory of KV cache
				"kv", format.HumanBytes2(kv),
				// Allocations across the GPUs
				"allocations", allocationsList,
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
		return MemoryEstimate{
			Layers:    0,
			Graph:     0,
			VRAMSize:  0,
			TotalSize: memoryRequiredTotal,
			GPUSizes:  []uint64{},
		}
	}
	if layerCount == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
		return MemoryEstimate{
			Layers:    0,
			Graph:     0,
			VRAMSize:  0,
			TotalSize: memoryRequiredTotal,
			GPUSizes:  []uint64{},
		}
	}

	return MemoryEstimate{
		Layers:      layerCount,
		Graph:       graphOffload,
		VRAMSize:    memoryRequiredPartial,
		TotalSize:   memoryRequiredTotal,
		TensorSplit: tensorSplit,
		GPUSizes:    gpuAllocations,
	}
}
