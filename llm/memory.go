package llm

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
)

// This algorithm looks for a complete fit to determine if we need to unload other models
func PredictServerFit(allGpus discover.GpuInfoList, f *ggml.GGML, adapters, projectors []string, opts api.Options) (bool, uint64) {
	// Split up the GPUs by type and try them
	var estimatedVRAM uint64
	for _, gpus := range allGpus.ByLibrary() {
		var layerCount int
		estimate := EstimateGPULayers(gpus, f, projectors, opts)
		layerCount, estimatedVRAM = estimate.Layers, estimate.VRAMSize
		if opts.NumGPU < 0 {
			if layerCount > 0 && layerCount >= int(f.KV().BlockCount()+1) {
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

	// internal fields for logging purposes
	inferenceLibrary    string
	layersRequested     int
	layersModel         int
	availableList       []string
	kv                  uint64
	allocationsList     []string
	memoryWeights       uint64
	memoryLayerOutput   uint64
	graphFullOffload    uint64
	graphPartialOffload uint64

	projectorWeights, projectorGraph uint64
}

// Given a model and one or more GPU targets, predict how many layers and bytes we can load, and the total size
// The GPUs provided must all be the same Library
func EstimateGPULayers(gpus []discover.GpuInfo, f *ggml.GGML, projectors []string, opts api.Options) MemoryEstimate {
	// Graph size for a partial offload, applies to all GPUs
	var graphPartialOffload uint64

	// Graph size when all layers are offloaded, applies to all GPUs
	var graphFullOffload uint64

	// Final graph offload once we know full or partial
	var graphOffload uint64

	// Projectors loaded into GPU0 only
	var projectorWeights uint64
	var projectorGraph uint64

	// Conditional output size on GPU 0
	var memoryLayerOutput uint64

	// The sizes of a layer
	var layerSize uint64

	// The sum of all the layer sizes (just for logging)
	var memoryWeights uint64

	// True if all the layers are loaded
	var fullyLoaded bool

	// Overflow that didn't fit into the GPU
	var overflow uint64

	overhead := envconfig.GpuOverhead()
	availableList := make([]string, len(gpus))
	for i, gpu := range gpus {
		availableList[i] = format.HumanBytes2(gpu.FreeMemory)
	}
	slog.Debug("evaluating", "library", gpus[0].Library, "gpu_count", len(gpus), "available", availableList)

	for _, projector := range projectors {
		weight, graph := projectorMemoryRequirements(projector)
		projectorWeights += weight
		projectorGraph += graph

		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	layers := f.Tensors().GroupLayers()
	// add one layer worth of memory as a buffer
	if blk0, ok := layers["blk.0"]; ok {
		layerSize = blk0.Size()
	} else {
		slog.Warn("model missing blk.0 layer size")
	}

	var kvct string
	if envconfig.FlashAttention() &&
		discover.GetGPUInfo().FlashAttentionSupported() &&
		f.SupportsFlashAttention() {
		requested := strings.ToLower(envconfig.KvCacheType())
		if requested != "" && f.SupportsKVCacheType(requested) {
			kvct = requested
		}
	}

	kv, graphPartialOffload, graphFullOffload := f.GraphSize(uint64(opts.NumCtx), uint64(min(opts.NumCtx, opts.NumBatch)), kvct)

	// KV is proportional to the number of layers
	layerSize += kv / f.KV().BlockCount()

	if graphPartialOffload == 0 {
		graphPartialOffload = f.KV().GQA() * kv / 6
	}
	if graphFullOffload == 0 {
		graphFullOffload = graphPartialOffload
	}

	// on metal there's no partial offload overhead
	if gpus[0].Library == "metal" {
		graphPartialOffload = graphFullOffload
	} else if len(gpus) > 1 {
		// multigpu should always use the partial graph size
		graphFullOffload = graphPartialOffload
	}

	if layer, ok := layers["output_norm"]; ok {
		memoryLayerOutput += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		memoryLayerOutput += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		memoryLayerOutput += layer.Size()
	}

	// Output layer handled at the end if we have space
	gpuZeroOverhead := projectorWeights + projectorGraph

	// Reduce set of GPUs to only those that have sufficient space to fit overhead and at least one layer
	var layerCount int
	layerCounts := make([]int, len(gpus))
	gpuAllocations := make([]uint64, len(gpus))
	type gs struct {
		i int
		g *discover.GpuInfo
	}
	gpusWithSpace := []gs{}
	for i := range gpus {
		var gzo uint64
		if len(gpusWithSpace) == 0 {
			gzo = gpuZeroOverhead
		}
		// Only include GPUs that can fit the graph, gpu minimum, the layer buffer and at least more layer
		if gpus[i].FreeMemory < overhead+gzo+max(graphPartialOffload, graphFullOffload)+gpus[i].MinimumMemory+2*layerSize {
			slog.Debug("gpu has too little memory to allocate any layers",
				"id", gpus[i].ID,
				"library", gpus[i].Library,
				"variant", gpus[i].Variant,
				"compute", gpus[i].Compute,
				"driver", fmt.Sprintf("%d.%d", gpus[i].DriverMajor, gpus[i].DriverMinor),
				"name", gpus[i].Name,
				"total", format.HumanBytes2(gpus[i].TotalMemory),
				"available", format.HumanBytes2(gpus[i].FreeMemory),
				"minimum_memory", gpus[i].MinimumMemory,
				"layer_size", format.HumanBytes2(layerSize),
				"gpu_zer_overhead", format.HumanBytes2(gzo),
				"partial_offload", format.HumanBytes2(graphPartialOffload),
				"full_offload", format.HumanBytes2(graphFullOffload),
			)
			continue
		}
		gpusWithSpace = append(gpusWithSpace, gs{i, &gpus[i]})
		gpuAllocations[i] += gpus[i].MinimumMemory + layerSize // We hold off on graph until we know partial vs. full
	}

	var gpuZeroID int
	if len(gpusWithSpace) > 0 {
		gpuZeroID = gpusWithSpace[0].i
		gpuAllocations[gpuZeroID] += gpuZeroOverhead
	}

	// For all the layers, find where they can fit on the GPU(s)
	for i := range int(f.KV().BlockCount()) {
		// Some models have inconsistent layer sizes
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			layerSize = blk.Size()
			layerSize += kv / f.KV().BlockCount()
		}
		memoryWeights += layerSize

		if opts.NumGPU >= 0 && layerCount >= opts.NumGPU {
			// Stop allocating on GPU(s) once we hit the users target NumGPU
			continue
		}

		// distribute the layers across the GPU(s) that have space
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[i%j]
			used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
			if g.g.FreeMemory > overhead+used+layerSize {
				gpuAllocations[g.i] += layerSize
				layerCounts[g.i]++
				layerCount++
				break
			} else {
				gpusWithSpace = append(gpusWithSpace[:i%j], gpusWithSpace[i%j+1:]...)
			}
		}
	}
	if layerCount >= int(f.KV().BlockCount()) {
		fullyLoaded = true
	} else {
		for i := layerCount; i < int(f.KV().BlockCount()); i++ {
			overflow += layerSize
		}
	}

	// Determine if we need to consider output then find where it fits
	if memoryLayerOutput > 0 && (opts.NumGPU < 0 || layerCount < opts.NumGPU) {
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[layerCount%j]
			used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
			if g.g.FreeMemory > overhead+used+memoryLayerOutput {
				gpuAllocations[g.i] += memoryLayerOutput
				layerCounts[g.i]++
				layerCount++
				break
			}
		}

		if layerCount < int(f.KV().BlockCount())+1 {
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

	estimate := MemoryEstimate{
		TotalSize: memoryRequiredTotal,
		Layers:    0,
		Graph:     0,
		VRAMSize:  0,
		GPUSizes:  []uint64{},

		inferenceLibrary:    gpus[0].Library,
		layersRequested:     opts.NumGPU,
		layersModel:         int(f.KV().BlockCount()) + 1,
		availableList:       availableList,
		kv:                  kv,
		allocationsList:     allocationsList,
		memoryWeights:       memoryWeights,
		memoryLayerOutput:   memoryLayerOutput,
		graphFullOffload:    graphFullOffload,
		graphPartialOffload: graphPartialOffload,
		projectorWeights:    projectorWeights,
		projectorGraph:      projectorGraph,
	}

	if gpus[0].Library == "cpu" {
		return estimate
	}
	if layerCount == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
		return estimate
	}
	estimate.Layers = layerCount
	estimate.Graph = graphOffload
	estimate.VRAMSize = memoryRequiredPartial
	estimate.TotalSize = memoryRequiredTotal
	estimate.TensorSplit = tensorSplit
	estimate.GPUSizes = gpuAllocations
	return estimate
}

func (m MemoryEstimate) LogValue() slog.Value {
	attrs := []slog.Attr{
		slog.String("library", m.inferenceLibrary),
		slog.Group(
			"layers",
			// requested number of layers to offload
			"requested", m.layersRequested,
			// The number of layers the model has (including output)
			"model", m.layersModel,
			// estimated number of layers that can be offloaded
			"offload", m.Layers,
			// multi-gpu split for tensors
			"split", m.TensorSplit,
		),
		slog.Group(
			"memory",
			// memory available by GPU for offloading
			"available", m.availableList,
			"gpu_overhead", format.HumanBytes2(envconfig.GpuOverhead()),
			slog.Group(
				"required",
				// memory required for full offloading
				"full", format.HumanBytes2(m.TotalSize),
				// memory required to offload layers.estimate layers
				"partial", format.HumanBytes2(m.VRAMSize),
				// memory of KV cache
				"kv", format.HumanBytes2(m.kv),
				// Allocations across the GPUs
				"allocations", m.allocationsList,
			),
			slog.Group(
				"weights",
				// memory of the weights
				"total", format.HumanBytes2(m.memoryWeights),
				// memory of repeating layers
				"repeating", format.HumanBytes2(m.memoryWeights-m.memoryLayerOutput),
				// memory of non-repeating layers
				"nonrepeating", format.HumanBytes2(m.memoryLayerOutput),
			),
			slog.Group(
				"graph",
				// memory of graph when fully offloaded
				"full", format.HumanBytes2(m.graphFullOffload),
				// memory of graph when not fully offloaded
				"partial", format.HumanBytes2(m.graphPartialOffload),
			),
		),
	}

	if m.projectorWeights > 0 {
		attrs = append(attrs, slog.Group(
			"projector",
			"weights", format.HumanBytes2(m.projectorWeights),
			"graph", format.HumanBytes2(m.projectorGraph),
		))
	}

	return slog.GroupValue(attrs...)
}

func projectorMemoryRequirements(filename string) (weights, graphSize uint64) {
	file, err := os.Open(filename)
	if err != nil {
		return 0, 0
	}
	defer file.Close()

	ggml, _, err := ggml.Decode(file, 0)
	if err != nil {
		return 0, 0
	}

	for _, layer := range ggml.Tensors().GroupLayers() {
		weights += layer.Size()
	}

	switch arch := ggml.KV().Architecture(); arch {
	case "mllama":
		kv := func(n string) uint64 {
			if v, ok := ggml.KV()[arch+".vision."+n].(uint32); ok {
				return uint64(v)
			}

			return 0
		}

		imageSize := kv("image_size")

		maxNumTiles := kv("max_num_tiles")
		embeddingLength := kv("embedding_length")
		headCount := kv("attention.head_count")

		numPatches := (imageSize / kv("patch_size")) * (imageSize / kv("patch_size"))
		if _, ok := ggml.Tensors().GroupLayers()["v"]["class_embd"]; ok {
			numPatches++
		}

		numPaddedPatches := numPatches + 8 - (numPatches%8)%8

		graphSize = 4 * (8 +
			imageSize*imageSize*kv("num_channels")*maxNumTiles +
			embeddingLength*numPatches*maxNumTiles +
			9*embeddingLength*numPaddedPatches*maxNumTiles +
			numPaddedPatches*maxNumTiles*numPaddedPatches*maxNumTiles*headCount)
	}

	return weights, graphSize
}
