package llm

import (
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
)

// pickBestFullFitByLibrary will try to find the optimal placement of the model in the available GPUs where the model fully fits
// The list of GPUs returned will always be the same brand (library)
// If the model can not be fit fully within the available GPU(s) nil is returned
func pickBestFullFitByLibrary(f *ggml.GGML, modelPath string, projectors []string, adapters []string, opts api.Options, gpus discover.GpuInfoList, numParallel int) discover.GpuInfoList {
	for _, gl := range gpus.ByLibrary() {
		sgl := append(make(discover.GpuInfoList, 0, len(gl)), gl...)

		// TODO - potentially sort by performance capability, existing models loaded, etc.
		// TODO - Eliminate any GPUs that already have envconfig.MaxRunners loaded on them
		// Note: at present, this will favor most current available VRAM descending and ignoring faster GPU speed in mixed setups
		sort.Sort(sort.Reverse(discover.ByFreeMemory(sgl)))

		if !envconfig.SchedSpread() {
			// Try to pack into as few GPUs as possible, starting from 1 GPU
			for numGPUs := 1; numGPUs <= len(sgl); numGPUs++ {
				gpuSubset := sgl[:numGPUs]
				ok, estimatedVRAM := predictServerFit(gpuSubset, f, adapters, projectors, opts, numParallel)

				if ok {
					slog.Info("new model will fit in available VRAM across minimum required GPUs, loading",
						"model", modelPath,
						"library", sgl[0].Library,
						"parallel", numParallel,
						"required", format.HumanBytes2(estimatedVRAM),
						"gpus", numGPUs)
					return gpuSubset
				}
			}
		} else {
			// TODO future refinements
			// - if multiple Libraries, see if any single GPU in any Library will fit
			// - try subsets of GPUs instead of just falling back to 1 or all in a family

			// Now try all the GPUS (OLLAMA_SCHED_SPREAD is set)
			if ok, estimatedVRAM := predictServerFit(sgl, f, adapters, projectors, opts, numParallel); ok {
				slog.Info("new model will fit in available VRAM, loading",
					"model", modelPath,
					"library", sgl[0].Library,
					"parallel", numParallel,
					"required", format.HumanBytes2(estimatedVRAM),
					"gpus", len(sgl))
				return sgl
			}
		}
	}
	return nil
}

// If multiple Libraries are detected, pick the Library which loads the most layers for the model
func pickBestPartialFitByLibrary(f *ggml.GGML, projectors []string, adapters []string, opts api.Options, gpus discover.GpuInfoList, numParallel int) discover.GpuInfoList {
	byLibrary := gpus.ByLibrary()
	if len(byLibrary) <= 1 {
		return gpus
	}
	var bestEstimate uint64
	var bestFit int
	for i, gl := range byLibrary {
		_, estimatedVRAM := predictServerFit(gl, f, adapters, projectors, opts, numParallel)
		if estimatedVRAM > bestEstimate {
			bestEstimate = estimatedVRAM
			bestFit = i
		}
	}
	return byLibrary[bestFit]
}

// This algorithm looks for a complete fit to determine if we need to unload other models
func predictServerFit(allGpus discover.GpuInfoList, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (bool, uint64) {
	// Split up the GPUs by type and try them
	var estimatedVRAM uint64
	for _, gpus := range allGpus.ByLibrary() {
		var layerCount int
		estimate := estimateGPULayers(gpus, f, projectors, opts, numParallel)
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

		if len(gpus) == 1 && gpus[0].Library == "cpu" && estimate.TotalSize <= gpus[0].FreeMemory {
			return true, estimatedVRAM
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
	TensorSplit []int

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
func estimateGPULayers(gpus []discover.GpuInfo, f *ggml.GGML, projectors []string, opts api.Options, numParallel int) MemoryEstimate {
	// Graph size for a partial offload, applies to all GPUs
	var graphPartialOffload uint64

	// Graph size when all layers are offloaded, applies to all GPUs
	var graphFullOffload uint64

	// Final graph offload once we know full or partial
	var graphOffload uint64

	// Projectors loaded into GPU0 only
	var llamaEngineProjectorWeights uint64

	// Projectors loaded with output layer
	var ollamaEngineProjectorWeights uint64
	var ollamaEngineProjectorGraph uint64

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
		llamaEngineProjectorWeights += projectorMemoryRequirements(projector)
	}
	if llamaEngineProjectorWeights == 0 {
		ollamaEngineProjectorWeights, ollamaEngineProjectorGraph = f.VisionGraphSize()
	}

	layers := f.Tensors().GroupLayers()
	// add one layer worth of memory as a buffer
	if blk0, ok := layers["blk.0"]; ok {
		layerSize = blk0.Size()
	} else {
		slog.Warn("model missing blk.0 layer size")
	}

	useFlashAttention := (envconfig.FlashAttention() || f.FlashAttention()) &&
		discover.GetGPUInfo().FlashAttentionSupported() &&
		f.SupportsFlashAttention()

	var kvct string
	if useFlashAttention {
		requested := strings.ToLower(envconfig.KvCacheType())
		if requested != "" && f.SupportsKVCacheType(requested) {
			kvct = requested
		}
	}

	kv, graphPartialOffload, graphFullOffload := f.GraphSize(uint64(opts.NumCtx), uint64(min(opts.NumCtx, opts.NumBatch)), numParallel, kvct, useFlashAttention)

	if len(kv) > 0 {
		layerSize += kv[0]
	}

	var kvTotal uint64
	for _, kvLayer := range kv {
		kvTotal += kvLayer
	}

	if graphPartialOffload == 0 {
		headsKV := f.KV().HeadCountKVMin()
		if headsKV == 0 {
			headsKV = 1
		}
		gqa := f.KV().HeadCountMax() / headsKV
		graphPartialOffload = gqa * kvTotal / 6
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

	// Output layer handled at the end if we have space
	if layer, ok := layers["output_norm"]; ok {
		memoryLayerOutput += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		memoryLayerOutput += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		memoryLayerOutput += layer.Size()
	}

	gpuZeroOverhead := llamaEngineProjectorWeights

	// Reduce set of GPUs to only those that have sufficient space to fit overhead and at least one layer
	var layerCount int
	tensorSplit := make([]int, len(gpus))
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
	} else {
		overflow += gpuZeroOverhead
	}

	// For all the layers, find where they can fit on the GPU(s)
	for i := int(f.KV().BlockCount()) - 1; i >= 0; i-- {
		// Some models have inconsistent layer sizes
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			layerSize = blk.Size()
			layerSize += kv[i]
			memoryWeights += blk.Size()
		}

		if opts.NumGPU >= 0 && layerCount >= opts.NumGPU {
			// Stop allocating on GPU(s) once we hit the users target NumGPU
			overflow += layerSize
			continue
		}

		// distribute the layers across the GPU(s) that have space
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[i%j]
			used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
			if g.g.FreeMemory > overhead+used+layerSize {
				gpuAllocations[g.i] += layerSize
				tensorSplit[g.i]++
				layerCount++
				break
			} else {
				gpusWithSpace = append(gpusWithSpace[:i%j], gpusWithSpace[i%j+1:]...)
			}
		}

		if len(gpusWithSpace) == 0 {
			overflow += layerSize
		}
	}
	if layerCount >= int(f.KV().BlockCount()) {
		fullyLoaded = true
	}

	// Determine if we need to consider output then find where it fits
	memoryLastLayer := memoryLayerOutput + ollamaEngineProjectorWeights + ollamaEngineProjectorGraph
	if memoryLastLayer > 0 {
		if opts.NumGPU < 0 || layerCount < opts.NumGPU {
			for j := len(gpusWithSpace); j > 0; j-- {
				g := gpusWithSpace[layerCount%j]
				used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
				if g.g.FreeMemory > overhead+used+memoryLastLayer {
					gpuAllocations[g.i] += memoryLastLayer
					tensorSplit[g.i]++
					layerCount++
					break
				}
			}
		}

		if layerCount < int(f.KV().BlockCount())+1 {
			fullyLoaded = false
			overflow += memoryLastLayer
		}
	}

	// Add the applicable (full or partial) graph allocations
	for i := range gpus {
		if tensorSplit[i] <= 0 {
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
		kv:                  kvTotal,
		allocationsList:     allocationsList,
		memoryWeights:       memoryWeights,
		memoryLayerOutput:   memoryLayerOutput,
		graphFullOffload:    graphFullOffload,
		graphPartialOffload: graphPartialOffload,
		projectorWeights:    llamaEngineProjectorWeights + ollamaEngineProjectorWeights,
		projectorGraph:      ollamaEngineProjectorGraph,
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
				"total", format.HumanBytes2(m.memoryWeights+m.memoryLayerOutput),
				// memory of repeating layers
				"repeating", format.HumanBytes2(m.memoryWeights),
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

func projectorMemoryRequirements(filename string) (weights uint64) {
	file, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer file.Close()

	ggml, err := ggml.Decode(file, 1024)
	if err != nil {
		return 0
	}

	for _, layer := range ggml.Tensors().GroupLayers() {
		weights += layer.Size()
	}

	return weights
}
