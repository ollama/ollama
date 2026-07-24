package server

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// llamaServerLoadPlan is the llama-server preflight result. It collects the
// GGUF metadata, placement, request options, launch config, and memory estimate
// needed by Scheduler.load while keeping llama-server sizing rules in one
// place.
type llamaServerLoadPlan struct {
	model        *ggml.GGML
	gpus         []ml.DeviceInfo
	requestOpts  api.Options
	launchOpts   api.Options
	config       llm.LlamaServerConfig
	systemInfo   ml.SystemInfo
	contextShift bool
	useMMapAuto  bool

	numParallel      int
	completion       bool
	effectiveContext int
	memory           loadMemoryAssessment
}

const (
	llamaServerGenerationBatchDefault     = 512
	llamaServerGenerationBatchConstrained = 256
	llamaServerGenerationBatchMedium      = 1024
	llamaServerGenerationBatchLarge       = 2048

	llamaServerGenerationBatchMediumHeadroomPercent = 75
	llamaServerGenerationBatchLargeHeadroomPercent  = 60
	llamaServerLoadHeadroomPercent                  = 80
	llamaServerDefaultFitTargetMiB                  = 1024

	llamaServerFitTargetEnv = "LLAMA_ARG_FIT_TARGET"
)

// newLlamaServerLoadPlan translates a scheduler load proposal into the concrete
// llama-server launch plan. It reads GGUF metadata up front so placement,
// automatic context/batch sizing, mmap defaults, and eviction checks all use
// the same predicted memory view before the subprocess is started.
func newLlamaServerLoadPlan(req *LlmRequest, proposal loadProposal) (llamaServerLoadPlan, error) {
	f, err := llm.LoadModel(req.model.ModelPath, 1024)
	if err != nil {
		return llamaServerLoadPlan{}, err
	}

	effectiveCtx := effectiveLlamaServerContext(req.opts.NumCtx, f, proposal.numParallel)
	predictedModel := llm.PredictServerVRAM(req.model.ModelPath, f, effectiveCtx)
	loadGpus, launchOpts := selectLlamaServerPlacement(proposal.systemInfo, proposal.gpus, predictedModel, req.opts)
	available, gpuFree, systemLimited := availableMemoryForPlacement(proposal.systemInfo, loadGpus, launchOpts)

	requestOpts := optionsWithAutomaticGenerationBatch(
		req.opts,
		req.numBatchAuto,
		proposal.completion,
		effectiveCtx,
		predictedModel,
		available,
		llm.LlamaServerFlashAttention(loadGpus),
		loadGpus,
	)
	launchOpts.NumBatch = requestOpts.NumBatch

	config := llamaServerConfigForModel(req.model)
	config.ContextShift = resolveContextShift(req.shift, req.model)
	config.PredictedVRAM = predictedModel + generationBatchSurchargeForCompletion(proposal.completion, launchOpts.NumBatch)
	config.AvailableVRAM = available
	config.FitTargetMiB = llamaServerFitTargetForRunner()

	return llamaServerLoadPlan{
		model:        f,
		gpus:         loadGpus,
		requestOpts:  requestOpts,
		launchOpts:   launchOpts,
		config:       config,
		systemInfo:   proposal.systemInfo,
		contextShift: config.ContextShift,

		numParallel:      proposal.numParallel,
		completion:       proposal.completion,
		effectiveContext: effectiveCtx,
		memory: loadMemoryAssessment{
			predictedModel: predictedModel,
			predictedLoad:  config.PredictedVRAM,
			available:      available,
			gpuFree:        gpuFree,
			systemLimited:  systemLimited,
		},
	}, nil
}

// maybeRetryLoadFailure handles llama-server OOMs that surface only after the
// runner starts. The fit estimate is imperfect, and auto selected context and
// batch size can be too aggressive even when preflight predicts the load will
// fit. It also covers the case where the pre-load fit check missed allocation
// pressure from resident runners. Projector GPU-offload OOMs are retried inside
// llamaServerRunner.Load before this hook sees the error.
func (p llamaServerLoadPlan) maybeRetryLoadFailure(req *LlmRequest, systemInfo ml.SystemInfo, loadedCount int, err error) bool {
	if req.loadRetryAttempted || !llm.IsOutOfMemory(err) {
		return false
	}
	if p.maybeReduceContextForLoadRetry(req, systemInfo, loadedCount, err) {
		return true
	}
	if loadedCount == 0 {
		return false
	}

	req.loadRetryAttempted = true
	slog.Warn("model load failed; evicting all other models and retrying once", "model", req.model.ModelPath, "error", err)
	return true
}

func (p llamaServerLoadPlan) maybeReduceContextForLoadRetry(req *LlmRequest, systemInfo ml.SystemInfo, loadedCount int, err error) bool {
	if !req.numCtxAuto {
		return false
	}

	oldNumCtx := req.opts.NumCtx
	oldNumBatch := req.opts.NumBatch
	effectiveNumCtx := oldNumCtx
	if trainCtx := int(p.model.KV().ContextLength()); trainCtx > 0 && effectiveNumCtx > trainCtx {
		effectiveNumCtx = trainCtx
	}

	newNumCtx, ok := nextLowerAutoNumCtx(effectiveNumCtx)
	if !ok || newNumCtx >= oldNumCtx {
		return false
	}

	opts := p.requestOpts
	opts.NumCtx = newNumCtx
	predictedCtx := effectiveLlamaServerContext(opts.NumCtx, p.model, p.numParallel)
	predictedVRAM := llm.PredictServerVRAM(req.model.ModelPath, p.model, predictedCtx)
	available, _, _ := availableMemoryForPlacement(systemInfo, p.gpus, p.launchOpts)
	opts = optionsWithAutomaticGenerationBatch(opts, req.numBatchAuto, p.completion, predictedCtx, predictedVRAM, available, llm.LlamaServerFlashAttention(p.gpus), p.gpus)

	req.loadRetryAttempted = true
	req.opts = opts
	slog.Warn("llama-server load failed; reducing automatic context and retrying once",
		"model", req.model.ModelPath,
		"old_num_ctx", oldNumCtx,
		"effective_num_ctx", effectiveNumCtx,
		"new_num_ctx", newNumCtx,
		"old_num_batch", oldNumBatch,
		"new_num_batch", opts.NumBatch,
		"loaded_count", loadedCount,
		"evict_all", loadedCount > 0,
		"error", err)
	return true
}

func (p llamaServerLoadPlan) assessLoadedRunnerFit(requireFull bool, loadedCount int) loadedRunnerFit {
	if !requireFull || loadedCount == 0 || len(p.gpus) == 0 {
		return loadedRunnerFitSkipped
	}
	if explicitPartialGPUOffload(p.launchOpts, p.model) {
		return loadedRunnerFitSkipped
	}
	if !p.memory.fitsWithHeadroom(llamaServerLoadHeadroomPercent) {
		return loadedRunnerNeedsEviction
	}
	return loadedRunnerFits
}

func (p llamaServerLoadPlan) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("backend", "llama-server"),
		slog.String("predicted", format.HumanBytes2(p.memory.predictedLoad)),
		slog.Int("predicted_num_ctx", p.effectiveContext),
		slog.Int("num_batch", p.launchOpts.NumBatch),
		slog.String("available", format.HumanBytes2(p.memory.available)),
		slog.String("gpu_free", format.HumanBytes2(p.memory.gpuFree)),
		slog.Bool("system_limited", p.memory.systemLimited),
	)
}

func (s *Scheduler) applyLlamaServerMmapDefaults(req *LlmRequest, plan llamaServerLoadPlan, systemInfo ml.SystemInfo) llamaServerLoadPlan {
	requestOpts := plan.requestOpts
	launchOpts := plan.launchOpts
	disableMmap := false

	if requestOpts.UseMMap == nil {
		reason := ""
		hasCUDA := false
		hasMetal := false
		allCPU := len(plan.gpus) > 0
		for _, gpu := range plan.gpus {
			hasCUDA = hasCUDA || strings.EqualFold(gpu.Library, "cuda")
			hasMetal = hasMetal || strings.EqualFold(gpu.Library, "metal")
			allCPU = allCPU && strings.EqualFold(gpu.Library, "cpu")
		}

		switch {
		case requestOpts.NumGPU == 0 || len(plan.gpus) == 0 || allCPU:
			reason = "cpu"
		case runtime.GOOS == "windows" && hasCUDA:
			reason = "windows_cuda"
		case hasMetal && requestOpts.NumGPU > 0 && uint64(requestOpts.NumGPU) < plan.model.KV().BlockCount()+1:
			reason = "metal_partial_offload"
		case hasMetal && requestOpts.NumGPU < 0 && plan.memory.predictedModel > 0 && plan.memory.available > 0 && plan.memory.predictedModel > plan.memory.available:
			reason = "metal_partial_offload"
		}

		if reason != "" {
			disableMmap = true
			slog.Info("disabling mmap for llama-server load by default",
				"model", req.model.ModelPath,
				"reason", reason)
		} else if runtime.GOOS == "linux" {
			placementGpus := gpusForPlacement(plan.gpus, launchOpts)
			allDiscrete := len(placementGpus) > 0
			for _, gpu := range placementGpus {
				allDiscrete = allDiscrete && !gpu.Integrated
			}

			modelSize := modelFileSize(req.model.ModelPath)
			loadedMmapSize := s.loadedMmapModelSizeLocked()
			predictedFits := plan.memory.predictedModel > 0 &&
				plan.memory.available > 0 &&
				plan.memory.predictedModel <= plan.memory.available*llamaServerLoadHeadroomPercent/100
			pressure := modelSize + loadedMmapSize + mmapHostPressureHeadroom(systemInfo.TotalMemory)

			// Only back off mmap when we still expect the model to fit on a
			// discrete GPU. If VRAM is already tight, disabling mmap can make
			// partial CPU offload worse by turning file-backed mappings into
			// anonymous memory.
			disableMmap = modelSize > 0 &&
				systemInfo.FreeMemory > 0 &&
				allDiscrete &&
				predictedFits &&
				systemInfo.FreeMemory < pressure
			if disableMmap {
				slog.Info("disabling mmap for llama-server load due to host memory pressure",
					"model", req.model.ModelPath,
					"model_size", format.HumanBytes2(modelSize),
					"loaded_mmap_size", format.HumanBytes2(loadedMmapSize),
					"headroom", format.HumanBytes2(mmapHostPressureHeadroom(systemInfo.TotalMemory)),
					"system_free", format.HumanBytes2(systemInfo.FreeMemory),
					"system_total", format.HumanBytes2(systemInfo.TotalMemory),
					"predicted_vram", format.HumanBytes2(plan.memory.predictedModel),
					"available_vram", format.HumanBytes2(plan.memory.available),
				)
			}
		}
	}
	if disableMmap {
		useMmap := false
		requestOpts.UseMMap = &useMmap
		plan.useMMapAuto = true
	}

	launchOpts.UseMMap = requestOpts.UseMMap
	plan.requestOpts = requestOpts
	plan.launchOpts = launchOpts
	return plan
}

func (p llamaServerLoadPlan) applyToRequest(req *LlmRequest) {
	req.opts = p.requestOpts
	req.useMMapAuto = p.useMMapAuto
	req.contextShift = p.contextShift
}

func (p llamaServerLoadPlan) gpusForLoad() []ml.DeviceInfo {
	return p.gpus
}

func (p llamaServerLoadPlan) newServer(s *Scheduler, req *LlmRequest) (llm.LlamaServer, error) {
	server, err := s.newServerFn(
		p.systemInfo,
		p.gpus,
		req.model.ModelPath,
		p.model,
		req.model.AdapterPaths,
		req.model.ProjectorPaths,
		p.launchOpts,
		p.numParallel,
		p.config,
	)
	if err == nil {
		return server, nil
	}

	// Some older models are not compatible with newer versions of llama.cpp.
	// Show a generalized compatibility error until there is a better way to
	// check for model compatibility.
	if errors.Is(err, ggml.ErrUnsupportedFormat) || strings.Contains(err.Error(), "failed to load model") {
		err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
	}
	return nil, err
}

func (p llamaServerLoadPlan) retryPlanner() loadRetryPlanner {
	return p
}

func (p llamaServerLoadPlan) trainContext() int {
	return modelTrainContext(p.model)
}

type llamaServerLayerOffloadReporter interface {
	LayerOffloadStatus() (gpuLayers, totalLayers uint64, overflow int, ok bool)
}

func (p llamaServerLoadPlan) maybeRetryCPUSpill(req *LlmRequest, runner llm.LlamaServer, requireFull bool, loadedCount int, totalSize, vramSize uint64) bool {
	if req.loadRetryAttempted || !requireFull || loadedCount == 0 || len(p.gpus) == 0 {
		return false
	}

	reporter, ok := runner.(llamaServerLayerOffloadReporter)
	if !ok {
		return false
	}

	gpuLayers, totalLayers, overflow, ok := reporter.LayerOffloadStatus()
	if !ok || (gpuLayers >= totalLayers && overflow == 0) {
		return false
	}
	// Explicit partial GPU offload intentionally leaves layers on CPU.
	if req.opts.NumGPU > 0 && uint64(req.opts.NumGPU) < totalLayers {
		return false
	}

	req.loadRetryAttempted = true
	slog.Warn("llama-server spilled layers to CPU with other models resident; evicting residents and retrying",
		"model", req.model.ModelPath,
		"loaded_count", loadedCount,
		"gpu_layers", gpuLayers,
		"total_layers", totalLayers,
		"fit_overflow_layers", overflow,
		"size", format.HumanBytes2(totalSize),
		"vram", format.HumanBytes2(vramSize))
	return true
}

func explicitPartialGPUOffload(opts api.Options, f *ggml.GGML) bool {
	if opts.NumGPU <= 0 || f == nil {
		return false
	}

	return uint64(opts.NumGPU) < f.KV().BlockCount()+1
}

func effectiveLlamaServerContext(numCtx int, f *ggml.GGML, numParallel int) int {
	return effectiveModelContext(numCtx, f) * max(numParallel, 1)
}

func optionsWithAutomaticGenerationBatch(opts api.Options, numBatchAuto, completion bool, effectiveCtx int, predictedVRAM, availableMemory uint64, flashAttention ml.FlashAttentionType, gpus []ml.DeviceInfo) api.Options {
	if completion && numBatchAuto {
		opts.NumBatch = automaticGenerationBatch(effectiveCtx, predictedVRAM, availableMemory, flashAttention, gpus)
	}
	return opts
}

func generationBatchSurchargeForCompletion(completion bool, batch int) uint64 {
	if !completion {
		return 0
	}
	return generationBatchSurcharge(batch)
}

func automaticGenerationBatch(effectiveCtx int, predictedVRAM, availableMemory uint64, flashAttention ml.FlashAttentionType, gpus []ml.DeviceInfo) int {
	if flashAttention == ml.FlashAttentionDisabled && hasCUDADevice(gpus) {
		if constrainedCUDAWithoutFlashAttention(effectiveCtx, gpus) {
			return llamaServerGenerationBatchConstrained
		}
		return llamaServerGenerationBatchDefault
	}

	batch := generationBatchForContext(effectiveCtx)
	for batch > llamaServerGenerationBatchDefault && !generationBatchFits(batch, predictedVRAM, availableMemory) {
		batch = nextLowerGenerationBatch(batch)
	}
	return batch
}

func hasCUDADevice(gpus []ml.DeviceInfo) bool {
	return slices.ContainsFunc(gpus, func(gpu ml.DeviceInfo) bool {
		return gpu.Library == "CUDA"
	})
}

func constrainedCUDAWithoutFlashAttention(effectiveCtx int, gpus []ml.DeviceInfo) bool {
	if effectiveCtx <= 4096 {
		return false
	}
	return slices.ContainsFunc(gpus, func(gpu ml.DeviceInfo) bool {
		if gpu.Library != "CUDA" {
			return false
		}
		memory := gpu.FreeMemory
		if memory == 0 || (gpu.TotalMemory > 0 && gpu.TotalMemory < memory) {
			memory = gpu.TotalMemory
		}
		return memory > 0 && memory <= 8*format.GibiByte
	})
}

func generationBatchForContext(effectiveCtx int) int {
	switch {
	case effectiveCtx > 32768:
		return llamaServerGenerationBatchLarge
	case effectiveCtx > 4096:
		return llamaServerGenerationBatchMedium
	default:
		return llamaServerGenerationBatchDefault
	}
}

func generationBatchFits(batch int, predictedVRAM, availableMemory uint64) bool {
	if predictedVRAM == 0 || availableMemory == 0 {
		return true
	}

	threshold := availableMemory * llamaServerLoadHeadroomPercent / 100
	if predictedVRAM > threshold {
		return false
	}
	if !generationBatchHasHeadroom(batch, predictedVRAM, availableMemory) {
		return false
	}

	return generationBatchSurcharge(batch) <= threshold-predictedVRAM
}

func generationBatchHasHeadroom(batch int, predictedVRAM, availableMemory uint64) bool {
	switch {
	case batch >= llamaServerGenerationBatchLarge:
		return predictedVRAM <= availableMemory*llamaServerGenerationBatchLargeHeadroomPercent/100
	case batch >= llamaServerGenerationBatchMedium:
		return predictedVRAM <= availableMemory*llamaServerGenerationBatchMediumHeadroomPercent/100
	default:
		return true
	}
}

func nextLowerGenerationBatch(batch int) int {
	switch {
	case batch > llamaServerGenerationBatchMedium:
		return llamaServerGenerationBatchMedium
	default:
		return llamaServerGenerationBatchDefault
	}
}

func generationBatchSurcharge(batch int) uint64 {
	switch {
	case batch >= llamaServerGenerationBatchLarge:
		return 2 * format.GibiByte
	case batch >= llamaServerGenerationBatchMedium:
		return 768 * format.MebiByte
	default:
		return 0
	}
}

func nextLowerAutoNumCtx(numCtx int) (int, bool) {
	switch {
	case numCtx > 32768:
		return 32768, true
	case numCtx > 4096:
		return 4096, true
	default:
		return 0, false
	}
}

func availableMemoryForLoad(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo) (available, gpuFree uint64, systemLimited bool) {
	var sharedGPUFree uint64
	var discreteGPUFree uint64
	var reserve uint64
	for i, gpu := range gpus {
		gpuFree += gpu.FreeMemory
		reserve += llamaServerDeviceReserve(gpu, i)
		if gpu.Integrated {
			sharedGPUFree += gpu.FreeMemory
		} else {
			discreteGPUFree += gpu.FreeMemory
		}
	}

	available = gpuFree
	// On iGPUs, GPU free memory can be a static or slowly refreshed device
	// baseline. updateFreeSpace has already subtracted known Ollama runner
	// allocations from that baseline. Current system free memory is a separate
	// live measurement that already includes those loaded runners, so use the
	// smaller value for shared-memory GPUs without discounting discrete VRAM.
	if systemInfo.FreeMemory > 0 && sharedGPUFree > 0 && systemInfo.FreeMemory < sharedGPUFree {
		available, systemLimited = discreteGPUFree+systemInfo.FreeMemory, true
	}

	if reserve >= available {
		available = 0
	} else {
		available -= reserve
	}
	return available, gpuFree, systemLimited
}

func availableMemoryForPlacement(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, opts api.Options) (available, gpuFree uint64, systemLimited bool) {
	placementGpus := gpusForPlacement(gpus, opts)
	if len(placementGpus) == 1 && opts.MainGPU != nil {
		gpu := placementGpus[0]
		gpuFree = gpu.FreeMemory
		available = availableMemoryForGPU(systemInfo, gpu, 0)
		systemLimited = gpu.Integrated && systemInfo.FreeMemory > 0 && systemInfo.FreeMemory < gpu.FreeMemory
		return available, gpuFree, systemLimited
	}

	return availableMemoryForLoad(systemInfo, placementGpus)
}

func gpusForPlacement(gpus []ml.DeviceInfo, opts api.Options) []ml.DeviceInfo {
	if opts.MainGPU != nil && *opts.MainGPU >= 0 && *opts.MainGPU < len(gpus) {
		return []ml.DeviceInfo{gpus[*opts.MainGPU]}
	}

	return gpus
}

func selectLlamaServerPlacement(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, predictedVRAM uint64, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	launchOpts := opts
	if len(gpus) <= 1 || opts.NumGPU == 0 {
		return gpus, launchOpts
	}

	groups := ml.ByLibrary(gpus)
	if len(groups) == 0 {
		return gpus, launchOpts
	}

	if opts.MainGPU != nil {
		gpu, available, ok := bestExplicitMainGPU(systemInfo, groups, *opts.MainGPU)
		if !ok {
			selected := bestGPUGroupByAvailableMemory(systemInfo, groups)
			slog.Warn("requested main_gpu is outside the selected GPU group; passing value through to llama-server",
				"main_gpu", *opts.MainGPU,
				"gpu_count", len(selected))
			logSelectedGPUGroup(gpus, selected)
			return selected, launchOpts
		}

		selected, launchOpts := singleLlamaServerGPUPlacement(gpu, launchOpts)
		slog.Info("selecting requested single GPU for llama-server model",
			"requested_main_gpu", *opts.MainGPU,
			"main_gpu", *launchOpts.MainGPU,
			"id", gpu.ID,
			"filter_id", gpu.FilterID,
			"library", gpu.Library,
			"name", gpu.Name,
			"description", gpu.Description,
			"integrated", gpu.Integrated,
			"available", format.HumanBytes2(available))
		logSelectedGPUGroup(gpus, selected)
		return selected, launchOpts
	}

	if !envconfig.SchedSpread() && predictedVRAM > 0 {
		gpu, available, ok := bestSingleGPUFit(systemInfo, groups, predictedVRAM)
		if ok {
			selected, launchOpts := singleLlamaServerGPUPlacement(gpu, launchOpts)
			slog.Info("selecting single GPU for llama-server model",
				"main_gpu", *launchOpts.MainGPU,
				"id", gpu.ID,
				"filter_id", gpu.FilterID,
				"library", gpu.Library,
				"name", gpu.Name,
				"description", gpu.Description,
				"integrated", gpu.Integrated,
				"predicted", format.HumanBytes2(predictedVRAM),
				"available", format.HumanBytes2(available))
			logSelectedGPUGroup(gpus, selected)
			return selected, launchOpts
		}
	}

	selected := bestGPUGroupByAvailableMemory(systemInfo, groups)
	logSelectedGPUGroup(gpus, selected)
	return selected, launchOpts
}

func singleLlamaServerGPUPlacement(gpu ml.DeviceInfo, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	// llama-server sees only this GPU after filtering, so the selected device is
	// index 0 for both main_gpu and per-device fit-target values.
	mainGPU := 0
	opts.MainGPU = &mainGPU
	return []ml.DeviceInfo{gpu}, opts
}

func bestExplicitMainGPU(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo, mainGPU int) (gpu ml.DeviceInfo, available uint64, ok bool) {
	if mainGPU < 0 {
		return ml.DeviceInfo{}, 0, false
	}

	for _, group := range groups {
		if mainGPU >= len(group) {
			continue
		}
		candidate := group[mainGPU]
		candidateAvailable := availableMemoryForGPU(systemInfo, candidate, 0)
		if !ok || betterPlacementGPU(candidate, candidateAvailable, gpu, available) {
			gpu = candidate
			available = candidateAvailable
			ok = true
		}
	}

	return gpu, available, ok
}

func bestSingleGPUFit(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo, predictedVRAM uint64) (gpu ml.DeviceInfo, available uint64, ok bool) {
	for _, group := range groups {
		for _, candidate := range group {
			candidateAvailable := availableMemoryForGPU(systemInfo, candidate, 0)
			if predictedVRAM > candidateAvailable {
				continue
			}
			if !ok || betterPlacementGPU(candidate, candidateAvailable, gpu, available) {
				gpu = candidate
				available = candidateAvailable
				ok = true
			}
		}
	}

	return gpu, available, ok
}

func betterPlacementGPU(candidate ml.DeviceInfo, candidateAvailable uint64, current ml.DeviceInfo, currentAvailable uint64) bool {
	if candidate.Integrated != current.Integrated {
		return !candidate.Integrated
	}

	return candidateAvailable > currentAvailable
}

func bestGPUGroupByAvailableMemory(systemInfo ml.SystemInfo, groups [][]ml.DeviceInfo) []ml.DeviceInfo {
	var best []ml.DeviceInfo
	var bestAvailable uint64
	for _, group := range groups {
		available, _, _ := availableMemoryForLoad(systemInfo, group)
		if best == nil || betterPlacementGroup(group, available, best, bestAvailable) {
			best = group
			bestAvailable = available
		}
	}

	return best
}

func betterPlacementGroup(candidate []ml.DeviceInfo, candidateAvailable uint64, current []ml.DeviceInfo, currentAvailable uint64) bool {
	candidateDiscrete := hasDiscreteGPU(candidate)
	currentDiscrete := hasDiscreteGPU(current)
	if candidateDiscrete != currentDiscrete {
		return candidateDiscrete
	}

	return candidateAvailable > currentAvailable
}

func hasDiscreteGPU(gpus []ml.DeviceInfo) bool {
	for _, gpu := range gpus {
		if !gpu.Integrated {
			return true
		}
	}
	return false
}

func availableMemoryForGPU(systemInfo ml.SystemInfo, gpu ml.DeviceInfo, visibleIndex int) uint64 {
	available := gpu.FreeMemory
	if gpu.Integrated && systemInfo.FreeMemory > 0 && systemInfo.FreeMemory < gpu.FreeMemory {
		available = systemInfo.FreeMemory
	}

	reserve := llamaServerDeviceReserve(gpu, visibleIndex)
	if reserve >= available {
		return 0
	}
	return available - reserve
}

func llamaServerDeviceReserve(gpu ml.DeviceInfo, visibleIndex int) uint64 {
	return gpu.MinimumMemory() + max(envconfig.GpuOverhead(), llamaServerFitTargetBytes(visibleIndex))
}

func llamaServerFitTargetBytes(visibleIndex int) uint64 {
	value := envconfig.Var(llamaServerFitTargetEnv)
	if value != "" {
		targets := strings.Split(value, ",")
		if len(targets) == 1 {
			visibleIndex = 0
		}
		if visibleIndex >= 0 && visibleIndex < len(targets) {
			if target, err := strconv.ParseUint(strings.TrimSpace(targets[visibleIndex]), 10, 64); err == nil {
				return target * format.MebiByte
			}
		}
	}

	return llamaServerDefaultFitTargetMiB * format.MebiByte
}

func llamaServerFitTargetForRunner() uint64 {
	if envconfig.Var(llamaServerFitTargetEnv) != "" {
		return 0
	}

	overhead := envconfig.GpuOverhead()
	defaultFitTarget := uint64(llamaServerDefaultFitTargetMiB * format.MebiByte)
	if overhead <= defaultFitTarget {
		return 0
	}

	return (overhead + format.MebiByte - 1) / format.MebiByte
}

func logSelectedGPUGroup(all, selected []ml.DeviceInfo) {
	if len(selected) == 0 || len(selected) == len(all) {
		return
	}

	slog.Info("selecting GPU backend for llama-server model",
		"library", selected[0].Library,
		"gpu_count", len(selected),
		"available_gpu_count", len(all))
}

func mmapHostPressureHeadroom(totalMemory uint64) uint64 {
	if totalMemory == 0 {
		return 8 * format.GigaByte
	}
	return max(8*format.GigaByte, totalMemory/10)
}

func modelFileSize(path string) uint64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return uint64(info.Size())
}

func (s *Scheduler) loadedMmapModelSizeLocked() uint64 {
	var total uint64
	for _, r := range s.loaded {
		if !runnerUsesMmap(r) {
			continue
		}
		if size := modelFileSize(r.modelPath); size > 0 {
			total += size
		} else {
			total += r.totalSize
		}
	}
	return total
}

func runnerUsesMmap(r *runnerRef) bool {
	if r == nil || r.Options == nil || r.Options.UseMMap == nil {
		return true
	}
	return *r.Options.UseMMap
}
