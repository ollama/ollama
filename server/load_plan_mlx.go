package server

import (
	"log/slog"
	"slices"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/imagegen"
	"github.com/ollama/ollama/x/mlxrunner"
)

// mlxLoadPlan is the MLX preflight result. MLX can estimate model size and
// context before loading, but startup allocation failures are reported later
// from runner readiness checks rather than synchronously from Load.
type mlxLoadPlan struct {
	modelName         string
	softContextLength int
	modelContext      int
	image             bool
	hasGPU            bool
	gpus              []ml.DeviceInfo
	memory            loadMemoryAssessment
}

// newMLXLoadPlan translates a scheduler load proposal into the MLX runner plan.
// It uses manifest metadata for context and memory preflight before the
// subprocess starts so scheduler eviction decisions use the same estimate MLX
// will recheck during Load.
func newMLXLoadPlan(req *LlmRequest, proposal loadProposal) (mlxLoadPlan, error) {
	plan := mlxLoadPlan{
		modelName:         req.model.ShortName,
		softContextLength: req.opts.NumCtx,
		image:             slices.Contains(req.model.Config.Capabilities, "image"),
		gpus:              slices.Clone(proposal.gpus),
	}
	if plan.image {
		return plan, nil
	}

	if err := mlxrunner.CheckPlatformSupport(); err != nil {
		return mlxLoadPlan{}, err
	}

	estimate, err := mlxrunner.EstimateLoad(plan.modelName, proposal.gpus)
	if err != nil {
		return mlxLoadPlan{}, err
	}
	plan.modelContext = estimate.ContextLength
	plan.softContextLength = effectiveContext(plan.softContextLength, plan.modelContext)
	plan.hasGPU = estimate.HasGPU

	plan.memory = loadMemoryAssessment{
		predictedModel: estimate.ModelSize,
		predictedLoad:  estimate.ModelSize,
		available:      estimate.Available,
		gpuFree:        estimate.GPUFree,
	}
	return plan, nil
}

func (p mlxLoadPlan) applyToRequest(_ *LlmRequest) {}

func (p mlxLoadPlan) gpusForLoad() []ml.DeviceInfo {
	return p.gpus
}

func (p mlxLoadPlan) newServer(_ *Scheduler, _ *LlmRequest) (llm.LlamaServer, error) {
	if p.image {
		return imagegen.NewServer(p.modelName)
	}
	return mlxrunner.NewClient(p.modelName, p.softContextLength)
}

func (p mlxLoadPlan) retryPlanner() loadRetryPlanner {
	// MLX Load only rechecks preflight fit and starts the subprocess. Real
	// startup allocation failures surface later from WaitUntilRunning, after
	// Scheduler.load has returned, so there is no synchronous Load retry hook
	// to wire here.
	return nil
}

func (p mlxLoadPlan) trainContext() int {
	return p.modelContext
}

func (p mlxLoadPlan) assessLoadedRunnerFit(requireFull bool, loadedCount int) loadedRunnerFit {
	if p.image || !p.hasGPU || !requireFull || loadedCount == 0 {
		return loadedRunnerFitSkipped
	}
	if !p.memory.fitsAvailable() {
		return loadedRunnerNeedsEviction
	}
	return loadedRunnerFits
}

func (p mlxLoadPlan) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("backend", "mlx"),
		slog.String("predicted", format.HumanBytes2(p.memory.predictedLoad)),
		slog.Int("num_ctx", p.softContextLength),
		slog.Int("model_context", p.modelContext),
		slog.String("available", format.HumanBytes2(p.memory.available)),
		slog.String("gpu_free", format.HumanBytes2(p.memory.gpuFree)),
	)
}
