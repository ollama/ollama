package server

import (
	"log/slog"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// loadMemoryAssessment is the scheduler-facing memory view for a proposed
// load. It keeps eviction decisions in Scheduler.load while hiding the
// backend-specific details used to produce the estimate.
type loadMemoryAssessment struct {
	predictedModel uint64
	predictedLoad  uint64
	available      uint64
	gpuFree        uint64
	systemLimited  bool
}

// loadProposal is the stable set of scheduler inputs that backend-specific
// planners need to estimate and prepare a runner load.
type loadProposal struct {
	systemInfo  ml.SystemInfo
	gpus        []ml.DeviceInfo
	numParallel int
	completion  bool
}

// loadRetryPlanner lets Scheduler.load ask the selected backend plan to apply
// a retry when backend-specific Load behavior proves the first attempt was too
// aggressive. Runners whose startup failures surface later from
// WaitUntilRunning should not wire this in.
type loadRetryPlanner interface {
	maybeRetryLoadFailure(req *LlmRequest, systemInfo ml.SystemInfo, loadedCount int, err error) bool
	maybeRetryCPUSpill(req *LlmRequest, runner llm.LlamaServer, requireFull bool, loadedCount int, totalSize, vramSize uint64) bool
}

// loadedRunnerFit reports whether a proposed load can safely coexist with
// already loaded runners, when that backend has enough information to tell.
type loadedRunnerFit int

const (
	loadedRunnerFitSkipped loadedRunnerFit = iota
	loadedRunnerFits
	loadedRunnerNeedsEviction
)

// runnerLoadPlan is the backend-specific preflight result Scheduler.load needs
// after selecting llama-server or MLX. It keeps runner construction and backend
// load details in the plan while leaving eviction execution in the scheduler.
type runnerLoadPlan interface {
	slog.LogValuer
	assessLoadedRunnerFit(requireFull bool, loadedCount int) loadedRunnerFit
	applyToRequest(req *LlmRequest)
	gpusForLoad() []ml.DeviceInfo
	newServer(s *Scheduler, req *LlmRequest) (llm.LlamaServer, error)
	retryPlanner() loadRetryPlanner
	trainContext() int
}

func (a loadMemoryAssessment) fitsAvailable() bool {
	if a.predictedLoad == 0 {
		return true
	}
	if a.available == 0 {
		return false
	}
	return a.predictedLoad <= a.available
}

func (a loadMemoryAssessment) fitsWithHeadroom(percent uint64) bool {
	if a.predictedLoad == 0 {
		return true
	}
	if a.available == 0 {
		return false
	}
	return a.predictedLoad <= a.available*percent/100
}
