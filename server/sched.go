package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint

	// loadRetryAttempted is set after a backend load outcome triggers a retry
	// through the eviction path. It prevents infinite retry on persistent load
	// failures or CPU spills.
	loadRetryAttempted bool

	// numCtxAuto is true when NumCtx came from Ollama's automatic VRAM-tier
	// default rather than explicit request, model, or environment config.
	numCtxAuto bool

	// numBatchAuto is true when NumBatch came from Ollama's default options
	// rather than an explicit request or model option.
	numBatchAuto bool

	// useMMapAuto is true when UseMMap was derived by the scheduler rather than
	// explicitly requested.
	useMMapAuto bool

	// contextShift is a llama-server launch attribute resolved from the
	// request-level shift option before scheduling.
	contextShift bool
	shift        *bool
}

func (pending *LlmRequest) fail(err error) {
	if err == nil {
		return
	}

	select {
	case pending.errCh <- err:
	default:
	}
}

func (pending *LlmRequest) failIfCanceled() bool {
	if err := pending.ctx.Err(); err != nil {
		pending.fail(err)
		return true
	}
	return false
}

type Scheduler struct {
	pendingReqCh  chan *LlmRequest
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan any

	// loadedMu protects loaded and activeLoading
	loadedMu sync.Mutex

	// activeLoading is the model that we are currently working on loading,
	// including by evicting one or more other models. We can only load
	// one model at a time but new requests to models that already loaded can
	// happen in parallel
	activeLoading llm.LlamaServer
	loaded        map[string]*runnerRef

	loadFn          func(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool
	newServerFn     func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error)
	getGpuFn        func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo
	getSystemInfoFn func() ml.SystemInfo
	waitForRecovery time.Duration
}

// Default automatic value for number of models we allow per GPU
// Model will still need to fit in VRAM, but loading many small models
// on a large GPU can cause stalling
var defaultModelsPerGPU = 3

var ErrMaxQueue = errors.New("server busy, please try again.  maximum pending requests exceeded")

func InitScheduler(ctx context.Context) *Scheduler {
	maxQueue := envconfig.MaxQueue()
	sched := &Scheduler{
		pendingReqCh:    make(chan *LlmRequest, maxQueue),
		finishedReqCh:   make(chan *LlmRequest, maxQueue),
		expiredCh:       make(chan *runnerRef, maxQueue),
		unloadedCh:      make(chan any, maxQueue),
		loaded:          make(map[string]*runnerRef),
		newServerFn:     llm.NewLlamaServer,
		getGpuFn:        discover.GPUDevices,
		getSystemInfoFn: discover.GetSystemInfo,
		waitForRecovery: 5 * time.Second,
	}
	sched.loadFn = sched.load
	return sched
}

// schedulerModelKey returns the scheduler map key for a model.
// GGUF-backed models use ModelPath; safetensors/image models without a
// ModelPath use manifest digest so distinct models don't collide.
func schedulerModelKey(m *Model) string {
	if m == nil {
		return ""
	}
	if m.ModelPath != "" {
		return m.ModelPath
	}
	if m.Digest != "" {
		return "digest:" + m.Digest
	}
	if m.Name != "" {
		return "name:" + m.Name
	}
	if m.ShortName != "" {
		return "short:" + m.ShortName
	}
	return ""
}

func resolveContextShift(shift *bool, m *Model) bool {
	if shift != nil {
		return *shift
	}

	return supportsContextShift(m)
}

func supportsContextShift(m *Model) bool {
	if m == nil {
		return true
	}

	if m.Config.ModelFamily == "deepseek2" || slices.Contains(m.Config.ModelFamilies, "deepseek2") {
		return false
	}

	return true
}

func effectiveModelContext(numCtx int, f *ggml.GGML) int {
	return effectiveContext(numCtx, modelTrainContext(f))
}

func modelTrainContext(f *ggml.GGML) int {
	if f == nil {
		return 0
	}

	return int(f.KV().ContextLength())
}

func effectiveContext(numCtx, trainCtx int) int {
	if trainCtx > 0 && numCtx > trainCtx {
		return trainCtx
	}

	return numCtx
}

func (s *Scheduler) getRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration, numCtxAuto bool, numBatchAuto bool, shift *bool) (chan *runnerRef, chan error) {
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	if m.CheckCapabilities(model.CapabilityVision) == nil {
		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	contextShift := false
	if m.ModelPath != "" {
		contextShift = resolveContextShift(shift, m)
	}

	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
		numCtxAuto:      numCtxAuto,
		numBatchAuto:    numBatchAuto,
		contextShift:    contextShift,
		shift:           shift,
	}

	key := schedulerModelKey(req.model)
	if req.failIfCanceled() {
		return req.successCh, req.errCh
	}

	s.loadedMu.Lock()
	runner := s.loaded[key]
	s.loadedMu.Unlock()
	if runner != nil && !runner.needsReload(c, req) {
		if req.tryUseLoadedRunner(runner, s.finishedReqCh) {
			return req.successCh, req.errCh
		}
	}

	if req.failIfCanceled() {
		return req.successCh, req.errCh
	} else {
		select {
		case s.pendingReqCh <- req:
		default:
			req.fail(ErrMaxQueue)
		}
	}
	return req.successCh, req.errCh
}

// acquireRunner schedules the model and blocks until it is loaded or fails,
// returning its server. The context must be canceled to decrement the ref
// count and release the runner.
func (s *Scheduler) acquireRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration, numCtxAuto bool, numBatchAuto bool, shift *bool) (llm.LlamaServer, error) {
	runnerCh, errCh := s.getRunner(c, m, opts, sessionDuration, numCtxAuto, numBatchAuto, shift)
	select {
	case runner := <-runnerCh:
		runner.refMu.Lock()
		llama := runner.llama
		runner.refMu.Unlock()
		return llama, nil
	case err := <-errCh:
		return nil, err
	case <-c.Done():
		return nil, c.Err()
	}
}

// Returns immediately, spawns go routines for the scheduler which will shutdown when ctx is done
func (s *Scheduler) Run(ctx context.Context) {
	slog.Debug("starting llm scheduler")
	go func() {
		s.processPending(ctx)
	}()

	go func() {
		s.processCompleted(ctx)
	}()
}

func (s *Scheduler) processPending(ctx context.Context) {
	maxRunners := envconfig.MaxRunners()

nextPending:
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case pending := <-s.pendingReqCh:
			// Block other requests until we get this pending request running
			pending.schedAttempts++

			if pending.failIfCanceled() {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			logutil.Trace("processing incoming request", "model", pending.model.ModelPath)

			for {
				if pending.failIfCanceled() {
					slog.Debug("pending request cancelled or timed out, stopping scheduling")
					continue nextPending
				}

				var runnerToExpire *runnerRef
				pendingKey := schedulerModelKey(pending.model)
				s.loadedMu.Lock()
				runner := s.loaded[pendingKey]
				loadedCount := len(s.loaded)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				s.loadedMu.Unlock()

				if runner != nil {
					if runner.needsReload(ctx, pending) {
						slog.Debug("reloading", "runner", runner)
						runnerToExpire = runner
					} else {
						// Runner is usable, return it
						logutil.Trace("using existing loaded runner", "model", pendingKey)
						if pending.tryUseLoadedRunner(runner, s.finishedReqCh) {
							break
						}
						continue
					}
				} else if maxRunners > 0 && loadedCount >= int(maxRunners) {
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload()
				} else {
					// Either no models are loaded or below envconfig.MaxRunners
					// Get a refreshed GPU list
					var gpus []ml.DeviceInfo
					if pending.opts.NumGPU == 0 {
						gpus = []ml.DeviceInfo{}
					} else {
						logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
						gpus = s.getGpuFn(ctx, runnersSnapshot)
					}
					logutil.Trace("refreshing system information", "model", pending.model.ModelPath)
					systemInfo := s.getSystemInfoFn()
					if maxRunners <= 0 {
						// No user specified MaxRunners, so figure out what automatic setting to use for the next load attempt
						if pending.opts.NumGPU == 0 {
							// Need to get actual GPU list to set the correct default max models
							logutil.Trace("refreshing GPU list", "model", pending.model.ModelPath)
							g := s.getGpuFn(ctx, runnersSnapshot)
							maxRunners = uint(defaultModelsPerGPU * max(len(g), 1))
						} else {
							maxRunners = uint(defaultModelsPerGPU * max(len(gpus), 1))
						}
						slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "gpu_count", len(gpus))
					}

					// Update free memory from currently loaded models
					logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
					s.updateFreeSpace(gpus)
					if pending.failIfCanceled() {
						slog.Debug("pending request cancelled or timed out before load")
						continue nextPending
					}

					if loadedCount == 0 {
						// No models loaded. Load the model but prefer the best fit.
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						if s.loadFn(pending, systemInfo, gpus, false) {
							slog.Debug("first model load requested retry", "model", pending.model.ModelPath)
							continue
						}
						break
					}

					// More than one loaded model, so we have to see if the
					// new one fits
					logutil.Trace("loading additional model", "model", pending.model.ModelPath)
					needEvict := s.loadFn(pending, systemInfo, gpus, true)
					if !needEvict {
						slog.Debug("new model fits with existing models, loading")
						break
					}

					// Load retry path: Load failed or llama-server spilled to
					// CPU with resident models still loaded. Evict all of them,
					// wait for every unload, then loop back to retry once.
					// load() has already set loadRetryAttempted so a second
					// failure falls through to the fail-fast path.
					if pending.loadRetryAttempted {
						if !s.evictAllAndWait(ctx, pendingKey) {
							return
						}
						continue
					}

					runnerToExpire = s.findRunnerToUnload()
				}

				if runnerToExpire == nil {
					// While we were performing load calculations, the loaded runner(s) unloaded in parallel
					// so findRunnerToUnload returned no runners.  We'll try again and the loadedCount should be zero
					slog.Debug("runner to expire was nil, retrying")
					continue
				}
				// Trigger an expiration to unload once it's done
				var expireNow bool
				runnerToExpire.refMu.Lock()
				slog.Debug("resetting model to expire immediately to make room", "runner", runnerToExpire, "refCount", runnerToExpire.refCount)
				expireNow = runnerToExpire.requestImmediateExpirationLocked()
				runnerToExpire.refMu.Unlock()
				if expireNow {
					s.expiredCh <- runnerToExpire
				}
				// Wait for the unload to happen
				slog.Debug("waiting for pending requests to complete and unload to occur", "runner", runnerToExpire)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-s.unloadedCh:
					slog.Debug("unload completed", "runner", runnerToExpire)
					if pending.failIfCanceled() {
						slog.Debug("pending request cancelled or timed out after unload")
						continue nextPending
					}
					continue
				}
			}
		case <-s.unloadedCh:
			// An unload request when there are no pending request can be ignored
			slog.Debug("ignoring unload event with no pending requests")
		}
	}
}

func (s *Scheduler) processCompleted(ctx context.Context) {
	// Process completed requests, expired timers, and unloading models
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler completed loop")
			return
		case finished := <-s.finishedReqCh:
			finishedKey := schedulerModelKey(finished.model)
			s.loadedMu.Lock()
			runner := s.loaded[finishedKey]
			s.loadedMu.Unlock()
			if runner == nil {
				slog.Error("finished request signal received after model unloaded", "modelPath", finishedKey)
				continue
			}
			runner.refMu.Lock()
			runner.refCount--
			expireNow := false
			if runner.refCount <= 0 {
				if runner.expireOnIdle || runner.sessionDuration <= 0 {
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					runner.expireOnIdle = false
					expireNow = true
				} else if runner.expireTimer == nil {
					slog.Debug("runner with non-zero duration has gone idle, adding timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
						slog.Debug("timer expired, expiring to unload", "runner", runner)
						runner.refMu.Lock()
						if runner.expireTimer == nil {
							runner.refMu.Unlock()
							return
						}
						runner.expireTimer = nil
						expireNow := runner.refCount <= 0
						if !expireNow {
							runner.expireOnIdle = true
						}
						runner.refMu.Unlock()
						if !expireNow {
							return
						}
						s.expiredCh <- runner
					})
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
				} else {
					slog.Debug("runner with non-zero duration has gone idle, resetting timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer.Reset(runner.sessionDuration)
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
				}
			}
			slog.Debug("after processing request finished event", "runner", runner, "refCount", runner.refCount)
			runner.refMu.Unlock()
			if expireNow {
				s.expiredCh <- runner
			}
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "runner", runner)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				runner.expireOnIdle = true
				runner.refMu.Unlock()
				slog.Debug("expired event with positive ref count, expiring when idle", "runner", runner, "refCount", runner.refCount)
				continue
			}
			if runner.unloading {
				runner.refMu.Unlock()
				slog.Debug("duplicate expired event, ignoring runner already unloading", "runner", runner)
				continue
			}
			runner.unloading = true
			runner.expireOnIdle = false
			runner.refMu.Unlock()

			s.loadedMu.Lock()
			slog.Debug("got lock to unload expired event", "runner", runner)
			runnerToUnload := s.loaded[runner.modelKey]
			if runnerToUnload == nil {
				// If runnerToUnload is nil, we already processed an event and
				// unloaded it. This double unload can happen if the initial
				// request is canceled and we're trying to load another model
				// that requires this one to be evicted, or the settings change
				// and require a reload
				s.loadedMu.Unlock()
				slog.Debug("duplicate expired event, ignoring", "runner", runner)
			} else if runner.pid != runnerToUnload.pid {
				// If the pids do not match, we likely had multiple load
				// failures for the same model in quick succession due to
				// request context canceled and are draining the queue of
				// events. Ensure the orphaned runner is properly shut down, but
				// do not delete the mismatched loaded runner, or wait for VRAM
				// convergence.
				slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
				s.loadedMu.Unlock()
				runner.unload()
			} else {
				slog.Debug("starting background wait for VRAM recovery", "runner", runner)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				delete(s.loaded, runner.modelKey)
				s.loadedMu.Unlock()
				finished := s.waitForVRAMRecovery(runner, runnersSnapshot)
				runner.unload()
				slog.Debug("runner terminated and removed from list, blocking for VRAM recovery", "runner", runner)
				<-finished
				slog.Debug("sending an unloaded event", "runner", runner)
				s.unloadedCh <- struct{}{}
			}
		}
	}
}

// Complete the pending request and send the runner back to the requester
// Wires up a finished event after the request context is completed
// Updates session duration, and resets expiration timer
func (pending *LlmRequest) tryUseLoadedRunner(runner *runnerRef, finished chan *LlmRequest) bool {
	if pending.failIfCanceled() {
		return true
	}

	runner.refMu.Lock()
	if pending.failIfCanceled() {
		runner.refMu.Unlock()
		return true
	}
	if runner.unloading {
		runner.refMu.Unlock()
		return false
	}
	runner.refCount++
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}
	runner.refMu.Unlock()

	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
	return true
}

// load creates a new model based on req and loads it. If requireFull is true then the model must be loaded fully onto GPUs
// (if any). Returns whether the scheduler needs to evict a model to make this one fit.
func (s *Scheduler) load(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
	numParallel := max(int(envconfig.NumParallel()), 1)
	completion := req.model.CheckCapabilities(model.CapabilityCompletion) == nil

	// Embedding models should always be loaded with parallel=1
	if !completion {
		numParallel = 1
	}

	// Some architectures are not safe with num_parallel > 1.
	// ref: https://github.com/ollama/ollama/issues/4165
	if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe", "nemotron_h_omni"}, req.model.Config.ModelFamily) && numParallel != 1 {
		numParallel = 1
		slog.Warn("model architecture does not currently support parallel requests", "architecture", req.model.Config.ModelFamily)
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	s.loadedMu.Lock()
	llama := s.activeLoading
	loadGpus := gpus
	var retryPlanner loadRetryPlanner
	var trainContext int
	loadedCountAtLoadStart := len(s.loaded)
	proposal := loadProposal{
		systemInfo:  systemInfo,
		gpus:        gpus,
		numParallel: numParallel,
		completion:  completion,
	}

	if llama == nil {
		var plan runnerLoadPlan
		var err error
		if req.model.IsMLX() {
			plan, err = newMLXLoadPlan(req, proposal)
		} else {
			var llamaPlan llamaServerLoadPlan
			llamaPlan, err = newLlamaServerLoadPlan(req, proposal)
			if err == nil {
				plan = s.applyLlamaServerMmapDefaults(req, llamaPlan, systemInfo)
			}
		}
		if err != nil {
			slog.Info("failed to plan model load", "model", req.model.ShortName, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			return false
		}

		switch plan.assessLoadedRunnerFit(requireFull, loadedCountAtLoadStart) {
		case loadedRunnerNeedsEviction:
			slog.Info("model predicted to exceed available memory, evicting",
				"load_plan", plan,
				"system_free", format.HumanBytes2(systemInfo.FreeMemory))
			s.loadedMu.Unlock()
			return true
		case loadedRunnerFits:
			slog.Info("model fits alongside existing models",
				"load_plan", plan,
				"system_free", format.HumanBytes2(systemInfo.FreeMemory))
		}

		plan.applyToRequest(req)
		loadGpus = plan.gpusForLoad()
		retryPlanner = plan.retryPlanner()
		trainContext = plan.trainContext()
		// newServer starts llama-server now; MLX constructs a client and starts during Load below.
		llama, err = plan.newServer(s, req)
		if err != nil {
			slog.Info("failed to create server", "model", req.model.ShortName, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			return false
		}

		s.activeLoading = llama
	} else {
		wantPath := req.model.ModelPath
		if wantPath == "" {
			wantPath = req.model.ShortName
		}
		if s.activeLoading.ModelPath() != wantPath {
			panic(fmt.Errorf("attempting to load different model after eviction (original %v new %v)", s.activeLoading.ModelPath(), wantPath))
		}
	}

	s.loadedMu.Unlock()

	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	for _, gpu := range loadGpus {
		available := gpu.FreeMemory - envconfig.GpuOverhead() - gpu.MinimumMemory()
		if gpu.FreeMemory < envconfig.GpuOverhead()+gpu.MinimumMemory() {
			available = 0
		}
		slog.Info("gpu memory", "id", gpu.ID, "library", gpu.Library,
			"available", format.HumanBytes2(available),
			"free", format.HumanBytes2(gpu.FreeMemory),
			"minimum", format.HumanBytes2(gpu.MinimumMemory()),
			"overhead", format.HumanBytes2(envconfig.GpuOverhead()))
	}

	// Load completes backend startup: llama-server waits for readiness, while MLX starts here.
	gpuIDs, err := llama.Load(req.ctx, systemInfo, loadGpus, requireFull)
	if err != nil {
		if errors.Is(err, llm.ErrLoadRequiredFull) {
			if !requireFull {
				// No other models loaded, yet we still don't fit, so report an error
				slog.Info("model is too large for system memory", "requireFull", requireFull)
				llama.Close()
				s.loadedMu.Lock()
				s.activeLoading = nil
				s.loadedMu.Unlock()
				req.errCh <- err
				return false
			}
			return true
		}

		slog.Info("Load failed", "model", req.model.ModelPath, "error", err)
		llama.Close()
		s.loadedMu.Lock()
		s.activeLoading = nil
		s.loadedMu.Unlock()

		s.loadedMu.Lock()
		loadedCount := len(s.loaded)
		s.loadedMu.Unlock()

		// MLX is a no-op here; llama-server may crash during Load, so retry with a smaller auto context or by evicting other models.
		if retryPlanner != nil && retryPlanner.maybeRetryLoadFailure(req, systemInfo, loadedCount, err) {
			return true
		}

		req.errCh <- err
		return false
	}

	totalSize, vramSize := llama.MemorySize()
	// After a successful llama-server Load, retry if it spilled to CPU while other models were resident.
	if retryPlanner != nil && retryPlanner.maybeRetryCPUSpill(req, llama, requireFull, loadedCountAtLoadStart, totalSize, vramSize) {
		llama.Close()
		s.loadedMu.Lock()
		s.activeLoading = nil
		s.loadedMu.Unlock()
		return true
	}
	logTemplateSelection(req.model)

	// Determine if we have discrete GPUs which we should monitor VRAM usage on during shutdown
	usesDiscreteGPU := false
iGPUScan:
	for _, devid := range gpuIDs {
		for _, dev := range loadGpus {
			if dev.DeviceID == devid {
				if !dev.Integrated {
					usesDiscreteGPU = true
					break iGPUScan
				}
			}
		}
	}

	if effectiveNumCtx := llama.ContextLength(); req.model.ModelPath != "" && effectiveNumCtx > 0 {
		req.opts.NumCtx = effectiveNumCtx
		req.contextShift = resolveContextShift(req.shift, req.model)
	}
	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		modelKey:        schedulerModelKey(req.model),
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpuIDs,
		usesDiscreteGPU: usesDiscreteGPU,
		totalSize:       totalSize,
		vramSize:        vramSize,
		loading:         true,
		pid:             llama.Pid(),
		numCtxAuto:      req.numCtxAuto,
		numBatchAuto:    req.numBatchAuto,
		useMMapAuto:     req.useMMapAuto,
		contextShift:    req.contextShift,
		trainContext:    trainContext,
	}
	runner.numParallel = numParallel
	runner.refMu.Lock() // hold lock until running or aborted

	s.loadedMu.Lock()
	var oldRunner *runnerRef
	if existing, ok := s.loaded[runner.modelKey]; ok {
		// Shouldn't happen, but safeguard against leaking a runner
		oldRunner = existing
		slog.Warn("model was still loaded", "old_runner", oldRunner, "new_runner", runner)
	}
	s.activeLoading = nil
	s.loaded[runner.modelKey] = runner
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()
	if oldRunner != nil {
		oldRunner.unload()
	}

	go func() {
		// llama-server usually returns immediately here because Load already
		// waited for startup. Keep the scheduler-level readiness gate for
		// backends whose Load only starts a subprocess, such as MLX.
		if err = llama.WaitUntilRunning(req.ctx); err != nil {
			slog.Error("error loading llama server", "error", err)
			req.errCh <- err
			slog.Debug("triggering expiration for failed load", "runner", runner)
			runner.refMu.Unlock()
			s.expiredCh <- runner
			return
		}
		slog.Debug("finished setting up", "runner", runner)
		if runner.pid < 0 {
			runner.pid = llama.Pid()
		}
		runner.refCount++
		runner.loading = false
		go func() {
			<-req.ctx.Done()
			slog.Debug("context for request finished")
			s.finishedReqCh <- req
		}()
		req.successCh <- runner
		runner.refMu.Unlock()
	}()

	return false
}

func (s *Scheduler) updateFreeSpace(allGpus []ml.DeviceInfo) {
	if len(allGpus) == 0 {
		return
	}
	predMap := map[ml.DeviceID]uint64{} // Sum up the total predicted usage per GPU for all runners
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()
	for _, r := range runners {
		r.refMu.Lock()
		llama := r.llama
		r.refMu.Unlock()
		if llama == nil {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
			continue
		}
		for _, gpu := range allGpus {
			predMap[gpu.DeviceID] += llama.VRAMByGPU(gpu.DeviceID)
		}
	}

	// Now that we've summed up all the GPU usage predictions across all the loaded runners, update the gpu list
	for i := range allGpus {
		if p, ok := predMap[allGpus[i].DeviceID]; ok {
			slog.Debug("gpu reported", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "available", format.HumanBytes2(allGpus[i].FreeMemory))
			if p > allGpus[i].TotalMemory {
				// Shouldn't happen
				slog.Warn("predicted usage exceeds VRAM", "gpu", allGpus[i].ID, "totalMemory", allGpus[i].TotalMemory, "predicted", p)
				allGpus[i].FreeMemory = 0
			} else if (allGpus[i].TotalMemory - p) < allGpus[i].FreeMemory { // predicted free is smaller than reported free, use it
				// TODO maybe we should just always trust our numbers, since cuda's free memory reporting is laggy
				// and we might unload models we didn't actually need to.  The risk is if some other GPU intensive app is loaded
				// after we start our first runner, then we'll never account for that, so picking the smallest free value seems prudent.
				allGpus[i].FreeMemory = allGpus[i].TotalMemory - p
			}
			slog.Info("updated VRAM based on existing loaded models", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "total", format.HumanBytes2(allGpus[i].TotalMemory), "available", format.HumanBytes2(allGpus[i].FreeMemory))
		}
	}
}

type runnerRef struct {
	// refMu guards refCount, loading, unloading, expireOnIdle,
	// sessionDuration, expireTimer, expiresAt, llama, model, Options, gpus, and
	// contextShift. Fields set once at construction, such as modelKey and
	// totalSize, may be read lock-free. LogValue uses a best-effort TryLock so
	// it can still render while refMu is already held.
	refMu sync.Mutex

	refCount     uint // prevent unloading if > 0
	loading      bool // True only during initial load, then false forever
	unloading    bool
	expireOnIdle bool

	sessionDuration time.Duration
	expireTimer     *time.Timer
	expiresAt       time.Time

	llama           llm.LlamaServer
	pid             int
	gpus            []ml.DeviceID // Recorded at time of provisioning
	usesDiscreteGPU bool          // Used to skip VRAM recovery for CPU and iGPU-only runners.
	vramSize        uint64
	totalSize       uint64

	model        *Model
	modelPath    string
	modelKey     string
	numParallel  int
	numCtxAuto   bool
	numBatchAuto bool
	useMMapAuto  bool
	contextShift bool
	trainContext int
	*api.Options
}

// requestImmediateExpirationLocked records that the runner should unload as
// soon as its refcount reaches zero. It returns true when the caller should
// send the runner to expiredCh immediately after releasing refMu.
func (runner *runnerRef) requestImmediateExpirationLocked() bool {
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	runner.sessionDuration = 0
	if runner.refCount > 0 {
		runner.expireOnIdle = true
		return false
	}
	return true
}

func (runner *runnerRef) unload() {
	runner.refMu.Lock()
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	llama := runner.llama
	runner.llama = nil
	runner.model = nil
	runner.Options = nil
	runner.gpus = nil
	runner.contextShift = false
	runner.loading = false
	runner.unloading = true
	runner.refMu.Unlock()

	if llama != nil {
		llama.Close()
	}
}

func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", schedulerModelKey(req.model))
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	if runner.unloading {
		return true
	}

	timeout := 10 * time.Second
	if runner.loading {
		timeout = 2 * time.Minute // Initial load can take a long time for big models on slow systems...
	}

	if runner.Options == nil {
		return true
	}

	// Don't reload runner if num_gpu=-1 was provided
	optsExisting := runner.Options.Runner
	optsNew := req.opts.Runner
	optsNew.NumCtx = effectiveContext(optsNew.NumCtx, runner.trainContext)
	if runner.numCtxAuto && req.numCtxAuto {
		optsNew.NumCtx = optsExisting.NumCtx
	}
	if runner.numBatchAuto && req.numBatchAuto {
		optsNew.NumBatch = optsExisting.NumBatch
	}
	if runner.useMMapAuto && optsNew.UseMMap == nil {
		optsNew.UseMMap = optsExisting.UseMMap
	}
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	contextShift := req.contextShift
	if req.model.ModelPath != "" {
		contextShift = resolveContextShift(req.shift, req.model)
	}
	if runner.contextShift != contextShift {
		return true
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if !reflect.DeepEqual(runner.model.AdapterPaths, req.model.AdapterPaths) || // have the adapters changed?
		!reflect.DeepEqual(runner.model.ProjectorPaths, req.model.ProjectorPaths) || // have the projectors changed?
		(!runner.model.IsMLX() && !reflect.DeepEqual(optsExisting, optsNew)) || // have the runner options changed?
		runner.llama.Ping(ctx) != nil {
		return true
	}

	return false
}

// Free memory reporting on GPUs can lag for a while even after the runner
// exits, so we have to keep checking until we see the available memory recover,
// otherwise subsequent model loads will get far less layers loaded or worse
// case, may completely fall back to CPU mode.
// This routine must be called before the runner unloads so it can establish
// a before and after GPU memory allocation.  The returned channel
// will be notified when we're done waiting, or have timed out and should
// proceed anyway
func (s *Scheduler) waitForVRAMRecovery(runner *runnerRef, runners []ml.FilteredRunnerDiscovery) chan any {
	finished := make(chan any, 1)

	runner.refMu.Lock()
	gpus := slices.Clone(runner.gpus)
	usesDiscreteGPU := runner.usesDiscreteGPU
	vramSize := runner.vramSize
	runner.refMu.Unlock()

	// CPU, Metal and iGPUs don't need checking, so no waiting required
	if len(gpus) == 0 || !usesDiscreteGPU ||
		(len(gpus) == 1 && gpus[0].Library == "Metal") {
		finished <- struct{}{}
		slog.Debug("no need to wait for VRAM recovery", "runner", runner)
		return finished
	}
	start := time.Now()

	// Establish a baseline before we unload
	gpusBefore := s.getGpuFn(context.Background(), runners)
	var totalMemoryBefore, freeMemoryBefore uint64
	for _, gpu := range gpusBefore {
		totalMemoryBefore += gpu.TotalMemory
		freeMemoryBefore += gpu.FreeMemory
	}
	totalMemoryNow := totalMemoryBefore
	freeMemoryNow := freeMemoryBefore

	go func() {
		// typical convergence is 0.5-1.5s - If it takes too long to discover and converge, let the scheduler estimate VRAM usage
		ctx, cancel := context.WithTimeout(context.Background(), s.waitForRecovery)
		defer cancel()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// Query GPUs, look for free to go back up
				gpusNow := s.getGpuFn(ctx, runners)
				totalMemoryNow = 0
				freeMemoryNow = 0
				for _, gpu := range gpusNow {
					totalMemoryNow += gpu.TotalMemory
					freeMemoryNow += gpu.FreeMemory
				}
				if freeMemoryNow > freeMemoryBefore {
					logutil.Trace("gpu VRAM convergence", "percent", int(float32(freeMemoryNow-freeMemoryBefore)/float32(vramSize)*100))
				} else {
					logutil.Trace("gpu VRAM convergence", "percent", 0)
				}
				// If we're within ~75% of the estimated memory usage recovered, bail out
				if float32(freeMemoryNow-freeMemoryBefore) > float32(vramSize)*0.75 {
					slog.Debug(fmt.Sprintf("gpu VRAM free memory converged after %0.2f seconds", time.Since(start).Seconds()), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
					finished <- struct{}{}
					return
				}
			case <-ctx.Done():
				slog.Debug("gpu VRAM usage didn't recover within timeout", "seconds", time.Since(start).Seconds(), "free_before", format.HumanBytes2(freeMemoryBefore), "free_now", format.HumanBytes2(freeMemoryNow), "runner", runner)
				finished <- struct{}{}
				return
			}
		}
	}()
	return finished
}

func (runner *runnerRef) LogValue() slog.Value {
	if runner == nil {
		return slog.StringValue("nil")
	}
	modelID := runner.modelPath
	if modelID == "" {
		modelID = runner.modelKey
	}
	attrs := []slog.Attr{}
	if runner.refMu.TryLock() {
		if runner.model != nil {
			attrs = append(attrs, slog.String("name", runner.model.Name))
		}
		if len(runner.gpus) > 0 {
			attrs = append(attrs,
				slog.Any("inference", slices.Clone(runner.gpus)),
			)
		}
		attrs = append(attrs, slog.Int("pid", runner.pid))
		if runner.Options != nil {
			attrs = append(attrs, slog.Int("num_ctx", runner.Options.NumCtx))
		}
		runner.refMu.Unlock()
	}
	attrs = append(attrs,
		slog.String("size", format.HumanBytes2(runner.totalSize)),
		slog.String("vram", format.HumanBytes2(runner.vramSize)),
		slog.Int("parallel", runner.numParallel),
		slog.String("model", modelID),
	)
	return slog.GroupValue(attrs...)
}

// Implements discover.RunnerDiscovery
func (runner *runnerRef) GetPort() int {
	runner.refMu.Lock()
	llama := runner.llama
	runner.refMu.Unlock()
	if llama != nil {
		return llama.GetPort()
	}
	return -1
}

func (runner *runnerRef) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	runner.refMu.Lock()
	llama := runner.llama
	runner.refMu.Unlock()
	if llama != nil {
		return llama.GetDeviceInfos(ctx)
	}
	return nil
}

func (runner *runnerRef) GetActiveDeviceIDs() []ml.DeviceID {
	runner.refMu.Lock()
	gpus := slices.Clone(runner.gpus)
	runner.refMu.Unlock()
	return gpus
}

func (runner *runnerRef) HasExited() bool {
	runner.refMu.Lock()
	llama := runner.llama
	runner.refMu.Unlock()
	if llama != nil {
		return llama.HasExited()
	}
	return true
}

type runnerUnloadCandidate struct {
	runner          *runnerRef
	refCount        uint
	sessionDuration time.Duration
	name            string
}

func newRunnerUnloadCandidate(runner *runnerRef) runnerUnloadCandidate {
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	name := runner.modelPath
	if name == "" {
		name = runner.modelKey
	}

	return runnerUnloadCandidate{
		runner:          runner,
		refCount:        runner.refCount,
		sessionDuration: runner.sessionDuration,
		name:            name,
	}
}

// TODO - future consideration to pick runners based on size
// type BySize []*runnerRef
// func (a BySize) Len() int           { return len(a) }
// func (a BySize) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
// func (a BySize) Less(i, j int) bool { return a[i].vramSize < a[j].vramSize }

// evictAllAndWait synchronously expires every currently loaded runner except
// the one being loaded (matched by modelKey) and waits for all unload events
// to drain. Returns false if the context was cancelled mid-wait so the caller
// can exit the scheduling loop. Used by the load retry path in processPending.
func (s *Scheduler) evictAllAndWait(ctx context.Context, keepKey string) bool {
	s.loadedMu.Lock()
	runnersToExpire := make([]*runnerRef, 0, len(s.loaded))
	for key, r := range s.loaded {
		if key == keepKey {
			continue
		}
		runnersToExpire = append(runnersToExpire, r)
	}
	s.loadedMu.Unlock()

	if len(runnersToExpire) == 0 {
		return true
	}

	slog.Info("evicting all other loaded models for load retry", "count", len(runnersToExpire))
	for _, runner := range runnersToExpire {
		var expireNow bool
		runner.refMu.Lock()
		expireNow = runner.requestImmediateExpirationLocked()
		runner.refMu.Unlock()
		if expireNow {
			s.expiredCh <- runner
		}
	}

	// Wait for every unload event. Each runner produces exactly one
	// unloadedCh signal when its cleanup finishes.
	for range runnersToExpire {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler during evict-all wait")
			return false
		case <-s.unloadedCh:
		}
	}
	return true
}

func (s *Scheduler) expireRunnersForRuntimeOOM(model *Model, err error) {
	if !llm.IsOutOfMemory(err) {
		return
	}

	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, runner := range s.loaded {
		runners = append(runners, runner)
	}
	s.loadedMu.Unlock()

	if len(runners) == 0 {
		return
	}

	slog.Warn("runtime OOM detected; expiring loaded models to clear memory before next request", "model", schedulerModelKey(model), "error", err)
	for _, runner := range runners {
		var expireNow bool
		runner.refMu.Lock()
		expireNow = runner.requestImmediateExpirationLocked()
		runner.refMu.Unlock()
		if expireNow {
			s.expiredCh <- runner
		}
	}
}

// findRunnerToUnload finds a runner to unload to make room for a new model
func (s *Scheduler) findRunnerToUnload() *runnerRef {
	s.loadedMu.Lock()
	runnerList := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnerList = append(runnerList, r)
	}
	s.loadedMu.Unlock()
	if len(runnerList) == 0 {
		slog.Debug("no loaded runner to unload")
		return nil
	}

	// In the future we can enhance the algorithm to be smarter about picking the optimal runner to unload
	// e.g., if we have multiple options, will one make room for the request?
	candidates := make([]runnerUnloadCandidate, 0, len(runnerList))
	for _, runner := range runnerList {
		candidates = append(candidates, newRunnerUnloadCandidate(runner))
	}
	sort.Slice(candidates, func(i, j int) bool {
		// Primary sort by session duration (uint64 to handle negatives)
		d1 := uint64(candidates[i].sessionDuration)
		d2 := uint64(candidates[j].sessionDuration)
		if d1 != d2 {
			return d1 < d2
		}
		// Secondary sort by model key/path lex order
		return candidates[i].name < candidates[j].name
	})

	// First try to find a runner that's already idle
	for _, candidate := range candidates {
		if candidate.refCount == 0 {
			slog.Debug("found an idle runner to unload", "runner", candidate.runner)
			return candidate.runner
		}
	}
	// None appear idle, just wait for the one with the shortest duration
	slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(candidates), "runner", candidates[0].runner)
	return candidates[0].runner
}

func (s *Scheduler) unloadAllRunners() {
	s.loadedMu.Lock()
	activeLoading := s.activeLoading
	s.activeLoading = nil
	runners := make([]*runnerRef, 0, len(s.loaded))
	for model, runner := range s.loaded {
		slog.Debug("shutting down runner", "model", model)
		runners = append(runners, runner)
		delete(s.loaded, model)
	}
	s.loadedMu.Unlock()

	if activeLoading != nil {
		slog.Debug("shutting down currently loading runner")
		activeLoading.Close()
	}
	for _, runner := range runners {
		runner.unload()
	}
}

func (s *Scheduler) expireRunner(model *Model) {
	modelKey := schedulerModelKey(model)
	s.loadedMu.Lock()
	runner, ok := s.loaded[modelKey]
	s.loadedMu.Unlock()
	if ok {
		var expireNow bool
		runner.refMu.Lock()
		runner.expiresAt = time.Now()
		expireNow = runner.requestImmediateExpirationLocked()
		runner.refMu.Unlock()
		if expireNow {
			s.expiredCh <- runner
		}
	}
}

// loadedModel is a point-in-time snapshot of a loaded runner's state, safe to
// use without holding any scheduler locks.
type loadedModel struct {
	model         *Model
	size          int64
	sizeVRAM      int64
	contextLength int
	expiresAt     time.Time
}

// loadedModels returns a snapshot of the currently loaded models for status
// reporting without exposing the scheduler's internal runner bookkeeping.
func (s *Scheduler) loadedModels() []loadedModel {
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	// Keep loadedMu scoped to the map snapshot so status reporting does not
	// block scheduler map updates while it inspects per-runner state.
	models := make([]loadedModel, 0, len(runners))
	for _, r := range runners {
		r.refMu.Lock()
		if r.model == nil {
			// Unloaded after the snapshot above was taken
			r.refMu.Unlock()
			continue
		}
		llama := r.llama
		sessionDuration := r.sessionDuration
		lm := loadedModel{
			model:     r.model,
			size:      int64(r.totalSize),
			sizeVRAM:  int64(r.vramSize),
			expiresAt: r.expiresAt,
		}
		r.refMu.Unlock()
		if llama != nil {
			lm.contextLength = llama.ContextLength()
			total, vram := llama.MemorySize()
			lm.size = int64(total)
			lm.sizeVRAM = int64(vram)
		}
		// The scheduler waits to set expiresAt, so a model that is still
		// loading may have the zero value. Estimate expiration from the
		// session duration instead.
		if lm.expiresAt.IsZero() {
			lm.expiresAt = time.Now().Add(sessionDuration)
		}
		models = append(models, lm)
	}
	return models
}
