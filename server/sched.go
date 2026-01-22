package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"sort"
	"strings"
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
	"github.com/ollama/ollama/x/imagegen"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint
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

	loadFn          func(req *LlmRequest, f *ggml.GGML, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool
	newServerFn     func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error)
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

// context must be canceled to decrement ref count and release the runner
func (s *Scheduler) GetRunner(c context.Context, m *Model, opts api.Options, sessionDuration *api.Duration) (chan *runnerRef, chan error) {
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	if m.CheckCapabilities(model.CapabilityVision) == nil {
		// multimodal models require at least 2048 context
		opts.NumCtx = max(opts.NumCtx, 2048)
	}

	req := &LlmRequest{
		ctx:             c,
		model:           m,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef, 1),
		errCh:           make(chan error, 1),
	}

	s.loadedMu.Lock()
	runner := s.loaded[req.model.ModelPath]
	s.loadedMu.Unlock()
	if runner != nil && !runner.needsReload(c, req) {
		req.useLoadedRunner(runner, s.finishedReqCh)
	} else {
		select {
		case s.pendingReqCh <- req:
		default:
			req.errCh <- ErrMaxQueue
		}
	}
	return req.successCh, req.errCh
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

	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case pending := <-s.pendingReqCh:
			// Block other requests until we get this pending request running
			pending.schedAttempts++

			if pending.ctx.Err() != nil {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			logutil.Trace("processing incoming request", "model", pending.model.ModelPath)

			for {
				var runnerToExpire *runnerRef
				s.loadedMu.Lock()
				runner := s.loaded[pending.model.ModelPath]
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
						logutil.Trace("using existing loaded runner", "model", pending.model.ModelPath)
						pending.useLoadedRunner(runner, s.finishedReqCh)
						break
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

					// Check for image generation model before attempting GGML load
					if slices.Contains(pending.model.Config.Capabilities, "image") {
						if s.loadImageGen(pending) {
							break
						}
						continue
					}

					// Load model for fitting
					logutil.Trace("loading model metadata", "model", pending.model.ModelPath)
					ggml, err := llm.LoadModel(pending.model.ModelPath, 1024)
					if err != nil {
						pending.errCh <- err
						break
					}

					// Update free memory from currently loaded models
					logutil.Trace("updating free space", "gpu_count", len(gpus), "model", pending.model.ModelPath)
					s.updateFreeSpace(gpus)

					if loadedCount == 0 {
						// No models loaded. Load the model but prefer the best fit.
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						s.loadFn(pending, ggml, systemInfo, gpus, false)
						break
					}

					// More than one loaded model, so we have to see if the
					// new one fits
					logutil.Trace("loading additional model", "model", pending.model.ModelPath)
					needEvict := s.loadFn(pending, ggml, systemInfo, gpus, true)
					if !needEvict {
						slog.Debug("new model fits with existing models, loading")
						break
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
				runnerToExpire.refMu.Lock()
				slog.Debug("resetting model to expire immediately to make room", "runner", runnerToExpire, "refCount", runnerToExpire.refCount)
				if runnerToExpire.expireTimer != nil {
					runnerToExpire.expireTimer.Stop()
					runnerToExpire.expireTimer = nil
				}
				runnerToExpire.sessionDuration = 0
				if runnerToExpire.refCount <= 0 {
					s.expiredCh <- runnerToExpire
				}
				runnerToExpire.refMu.Unlock()
				// Wait for the unload to happen
				slog.Debug("waiting for pending requests to complete and unload to occur", "runner", runnerToExpire)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-s.unloadedCh:
					slog.Debug("unload completed", "runner", runnerToExpire)
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
			s.loadedMu.Lock()
			runner := s.loaded[finished.model.ModelPath]
			s.loadedMu.Unlock()
			if runner == nil {
				slog.Error("finished request signal received after model unloaded", "modelPath", finished.model.ModelPath)
				continue
			}
			runner.refMu.Lock()
			runner.refCount--
			if runner.refCount <= 0 {
				if runner.sessionDuration <= 0 {
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "runner", runner)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					s.expiredCh <- runner
				} else if runner.expireTimer == nil {
					slog.Debug("runner with non-zero duration has gone idle, adding timer", "runner", runner, "duration", runner.sessionDuration)
					runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
						slog.Debug("timer expired, expiring to unload", "runner", runner)
						runner.refMu.Lock()
						defer runner.refMu.Unlock()
						if runner.expireTimer != nil {
							runner.expireTimer.Stop()
							runner.expireTimer = nil
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
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "runner", runner)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				slog.Debug("expired event with positive ref count, retrying", "runner", runner, "refCount", runner.refCount)
				go func(runner *runnerRef) {
					// We can't unload yet, but want to as soon as the current request completes
					// So queue up another expired event
					time.Sleep(10 * time.Millisecond)
					s.expiredCh <- runner
				}(runner)
				runner.refMu.Unlock()
				continue
			}

			s.loadedMu.Lock()
			slog.Debug("got lock to unload expired event", "runner", runner)
			runnerToUnload := s.loaded[runner.modelPath]
			if runnerToUnload == nil {
				// If runnerToUnload is nil, we already processed an event and
				// unloaded it. This double unload can happen if the initial
				// request is canceled and we're trying to load another model
				// that requires this one to be evicted, or the settings change
				// and require a reload
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
				slog.Debug("duplicate expired event, ignoring", "runner", runner)
			} else if runner.pid != runnerToUnload.pid {
				// If the pids do not match, we likely had multiple load
				// failures for the same model in quick succession due to
				// request context canceled and are draining the queue of
				// events. Ensure the orphaned runner is properly shut down, but
				// do not delete the mismatched loaded runner, or wait for VRAM
				// convergence.
				slog.Debug("orphaned runner shutting down", "orphan", runner, "loaded", runnerToUnload)
				runner.unload()
				s.loadedMu.Unlock()
				runner.refMu.Unlock()
			} else {
				slog.Debug("starting background wait for VRAM recovery", "runner", runner)
				runnersSnapshot := make([]ml.FilteredRunnerDiscovery, 0, len(s.loaded))
				for _, r := range s.loaded {
					runnersSnapshot = append(runnersSnapshot, r)
				}
				finished := s.waitForVRAMRecovery(runner, runnersSnapshot)
				runner.unload()
				delete(s.loaded, runner.modelPath)
				s.loadedMu.Unlock()
				slog.Debug("runner terminated and removed from list, blocking for VRAM recovery", "runner", runner)
				<-finished
				runner.refMu.Unlock()
				slog.Debug("sending an unloaded event", "runner", runner)
				s.unloadedCh <- struct{}{}
			}
		}
	}
}

// Complete the pending request and send the runner back to the requester
// Wires up a finished event after the request context is completed
// Updates session duration, and resets expiration timer
func (pending *LlmRequest) useLoadedRunner(runner *runnerRef, finished chan *LlmRequest) {
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	runner.refCount++
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if pending.sessionDuration != nil {
		runner.sessionDuration = pending.sessionDuration.Duration
	}
	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished", "runner", runner)
		finished <- pending
	}()
}

// load creates a new model based on req and loads it. If requireFull is true then the model must be loaded fully onto GPUs
// (if any). Returns whether the scheduler needs to evict a model to make this one fit.
func (s *Scheduler) load(req *LlmRequest, f *ggml.GGML, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
	numParallel := max(int(envconfig.NumParallel()), 1)

	// Embedding models should always be loaded with parallel=1
	if req.model.CheckCapabilities(model.CapabilityCompletion) != nil {
		numParallel = 1
	}

	// `mllama`, `qwen3vl`, and `qwen3vlmoe` are snowflakes and uses an encoder cache which cannot be used with num_parallel > 1
	// ref: https://github.com/ollama/ollama/issues/4165
	if slices.Contains([]string{"mllama", "qwen3vl", "qwen3vlmoe"}, req.model.Config.ModelFamily) && numParallel != 1 {
		numParallel = 1
		slog.Warn("model architecture does not currently support parallel requests", "architecture", req.model.Config.ModelFamily)
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	s.loadedMu.Lock()
	llama := s.activeLoading

	if llama == nil {
		var err error
		llama, err = s.newServerFn(systemInfo, gpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts, numParallel)
		if err != nil {
			// some older models are not compatible with newer versions of llama.cpp
			// show a generalized compatibility error until there is a better way to
			// check for model compatibility
			if errors.Is(err, ggml.ErrUnsupportedFormat) || strings.Contains(err.Error(), "failed to load model") {
				err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
			}
			slog.Info("NewLlamaServer failed", "model", req.model.ModelPath, "error", err)
			req.errCh <- err
			s.loadedMu.Unlock()
			return false
		}

		s.activeLoading = llama
	} else {
		if s.activeLoading.ModelPath() != req.model.ModelPath {
			panic(fmt.Errorf("attempting to load different model after eviction (original %v new %v)", s.activeLoading.ModelPath(), req.model.ModelPath))
		}
	}

	s.loadedMu.Unlock()

	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	for _, gpu := range gpus {
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

	gpuIDs, err := llama.Load(req.ctx, systemInfo, gpus, requireFull)
	if err != nil {
		if errors.Is(err, llm.ErrLoadRequiredFull) {
			if !requireFull {
				// No other models loaded, yet we still don't fit, so report an error
				slog.Info("model is too large for system memory", "requireFull", requireFull)
				s.activeLoading.Close()
				s.activeLoading = nil
				req.errCh <- err
			}
			return true
		}

		slog.Info("Load failed", "model", req.model.ModelPath, "error", err)
		s.activeLoading.Close()
		s.activeLoading = nil
		req.errCh <- err
		return false
	}

	// Determine if we have discrete GPUs which we should monitor VRAM usage on during shutdown
	discreteGPUs := false
iGPUScan:
	for _, devid := range gpuIDs {
		for _, dev := range gpus {
			if dev.DeviceID == devid {
				if !dev.Integrated {
					discreteGPUs = true
					break iGPUScan
				}
			}
		}
	}

	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpuIDs,
		discreteGPUs:    discreteGPUs,
		vramSize:        llama.VRAMSize(),
		totalSize:       llama.TotalSize(),
		loading:         true,
		pid:             llama.Pid(),
	}
	runner.numParallel = numParallel
	runner.refMu.Lock() // hold lock until running or aborted

	s.loadedMu.Lock()
	if oldRunner, ok := s.loaded[req.model.ModelPath]; ok {
		// Shouldn't happen, but safeguard against leaking a runner
		slog.Warn("model was still loaded", "old_runner", oldRunner, "new_runner", runner)
		oldRunner.refMu.Lock()
		oldRunner.unload()
		oldRunner.refMu.Unlock()
	}
	s.activeLoading = nil
	s.loaded[req.model.ModelPath] = runner
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()

	go func() {
		defer runner.refMu.Unlock()
		if err = llama.WaitUntilRunning(req.ctx); err != nil {
			slog.Error("error loading llama server", "error", err)
			req.errCh <- err
			slog.Debug("triggering expiration for failed load", "runner", runner)
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
	}()

	return false
}

// loadImageGen loads an image generation model.
func (s *Scheduler) loadImageGen(req *LlmRequest) bool {
	// Use model name for imagegen (it resolves manifests by name, not file path)
	modelName := req.model.ShortName
	server, err := imagegen.NewServer(modelName)
	if err != nil {
		req.errCh <- err
		return true
	}

	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}

	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		llama:           server,
		Options:         &req.opts,
		loading:         false,
		sessionDuration: sessionDuration,
		totalSize:       server.TotalSize(),
		vramSize:        server.VRAMSize(),
	}

	s.loadedMu.Lock()
	s.loaded[req.model.ModelPath] = runner
	s.loadedMu.Unlock()

	// Set up expiration timer
	runner.refMu.Lock()
	if sessionDuration > 0 {
		runner.expireTimer = time.AfterFunc(sessionDuration, func() {
			s.expiredCh <- runner
		})
	}
	runner.refMu.Unlock()

	req.useLoadedRunner(runner, s.finishedReqCh)
	return true
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
		if r.llama != nil {
			for _, gpu := range allGpus {
				predMap[gpu.DeviceID] += r.llama.VRAMByGPU(gpu.DeviceID)
			}
		} else {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
		}
		r.refMu.Unlock()
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

// TODO consolidate sched_types.go
type runnerRef struct {
	refMu    sync.Mutex
	refCount uint // prevent unloading if > 0

	llama        llm.LlamaServer
	pid          int
	loading      bool          // True only during initial load, then false forever
	gpus         []ml.DeviceID // Recorded at time of provisioning
	discreteGPUs bool          // True if all devices are discrete GPUs - used to skip VRAM recovery check for iGPUs
	vramSize     uint64
	totalSize    uint64

	sessionDuration time.Duration
	expireTimer     *time.Timer
	expiresAt       time.Time

	model       *Model
	modelPath   string
	numParallel int
	*api.Options
}

// The refMu must already be held when calling unload
func (runner *runnerRef) unload() {
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	if runner.llama != nil {
		runner.llama.Close()
	}
	runner.model = nil
	runner.Options = nil
	runner.gpus = nil
}

func (runner *runnerRef) needsReload(ctx context.Context, req *LlmRequest) bool {
	slog.Debug("evaluating already loaded", "model", req.model.ModelPath)
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

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
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if !reflect.DeepEqual(runner.model.AdapterPaths, req.model.AdapterPaths) || // have the adapters changed?
		!reflect.DeepEqual(runner.model.ProjectorPaths, req.model.ProjectorPaths) || // have the projectors changed?
		!reflect.DeepEqual(optsExisting, optsNew) || // have the runner options changed?
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

	// CPU, Metal and iGPUs don't need checking, so no waiting required
	if len(runner.gpus) == 0 || !runner.discreteGPUs ||
		(len(runner.gpus) == 1 && runner.gpus[0].Library == "Metal") {
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
					logutil.Trace("gpu VRAM convergence", "percent", int(float32(freeMemoryNow-freeMemoryBefore)/float32(runner.vramSize)*100))
				} else {
					logutil.Trace("gpu VRAM convergence", "percent", 0)
				}
				// If we're within ~75% of the estimated memory usage recovered, bail out
				if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.vramSize)*0.75 {
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
	attrs := []slog.Attr{}
	if runner.model != nil {
		attrs = append(attrs, slog.String("name", runner.model.Name))
	}
	if len(runner.gpus) > 0 {
		attrs = append(attrs,
			slog.Any("inference", runner.gpus),
		)
	}
	attrs = append(attrs,
		slog.String("size", format.HumanBytes2(runner.totalSize)),
		slog.String("vram", format.HumanBytes2(runner.vramSize)),
		slog.Int("parallel", runner.numParallel),
		slog.Int("pid", runner.pid),
		slog.String("model", runner.modelPath),
	)
	if runner.Options != nil {
		attrs = append(attrs, slog.Int("num_ctx", runner.Options.NumCtx))
	}
	return slog.GroupValue(attrs...)
}

// Implements discover.RunnerDiscovery
func (runner *runnerRef) GetPort() int {
	if runner.llama != nil {
		return runner.llama.GetPort()
	}
	return -1
}

func (runner *runnerRef) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if runner.llama != nil {
		return runner.llama.GetDeviceInfos(ctx)
	}
	return nil
}

func (runner *runnerRef) GetActiveDeviceIDs() []ml.DeviceID {
	return runner.gpus
}

func (runner *runnerRef) HasExited() bool {
	if runner.llama != nil {
		return runner.llama.HasExited()
	}
	return true
}

type ByDurationAndName []*runnerRef

func (a ByDurationAndName) Len() int      { return len(a) }
func (a ByDurationAndName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByDurationAndName) Less(i, j int) bool {
	// Primary sort by session duration (uint64 to handle negatives)
	d1 := uint64(a[i].sessionDuration)
	d2 := uint64(a[j].sessionDuration)
	if d1 != d2 {
		return d1 < d2
	}
	// Secondary sort by model path lex order
	return a[i].modelPath < a[j].modelPath
}

// TODO - future consideration to pick runners based on size
// type BySize []*runnerRef
// func (a BySize) Len() int           { return len(a) }
// func (a BySize) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
// func (a BySize) Less(i, j int) bool { return a[i].vramSize < a[j].vramSize }

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
	sort.Sort(ByDurationAndName(runnerList))

	// First try to find a runner that's already idle
	for _, runner := range runnerList {
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner to unload", "runner", runner)
			return runner
		}
	}
	// None appear idle, just wait for the one with the shortest duration
	slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(runnerList), "runner", runnerList[0])
	return runnerList[0]
}

func (s *Scheduler) unloadAllRunners() {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()

	if s.activeLoading != nil {
		slog.Debug("shutting down currently loading runner")
		s.activeLoading.Close()
		s.activeLoading = nil
	}

	for model, runner := range s.loaded {
		if runner.llama != nil {
			slog.Debug("shutting down runner", "model", model)
			runner.llama.Close()
		}
	}
}

func (s *Scheduler) expireRunner(model *Model) {
	s.loadedMu.Lock()
	runner, ok := s.loaded[model.ModelPath]
	s.loadedMu.Unlock()
	if ok {
		runner.refMu.Lock()
		runner.expiresAt = time.Now()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.sessionDuration = 0
		if runner.refCount <= 0 {
			s.expiredCh <- runner
		}
		runner.refMu.Unlock()
	}
}
