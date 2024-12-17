package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	origNumCtx      int // Track the initial ctx request
	sessionDuration *api.Duration
	successCh       chan *runnerRef
	errCh           chan error
	schedAttempts   uint
}

type Scheduler struct {
	pendingReqCh  chan *LlmRequest
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan interface{}

	loaded   map[string]*runnerRef
	loadedMu sync.Mutex

	loadFn       func(req *LlmRequest, f *ggml.GGML, gpus discover.GpuInfoList, numParallel int)
	newServerFn  func(gpus discover.GpuInfoList, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error)
	getGpuFn     func() discover.GpuInfoList
	getCpuFn     func() discover.GpuInfoList
	reschedDelay time.Duration
}

// Default automatic value for number of models we allow per GPU
// Model will still need to fit in VRAM, but loading many small models
// on a large GPU can cause stalling
var defaultModelsPerGPU = 3

// Default automatic value for parallel setting
// Model will still need to fit in VRAM.  If this setting won't fit
// we'll back off down to 1 to try to get it to fit
var defaultParallel = 4

var ErrMaxQueue = errors.New("server busy, please try again.  maximum pending requests exceeded")

func InitScheduler(ctx context.Context) *Scheduler {
	maxQueue := envconfig.MaxQueue()
	sched := &Scheduler{
		pendingReqCh:  make(chan *LlmRequest, maxQueue),
		finishedReqCh: make(chan *LlmRequest, maxQueue),
		expiredCh:     make(chan *runnerRef, maxQueue),
		unloadedCh:    make(chan interface{}, maxQueue),
		loaded:        make(map[string]*runnerRef),
		newServerFn:   llm.NewLlamaServer,
		getGpuFn:      discover.GetGPUInfo,
		getCpuFn:      discover.GetCPUInfo,
		reschedDelay:  250 * time.Millisecond,
	}
	sched.loadFn = sched.load
	return sched
}

// context must be canceled to decrement ref count and release the runner
func (s *Scheduler) GetRunner(c context.Context, model *Model, opts api.Options, sessionDuration *api.Duration) (chan *runnerRef, chan error) {
	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	req := &LlmRequest{
		ctx:             c,
		model:           model,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef),
		errCh:           make(chan error, 1),
	}

	select {
	case s.pendingReqCh <- req:
	default:
		req.errCh <- ErrMaxQueue
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
	for {
		select {
		case <-ctx.Done():
			slog.Debug("shutting down scheduler pending loop")
			return
		case pending := <-s.pendingReqCh:
			// Block other requests until we get this pending request running
			pending.schedAttempts++
			if pending.origNumCtx == 0 {
				pending.origNumCtx = pending.opts.NumCtx
			}

			if pending.ctx.Err() != nil {
				slog.Debug("pending request cancelled or timed out, skipping scheduling")
				continue
			}
			numParallel := int(envconfig.NumParallel())
			// TODO (jmorganca): mllama doesn't support parallel yet
			// see https://github.com/ollama/ollama/issues/4165
			if checkMllamaModelFamily(pending.model) && numParallel != 1 {
				numParallel = 1
				slog.Warn("mllama doesn't support parallel requests yet")
			}

			for {
				var runnerToExpire *runnerRef
				s.loadedMu.Lock()
				runner := s.loaded[pending.model.ModelPath]
				loadedCount := len(s.loaded)
				s.loadedMu.Unlock()
				if runner != nil {
					if runner.needsReload(ctx, pending) {
						runnerToExpire = runner
					} else {
						// Runner is usable, return it
						pending.useLoadedRunner(runner, s.finishedReqCh)
						break
					}
				} else if envconfig.MaxRunners() > 0 && loadedCount >= int(envconfig.MaxRunners()) {
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload()
				} else {
					// Either no models are loaded or below envconfig.MaxRunners
					// Get a refreshed GPU list
					var gpus discover.GpuInfoList
					if pending.opts.NumGPU == 0 {
						gpus = s.getCpuFn()
					} else {
						gpus = s.getGpuFn()
					}

					if envconfig.MaxRunners() <= 0 {
						// No user specified MaxRunners, so figure out what automatic setting to use
						// If all GPUs have reliable free memory reporting, defaultModelsPerGPU * the number of GPUs
						// if any GPU has unreliable free memory reporting, 1x the number of GPUs
						allReliable := true
						for _, gpu := range gpus {
							if gpu.UnreliableFreeMemory {
								allReliable = false
								break
							}
						}
						if allReliable {
							// HACK
							os.Setenv("OLLAMA_MAX_LOADED_MODELS", strconv.Itoa(defaultModelsPerGPU*len(gpus)))
							slog.Debug("updating default concurrency", "OLLAMA_MAX_LOADED_MODELS", envconfig.MaxRunners, "gpu_count", len(gpus))
						} else {
							// HACK
							os.Setenv("OLLAMA_MAX_LOADED_MODELS", strconv.Itoa(len(gpus)))
							slog.Info("one or more GPUs detected that are unable to accurately report free memory - disabling default concurrency")
						}
					}

					// Load model for fitting
					ggml, err := llm.LoadModel(pending.model.ModelPath, 0)
					if err != nil {
						pending.errCh <- err
						break
					}

					// Embedding models should always be loaded with parallel=1
					if pending.model.CheckCapabilities(CapabilityCompletion) != nil {
						numParallel = 1
					}

					// Evaluate if the model will fit in the available system memory, or if we should unload a model first
					if len(gpus) == 1 && gpus[0].Library == "cpu" {
						// simplifying assumption of defaultParallel when in CPU mode
						if numParallel <= 0 {
							numParallel = defaultParallel
						}

						pending.opts.NumCtx = pending.origNumCtx * numParallel

						if loadedCount == 0 {
							slog.Debug("cpu mode with first model, loading")
							s.loadFn(pending, ggml, gpus, numParallel)
							break
						}
						runnerToExpire = s.maybeFindCPURunnerToUnload(pending, ggml, gpus)
						if runnerToExpire == nil {
							slog.Debug("cpu mode with available system memory or first model, loading")
							s.loadFn(pending, ggml, gpus, numParallel)
							break
						}
						// else we need to expire a runner
					} else if loadedCount == 0 {
						// No models loaded. Load the model but prefer the best fit.
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						g := pickBestFullFitByLibrary(pending, ggml, gpus, &numParallel)
						if g != nil {
							gpus = g
						} else {
							// Only allow partial loads when this is the first model
							gpus = pickBestPartialFitByLibrary(pending, ggml, gpus, &numParallel)
						}
						s.loadFn(pending, ggml, gpus, numParallel)
						break
					}

					if runnerToExpire == nil {
						// More than one loaded model, so we have to see if the
						// new one fits
						//
						// We want to avoid loading on any GPUs that have other
						// models still loading on them to avoid potential races
						// with VRAM consumption ramping up during load
						availGpus := s.filterGPUsWithoutLoadingModels(gpus)

						// Update free memory from currently loaded models
						s.updateFreeSpace(availGpus)
						fitGpus := pickBestFullFitByLibrary(pending, ggml, availGpus, &numParallel)
						if fitGpus != nil {
							slog.Debug("new model fits with existing models, loading")
							s.loadFn(pending, ggml, fitGpus, numParallel)
							break
						}

						// We couldn't find a set of GPUs to fully load the new
						// model. If no other models are loading (both GPU lists
						// are the same) then we need to unload another model to
						// make room
						if len(availGpus) < len(gpus) {
							// There are other requests pending, and this one
							// needs more time, so put it on the back of the
							// queue so that we might satisfy other pending
							// requests that aren't blocked
							go func() {
								// Process in a go routine to avoid deadlocking
								// the scheduler if our queue is full
								slog.Debug("delaying scheduling while other models finish loading", "attempts", pending.schedAttempts, "model", pending.model.ModelPath)
								time.Sleep(s.reschedDelay)
								s.pendingReqCh <- pending
							}()
							break
						}
						runnerToExpire = s.findRunnerToUnload()
					}
				}

				if runnerToExpire == nil {
					// Shouildn't happen
					slog.Error("runner to expire was nil!")
					continue
				}
				// Trigger an expiration to unload once it's done
				runnerToExpire.refMu.Lock()
				slog.Debug("resetting model to expire immediately to make room", "modelPath", runnerToExpire.modelPath, "refCount", runnerToExpire.refCount)
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
				// Note: at this point we're queueing up all incoming requests, even if they were for
				// a different model that's loaded and not scheduled to be removed.
				slog.Debug("waiting for pending requests to complete and unload to occur", "modelPath", runnerToExpire.modelPath)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-s.unloadedCh:
					slog.Debug("unload completed", "modelPath", runnerToExpire.modelPath)
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
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "modelPath", runner.modelPath)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					s.expiredCh <- runner
				} else if runner.expireTimer == nil {
					slog.Debug("runner with non-zero duration has gone idle, adding timer", "modelPath", runner.modelPath, "duration", runner.sessionDuration)
					runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
						slog.Debug("timer expired, expiring to unload", "modelPath", runner.modelPath)
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
					slog.Debug("runner with non-zero duration has gone idle, resetting timer", "modelPath", runner.modelPath, "duration", runner.sessionDuration)
					runner.expireTimer.Reset(runner.sessionDuration)
					runner.expiresAt = time.Now().Add(runner.sessionDuration)
				}
			}
			slog.Debug("after processing request finished event", "modelPath", runner.modelPath, "refCount", runner.refCount)
			runner.refMu.Unlock()
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "modelPath", runner.modelPath)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				slog.Debug("expired event with positive ref count, retrying", "modelPath", runner.modelPath, "refCount", runner.refCount)
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
			slog.Debug("got lock to unload", "modelPath", runner.modelPath)
			finished := runner.waitForVRAMRecovery()
			runner.unload()
			delete(s.loaded, runner.modelPath)
			s.loadedMu.Unlock()
			slog.Debug("runner released", "modelPath", runner.modelPath)
			runner.refMu.Unlock()

			<-finished
			slog.Debug("sending an unloaded event", "modelPath", runner.modelPath)
			s.unloadedCh <- struct{}{}
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
		slog.Debug("context for request finished")
		finished <- pending
	}()
}

func (s *Scheduler) load(req *LlmRequest, f *ggml.GGML, gpus discover.GpuInfoList, numParallel int) {
	if numParallel < 1 {
		numParallel = 1
	}
	sessionDuration := envconfig.KeepAlive()
	if req.sessionDuration != nil {
		sessionDuration = req.sessionDuration.Duration
	}
	llama, err := s.newServerFn(gpus, req.model.ModelPath, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts, numParallel)
	if err != nil {
		// some older models are not compatible with newer versions of llama.cpp
		// show a generalized compatibility error until there is a better way to
		// check for model compatibility
		if errors.Is(err, ggml.ErrUnsupportedFormat) || strings.Contains(err.Error(), "failed to load model") {
			err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
		}
		slog.Info("NewLlamaServer failed", "model", req.model.ModelPath, "error", err)
		req.errCh <- err
		return
	}
	runner := &runnerRef{
		model:           req.model,
		modelPath:       req.model.ModelPath,
		llama:           llama,
		Options:         &req.opts,
		sessionDuration: sessionDuration,
		gpus:            gpus,
		estimatedVRAM:   llama.EstimatedVRAM(),
		estimatedTotal:  llama.EstimatedTotal(),
		loading:         true,
		refCount:        1,
	}
	runner.numParallel = numParallel
	runner.refMu.Lock()

	s.loadedMu.Lock()
	s.loaded[req.model.ModelPath] = runner
	slog.Info("loaded runners", "count", len(s.loaded))
	s.loadedMu.Unlock()

	go func() {
		defer runner.refMu.Unlock()
		if err = llama.WaitUntilRunning(req.ctx); err != nil {
			slog.Error("error loading llama server", "error", err)
			runner.refCount--
			req.errCh <- err
			slog.Debug("triggering expiration for failed load", "model", runner.modelPath)
			s.expiredCh <- runner
			return
		}
		slog.Debug("finished setting up runner", "model", req.model.ModelPath)
		runner.loading = false
		go func() {
			<-req.ctx.Done()
			slog.Debug("context for request finished")
			s.finishedReqCh <- req
		}()
		req.successCh <- runner
	}()
}

func (s *Scheduler) updateFreeSpace(allGpus discover.GpuInfoList) {
	type predKey struct {
		Library string
		ID      string
	}
	predMap := map[predKey]uint64{} // Sum up the total predicted usage per GPU for all runners
	s.loadedMu.Lock()
	for _, r := range s.loaded {
		r.refMu.Lock()
		if r.llama != nil {
			for _, gpu := range allGpus {
				predMap[predKey{gpu.Library, gpu.ID}] += r.llama.EstimatedVRAMByGPU(gpu.ID)
			}
		} else {
			slog.Warn("unexpected nil runner reference, memory prediction may be incorrect")
		}
		r.refMu.Unlock()
	}
	s.loadedMu.Unlock()

	// Now that we've summed up all the GPU usage predictions across all the loaded runners, update the gpu list
	for i := range allGpus {
		if p, ok := predMap[predKey{allGpus[i].Library, allGpus[i].ID}]; ok {
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

// While models are loading the VRAM consumption numbers will be indeterminate, so we have
// to avoid scheduling another model on the same GPU(s) that haven't stabilized.
// This routine returns the set of GPUs that do not have an active loading model.
// If all GPUs have loading models, an empty list will be returned (not a single CPU entry)
func (s *Scheduler) filterGPUsWithoutLoadingModels(allGpus discover.GpuInfoList) discover.GpuInfoList {
	ret := append(discover.GpuInfoList{}, allGpus...)
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	for _, runner := range s.loaded {
		if runner.loading {
			slog.Debug("overlapping loads detected", "gpus", runner.gpus, "model", runner.modelPath)
			for _, busyGPU := range runner.gpus {
				for i := range ret {
					if ret[i].ID == busyGPU.ID {
						ret = append(ret[:i], ret[i+1:]...)
						break
					}
				}
			}
		}
	}
	return ret
}

// TODO consolidate sched_types.go
type runnerRef struct {
	refMu sync.Mutex
	// refCond   sync.Cond // Signaled on transition from 1 -> 0 refCount
	refCount uint // prevent unloading if > 0
	// unloading bool      // set to true when we are trying to unload the runner

	llama          llm.LlamaServer
	loading        bool                 // True only during initial load, then false forever
	gpus           discover.GpuInfoList // Recorded at time of provisioning
	estimatedVRAM  uint64
	estimatedTotal uint64

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
	runner.llama = nil
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

	// Normalize the NumCtx for parallelism
	optsExisting.NumCtx = optsExisting.NumCtx / runner.numParallel

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
func (runner *runnerRef) waitForVRAMRecovery() chan interface{} {
	finished := make(chan interface{}, 1)

	// CPU or Metal don't need checking, so no waiting required
	// windows can page VRAM, only cuda currently can report accurate used vram usage
	if len(runner.gpus) == 0 ||
		(len(runner.gpus) == 1 && (runner.gpus[0].Library == "cpu" || runner.gpus[0].Library == "metal")) ||
		(runtime.GOOS == "windows" && runner.gpus[0].Library != "cuda") {
		finished <- struct{}{}
		return finished
	}
	start := time.Now()

	// Establish a baseline before we unload
	gpusBefore := discover.GetGPUInfo()
	var totalMemoryBefore, freeMemoryBefore uint64
	for _, gpu := range gpusBefore {
		totalMemoryBefore += gpu.TotalMemory
		freeMemoryBefore += gpu.FreeMemory
	}
	go func() {
		expiresAt := start.Add(5 * time.Second) // typical convergence is 0.5-1.5s
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			<-ticker.C
			if time.Now().After(expiresAt) {
				slog.Warn("gpu VRAM usage didn't recover within timeout", "seconds", time.Since(start).Seconds(), "model", runner.modelPath)
				finished <- struct{}{}
			}

			// Query GPUs, look for free to go back up
			gpusNow := discover.GetGPUInfo()
			var totalMemoryNow, freeMemoryNow uint64
			for _, gpu := range gpusNow {
				totalMemoryNow += gpu.TotalMemory
				freeMemoryNow += gpu.FreeMemory
			}
			// If we're within ~80% of the estimated memory usage recovered, bail out
			if float32(freeMemoryNow-freeMemoryBefore) > float32(runner.estimatedVRAM)*0.8 {
				slog.Debug(fmt.Sprintf("gpu VRAM free memory converged after %0.2f seconds", time.Since(start).Seconds()), "model", runner.modelPath)
				finished <- struct{}{}
				return
			}
		}
	}()
	return finished
}

type ByDuration []*runnerRef

func (a ByDuration) Len() int      { return len(a) }
func (a ByDuration) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByDuration) Less(i, j int) bool {
	// uint64 to turn negative time (never unload) to largest
	return uint64(a[i].sessionDuration) < uint64(a[j].sessionDuration)
}

// TODO - future consideration to pick runners based on size
// type BySize []*runnerRef
// func (a BySize) Len() int           { return len(a) }
// func (a BySize) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
// func (a BySize) Less(i, j int) bool { return a[i].estimatedVRAM < a[j].estimatedVRAM }

// pickBestFullFitByLibrary will try to find the optimal placement of the model in the available GPUs where the model fully fits
// The list of GPUs returned will always be the same brand (library)
// If the model can not be fit fully within the available GPU(s) nil is returned
// If numParallel is <= 0, this will attempt try to optimize parallelism based on available VRAM, and adjust
// opts.NumCtx accordingly
func pickBestFullFitByLibrary(req *LlmRequest, f *ggml.GGML, gpus discover.GpuInfoList, numParallel *int) discover.GpuInfoList {
	var estimatedVRAM uint64

	var numParallelToTry []int
	if *numParallel <= 0 {
		// If no specific parallel setting was provided, try larger then smaller, always end with 1
		numParallelToTry = append(numParallelToTry, defaultParallel, 1)
	} else {
		numParallelToTry = []int{*numParallel}
	}

	for _, gl := range gpus.ByLibrary() {
		var ok bool
		sgl := append(make(discover.GpuInfoList, 0, len(gl)), gl...)

		// TODO - potentially sort by performance capability, existing models loaded, etc.
		// TODO - Eliminate any GPUs that already have envconfig.MaxRunners loaded on them
		// Note: at present, this will favor more VRAM over faster GPU speed in mixed setups
		sort.Sort(sort.Reverse(discover.ByFreeMemory(sgl)))

		// First attempt to fit the model into a single GPU
		for _, p := range numParallelToTry {
			req.opts.NumCtx = req.origNumCtx * p
			if !envconfig.SchedSpread() {
				for _, g := range sgl {
					if ok, estimatedVRAM = llm.PredictServerFit([]discover.GpuInfo{g}, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts); ok {
						slog.Info("new model will fit in available VRAM in single GPU, loading", "model", req.model.ModelPath, "gpu", g.ID, "parallel", p, "available", g.FreeMemory, "required", format.HumanBytes2(estimatedVRAM))
						*numParallel = p
						return []discover.GpuInfo{g}
					}
				}
			}
		}

		// TODO future refinements
		// - if multiple Libraries, see if any single GPU in any Library will fit
		// - try subsets of GPUs instead of just falling back to 1 or all in a family

		// Now try all the GPUs
		for _, p := range numParallelToTry {
			req.opts.NumCtx = req.origNumCtx * p
			if ok, estimatedVRAM = llm.PredictServerFit(sgl, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts); ok {
				slog.Info("new model will fit in available VRAM, loading", "model", req.model.ModelPath, "library", sgl[0].Library, "parallel", p, "required", format.HumanBytes2(estimatedVRAM))
				*numParallel = p
				return sgl
			}
		}
	}
	return nil
}

// If multiple Libraries are detected, pick the Library which loads the most layers for the model
func pickBestPartialFitByLibrary(req *LlmRequest, f *ggml.GGML, gpus discover.GpuInfoList, numParallel *int) discover.GpuInfoList {
	if *numParallel <= 0 {
		*numParallel = 1
		req.opts.NumCtx = req.origNumCtx
	}
	byLibrary := gpus.ByLibrary()
	if len(byLibrary) <= 1 {
		return gpus
	}
	var bestEstimate uint64
	var bestFit int
	for i, gl := range byLibrary {
		_, estimatedVRAM := llm.PredictServerFit(gl, f, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts)
		if estimatedVRAM > bestEstimate {
			bestEstimate = estimatedVRAM
			bestFit = i
		}
	}
	return byLibrary[bestFit]
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
	sort.Sort(ByDuration(runnerList))

	// First try to find a runner that's already idle
	for _, runner := range runnerList {
		runner.refMu.Lock()
		rc := runner.refCount
		runner.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner to unload")
			return runner
		}
	}
	// None appear idle, just wait for the one with the shortest duration
	slog.Debug("no idle runners, picking the shortest duration", "count", len(runnerList))
	return runnerList[0]
}

func (s *Scheduler) unloadAllRunners() {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	for model, runner := range s.loaded {
		if runner.llama != nil {
			slog.Debug("shutting down runner", "model", model)
			runner.llama.Close()
		}
	}
}

func (s *Scheduler) expireRunner(model *Model) {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	runner, ok := s.loaded[model.ModelPath]
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

// If other runners are loaded, make sure the pending request will fit in system memory
// If not, pick a runner to unload, else return nil and the request can be loaded
func (s *Scheduler) maybeFindCPURunnerToUnload(req *LlmRequest, f *ggml.GGML, gpus discover.GpuInfoList) *runnerRef {
	slog.Debug("evaluating if CPU model load will fit in available system memory")
	estimate := llm.EstimateGPULayers(gpus, f, req.model.ProjectorPaths, req.opts)
	if estimate.TotalSize <= gpus[0].FreeMemory {
		slog.Debug("cpu inference mode, model fits in available system memory", "model", format.HumanBytes2(estimate.TotalSize), "available", format.HumanBytes2(gpus[0].FreeMemory))
		return nil
	}

	// TODO - optimization: try to find CPU only runners first, or partial offloads with enough in system memory to make room

	return s.findRunnerToUnload()
}
