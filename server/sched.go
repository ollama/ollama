package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
	"github.com/ollama/ollama/llm"
	"golang.org/x/exp/slices"
)

type LlmRequest struct {
	ctx             context.Context //nolint:containedctx
	model           *Model
	opts            api.Options
	sessionDuration time.Duration
	successCh       chan *runnerRef
	errCh           chan error
}

type Scheduler struct {
	pendingReqCh  chan *LlmRequest
	finishedReqCh chan *LlmRequest
	expiredCh     chan *runnerRef
	unloadedCh    chan interface{}

	loaded   map[string]*runnerRef
	loadedMu sync.Mutex

	loadFn      func(req *LlmRequest, ggml *llm.GGML, gpus gpu.GpuInfoList)
	newServerFn func(gpus gpu.GpuInfoList, model string, ggml *llm.GGML, adapters []string, projectors []string, opts api.Options) (llm.LlamaServer, error)
	getGpuFn    func() gpu.GpuInfoList
}

// TODO set this to zero after a release or two, to enable multiple models by default
var loadedMax = 1          // Maximum runners; < 1 maps to as many as will fit in VRAM (unlimited for CPU runners)
var maxQueuedRequests = 10 // TODO configurable
var numParallel = 1

func InitScheduler(ctx context.Context) *Scheduler {
	maxRunners := os.Getenv("OLLAMA_MAX_LOADED_MODELS")
	if maxRunners != "" {
		m, err := strconv.Atoi(maxRunners)
		if err != nil {
			slog.Error("invalid setting", "OLLAMA_MAX_LOADED_MODELS", maxRunners, "error", err)
		} else {
			loadedMax = m
		}
	}
	if onp := os.Getenv("OLLAMA_NUM_PARALLEL"); onp != "" {
		p, err := strconv.Atoi(onp)
		if err != nil || p <= 0 {
			slog.Error("invalid parallel setting, must be greater than zero", "OLLAMA_NUM_PARALLEL", onp, "error", err)
		} else {
			numParallel = p
		}
	}

	sched := &Scheduler{
		pendingReqCh:  make(chan *LlmRequest, maxQueuedRequests),
		finishedReqCh: make(chan *LlmRequest, maxQueuedRequests),
		expiredCh:     make(chan *runnerRef, maxQueuedRequests),
		unloadedCh:    make(chan interface{}, maxQueuedRequests),
		loaded:        make(map[string]*runnerRef),
		newServerFn:   llm.NewLlamaServer,
		getGpuFn:      gpu.GetGPUInfo,
	}
	sched.loadFn = sched.load
	return sched
}

// context must be canceled to decrement ref count and release the runner
func (s *Scheduler) GetRunner(c context.Context, model *Model, opts api.Options, sessionDuration time.Duration) (chan *runnerRef, chan error) {
	req := &LlmRequest{
		ctx:             c,
		model:           model,
		opts:            opts,
		sessionDuration: sessionDuration,
		successCh:       make(chan *runnerRef),
		errCh:           make(chan error, 1),
	}
	// context split across parallel threads
	opts.NumCtx = opts.NumCtx * numParallel
	select {
	case s.pendingReqCh <- req:
	default:
		req.errCh <- fmt.Errorf("server busy, please try again.  maximum pending requests exceeded")
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
				} else if loadedMax > 0 && loadedCount >= loadedMax {
					slog.Debug("max runners achieved, unloading one to make room", "runner_count", loadedCount)
					runnerToExpire = s.findRunnerToUnload(pending)
				} else {
					// Either no models are loaded or below loadedMax
					// Get a refreshed GPU list
					gpus := s.getGpuFn()

					// Load model for fitting
					ggml, err := llm.LoadModel(pending.model.ModelPath)
					if err != nil {
						pending.errCh <- err
						break
					}

					// If we're CPU only mode, just limit by loadedMax above
					// TODO handle system memory exhaustion
					if (len(gpus) == 1 && gpus[0].Library == "cpu") || pending.opts.NumGPU == 0 {
						slog.Debug("cpu mode with existing models, loading")
						s.loadFn(pending, ggml, gpus)
						break
					}

					// No models loaded. Load the model but prefer the best fit.
					if loadedCount == 0 {
						slog.Debug("loading first model", "model", pending.model.ModelPath)
						g := pickBestFitGPUs(pending, ggml, gpus)
						if g != nil {
							gpus = g
						}
						s.loadFn(pending, ggml, gpus)
						break
					}

					// More than one loaded model, so we have to see if the new one fits
					// Update free memory from currently loaded models
					s.updateFreeSpace(gpus)
					gpus = pickBestFitGPUs(pending, ggml, gpus)
					if gpus != nil {
						slog.Debug("new model fits with existing models, loading")
						s.loadFn(pending, ggml, gpus)
						break
					}
					runnerToExpire = s.findRunnerToUnload(pending)
				}

				if runnerToExpire == nil {
					// Shouildn't happen
					slog.Error("runner to expire was nil!")
					continue
				}
				// Trigger an expiration to unload once it's done
				runnerToExpire.refMu.Lock()
				slog.Debug("resetting model to expire immediately to make room", "model", runnerToExpire.model, "refCount", runnerToExpire.refCount)
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
				slog.Debug("waiting for pending requests to complete and unload to occur", "model", runnerToExpire.model)
				select {
				case <-ctx.Done():
					slog.Debug("shutting down scheduler pending loop")
					return
				case <-s.unloadedCh:
					slog.Debug("unload completed", "model", runnerToExpire.model)
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
				slog.Error("finished requeset signal received after model unloaded", "model", finished.model.ModelPath)
				continue
			}
			runner.refMu.Lock()
			runner.refCount--
			if runner.refCount <= 0 {
				if runner.sessionDuration <= 0 {
					slog.Debug("runner with zero duration has gone idle, expiring to unload", "model", runner.model)
					if runner.expireTimer != nil {
						runner.expireTimer.Stop()
						runner.expireTimer = nil
					}
					s.expiredCh <- runner
				} else if runner.expireTimer == nil {
					slog.Debug("runner with non-zero duration has gone idle, adding timer", "model", runner.model, "duration", runner.sessionDuration)
					runner.expireTimer = time.AfterFunc(runner.sessionDuration, func() {
						slog.Debug("timer expired, expiring to unload", "model", runner.model)
						runner.refMu.Lock()
						defer runner.refMu.Unlock()
						if runner.expireTimer != nil {
							runner.expireTimer.Stop()
						}
						s.expiredCh <- runner
					})
				} else {
					slog.Debug("runner with non-zero duration has gone idle, resetting timer", "model", runner.model, "duration", runner.sessionDuration)
					runner.expireTimer.Reset(runner.sessionDuration)
				}
			}
			slog.Debug("after processing request finished event", "model", runner.model, "refCount", runner.refCount)
			runner.refMu.Unlock()
		case runner := <-s.expiredCh:
			slog.Debug("runner expired event received", "model", runner.model)
			runner.refMu.Lock()
			if runner.refCount > 0 {
				// Shouldn't happen, but safeguard to ensure no leaked runners
				slog.Debug("expired event with positive ref count, retrying", "model", runner.model, "refCount", runner.refCount)
				go func(runner *runnerRef) {
					// We can't unload yet, but want to as soon as the current request completes
					// So queue up another expired event
					time.Sleep(10 * time.Millisecond)
					s.expiredCh <- runner
				}(runner)
				runner.refMu.Unlock()
				continue
			}

			slog.Debug("got lock to unload", "model", runner.model)
			runner.unload()
			s.loadedMu.Lock()
			delete(s.loaded, runner.model)
			s.loadedMu.Unlock()
			slog.Debug("runner released", "model", runner.model)
			runner.refMu.Unlock()
			slog.Debug("sending an unloaded event", "model", runner.model)
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
	runner.sessionDuration = pending.sessionDuration
	pending.successCh <- runner
	go func() {
		<-pending.ctx.Done()
		slog.Debug("context for request finished")
		finished <- pending
	}()
}

func (s *Scheduler) load(req *LlmRequest, ggml *llm.GGML, gpus gpu.GpuInfoList) {
	llama, err := s.newServerFn(gpus, req.model.ModelPath, ggml, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts)
	if err != nil {
		// some older models are not compatible with newer versions of llama.cpp
		// show a generalized compatibility error until there is a better way to
		// check for model compatibility
		if errors.Is(llm.ErrUnsupportedFormat, err) || strings.Contains(err.Error(), "failed to load model") {
			err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, req.model.ShortName)
		}
		slog.Info("NewLlamaServer failed", "model", req.model.ModelPath, "error", err)
		req.errCh <- err
		return
	}
	runner := &runnerRef{}
	runner.model = req.model.ModelPath
	runner.adapters = req.model.AdapterPaths
	runner.projectors = req.model.ProjectorPaths
	runner.llama = llama
	runner.Options = &req.opts
	runner.sessionDuration = req.sessionDuration
	runner.gpus = gpus
	runner.estimatedVRAM = llama.EstimatedVRAM()
	runner.loading = true
	runner.refCount = 1
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
			slog.Debug("triggering expiration for failed load", "model", runner.model)
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

func (s *Scheduler) updateFreeSpace(allGpus gpu.GpuInfoList) {
	type predKey struct {
		Library string
		ID      string
	}
	predMap := map[predKey]uint64{} // Sum up the total predicted usage per GPU for all runners
	s.loadedMu.Lock()
	for _, r := range s.loaded {
		r.refMu.Lock()
		gpuIDs := make([]string, 0, len(r.gpus))
		if r.llama != nil {

			// TODO this should be broken down by GPU instead of assuming uniform spread
			estimatedVRAMPerGPU := r.llama.EstimatedVRAM() / uint64(len(r.gpus))
			for _, gpu := range r.gpus {
				gpuIDs = append(gpuIDs, gpu.ID)
			}
			for _, gpu := range allGpus {
				if slices.Contains(gpuIDs, gpu.ID) {
					predMap[predKey{gpu.Library, gpu.ID}] += estimatedVRAMPerGPU
				}
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
				// after we start our first runner, then we'll never acount for that, so picking the smallest free value seems prudent.
				allGpus[i].FreeMemory = allGpus[i].TotalMemory - p
			}
			slog.Info("updated VRAM", "gpu", allGpus[i].ID, "library", allGpus[i].Library, "total", format.HumanBytes2(allGpus[i].TotalMemory), "available", format.HumanBytes2(allGpus[i].FreeMemory))
		}
	}
}

type runnerRef struct {
	refMu sync.Mutex
	// refCond   sync.Cond // Signaled on transition from 1 -> 0 refCount
	refCount uint // prevent unloading if > 0
	// unloading bool      // set to true when we are trying to unload the runner

	llama         llm.LlamaServer
	loading       bool            // True only during initial load, then false forever
	gpus          gpu.GpuInfoList // Recorded at time of provisioning
	estimatedVRAM uint64

	sessionDuration time.Duration
	expireTimer     *time.Timer

	model      string
	adapters   []string
	projectors []string
	*api.Options
}

// The refMu must already be held when calling unload
func (runner *runnerRef) unload() {
	if runner.llama != nil {
		runner.llama.Close()
	}
	runner.llama = nil
	runner.adapters = nil
	runner.projectors = nil
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

	// Don't reload runner if num_gpu=-1 was provided
	optsExisting := runner.Options.Runner
	optsNew := req.opts.Runner
	if optsNew.NumGPU < 0 {
		optsExisting.NumGPU = -1
		optsNew.NumGPU = -1
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if !reflect.DeepEqual(runner.adapters, req.model.AdapterPaths) || // have the adapters changed?
		!reflect.DeepEqual(runner.projectors, req.model.ProjectorPaths) || // have the projectors changed?
		!reflect.DeepEqual(optsExisting, optsNew) || // have the runner options changed?
		runner.llama.Ping(ctx) != nil {
		return true
	}

	return false
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

// pickBestFitGPUs will try to find the optimal placement of the model in the available GPUs where the model fully fits
// If the model can not be fit fully within the available GPU(s) nil is returned
func pickBestFitGPUs(req *LlmRequest, ggml *llm.GGML, gpus gpu.GpuInfoList) gpu.GpuInfoList {
	var estimatedVRAM uint64
	for _, gl := range gpus.ByLibrary() {
		var ok bool
		sgl := append(make(gpu.GpuInfoList, 0, len(gl)), gl...)

		// TODO - potentially sort by performance capability, existing models loaded, etc.
		// Note: at present, this will favor more VRAM over faster GPU speed in mixed setups
		sort.Sort(sort.Reverse(gpu.ByFreeMemory(sgl)))

		// First attempt to fit the model into a single GPU
		for _, g := range sgl {
			if ok, estimatedVRAM = llm.PredictServerFit([]gpu.GpuInfo{g}, ggml, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts); ok {
				slog.Debug("new model will fit in available VRAM in single GPU, loading", "model", req.model.ModelPath, "gpu", g.ID, "available", g.FreeMemory, "required", format.HumanBytes2(estimatedVRAM))
				return []gpu.GpuInfo{g}
			}
		}

		// TODO future refinements
		// - if multiple Libraries, see if any single GPU in any Library will fit
		// - try subsets of GPUs instead of just falling back to 1 or all in a family

		// Now try all the GPUs
		if ok, estimatedVRAM = llm.PredictServerFit(gl, ggml, req.model.AdapterPaths, req.model.ProjectorPaths, req.opts); ok {
			slog.Debug("new model will fit in available VRAM, loading", "model", req.model.ModelPath, "library", gl[0].Library, "required", format.HumanBytes2(estimatedVRAM))
			return gl
		}
	}
	return nil
}

// findRunnerToUnload finds a runner to unload to make room for a new model
func (s *Scheduler) findRunnerToUnload(req *LlmRequest) *runnerRef {
	s.loadedMu.Lock()
	runnerList := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runnerList = append(runnerList, r)
	}
	s.loadedMu.Unlock()

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
