package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

type filteredEnv []string

func (e filteredEnv) LogValue() slog.Value {
	var attrs []slog.Attr
	for _, env := range e {
		if key, value, ok := strings.Cut(env, "="); ok {
			switch {
			case strings.HasPrefix(key, "OLLAMA_"),
				strings.HasPrefix(key, "CUDA_"),
				strings.HasPrefix(key, "ROCR_"),
				strings.HasPrefix(key, "ROCM_"),
				strings.HasPrefix(key, "HIP_"),
				strings.HasPrefix(key, "GPU_"),
				strings.HasPrefix(key, "HSA_"),
				strings.HasPrefix(key, "GGML_"),
				slices.Contains([]string{
					"PATH",
					"LD_LIBRARY_PATH",
					"DYLD_LIBRARY_PATH",
				}, key):
				attrs = append(attrs, slog.String(key, value))
			}
		}
	}
	return slog.GroupValue(attrs...)
}

type LlamaServer interface {
	Load(ctx context.Context, gpus discover.GpuInfoList, requireFull bool) error
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Embedding(ctx context.Context, input string) ([]float32, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	EstimatedVRAM() uint64 // Total VRAM across all GPUs
	EstimatedTotal() uint64
	EstimatedVRAMByGPU(gpuID string) uint64
	Pid() int
}

// llmServer is an instance of the llama.cpp server
type llmServer struct {
	port        int
	cmd         *exec.Cmd
	done        chan error // Channel to signal when the process exits
	status      *StatusWriter
	options     api.Options
	numParallel int
	modelPath   string

	discover.SystemInfo
	loadRequest LoadRequest

	// llamaModel is an instance of the cgo llama.cpp model definition
	// nil if this server is running the new engine
	llamaModel     *llama.Model
	llamaModelLock *sync.Mutex

	// textProcessor handles text encoding/decoding for the model in the Ollama engine
	// nil if this server is running the llama.cpp based engine
	textProcessor model.TextProcessor

	totalLayers uint64
	// gpuCount     int
	loadStart    time.Time
	loadProgress float32

	sem *semaphore.Weighted
}

type llamaServer struct {
	llmServer

	ggml     *ggml.GGML
	gpus     discover.GpuInfoList
	estimate MemoryEstimate
}

type ollamaServer struct {
	llmServer

	mem *ml.BackendMemory
}

// LoadModel will load a model from disk. The model must be in the GGML format.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func LoadModel(model string, maxArraySize int) (*ggml.GGML, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := ggml.Decode(f, maxArraySize)
	return ggml, err
}

// NewLlamaServer will run a server for the given GPUs
func NewLlamaServer(gpus discover.GpuInfoList, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	systemInfo := discover.GetSystemInfo()
	systemTotalMemory := systemInfo.System.TotalMemory
	systemFreeMemory := systemInfo.System.FreeMemory
	systemSwapFreeMemory := systemInfo.System.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	// Verify the requested context size is <= the model training size
	trainCtx := f.KV().ContextLength()
	if opts.NumCtx/numParallel > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "num_parallel", numParallel, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx) * numParallel
	}

	loadRequest := LoadRequest{LoraPath: adapters, KvSize: opts.NumCtx, BatchSize: opts.NumBatch, Parallel: numParallel, MultiUserCache: envconfig.MultiUserCache()}

	var llamaModel *llama.Model
	var textProcessor model.TextProcessor
	var err error
	if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
		textProcessor, err = model.NewTextProcessor(modelPath)
		if err != nil {
			// To prepare for opt-out mode, instead of treating this as an error, we fallback to the old runner
			slog.Debug("model not yet supported by Ollama engine, switching to compatibility mode", "model", modelPath, "error", err)
		}
	}
	if textProcessor == nil {
		llamaModel, err = llama.LoadModelFromFile(modelPath, llama.ModelParams{VocabOnly: true})
		if err != nil {
			return nil, err
		}
	}

	newEstimates := textProcessor != nil && envconfig.NewMemoryEstimates()

	defaultThreads := discover.GetSystemInfo().GetOptimalThreadCount()
	if opts.NumThread > 0 {
		loadRequest.NumThreads = opts.NumThread
	} else if defaultThreads > 0 {
		loadRequest.NumThreads = defaultThreads
	}

	// TODO - NUMA support currently doesn't work properly

	if opts.MainGPU > 0 {
		loadRequest.MainGPU = opts.MainGPU
	}

	if len(projectors) > 0 && llamaModel != nil {
		loadRequest.ProjectorPath = projectors[0]
	}

	// This will disable flash attention unless all GPUs on the system support it, even if we end up selecting a subset
	// that can handle it.
	fa := envconfig.FlashAttention()
	if fa && !gpus.FlashAttentionSupported() {
		slog.Warn("flash attention enabled but not supported by gpu")
		fa = false
	}

	if fa && !f.SupportsFlashAttention() {
		slog.Warn("flash attention enabled but not supported by model")
		fa = false
	}

	kvct := strings.ToLower(envconfig.KvCacheType())

	if fa {
		slog.Info("enabling flash attention")
		loadRequest.FlashAttention = true

		// Flash Attention also supports kv cache quantization
		// Enable if the requested and kv cache type is supported by the model
		if kvct != "" && f.SupportsKVCacheType(kvct) {
			loadRequest.KvCacheType = kvct
		} else {
			slog.Warn("kv cache type not supported by model", "type", kvct)
		}
	} else if kvct != "" && kvct != "f16" {
		slog.Warn("quantized kv cache requested but flash attention disabled", "type", kvct)
	}

	avilableLibs := make(map[string]string)
	if entries, err := os.ReadDir(discover.LibOllamaPath); err == nil {
		for _, entry := range entries {
			avilableLibs[entry.Name()] = filepath.Join(discover.LibOllamaPath, entry.Name())
		}
	}

	var gpuLibs []string
	for _, gpu := range gpus {
		gpuLibs = append(gpuLibs, gpu.RunnerName())
	}

	requested := envconfig.LLMLibrary()
	if avilableLibs[requested] != "" {
		slog.Info("using requested gpu library", "requested", requested)
		gpuLibs = []string{requested}
	}

	var compatible []string
	for _, gpuLib := range gpuLibs {
		var matchingLibs []string
		for k := range avilableLibs {
			// exact match first
			if k == gpuLib {
				matchingLibs = append([]string{k}, matchingLibs...)
				continue
			}

			// then match the family (e.g. 'cuda')
			if strings.Split(k, "_")[0] == strings.Split(gpuLib, "_")[0] {
				matchingLibs = append(matchingLibs, k)
			}
		}

		if len(matchingLibs) > 0 {
			compatible = append(compatible, matchingLibs[0])
		}
	}

	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}

	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// iterate through compatible GPU libraries such as 'cuda_v12', 'rocm', etc.
	// adding each library's respective path to the LD_LIBRARY_PATH, until finally running
	// without any LD_LIBRARY_PATH flags
	for {
		port := 0
		if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
			var l *net.TCPListener
			if l, err = net.ListenTCP("tcp", a); err == nil {
				port = l.Addr().(*net.TCPAddr).Port
				l.Close()
			}
		}
		if port == 0 {
			slog.Debug("ResolveTCPAddr failed, using random port")
			port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
		}
		params := []string{"runner"}
		if textProcessor != nil {
			// New engine
			// TODO - if we have failure to load scenarios, add logic to retry with the old runner
			params = append(params, "--ollama-engine")
		}
		params = append(params, "--model", modelPath)
		params = append(params, "--port", strconv.Itoa(port))

		var pathEnv string
		switch runtime.GOOS {
		case "windows":
			pathEnv = "PATH"
		case "darwin":
			pathEnv = "DYLD_LIBRARY_PATH"
		default:
			pathEnv = "LD_LIBRARY_PATH"
		}

		// Note: we always put our dependency paths first
		// since these are the exact version we compiled/linked against
		libraryPaths := []string{discover.LibOllamaPath}
		if libraryPath, ok := os.LookupEnv(pathEnv); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
		}

		ggmlPaths := []string{discover.LibOllamaPath}
		for _, c := range compatible {
			if libpath, ok := avilableLibs[c]; ok {
				slog.Debug("adding gpu library", "path", libpath)
				libraryPaths = append([]string{libpath}, libraryPaths...)
				ggmlPaths = append(ggmlPaths, libpath)
			}
		}

		for _, gpu := range gpus {
			if gpu.DependencyPath != nil {
				slog.Debug("adding gpu dependency paths", "paths", gpu.DependencyPath)
				libraryPaths = append(gpu.DependencyPath, libraryPaths...)
			}
		}

		// finally, add the root library path
		libraryPaths = append(libraryPaths, discover.LibOllamaPath)

		s := llmServer{
			SystemInfo:     systemInfo,
			port:           port,
			cmd:            exec.Command(exe, params...),
			status:         NewStatusWriter(os.Stderr),
			options:        opts,
			modelPath:      modelPath,
			loadRequest:    loadRequest,
			llamaModel:     llamaModel,
			llamaModelLock: &sync.Mutex{},
			textProcessor:  textProcessor,
			numParallel:    numParallel,
			sem:            semaphore.NewWeighted(int64(numParallel)),
			totalLayers:    f.KV().BlockCount() + 1,
			loadStart:      time.Now(),
			done:           make(chan error, 1),
		}

		s.cmd.Env = os.Environ()
		s.cmd.Stdout = os.Stdout
		s.cmd.Stderr = s.status
		s.cmd.SysProcAttr = LlamaServerSysProcAttr

		s.cmd.Env = append(s.cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(ggmlPaths, string(filepath.ListSeparator)))

		envWorkarounds := [][2]string{}
		for _, gpu := range gpus {
			envWorkarounds = append(envWorkarounds, gpu.EnvWorkarounds...)
		}
		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		// Update or add the path variable with our adjusted version
		pathNeeded := true
		for i := range s.cmd.Env {
			cmp := strings.SplitN(s.cmd.Env[i], "=", 2)
			if strings.EqualFold(cmp[0], pathEnv) {
				s.cmd.Env[i] = pathEnv + "=" + pathEnvVal
				pathNeeded = false
			} else if len(envWorkarounds) != 0 {
				for _, kv := range envWorkarounds {
					if strings.EqualFold(cmp[0], kv[0]) {
						s.cmd.Env[i] = kv[0] + "=" + kv[1]
					}
				}
			}
		}
		if pathNeeded {
			s.cmd.Env = append(s.cmd.Env, pathEnv+"="+pathEnvVal)
		}

		slog.Info("starting llama server", "cmd", s.cmd)
		slog.Debug("subprocess", "", filteredEnv(s.cmd.Env))

		if err = s.cmd.Start(); err != nil {
			var msg string
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			err := fmt.Errorf("error starting runner: %v %s", err, msg)
			if len(compatible) == 0 {
				if llamaModel != nil {
					llama.FreeModel(llamaModel)
				}
				return nil, err
			}

			slog.Warn("unable to start runner with compatible gpu", "error", err, "compatible", compatible)
			compatible = compatible[1:]
			continue
		}

		// reap subprocess when it exits
		go func() {
			err := s.cmd.Wait()
			// Favor a more detailed message over the process exit status
			if err != nil && s.status != nil && s.status.LastErrMsg != "" {
				slog.Error("llama runner terminated", "error", err)
				if strings.Contains(s.status.LastErrMsg, "unknown model") {
					s.status.LastErrMsg = "this model is not supported by your version of Ollama. You may need to upgrade"
				}
				s.done <- errors.New(s.status.LastErrMsg)
			} else {
				s.done <- err
			}
		}()

		if newEstimates {
			return &ollamaServer{llmServer: s}, nil
		} else {
			return &llamaServer{llmServer: s, ggml: f}, nil
		}
	}
}

var ErrRequiredFull = errors.New("unable to load full model on GPU")

func createGPULayers(estimate MemoryEstimate, ggml *ggml.GGML, gpus discover.GpuInfoList, numGPU int) ml.GPULayersList {
	if numGPU <= 0 {
		return nil
	}

	gpuLayers := make(ml.GPULayersList, len(gpus))
	for i := range gpuLayers {
		gpuLayers[i].UUID = gpus[i].ID
	}

	var sum float32
	splits := make([]float32, len(estimate.TensorSplit))
	// cumulative sum of all splits
	for i := range splits {
		sum += float32(estimate.TensorSplit[i])
		splits[i] = sum
	}

	// normalize splits
	for i := range splits {
		splits[i] /= sum
	}

	blocks := int(ggml.KV().BlockCount())
	gpuRangeStart := max(0, blocks-numGPU)
	gpuRangeStop := min(gpuRangeStart+numGPU, blocks+1)
	for i := range blocks + 1 {
		if i < gpuRangeStart || i >= gpuRangeStop {
			continue
		}

		index := slices.IndexFunc(splits, func(f float32) bool { return float32(i-gpuRangeStart)/float32(gpuRangeStop-gpuRangeStart) < f })
		if index < 0 || index >= len(gpus) {
			continue
		}

		gpuLayers[index].Layers = append(gpuLayers[index].Layers, i)
	}

	return gpuLayers
}

func (s *llamaServer) Load(ctx context.Context, gpus discover.GpuInfoList, requireFull bool) error {
	g := pickBestFullFitByLibrary(s.ggml, []string{s.loadRequest.ProjectorPath}, s.loadRequest.LoraPath, s.options, gpus, &s.numParallel)
	if g == nil {
		if !requireFull {
			g = pickBestPartialFitByLibrary(s.ggml, []string{s.loadRequest.ProjectorPath}, s.loadRequest.LoraPath, s.options, gpus, &s.numParallel)
		} else {
			return ErrRequiredFull
		}
	}

	gpus = g
	s.estimate = estimateGPULayers(gpus, s.ggml, []string{s.loadRequest.ProjectorPath}, s.options, s.numParallel)

	if len(gpus) > 1 || gpus[0].Library != "cpu" {
		switch {
		case gpus[0].Library == "metal" && s.estimate.VRAMSize > s.SystemInfo.System.TotalMemory:
			// disable partial offloading when model is greater than total system memory as this
			// can lead to locking up the system
			s.options.NumGPU = 0
		case gpus[0].Library != "metal" && s.estimate.Layers == 0:
			// Don't bother loading into the GPU if no layers can fit
			gpus = discover.GetCPUInfo()
		case s.options.NumGPU < 0 && s.estimate.Layers > 0 && gpus[0].Library != "cpu":
			s.options.NumGPU = s.estimate.Layers
		}
	}

	// On linux and windows, over-allocating CPU memory will almost always result in an error
	// Darwin has fully dynamic swap so has no direct concept of free swap space
	if runtime.GOOS != "darwin" {
		systemMemoryRequired := s.estimate.TotalSize - s.estimate.VRAMSize
		available := s.SystemInfo.System.FreeMemory + s.SystemInfo.System.FreeSwap
		if systemMemoryRequired > available {
			slog.Warn("model request too large for system", "requested", format.HumanBytes2(systemMemoryRequired), "available", available, "total", format.HumanBytes2(s.SystemInfo.System.TotalMemory), "free", format.HumanBytes2(s.SystemInfo.System.FreeMemory), "swap", format.HumanBytes2(s.SystemInfo.System.FreeSwap))
			return fmt.Errorf("model requires more system memory (%s) than is available (%s)", format.HumanBytes2(systemMemoryRequired), format.HumanBytes2(available))
		}
	}

	if requireFull && len(gpus) == 1 && gpus[0].Library == "cpu" && s.estimate.TotalSize > gpus[0].FreeMemory {
		return ErrRequiredFull
	}

	slog.Info("offload", "", s.estimate)

	s.gpus = gpus
	s.loadRequest.GPULayers = createGPULayers(s.estimate, s.ggml, gpus, s.options.NumGPU)

	// Mmap is only supported on the llama engine
	if s.textProcessor == nil {
		s.loadRequest.UseMmap = true

		// mmap has issues with partial offloading on metal
		for _, g := range gpus {
			if g.Library == "metal" &&
				uint64(s.options.NumGPU) > 0 &&
				uint64(s.options.NumGPU) < s.ggml.KV().BlockCount()+1 {
				s.options.UseMMap = new(bool)
				*s.options.UseMMap = false
			}
		}

		// Windows CUDA should not use mmap for best performance
		// Linux  with a model larger than free space, mmap leads to thrashing
		// For CPU loads we want the memory to be allocated, not FS cache
		if (runtime.GOOS == "windows" && gpus[0].Library == "cuda" && s.options.UseMMap == nil) ||
			(runtime.GOOS == "linux" && s.SystemInfo.System.FreeMemory < s.estimate.TotalSize && s.options.UseMMap == nil) ||
			(gpus[0].Library == "cpu" && s.options.UseMMap == nil) ||
			(s.options.UseMMap != nil && !*s.options.UseMMap) {
			s.loadRequest.UseMmap = false
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		_, err := s.getServerStatus(ctx)
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	resp, err := s.initModel(ctx, s.loadRequest, true)
	if err != nil {
		return err
	}

	if !resp.Success {
		slog.Warn("failed to allocate memory for model", "memory", resp.Memory)
		return errors.New("failed to allocate memory for model")
	}

	// The llama engine does its memory allocations together with model loading, so we
	// need to wait until it is done to ensure that we have accurate memory data before
	// loading the next model
	if s.textProcessor == nil {
		return s.WaitUntilRunning(ctx)
	} else {
		return nil
	}
}

func (s *ollamaServer) Load(ctx context.Context, gpus discover.GpuInfoList, requireFull bool) error {
	var success bool

	defer func() {
		if !success {
			s.initModel(ctx, LoadRequest{Close: true}, false)
		}
	}()

	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		_, err := s.getServerStatus(ctx)
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// TODO: Adjust numParallel

	pastAllocations := make(map[uint64]bool)
	var bestAllocation ml.GPULayersList
	var backoff float32

	gpuLayers, err := s.fitGPU(gpus, s.mem, requireFull, backoff)
	if err != nil {
		return err
	}

	for {
		s.loadRequest.GPULayers = gpuLayers
		resp, err := s.initModel(ctx, s.loadRequest, false)
		if err != nil {
			return err
		}

		slog.Debug("memory", "success", resp.Success, "required", resp.Memory)

		pastAllocations[gpuLayers.Hash()] = resp.Success

		// Don't consider layouts done without model memory data as candidates for best
		if s.mem != nil && resp.Success && gpuLayers.Sum() > bestAllocation.Sum() {
			bestAllocation = gpuLayers
		}

		s.mem = &resp.Memory

	layout:
		newGPULayers, err := s.fitGPU(gpus, s.mem, requireFull, backoff)
		if err != nil {
			return err
		}

		// If the previous allocation was successful and we came up with the same layout again, we've converged.
		if resp.Success && reflect.DeepEqual(gpuLayers, newGPULayers) {
			break
		}

		if success, ok := pastAllocations[newGPULayers.Hash()]; ok {
			if bestAllocation != nil {
				// We repeated a previous memory layout - use the best allocation that we've seen so far.
				s.loadRequest.GPULayers = bestAllocation
				break
			} else if success {
				// We didn't consider this layout the best but it succeeded, so take it anyways
				s.loadRequest.GPULayers = newGPULayers
				break
			}

			// Memory allocation failed even though we created a layout that we thought should
			// fit in available memory. This could happen if either our free memory reports
			// are incorrect or if available memory is changing between layout and allocation
			// time. Apply an exponential backoff to try to find the real amount of available
			// space.
			if backoff == 0 {
				backoff = 0.01
			} else {
				backoff *= 2
			}

			if backoff > 1 {
				return errors.New("memory layout cannot be allocated")
			}

			slog.Info("model layout did not fit, applying backoff", "backoff", backoff)
			clear(pastAllocations)
			goto layout
		}

		gpuLayers = newGPULayers
	}

	resp, err := s.initModel(ctx, s.loadRequest, true)
	if err != nil {
		return err
	}

	success = resp.Success

	if !resp.Success {
		slog.Warn("failed to allocate memory for model", "memory", resp.Memory)
		return errors.New("failed to allocate memory for model")
	}

	return nil
}

func fit(layers []uint64, gpus discover.GpuInfoList, capacity float32, requestedLayers int) (gpuLayers ml.GPULayersList) {
	device := len(gpus) - 1
	gpuLayers = ml.GPULayersList{{UUID: gpus[device].ID}}
	freeSpace := uint64(float32(gpus[device].FreeMemory) * capacity)
	for i := len(layers) - 1; i >= 0; i-- {
		if requestedLayers >= 0 && len(layers)-1-i >= requestedLayers {
			break
		}

		for {
			if layers[i] <= freeSpace {
				gpuLayers[0].Layers = append([]int{i}, gpuLayers[0].Layers...)
				freeSpace -= layers[i]
				break
			}

			device--
			if device < 0 {
				return gpuLayers
			}
			gpuLayers = append(ml.GPULayersList{{UUID: gpus[device].ID}}, gpuLayers...)
			freeSpace = uint64(float32(gpus[device].FreeMemory) * capacity)
		}
	}

	return gpuLayers
}

func distributeGPU(layers []uint64, gpus discover.GpuInfoList, requestedLayers int, forceRequest bool) (gpuLayers ml.GPULayersList) {
	var high float32 = 1
	var low float32 = 0

	// If we need to fulfill the requested number of layers, pretend we have almost infinite VRAM
	if requestedLayers >= 0 && forceRequest {
		high = 1000
	}

	bestAssignments := fit(layers, gpus, high, requestedLayers)
	maxNumGPU := bestAssignments.Sum()
	if maxNumGPU == 0 {
		return bestAssignments
	}

	for high-low > 1e-6 {
		mid := (low + high) / 2
		assignments := fit(layers, gpus, mid, requestedLayers)
		if assignments.Sum() == maxNumGPU {
			high = mid
			bestAssignments = assignments
		} else {
			low = mid
		}
	}

	return bestAssignments
}

func assignLayers(layers []uint64, gpus discover.GpuInfoList, requestedLayers int, lastUsedGPU int) (gpuLayers ml.GPULayersList) {
	requestedLayers = min(len(layers), requestedLayers)

	if !envconfig.SchedSpread() {
		for i := lastUsedGPU; i < len(gpus); i++ {
			// Try to pack things into as few GPUs as possible
			forceRequest := i == len(gpus)-1
			gpuLayers = distributeGPU(layers, gpus[:i+1], requestedLayers, forceRequest)
			if gpuLayers.Sum() == len(layers) {
				break
			}
		}
	} else {
		gpuLayers = distributeGPU(layers, gpus, requestedLayers, true)
	}

	if gpuLayers.Sum() < len(layers) {
		// If we can't fit everything then prefer offloading layers other than the output layer
		gpuLayers = distributeGPU(layers[:len(layers)-1], gpus, requestedLayers, true)
	}

	return gpuLayers
}

func (s *ollamaServer) fitGPU(systemGPUs discover.GpuInfoList, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, error) {
	if s.totalLayers == 0 || s.options.NumGPU == 0 || len(systemGPUs) == 0 || (len(systemGPUs) == 1 && systemGPUs[0].Library == "cpu") {
		return nil, nil
	}

	gpus := append(make(discover.GpuInfoList, 0, len(systemGPUs)), systemGPUs...)
	sort.Sort(sort.Reverse(discover.ByFreeMemory(gpus)))

	// TODO: SCHED_SPREAD first bounce?
	if memory == nil {
		memory = &ml.BackendMemory{CPU: ml.DeviceMemory{
			Weights: make([]ml.Memory, s.totalLayers),
			Cache:   make([]ml.Memory, s.totalLayers),
		}}
	}

	// TODO: The graph placement needs to be improved to prevent thrashing
	layers := make([]uint64, len(memory.CPU.Weights))
	for i := range layers {
		for j := range memory.GPUs {
			layers[i] += memory.GPUs[j].Weights[i].Size
			layers[i] += memory.GPUs[j].Cache[i].Size
		}
		layers[i] += memory.CPU.Weights[i].Size
		layers[i] += memory.CPU.Cache[i].Size
	}

	var gpuLayers ml.GPULayersList
	for _, gl := range gpus.ByLibrary() {
		// If a GPU already has a graph allocated on it, then we should continue to use it.
		// Otherwise, we lose information that we got from previous allocations, which can
		// cause cycling. Plus, we get more information about required allocation from each
		// iteration, so it doesn't make sense that a later iteration would use fewer GPUs.
		lastUsedGPU := 0
		for i := range gl {
			found := false
			for j := range memory.GPUs {
				if gl[i].ID == memory.GPUs[j].UUID {
					if memory.GPUs[j].Graph.Size != 0 {
						lastUsedGPU = i
					}

					reserved := uint64(float32(gl[i].FreeMemory)*backoff) + gl[i].MinimumMemory + envconfig.GpuOverhead() + memory.GPUs[j].Graph.Size
					if gl[i].FreeMemory > reserved {
						gl[i].FreeMemory -= reserved
					} else {
						gl[i].FreeMemory = 0
					}

					slog.Debug("gpu", "id", gl[i].ID, "free", format.HumanBytes2(gl[i].FreeMemory))
					found = true
					break
				}
			}
			if !found {
				// The runner doesn't report seeing this GPU
				gl[i].FreeMemory = 0
			}
		}

		libraryGpuLayers := assignLayers(layers, gl, s.options.NumGPU, lastUsedGPU)
		if libraryGpuLayers.Sum() > gpuLayers.Sum() {
			gpuLayers = libraryGpuLayers
		}
	}

	// These sizes will only increase as we go through additional iterations and get additional information.
	cpuSize := memory.InputWeights.Size + memory.CPU.Graph.Size
	var vramSize uint64
	for _, gl := range gpuLayers {
		for _, gpu := range memory.GPUs {
			if gl.UUID == gpu.UUID {
				vramSize += gpu.Graph.Size
				break
			}
		}
	}

nextLayer:
	for i := range layers {
		for _, g := range gpuLayers {
			for _, gl := range g.Layers {
				if i == gl {
					vramSize += layers[i]
					continue nextLayer
				}
			}
		}
		cpuSize += layers[i]
	}

	if requireFull {
		if gpuLayers.Sum() < len(layers) && (s.options.NumGPU < 0 || gpuLayers.Sum() < s.options.NumGPU) {
			return nil, ErrRequiredFull
		}

		if cpuSize > s.SystemInfo.System.FreeMemory {
			return nil, ErrRequiredFull
		}
	}

	// On linux and windows, over-allocating CPU memory will almost always result in an error
	// Darwin has fully dynamic swap so has no direct concept of free swap space
	if runtime.GOOS != "darwin" {
		available := s.SystemInfo.System.FreeMemory + s.SystemInfo.System.FreeSwap
		if cpuSize > available {
			slog.Warn("model request too large for system", "requested", format.HumanBytes2(cpuSize), "available", available, "total", format.HumanBytes2(s.SystemInfo.System.TotalMemory), "free", format.HumanBytes2(s.SystemInfo.System.FreeMemory), "swap", format.HumanBytes2(s.SystemInfo.System.FreeSwap))
			return nil, fmt.Errorf("model requires more system memory (%s) than is available (%s)", format.HumanBytes2(cpuSize), format.HumanBytes2(available))
		}
	} else {
		if vramSize > s.SystemInfo.System.TotalMemory {
			// disable partial offloading when model is greater than total system memory as this
			// can lead to locking up the system
			s.options.NumGPU = 0
			gpuLayers = nil
		}
	}

	if gpuLayers.Sum() == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
	}

	return gpuLayers, nil
}

func (s *llmServer) initModel(ctx context.Context, req LoadRequest, commit bool) (*LoadResponse, error) {
	req.Commit = commit

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling load data: %w", err)
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/load", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("error creating load request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, fmt.Errorf("do load request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read load request: %w", err)
	}

	if resp.StatusCode >= 400 {
		log.Printf("llm load error: %s", body)
		return nil, fmt.Errorf("%s", body)
	}

	var llmResp LoadResponse
	if err := json.Unmarshal(body, &llmResp); err != nil {
		return nil, fmt.Errorf("health unmarshal load response: %w", err)
	}

	return &llmResp, nil
}

type ServerStatus int

const ( // iota is reset to 0
	ServerStatusReady ServerStatus = iota
	ServerStatusNoSlotsAvailable
	ServerStatusLaunched
	ServerStatusLoadingModel
	ServerStatusNotResponding
	ServerStatusError
)

func (s ServerStatus) String() string {
	switch s {
	case ServerStatusReady:
		return "llm server ready"
	case ServerStatusNoSlotsAvailable:
		return "llm busy - no slots available"
	case ServerStatusLaunched:
		return "llm server launched"
	case ServerStatusLoadingModel:
		return "llm server loading model"
	case ServerStatusNotResponding:
		return "llm server not responding"
	default:
		return "llm server error"
	}
}

type ServerStatusResponse struct {
	Status   ServerStatus `json:"status"`
	Progress float32      `json:"progress"`
}

func (s *llmServer) getServerStatus(ctx context.Context) (ServerStatus, error) {
	// Fail fast if its exited
	if s.cmd.ProcessState != nil {
		msg := ""
		if s.status != nil && s.status.LastErrMsg != "" {
			msg = s.status.LastErrMsg
		}
		if s.cmd.ProcessState.ExitCode() == -1 {
			// Most likely a signal killed it, log some more details to try to help troubleshoot
			slog.Warn("llama runner process no longer running", "sys", s.cmd.ProcessState.Sys(), "string", s.cmd.ProcessState)
		}
		return ServerStatusError, fmt.Errorf("llama runner process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return ServerStatusError, fmt.Errorf("error creating GET request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return ServerStatusNotResponding, errors.New("server not responding")
		}
		if strings.Contains(err.Error(), "connection refused") {
			return ServerStatusNotResponding, errors.New("connection refused")
		}
		return ServerStatusError, fmt.Errorf("health resp: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServerStatusError, fmt.Errorf("read health request: %w", err)
	}

	var ssr ServerStatusResponse
	if err := json.Unmarshal(body, &ssr); err != nil {
		return ServerStatusError, fmt.Errorf("health unmarshal encode response: %w", err)
	}

	switch ssr.Status {
	case ServerStatusLoadingModel:
		s.loadProgress = ssr.Progress
		return ssr.Status, nil
	case ServerStatusLaunched, ServerStatusReady, ServerStatusNoSlotsAvailable:
		return ssr.Status, nil
	default:
		return ssr.Status, fmt.Errorf("server error: %+v", ssr)
	}
}

// getServerStatusRetry will retry if ServerStatusNoSlotsAvailable is received
func (s *llmServer) getServerStatusRetry(ctx context.Context) (ServerStatus, error) {
	var retries int
	for {
		status, err := s.getServerStatus(ctx)
		if err != nil {
			return status, err
		}

		if status == ServerStatusNoSlotsAvailable {
			if retries >= 10 {
				return status, fmt.Errorf("no slots available after %d retries", retries)
			}

			time.Sleep(5 * time.Millisecond)
			retries++
			continue
		}

		return status, nil
	}
}

func (s *llmServer) Ping(ctx context.Context) error {
	_, err := s.getServerStatus(ctx)
	if err != nil {
		slog.Debug("server unhealthy", "error", err)
		return err
	}
	return nil
}

func (s *llmServer) WaitUntilRunning(ctx context.Context) error {
	stallDuration := envconfig.LoadTimeout()    // If no progress happens
	stallTimer := time.Now().Add(stallDuration) // give up if we stall

	slog.Info("waiting for llama runner to start responding")
	var lastStatus ServerStatus = -1
	fullyLoaded := false

	for {
		select {
		case <-ctx.Done():
			slog.Warn("client connection closed before server finished loading, aborting load")
			return fmt.Errorf("timed out waiting for llama runner to start: %w", ctx.Err())
		case err := <-s.done:
			return fmt.Errorf("llama runner process has terminated: %w", err)
		default:
		}
		if time.Now().After(stallTimer) {
			// timeout
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			return fmt.Errorf("timed out waiting for llama runner to start - progress %0.2f - %s", s.loadProgress, msg)
		}
		if s.cmd.ProcessState != nil {
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			return fmt.Errorf("llama runner process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
		}
		ctx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		defer cancel()
		priorProgress := s.loadProgress
		status, _ := s.getServerStatus(ctx)
		if lastStatus != status && status != ServerStatusReady {
			// Only log on status changes
			slog.Info("waiting for server to become available", "status", status)
		}
		switch status {
		case ServerStatusReady:
			slog.Info(fmt.Sprintf("llama runner started in %0.2f seconds", time.Since(s.loadStart).Seconds()))
			return nil
		default:
			lastStatus = status
			// Reset the timer as long as we're making forward progress on the load
			if priorProgress != s.loadProgress {
				slog.Debug(fmt.Sprintf("model load progress %0.2f", s.loadProgress))
				stallTimer = time.Now().Add(stallDuration)
			} else if !fullyLoaded && int(s.loadProgress*100.0) >= 100 {
				slog.Debug("model load completed, waiting for server to become available", "status", status)
				stallTimer = time.Now().Add(stallDuration)
				fullyLoaded = true
			}
			time.Sleep(time.Millisecond * 250)
			continue
		}
	}
}

func (s *llmServer) Pid() int {
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return -1
}

var grammarJSON = `
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
         string ":" ws value
    ("," ws string ":" ws value)*
  )? ws "}" 
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? ws "]" 
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" 
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? 
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

const maxBufferSize = 512 * format.KiloByte

type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Images  []ImageData
	Options *api.Options

	Grammar string // set before sending the request to the subprocess
}

type LoadRequest struct {
	Commit bool
	Close  bool

	LoraPath       []string
	Parallel       int
	BatchSize      int
	FlashAttention bool
	KvSize         int
	KvCacheType    string
	NumThreads     int
	GPULayers      ml.GPULayersList
	MultiUserCache bool

	// Legacy fields - not used with the Ollama engine
	ProjectorPath string
	MainGPU       int
	UseMmap       bool
}

type LoadResponse struct {
	Success bool
	Memory  ml.BackendMemory
}

// DoneReason represents the reason why a completion response is done
type DoneReason int

const (
	// DoneReasonStop indicates the completion stopped naturally
	DoneReasonStop DoneReason = iota
	// DoneReasonLength indicates the completion stopped due to length limits
	DoneReasonLength
	// DoneReasonConnectionClosed indicates the completion stopped due to the connection being closed
	DoneReasonConnectionClosed
)

func (d DoneReason) String() string {
	switch d {
	case DoneReasonLength:
		return "length"
	case DoneReasonStop:
		return "stop"
	default:
		return "" // closed
	}
}

type CompletionResponse struct {
	Content            string        `json:"content"`
	DoneReason         DoneReason    `json:"done_reason"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
}

func (s *llmServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	slog.Debug("completion request", "images", len(req.Images), "prompt", len(req.Prompt), "format", string(req.Format))
	slog.Log(ctx, logutil.LevelTrace, "completion request", "prompt", req.Prompt)

	if len(req.Format) > 0 {
		switch string(req.Format) {
		case `null`, `""`:
			// Field was set, but "missing" a value. We accept
			// these as "not set".
			break
		case `"json"`:
			req.Grammar = grammarJSON
		default:
			if req.Format[0] != '{' {
				return fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", req.Format)
			}

			// User provided a JSON schema
			g := llama.SchemaToGrammar(req.Format)
			if g == nil {
				return fmt.Errorf("invalid JSON schema in format")
			}
			req.Grammar = string(g)
		}
	}

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return err
	}
	defer s.sem.Release(1)

	// put an upper limit on num_predict to avoid the model running on forever
	if req.Options.NumPredict < 0 || req.Options.NumPredict > 10*s.options.NumCtx {
		req.Options.NumPredict = 10 * s.options.NumCtx
	}

	// Make sure the server is ready
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	// Handling JSON marshaling with special characters unescaped.
	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(req); err != nil {
		return fmt.Errorf("failed to marshal data: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating POST request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(serverReq)
	if err != nil {
		slog.Error("post predict", "error", err)
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llm error response: %w", err)
		}
		log.Printf("llm predict error: %s", bodyBytes)
		return fmt.Errorf("%s", bodyBytes)
	}

	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(buf, maxBufferSize)

	// keep track of the last token generated, this is used to abort if the model starts looping
	var lastToken string
	var tokenRepeat int

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			// This handles the request cancellation
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}

			var c CompletionResponse
			if err := json.Unmarshal(evt, &c); err != nil {
				return fmt.Errorf("error unmarshalling llm prediction response: %v", err)
			}
			switch {
			case strings.TrimSpace(c.Content) == lastToken:
				tokenRepeat++
			default:
				lastToken = strings.TrimSpace(c.Content)
				tokenRepeat = 0
			}

			// 30 picked as an arbitrary max token repeat limit, modify as needed
			if tokenRepeat > 30 {
				slog.Debug("prediction aborted, token repeat limit reached")
				return ctx.Err()
			}

			if c.Content != "" {
				fn(CompletionResponse{
					Content: c.Content,
				})
			}

			if c.Done {
				fn(c)
				return nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			var msg string
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			} else {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}

		return fmt.Errorf("error reading llm response: %v", err)
	}

	return nil
}

type EmbeddingRequest struct {
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

func (s *llmServer) Embedding(ctx context.Context, input string) ([]float32, error) {
	slog.Log(ctx, logutil.LevelTrace, "embedding request", "input", input)

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting embedding request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return nil, err
	}
	defer s.sem.Release(1)

	// Make sure the server is ready
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return nil, err
	} else if status != ServerStatusReady {
		return nil, fmt.Errorf("unexpected server status: %s", status)
	}

	data, err := json.Marshal(EmbeddingRequest{Content: input})
	if err != nil {
		return nil, fmt.Errorf("error marshaling embed data: %w", err)
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/embedding", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("error creating embed request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, fmt.Errorf("do embedding request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading embed response: %w", err)
	}

	if resp.StatusCode >= 400 {
		log.Printf("llm embedding error: %s", body)
		return nil, fmt.Errorf("%s", body)
	}

	var e EmbeddingResponse
	if err := json.Unmarshal(body, &e); err != nil {
		return nil, fmt.Errorf("unmarshal tokenize response: %w", err)
	}

	return e.Embedding, nil
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func (s *llmServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	s.llamaModelLock.Lock()
	defer s.llamaModelLock.Unlock()

	if s.llamaModel != nil {
		return s.llamaModel.Tokenize(content, false, true)
	}
	if s.textProcessor != nil {
		tokens, err := s.textProcessor.Encode(content, false)
		if err != nil {
			return nil, err
		}
		toks := make([]int, len(tokens))
		for i, t := range tokens {
			toks[i] = int(t)
		}
		return toks, nil
	}
	// not reached
	return nil, fmt.Errorf("no tokenizer configured")
}

type DetokenizeRequest struct {
	Tokens []int `json:"tokens"`
}

type DetokenizeResponse struct {
	Content string `json:"content"`
}

func (s *llmServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	s.llamaModelLock.Lock()
	defer s.llamaModelLock.Unlock()

	if s.llamaModel != nil {
		var resp string
		for _, token := range tokens {
			resp += s.llamaModel.TokenToPiece(token)
		}
		return resp, nil
	}
	if s.textProcessor != nil {
		toks := make([]int32, len(tokens))
		for i, t := range tokens {
			toks[i] = int32(t)
		}
		content, err := s.textProcessor.Decode(toks)
		if err != nil {
			return "", err
		}
		return content, nil
	}
	// not reached
	return "", fmt.Errorf("no tokenizer configured")
}

func (s *llmServer) Close() error {
	s.llamaModelLock.Lock()
	if s.llamaModel != nil {
		llama.FreeModel(s.llamaModel)
		s.llamaModel = nil
	}
	s.llamaModelLock.Unlock()

	if s.cmd != nil {
		slog.Debug("stopping llama server", "pid", s.Pid())
		if err := s.cmd.Process.Kill(); err != nil {
			return err
		}
		// if ProcessState is already populated, Wait already completed, no need to wait again
		if s.cmd.ProcessState == nil {
			slog.Debug("waiting for llama server to exit", "pid", s.Pid())
			<-s.done
		}

		slog.Debug("llama server stopped", "pid", s.Pid())
	}

	return nil
}

func (s *llamaServer) EstimatedVRAM() uint64 {
	return s.estimate.VRAMSize
}

func (s *llamaServer) EstimatedTotal() uint64 {
	return s.estimate.TotalSize
}

func (s *llamaServer) EstimatedVRAMByGPU(gpuID string) uint64 {
	for i, gpu := range s.gpus {
		if gpu.ID == gpuID {
			if i < len(s.estimate.GPUSizes) {
				return s.estimate.GPUSizes[i]
			}
		}
	}
	return 0
}

func (s *ollamaServer) EstimatedVRAM() uint64 {
	if s.mem == nil {
		return 0
	}

	var mem uint64

	for _, g := range s.mem.GPUs {
		mem += g.SumAllocated()
	}

	// Some elements are always on CPU. However, if we have allocated the first layer
	// on the GPU then include these as well, to represent complete offloading.
	if len(s.mem.CPU.Weights) > 0 {
		if s.mem.CPU.Weights[0].Size == 0 && s.mem.CPU.Cache[0].Size == 0 {
			mem += s.mem.InputWeights.Size
			mem += s.mem.CPU.Graph.Size
		}
	}

	return mem
}

func (s *ollamaServer) EstimatedTotal() uint64 {
	if s.mem == nil {
		return 0
	}

	mem := s.mem.InputWeights.Size
	mem += s.mem.CPU.SumAllocated()
	for _, g := range s.mem.GPUs {
		mem += g.SumAllocated()
	}

	return mem
}

func (s *ollamaServer) EstimatedVRAMByGPU(gpuID string) uint64 {
	if s.mem == nil {
		return 0
	}

	for _, g := range s.mem.GPUs {
		if g.UUID == gpuID {
			return g.SumAllocated()
		}
	}

	return 0
}
