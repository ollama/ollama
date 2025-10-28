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
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
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
	ModelPath() string
	Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Embedding(ctx context.Context, input string) ([]float32, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	VRAMSize() uint64 // Total VRAM across all GPUs
	TotalSize() uint64
	VRAMByGPU(id ml.DeviceID) uint64
	Pid() int
	GetPort() int
	GetDeviceInfos(ctx context.Context) []ml.DeviceInfo
	HasExited() bool
}

// llmServer is an instance of a runner hosting a single model
type llmServer struct {
	port        int
	cmd         *exec.Cmd
	done        chan error // Channel to signal when the process exits
	status      *StatusWriter
	options     api.Options
	numParallel int
	modelPath   string

	loadRequest LoadRequest // Parameters used to initialize the runner

	// llamaModel is an instance of the cgo llama.cpp model definition
	// nil if this server is running the new engine
	llamaModel     *llama.Model
	llamaModelLock *sync.Mutex

	// textProcessor handles text encoding/decoding for the model in the Ollama engine
	// nil if this server is running the llama.cpp based engine
	textProcessor model.TextProcessor

	totalLayers  uint64
	loadStart    time.Time // Record how long it took the model to load
	loadProgress float32

	sem *semaphore.Weighted
}

type llamaServer struct {
	llmServer

	ggml     *ggml.GGML
	gpus     []ml.DeviceInfo // The set of GPUs covered by the memory estimate
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
func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	var llamaModel *llama.Model
	var textProcessor model.TextProcessor
	var err error
	if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
		if len(projectors) == 0 {
			textProcessor, err = model.NewTextProcessor(modelPath)
		} else {
			err = errors.New("split vision models aren't supported")
		}
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

	// Verify the requested context size is <= the model training size
	trainCtx := f.KV().ContextLength()
	if opts.NumCtx > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx)
	}

	opts.NumBatch = min(opts.NumBatch, opts.NumCtx)

	loadRequest := LoadRequest{LoraPath: adapters, KvSize: opts.NumCtx * numParallel, BatchSize: opts.NumBatch, Parallel: numParallel, MultiUserCache: envconfig.MultiUserCache()}

	defaultThreads := systemInfo.ThreadCount
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

	fa := envconfig.FlashAttention(f.FlashAttention())

	// This will disable flash attention unless all GPUs on the system support it, even if we end up selecting a subset
	// that can handle it.
	if fa && !ml.FlashAttentionSupported(gpus) {
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
		if f.SupportsKVCacheType(kvct) {
			loadRequest.KvCacheType = kvct
		} else {
			slog.Warn("kv cache type not supported by model", "type", kvct)
		}
	} else if kvct != "" && kvct != "f16" {
		slog.Warn("quantized kv cache requested but flash attention disabled", "type", kvct)
	}

	gpuLibs := ml.LibraryPaths(gpus)
	status := NewStatusWriter(os.Stderr)
	cmd, port, err := StartRunner(
		textProcessor != nil,
		modelPath,
		gpuLibs,
		status,
		ml.GetVisibleDevicesEnv(gpus),
	)

	s := llmServer{
		port:           port,
		cmd:            cmd,
		status:         status,
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

	if err != nil {
		var msg string
		if s.status != nil && s.status.LastErrMsg != "" {
			msg = s.status.LastErrMsg
		}
		err := fmt.Errorf("error starting runner: %v %s", err, msg)
		if llamaModel != nil {
			llama.FreeModel(llamaModel)
		}
		return nil, err
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

	if textProcessor != nil {
		return &ollamaServer{llmServer: s}, nil
	} else {
		return &llamaServer{llmServer: s, ggml: f}, nil
	}
}

func StartRunner(ollamaEngine bool, modelPath string, gpuLibs []string, out io.Writer, extraEnvs map[string]string) (cmd *exec.Cmd, port int, err error) {
	var exe string
	exe, err = os.Executable()
	if err != nil {
		return nil, 0, fmt.Errorf("unable to lookup executable path: %w", err)
	}

	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	port = 0
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
	if ollamaEngine {
		params = append(params, "--ollama-engine")
	}
	if modelPath != "" {
		params = append(params, "--model", modelPath)
	}
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
	libraryPaths := append([]string{}, gpuLibs...)
	if libraryPath, ok := os.LookupEnv(pathEnv); ok {
		libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
	}

	cmd = exec.Command(exe, params...)

	cmd.Env = os.Environ()
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.SysProcAttr = LlamaServerSysProcAttr

	// Always filter down the set of GPUs in case there are any unsupported devices that might crash
	pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

	// Update or add the path variable with our adjusted version
	pathNeeded := true
	ollamaPathNeeded := true
	extraEnvsDone := map[string]bool{}
	for k := range extraEnvs {
		extraEnvsDone[k] = false
	}
	for i := range cmd.Env {
		cmp := strings.SplitN(cmd.Env[i], "=", 2)
		if strings.EqualFold(cmp[0], pathEnv) {
			cmd.Env[i] = pathEnv + "=" + pathEnvVal
			pathNeeded = false
		} else if strings.EqualFold(cmp[0], "OLLAMA_LIBRARY_PATH") {
			cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + strings.Join(gpuLibs, string(filepath.ListSeparator))
			ollamaPathNeeded = false
		} else if len(extraEnvs) != 0 {
			for k, v := range extraEnvs {
				if strings.EqualFold(cmp[0], k) {
					cmd.Env[i] = k + "=" + v
					extraEnvsDone[k] = true
				}
			}
		}
	}
	if pathNeeded {
		cmd.Env = append(cmd.Env, pathEnv+"="+pathEnvVal)
	}
	if ollamaPathNeeded {
		cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(gpuLibs, string(filepath.ListSeparator)))
	}
	for k, done := range extraEnvsDone {
		if !done {
			cmd.Env = append(cmd.Env, k+"="+extraEnvs[k])
		}
	}

	slog.Info("starting runner", "cmd", cmd)
	slog.Debug("subprocess", "", filteredEnv(cmd.Env))

	if err = cmd.Start(); err != nil {
		return nil, 0, err
	}
	err = nil
	return
}

func (s *llmServer) ModelPath() string {
	return s.modelPath
}

type LoadOperation int

// The order of these constants are significant because we iterate over the operations. They
// should be in order of increasingly loading the model.
const (
	LoadOperationFit    LoadOperation = iota // Return memory requirements but do not allocate
	LoadOperationAlloc                       // Allocate memory but do not load the weights
	LoadOperationCommit                      // Load weights - further changes cannot be made after this
	LoadOperationClose                       // Close model and free memory
)

func (o LoadOperation) String() string {
	switch o {
	case LoadOperationFit:
		return "fit"
	case LoadOperationAlloc:
		return "alloc"
	case LoadOperationCommit:
		return "commit"
	case LoadOperationClose:
		return "close"
	default:
		return "unknown"
	}
}

type LoadRequest struct {
	Operation LoadOperation

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

var ErrLoadRequiredFull = errors.New("unable to load full model on GPU")

func (s *llamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	systemTotalMemory := systemInfo.TotalMemory
	systemFreeMemory := systemInfo.FreeMemory
	systemSwapFreeMemory := systemInfo.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	if len(gpus) == 0 || s.options.NumGPU == 0 {
		if !verifyCPUFit(s.ggml, s.modelPath, []string{s.loadRequest.ProjectorPath}, s.loadRequest.LoraPath, s.options, systemInfo, s.numParallel) {
			slog.Info("model requires more memory than is currently available, evicting a model to make space", "estimate", s.estimate)
			return nil, fmt.Errorf("model requires more system memory than is currently available %w", ErrLoadRequiredFull)
		}
	} else {
		g := pickBestFullFitByLibrary(s.ggml, s.modelPath, []string{s.loadRequest.ProjectorPath}, s.loadRequest.LoraPath, s.options, gpus, s.numParallel)
		if g == nil {
			if !requireFull {
				g = pickBestPartialFitByLibrary(s.ggml, []string{s.loadRequest.ProjectorPath}, s.loadRequest.LoraPath, s.options, gpus, s.numParallel)
			} else {
				slog.Info("model requires more memory than is currently available, evicting a model to make space", "estimate", s.estimate)
				return nil, ErrLoadRequiredFull
			}
		}
		gpus = g
	}

	s.estimate = estimateGPULayers(gpus, s.ggml, []string{s.loadRequest.ProjectorPath}, s.options, s.numParallel)

	if len(gpus) >= 1 {
		switch {
		case s.options.NumGPU == 0:
			gpus = []ml.DeviceInfo{}
		case gpus[0].Library == "Metal" && s.estimate.VRAMSize > systemInfo.TotalMemory:
			// disable partial offloading when model is greater than total system memory as this
			// can lead to locking up the system
			s.options.NumGPU = 0
			gpus = []ml.DeviceInfo{}
		case gpus[0].Library != "Metal" && s.estimate.Layers == 0:
			// Don't bother loading into the GPU if no layers can fit
			gpus = []ml.DeviceInfo{}
		case s.options.NumGPU < 0 && s.estimate.Layers > 0:
			s.options.NumGPU = s.estimate.Layers
		}
	} else {
		s.options.NumGPU = 0
	}

	// On linux and windows, over-allocating CPU memory will almost always result in an error
	// Darwin has fully dynamic swap so has no direct concept of free swap space
	if runtime.GOOS != "darwin" {
		systemMemoryRequired := s.estimate.TotalSize - s.estimate.VRAMSize
		available := systemInfo.FreeMemory + systemInfo.FreeSwap
		if systemMemoryRequired > available {
			slog.Warn("model request too large for system", "requested", format.HumanBytes2(systemMemoryRequired), "available", format.HumanBytes2(available), "total", format.HumanBytes2(systemInfo.TotalMemory), "free", format.HumanBytes2(systemInfo.FreeMemory), "swap", format.HumanBytes2(systemInfo.FreeSwap))
			return nil, fmt.Errorf("model requires more system memory (%s) than is available (%s)", format.HumanBytes2(systemMemoryRequired), format.HumanBytes2(available))
		}
	}

	slog.Info("offload", "", s.estimate)

	s.gpus = gpus
	s.loadRequest.GPULayers = createGPULayers(s.estimate, s.ggml, gpus, s.options.NumGPU)

	// Mmap is only supported on the llama engine
	if s.textProcessor == nil {
		s.loadRequest.UseMmap = true

		// mmap has issues with partial offloading on metal
		for _, g := range gpus {
			if g.Library == "Metal" &&
				uint64(s.options.NumGPU) > 0 &&
				uint64(s.options.NumGPU) < s.ggml.KV().BlockCount()+1 {
				s.options.UseMMap = new(bool)
				*s.options.UseMMap = false
			}
		}

		// Windows CUDA should not use mmap for best performance
		// Linux  with a model larger than free space, mmap leads to thrashing
		// For CPU loads we want the memory to be allocated, not FS cache
		if (runtime.GOOS == "windows" && len(gpus) > 0 && gpus[0].Library == "CUDA" && s.options.UseMMap == nil) ||
			(runtime.GOOS == "linux" && systemInfo.FreeMemory < s.estimate.TotalSize && s.options.UseMMap == nil) ||
			(len(gpus) == 0 && s.options.UseMMap == nil) ||
			(len(gpus) > 0 && gpus[0].Library == "Vulkan" && s.options.UseMMap == nil) ||
			(s.options.UseMMap != nil && !*s.options.UseMMap) {
			s.loadRequest.UseMmap = false
		}
	}

	if err := s.waitUntilRunnerLaunched(ctx); err != nil {
		return nil, err
	}

	resp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)
	if err != nil {
		return nil, err
	}

	// On the Ollama engine, we can print out a summary of the memory allocations.
	// We don't have this for the llama engine but it does something similar itself.
	if s.textProcessor != nil {
		resp.Memory.Log(slog.LevelInfo)
	}

	if !resp.Success {
		slog.Warn("failed to allocate memory for model", "memory", resp.Memory)
		return nil, errors.New("failed to allocate memory for model")
	}

	// The llama engine does its memory allocations together with model loading, so we
	// need to wait until it is done to ensure that we have accurate memory data before
	// loading the next model
	if s.textProcessor == nil {
		return uniqueDeviceIDs(s.loadRequest.GPULayers), s.WaitUntilRunning(ctx)
	} else {
		return uniqueDeviceIDs(s.loadRequest.GPULayers), nil
	}
}

// createGPULayers maps from the tensor splits assigned by the memory estimates to explicit assignment
// of particular layers onto GPUs
func createGPULayers(estimate MemoryEstimate, ggml *ggml.GGML, gpus []ml.DeviceInfo, numGPU int) ml.GPULayersList {
	if numGPU <= 0 || len(gpus) == 0 {
		return nil
	}

	gpuLayers := make(ml.GPULayersList, len(gpus))
	for i := range gpuLayers {
		gpuLayers[i].DeviceID = gpus[i].DeviceID
	}

	var sum float32
	splits := make([]float32, len(estimate.TensorSplit))
	// cumulative sum of all splits
	for i := range splits {
		sum += float32(estimate.TensorSplit[i])
		splits[i] = sum
	}

	if sum <= 0 {
		return nil
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

// Load finds the optimal layout of layers to offload on GPUs based on no initial information about the size of the model
// It does this by:
// 1. Assigning the full model to the GPU with the largest available free memory
// 2. Attempting to allocate the layout and receiving the memory requirements in response
// 3. Creating a new layout based on the updated memory information
// 4. Going back to step 2 and looping until we either stabilize on a particular layout or discover that we have entered a cycle
//
// This process is repeated for higher levels of loading the model (fit, allocate, commit). The earlier levels are quicker,
// allowing for faster iteration, but may return less information.
//
// Returns the list of GPU IDs that were used in the final allocation on success
func (s *ollamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	var success bool
	defer func() {
		if !success {
			s.initModel(ctx, LoadRequest{}, LoadOperationClose)
		}
		if s.mem != nil {
			s.mem.Log(slog.LevelInfo)
		}
	}()

	slog.Info("loading model", "model layers", s.totalLayers, "requested", s.options.NumGPU)

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

	pastAllocations := make(map[uint64]struct{})
	var backoff float32

	gpuLayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
	if err != nil {
		return nil, err
	}

	if err := s.waitUntilRunnerLaunched(ctx); err != nil {
		return nil, err
	}

nextOperation:
	for operation := LoadOperationFit; operation < LoadOperationCommit; operation++ {
	nextLoad:
		for {
			s.loadRequest.GPULayers = gpuLayers
			resp, err := s.initModel(ctx, s.loadRequest, operation)
			if err != nil {
				return nil, err
			}

			resp.Memory.Log(slog.LevelDebug)
			slog.Debug("memory", "success", resp.Success, "required", resp.Memory)

			pastAllocations[gpuLayers.Hash()] = struct{}{}
			s.mem = &resp.Memory

			for {
				newGPULayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
				if err != nil {
					return nil, err
				}

				slog.Debug("new layout created", "layers", newGPULayers)

				// We get additional memory information over time, which will reduce the number of
				// layers that can fit, so fewer layers is actually better. As long as we haven't seen
				// this layout before and it doesn't have more layers than the last one, we can keep
				// trying to see if we can do better.
				if _, ok := pastAllocations[newGPULayers.Hash()]; !ok && newGPULayers.Sum() <= gpuLayers.Sum() {
					gpuLayers = newGPULayers
					continue nextLoad
				}

				// If we are looping around a few different layouts due to graphs moving off and on
				// GPUs, make sure that we try out the intermediate states. For example, if we are
				// looping between offloading 39 and 41 layers, we should also check 40.
				//
				// This switches strategies to force an incremental number of layers to be offloaded
				// and checking the memory layout. If the allocation succeeds and creating a new layout
				// without forcing offload yields the same or greater number of layers offloaded, then
				// the trial is successful.
				//
				// This alternate strategy does not introduce the possibility of loops with the overall
				// state machine, as it exits this code block either with a successful result, moving
				// to the next operation or the original number of layers offloaded.
				if s.options.NumGPU < 0 && newGPULayers.Sum()-gpuLayers.Sum() > 1 {
					for i := newGPULayers.Sum() - 1; i >= gpuLayers.Sum(); i-- {
						slog.Debug("exploring intermediate layers", "layer", i)

						s.options.NumGPU = i
						newGPULayers, err = s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
						s.options.NumGPU = -1
						if err != nil {
							return nil, err
						}
						slog.Debug("new layout created", "layers", newGPULayers)

						s.loadRequest.GPULayers = newGPULayers
						resp, err = s.initModel(ctx, s.loadRequest, operation)
						if err != nil {
							return nil, err
						}

						resp.Memory.Log(slog.LevelDebug)
						slog.Debug("memory", "success", resp.Success, "required", resp.Memory)

						if resp.Success {
							verifyGPULayers, err := s.createLayout(systemInfo, gpus, &resp.Memory, requireFull, backoff)
							if err != nil {
								return nil, err
							}

							slog.Debug("verifying layout", "layers", verifyGPULayers)

							if newGPULayers.Sum() <= verifyGPULayers.Sum() {
								gpuLayers = newGPULayers

								// Since we are going backwards (increasing the number of layers), ensure that
								// we can come back down if needed
								clear(pastAllocations)

								continue nextOperation
							}
						}
					}
				}

				// If we generated a layout a second time or go backwards, then we've converged. Use the last
				// layout before the repeat, which is already allocated.
				if resp.Success {
					continue nextOperation
				}

				if s.options.NumGPU >= 0 {
					return nil, fmt.Errorf("memory layout cannot be allocated with num_gpu = %v", s.options.NumGPU)
				}

				// Memory allocation failed even though we created a layout that we thought should
				// fit in available memory. This could happen if either our free memory reports
				// are incorrect or if available memory is changing between layout and allocation
				// time. Apply a backoff to try to find the real amount of available space.
				if backoff > 1 {
					slog.Warn("memory layout cannot be allocated", "memory", resp.Memory)
					return nil, errors.New("memory layout cannot be allocated")
				} else {
					backoff += 0.1
				}

				slog.Info("model layout did not fit, applying backoff", "backoff", fmt.Sprintf("%.2f", backoff))
			}
		}
	}

	s.loadRequest.GPULayers = gpuLayers
	resp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)
	if err != nil {
		return nil, err
	}

	success = resp.Success
	s.mem = &resp.Memory

	if !success {
		slog.Warn("failed to commit memory for model", "memory", resp.Memory)
		return nil, errors.New("failed to commit memory for model")
	}

	return uniqueDeviceIDs(gpuLayers), nil
}

func uniqueDeviceIDs(gpuLayers ml.GPULayersList) []ml.DeviceID {
	devices := []ml.DeviceID{}
	for _, layer := range gpuLayers {
		new := true
		for _, ID := range devices {
			if layer.DeviceID == ID {
				new = false
				break
			}
		}
		if new {
			devices = append(devices, layer.DeviceID)
		}
	}
	return devices
}

// createLayout uses the current best view of memory requirements and creates a layout of model layers on GPUs.
// It does this by:
// - Calculating how much space each layer requires
// - Calculating how much space each GPU has available for layers, based on free memory and space occupied by the graph
// - Assigning layers
// - Ensuring that we don't exceed limits, such as requirements about partial offloading or system memory
func (s *ollamaServer) createLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, error) {
	if memory == nil {
		memory = &ml.BackendMemory{CPU: ml.DeviceMemory{
			Weights: make([]uint64, s.totalLayers),
			Cache:   make([]uint64, s.totalLayers),
		}}
	}
	gpuLayers, layers, err := s.buildLayout(systemGPUs, memory, requireFull, backoff)
	if err != nil {
		return nil, err
	}
	err = s.verifyLayout(systemInfo, memory, requireFull, gpuLayers, layers)
	if err != nil {
		return nil, err
	}
	return gpuLayers, nil
}

func (s *ollamaServer) buildLayout(systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, []uint64, error) {
	gpus := append(make([]ml.DeviceInfo, 0, len(systemGPUs)), systemGPUs...)
	sort.Sort(sort.Reverse(ml.ByFreeMemory(gpus)))

	layers := make([]uint64, len(memory.CPU.Weights))
	for i := range layers {
		for j := range memory.GPUs {
			layers[i] += memory.GPUs[j].Weights[i]
			layers[i] += memory.GPUs[j].Cache[i]
		}
		layers[i] += memory.CPU.Weights[i]
		layers[i] += memory.CPU.Cache[i]
		logutil.Trace("layer to assign", "layer", i, "size", format.HumanBytes2(layers[i]))
	}

	gpuLayers := ml.GPULayersList{}
	for _, gl := range ml.ByLibrary(gpus) {
		// If a GPU already has a graph allocated on it, then we should continue to use it.
		// Otherwise, we lose information that we got from previous allocations, which can
		// cause cycling. Plus, we get more information about required allocation from each
		// iteration, so it doesn't make sense that a later iteration would use fewer GPUs.
		lastUsedGPU := 0
		for i := range gl {
			found := false
			for j := range memory.GPUs {
				if gl[i].DeviceID == memory.GPUs[j].DeviceID {
					if memory.GPUs[j].Graph != 0 {
						lastUsedGPU = i
					}

					reserved := uint64(float32(gl[i].FreeMemory)*backoff) + gl[i].MinimumMemory() + envconfig.GpuOverhead() + memory.GPUs[j].Graph
					if gl[i].FreeMemory > reserved {
						gl[i].FreeMemory -= reserved
					} else {
						gl[i].FreeMemory = 0
					}

					slog.Debug("available gpu", "id", gl[i].ID, "library", gl[i].Library,
						"available layer vram", format.HumanBytes2(gl[i].FreeMemory),
						"backoff", fmt.Sprintf("%.2f", backoff), "minimum", format.HumanBytes2(gl[i].MinimumMemory()),
						"overhead", format.HumanBytes2(envconfig.GpuOverhead()),
						"graph", format.HumanBytes2(memory.GPUs[j].Graph))

					found = true
					break
				}
			}
			if !found {
				// The runner doesn't report seeing this GPU
				gl[i].FreeMemory = 0
			}
		}

		libraryGpuLayers := assignLayers(layers, gl, requireFull, s.options.NumGPU, lastUsedGPU)
		if libraryGpuLayers.Sum() > gpuLayers.Sum() {
			gpuLayers = libraryGpuLayers
		}
	}
	return gpuLayers, layers, nil
}

// verifyLayout ensures that we don't exceed limits, such as requirements about partial offloading or system memory
func (s *ollamaServer) verifyLayout(systemInfo ml.SystemInfo, memory *ml.BackendMemory, requireFull bool, gpuLayers ml.GPULayersList, layers []uint64) error {
	// These sizes will only increase as we go through additional iterations and get additional information.
	cpuSize := memory.InputWeights + memory.CPU.Graph
	var vramSize uint64
	for _, gl := range gpuLayers {
		for _, gpu := range memory.GPUs {
			if gl.DeviceID == gpu.DeviceID {
				vramSize += gpu.Graph
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
			return ErrLoadRequiredFull
		}

		if cpuSize > systemInfo.FreeMemory {
			return ErrLoadRequiredFull
		}
	}

	// On linux and windows, over-allocating CPU memory will almost always result in an error
	// Darwin has fully dynamic swap so has no direct concept of free swap space
	if runtime.GOOS != "darwin" {
		available := systemInfo.FreeMemory + systemInfo.FreeSwap
		if cpuSize > available {
			slog.Warn("model request too large for system", "requested", format.HumanBytes2(cpuSize), "available", format.HumanBytes2(available), "total", format.HumanBytes2(systemInfo.TotalMemory), "free", format.HumanBytes2(systemInfo.FreeMemory), "swap", format.HumanBytes2(systemInfo.FreeSwap))
			return fmt.Errorf("model requires more system memory (%s) than is available (%s)", format.HumanBytes2(cpuSize), format.HumanBytes2(available))
		}
	} else {
		if vramSize > systemInfo.TotalMemory {
			// disable partial offloading when model is greater than total system memory as this
			// can lead to locking up the system
			s.options.NumGPU = 0
			gpuLayers = ml.GPULayersList{}
		}
	}

	if gpuLayers.Sum() == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
	}

	return nil
}

// assignLayers packs the maximum number of layers onto the smallest set of GPUs and comes up with a layer assignment
func assignLayers(layers []uint64, gpus []ml.DeviceInfo, requireFull bool, requestedLayers int, lastUsedGPU int) (gpuLayers ml.GPULayersList) {
	// If we can't fit everything then prefer offloading layers other than the output layer
	for range 2 {
		// requestedLayers may be -1 if nothing was requested
		requestedLayers = min(len(layers), requestedLayers)

		if !envconfig.SchedSpread() {
			for i := lastUsedGPU; i < len(gpus); i++ {
				// Try to pack things into as few GPUs as possible
				forceRequest := i == len(gpus)-1 && !requireFull
				gpuLayers = findBestFit(layers, gpus[:i+1], requestedLayers, forceRequest)
				if gpuLayers.Sum() == len(layers) || gpuLayers.Sum() == requestedLayers {
					break
				}
			}
		} else {
			gpuLayers = findBestFit(layers, gpus, requestedLayers, !requireFull)
		}

		// We only stop if we've gotten all of the layers - even if we got requestedLayers, we still
		// might want to try dropping the output layer.
		if gpuLayers.Sum() == len(layers) {
			return gpuLayers
		}

		layers = layers[:len(layers)-1]
	}

	return gpuLayers
}

// findBestFit binary searches to find the smallest capacity factor that can fit
// the max number of layers. The capacity factor is multiplied by the free space on
// each GPU and a small one will force even balancing.
func findBestFit(layers []uint64, gpus []ml.DeviceInfo, requestedLayers int, forceRequest bool) (gpuLayers ml.GPULayersList) {
	var high float32 = 1
	var low float32 = 0

	// If we need to fulfill the requested number of layers, pretend we have almost infinite VRAM
	if requestedLayers >= 0 && forceRequest {
		high = 1000
	}

	bestAssignments := greedyFit(layers, gpus, high, requestedLayers)
	maxNumGPU := bestAssignments.Sum()
	if maxNumGPU == 0 {
		return bestAssignments
	}

	for high-low > 1e-6 {
		mid := (low + high) / 2
		assignments := greedyFit(layers, gpus, mid, requestedLayers)
		if assignments.Sum() == maxNumGPU {
			high = mid
			bestAssignments = assignments
		} else {
			low = mid
		}
	}
	return bestAssignments
}

// greedyFit assigns layers incrementally to GPUs, spilling over as each runs out of free space
func greedyFit(layers []uint64, gpus []ml.DeviceInfo, capacity float32, requestedLayers int) (gpuLayers ml.GPULayersList) {
	device := len(gpus) - 1
	gpuLayers = ml.GPULayersList{{DeviceID: gpus[device].DeviceID}}
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
			gpuLayers = append(ml.GPULayersList{{DeviceID: gpus[device].DeviceID}}, gpuLayers...)
			freeSpace = uint64(float32(gpus[device].FreeMemory) * capacity)
		}
	}
	return gpuLayers
}

// waitUntilRunnerLaunched sleeps until the runner subprocess is alive enough
// to respond to status requests
func (s *llmServer) waitUntilRunnerLaunched(ctx context.Context) error {
	for {
		_, err := s.getServerStatus(ctx)
		if err == nil {
			break
		}

		t := time.NewTimer(10 * time.Millisecond)
		select {
		case <-t.C:
			continue
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

// initModel sends a load request to the runner based on the request operation (fit, alloc, commit)
// and parameters
func (s *llmServer) initModel(ctx context.Context, req LoadRequest, operation LoadOperation) (*LoadResponse, error) {
	req.Operation = operation

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
		return nil, fmt.Errorf("load unmarshal encode response: %w", err)
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

func (s *llmServer) GetPort() int {
	return s.port
}

func (s *llmServer) HasExited() bool {
	if s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0 {
		return true
	}
	return false
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

	Grammar  string // set before sending the request to the subprocess
	Shift    bool
	Truncate bool
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
	logutil.Trace("completion request", "prompt", req.Prompt)

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
	if err != nil && errors.Is(err, context.Canceled) {
		// client closed connection
		return err
	} else if err != nil {
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
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(bodyBytes))}
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
	logutil.Trace("embedding request", "input", input)

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

func (s *llamaServer) VRAMSize() uint64 {
	return s.estimate.VRAMSize
}

func (s *llamaServer) TotalSize() uint64 {
	return s.estimate.TotalSize
}

func (s *llamaServer) VRAMByGPU(id ml.DeviceID) uint64 {
	for i, gpu := range s.gpus {
		if gpu.DeviceID == id {
			if i < len(s.estimate.GPUSizes) {
				return s.estimate.GPUSizes[i]
			}
		}
	}
	return 0
}

func (s *llamaServer) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	slog.Debug("llamarunner free vram reporting not supported")
	return nil
}

func (s *ollamaServer) VRAMSize() uint64 {
	if s.mem == nil {
		return 0
	}

	var mem uint64

	for _, g := range s.mem.GPUs {
		mem += g.Size()
	}

	// Some elements are always on CPU. However, if we have allocated all layers
	// on the GPU then include the CPU components as well, to represent complete offloading.
	noCPULayers := true
	for i := range s.mem.CPU.Weights {
		if s.mem.CPU.Weights[i] != 0 || s.mem.CPU.Cache[i] != 0 {
			noCPULayers = false
			break
		}
	}
	if noCPULayers {
		mem += s.mem.InputWeights
		mem += s.mem.CPU.Graph
	}

	return mem
}

func (s *ollamaServer) TotalSize() uint64 {
	if s.mem == nil {
		return 0
	}

	mem := s.mem.InputWeights
	mem += s.mem.CPU.Size()
	for _, g := range s.mem.GPUs {
		mem += g.Size()
	}

	return mem
}

func (s *ollamaServer) VRAMByGPU(id ml.DeviceID) uint64 {
	if s.mem == nil {
		return 0
	}

	for _, g := range s.mem.GPUs {
		if g.DeviceID == id {
			return g.Size()
		}
	}

	return 0
}

func (s *ollamaServer) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	devices, err := ml.GetDevicesFromRunner(ctx, s)
	if err != nil {
		if s.cmd != nil && s.cmd.ProcessState == nil {
			// Still running but hit an error, log
			slog.Debug("failure refreshing GPU information", "error", err)
		}
		// else no longer running so suppress logging as a failure is expected
	}
	return devices
}
