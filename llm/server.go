package llm

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
<<<<<<< Updated upstream
=======
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
>>>>>>> Stashed changes
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/ml"
)

var ErrLoadRequiredFull = errors.New("unable to load full model on GPU")

type filteredEnv []string

func (e filteredEnv) LogValue() slog.Value {
	var attrs []slog.Attr
	for _, env := range e {
		if key, value, ok := strings.Cut(env, "="); ok {
			if filteredEnvLogKey(key) {
				attrs = append(attrs, slog.String(key, filteredEnvLogValue(key, value)))
			}
		}
	}
	return slog.GroupValue(attrs...)
}

func filteredEnvLogKey(key string) bool {
	return strings.HasPrefix(key, "CUDA_") ||
		strings.HasPrefix(key, "ROCR_") ||
		strings.HasPrefix(key, "ROCM_") ||
		strings.HasPrefix(key, "HIP_") ||
		strings.HasPrefix(key, "HSA_") ||
		strings.HasPrefix(key, "GGML_") ||
		slices.Contains([]string{
			"PATH",
			"LD_LIBRARY_PATH",
			"DYLD_LIBRARY_PATH",
		}, key)
}

func filteredEnvLogValue(key, value string) string {
	for _, token := range []string{"API", "KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "CREDENTIAL", "AUTH"} {
		if strings.Contains(strings.ToUpper(key), token) {
			return "[redacted]"
		}
	}
	return value
}

type LlamaServer interface {
	ModelPath() string
	Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse)) error
	ApplyChatTemplate(ctx context.Context, req ChatRequest) (string, error)
	Embedding(ctx context.Context, input string) ([]float32, int, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	MemorySize() (total, vram uint64)
	VRAMByGPU(id ml.DeviceID) uint64
	Pid() int
	GetPort() int
	GetDeviceInfos(ctx context.Context) []ml.DeviceInfo
	HasExited() bool
	ContextLength() int
}

type LlamaServerConfig struct {
	DisableJinja         bool
	ContextShift         bool
	EnableMTP            bool
	ManifestDigest       string
	DraftModelPath       string
	DraftModelShardPaths []string
}

// LoadModel loads GGUF model metadata from disk.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func LoadModel(model string, maxArraySize int, shards ...string) (*gguf.Model, error) {
	return gguf.ReadModel(model, maxArraySize, shards...)
}

// NewLlamaServer creates a new llama-server runner for the given model.
// All GGUF models are served via the upstream llama-server subprocess.
func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *gguf.Model, adapters, projectors []string, opts api.Options, numParallel int, config LlamaServerConfig) (LlamaServer, error) {
	slog.Info("using llama-server for model", "model", modelPath)

	// Verify the requested context size is <= the model training size
	trainCtx := f.KV().ContextLength()
	if opts.NumCtx > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx)
	}

<<<<<<< Updated upstream
	kvct := strings.ToLower(envconfig.KvCacheType())
	return NewLlamaServerRunner(gpus, modelPath, f, adapters, projectors, opts, numParallel, kvct, config)
}

// Server status types
=======
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
	// Determine if the user has forced FA on or off
	faUserSet := false
	if envconfig.FlashAttention(true) == envconfig.FlashAttention(false) {
		faUserSet = true
	}

	fa := envconfig.FlashAttention(f.SupportsFlashAttention())

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

	// Gemma 4's 512-dim attention heads require MMA FA kernels (Turing+, compute >= 7.5).
	// Older CUDA GPUs only have tile/vec FA kernels which abort on dk512 non-GQA attention.
	if fa && f.KV().Architecture() == "gemma4" {
		for _, gpu := range gpus {
			if gpu.Library == "CUDA" && (gpu.ComputeMajor < 7 || (gpu.ComputeMajor == 7 && gpu.ComputeMinor < 5)) {
				slog.Debug("disabling flash attention for gemma4 on pre-Turing GPU", "compute", fmt.Sprintf("%d.%d", gpu.ComputeMajor, gpu.ComputeMinor))
				fa = false
				break
			}
		}
	}

	kvct := strings.ToLower(envconfig.KvCacheType())

	if tok == nil {
		flashAttention := ml.FlashAttentionAuto
		if faUserSet {
			if fa {
				flashAttention = ml.FlashAttentionEnabled
			} else {
				flashAttention = ml.FlashAttentionDisabled
			}
		}

		if kvct != "" {
			if f.KVCacheTypeIsQuantized(kvct) {
				if flashAttention != ml.FlashAttentionEnabled {
					slog.Warn("OLLAMA_FLASH_ATTENTION must be enabled to use a quantized OLLAMA_KV_CACHE_TYPE", "type", kvct)
					loadRequest.KvCacheType = ""
				} else if f.SupportsKVCacheType(kvct) {
					loadRequest.KvCacheType = kvct
				} else {
					slog.Warn("unsupported OLLAMA_KV_CACHE_TYPE", "type", kvct)
				}
			} else {
				if f.SupportsKVCacheType(kvct) {
					loadRequest.KvCacheType = kvct
				} else {
					slog.Warn("unsupported OLLAMA_KV_CACHE_TYPE", "type", kvct)
				}
			}
		}
		loadRequest.FlashAttention = flashAttention
	} else {
		// For Ollama engine, use our SupportsFlashAttention logic
		if fa {
			slog.Info("enabling flash attention")
			loadRequest.FlashAttention = ml.FlashAttentionEnabled

			// Flash Attention also supports kv cache quantization
			// Enable if the requested and kv cache type is supported by the model
			if f.SupportsKVCacheType(kvct) {
				loadRequest.KvCacheType = kvct
			} else {
				slog.Warn("kv cache type not supported by model", "type", kvct)
			}
		} else {
			loadRequest.FlashAttention = ml.FlashAttentionDisabled
			if kvct != "" && kvct != "f16" {
				slog.Warn("quantized kv cache requested but flash attention disabled", "type", kvct)
			}
		}
	}

	gpuLibs := ml.LibraryPaths(gpus)
	status := NewStatusWriter(os.Stderr)
	cmd, port, err := StartRunner(
		tok != nil,
		modelPath,
		gpuLibs,
		status,
		ml.GetDevicesEnv(gpus, false),
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
		sem:            semaphore.NewWeighted(int64(numParallel)),
		totalLayers:    f.KV().BlockCount() + 1,
		loadStart:      time.Now(),
		done:           make(chan struct{}),
	}

	if err != nil {
		var msg string
		if s.status != nil && s.status.LastError() != "" {
			msg = s.status.LastError()
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
		if err != nil && s.status != nil && s.status.LastError() != "" {
			slog.Error("llama runner terminated", "error", err)
			if strings.Contains(s.status.LastError(), "unknown model") {
				s.status.SetLastError("this model is not supported by your version of Ollama. You may need to upgrade")
			}
			s.doneErr = errors.New(s.status.LastError())
		} else {
			s.doneErr = err
		}
		close(s.done)
	}()

	if tok != nil {
		return &ollamaServer{llmServer: s, tokenizer: tok}, nil
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

	if out != nil {
		// os/exec serializes Write calls when shared
		cmd.Stdout = out
		cmd.Stderr = out
	}
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

// Workaround possible runtime crash where the probe incorrectly
// enables metal tensor, but fails at runtime
func ShouldRetryWithMetalTensorDisabled(err error, status *StatusWriter) bool {
	if runtime.GOOS != "darwin" {
		return false
	}

	var msg strings.Builder
	msg.WriteString(strings.ToLower(err.Error()))
	if status != nil && status.LastError() != "" {
		msg.WriteByte(' ')
		msg.WriteString(strings.ToLower(status.LastError()))
	}
	text := msg.String()

	for _, needle := range []string{
		"failed to initialize ggml backend device: metal",
		"failed to initialize metal backend",
		"failed to initialize the metal library",
		"failed to allocate context",
		"unable to create llama context",
		"signal arrived during cgo execution",
		"input types must match cooperative tensor types",
	} {
		if strings.Contains(text, needle) {
			return true
		}
	}

	return false
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
	FlashAttention ml.FlashAttentionType
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

func (s *llamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	slog.Info("loading model", "model layers", s.totalLayers, "requested", s.options.NumGPU)

	gpus := append(make([]ml.DeviceInfo, 0, len(systemGPUs)), systemGPUs...)

	// Synthesize memory allocation information based on our estimates
	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Name:    "CPU",
		Weights: make([]uint64, s.totalLayers),
		Cache:   make([]uint64, s.totalLayers),
	}, GPUs: make([]ml.DeviceMemory, len(gpus))}

	for i := range s.mem.GPUs {
		s.mem.GPUs[i].Name = gpus[i].Name
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, s.totalLayers)
		s.mem.GPUs[i].Cache = make([]uint64, s.totalLayers)
	}

	// Check if embedding model and adjust batch size accordingly
	_, isEmbedding := s.ggml.KV()[fmt.Sprintf("%s.pooling_type", s.ggml.KV().Architecture())]
	if isEmbedding && s.loadRequest.BatchSize < s.options.NumCtx {
		s.loadRequest.BatchSize = s.options.NumCtx
		slog.Info("embedding model detected, setting batch size to context length", "batch_size", s.loadRequest.BatchSize)
	}

	kv, graphPartialOffload, graphFullOffload := s.ggml.GraphSize(uint64(s.options.NumCtx), uint64(s.loadRequest.BatchSize),
		s.loadRequest.Parallel, s.loadRequest.KvCacheType, s.loadRequest.FlashAttention)

	// Use the size of one layer as a buffer
	layers := s.ggml.Tensors().GroupLayers()
	if blk0, ok := layers["blk.0"]; ok {
		buffer := blk0.Size() + kv[0]
		for i := range gpus {
			if gpus[i].FreeMemory > buffer {
				gpus[i].FreeMemory -= buffer
			} else {
				gpus[i].FreeMemory = 0
			}
		}
	} else {
		slog.Warn("model missing blk.0 layer size")
	}

	// Assign all the layers to the CPU for now, they will get reassigned later
	for i := range s.ggml.KV().BlockCount() {
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			s.mem.CPU.Weights[i] = blk.Size()
			s.mem.CPU.Cache[i] += kv[i]
		}
	}

	// We historically haven't included InputWeights in the model size
	var outputWeights uint64
	if layer, ok := layers["output_norm"]; ok {
		outputWeights += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		outputWeights += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		outputWeights += layer.Size()
	}
	s.mem.CPU.Weights[s.totalLayers-1] = outputWeights

	// The vision projector is always loaded on the first GPU if available.
	// This can't be assigned by us, so just subtract it from free space
	projectorGPU := -1
	var projectorWeights uint64
	if len(gpus) > 0 {
		for _, projector := range s.loadRequest.LoraPath {
			projectorWeights += projectorMemoryRequirements(projector)
		}

		// llama.cpp uses the first discrete GPU if available, otherwise the first iGPU
		firstIntegrated := -1
		for i := range gpus {
			if !gpus[i].Integrated {
				projectorGPU = i
				break
			}
			if firstIntegrated == -1 {
				firstIntegrated = i
			}
		}
		if projectorGPU == -1 {
			projectorGPU = firstIntegrated
		}

		if gpus[projectorGPU].FreeMemory > projectorWeights {
			gpus[projectorGPU].FreeMemory -= projectorWeights
		} else {
			gpus[projectorGPU].FreeMemory = 0
		}
	}

	var kvTotal uint64
	for _, kvLayer := range kv {
		kvTotal += kvLayer
	}

	if graphPartialOffload == 0 {
		headsKV := s.ggml.KV().HeadCountKVMin()
		if headsKV == 0 {
			headsKV = 1
		}
		gqa := s.ggml.KV().HeadCountMax() / headsKV
		graphPartialOffload = gqa * kvTotal / 6
	}
	if graphFullOffload == 0 {
		graphFullOffload = graphPartialOffload
	}

	// On Metal there's no partial offload overhead
	if len(gpus) > 0 && gpus[0].Library == "Metal" {
		graphPartialOffload = graphFullOffload
	}

	// Create a layout based on the memory data that we've built. The compute graph
	// for GPUs is iteratively assigned based on the number of GPUs that are required.
	var gpuLayers ml.GPULayersList
	for {
		prevGPULayers := gpuLayers

		var err error
		gpuLayers, err = s.createLayout(systemInfo, gpus, s.mem, requireFull, 0)
		if err != nil {
			return nil, err
		}

		if len(gpuLayers) > len(prevGPULayers) {
			for _, gl := range gpuLayers {
				for i := range s.mem.GPUs {
					if gl.DeviceID == s.mem.GPUs[i].DeviceID {
						s.mem.GPUs[i].Graph = max(graphPartialOffload, graphFullOffload)
						break
					}
				}
			}
		} else {
			break
		}
	}

	// This maintains the historical assignment of graph sizes, though it isn't fully accurate
	graphSize := graphFullOffload
	if gpuLayers.Sum() < int(s.totalLayers) {
		graphSize = graphPartialOffload
	}

	// For all layers that we have assigned to GPUs, move them in the memory data so
	// that it is reported accurately
	for _, gl := range gpuLayers {
		for i := range s.mem.GPUs {
			if gl.DeviceID == s.mem.GPUs[i].DeviceID {
				for _, l := range gl.Layers {
					s.mem.GPUs[i].Weights[l] = s.mem.CPU.Weights[l]
					s.mem.GPUs[i].Cache[l] = s.mem.CPU.Cache[l]

					s.mem.CPU.Weights[l] = 0
					s.mem.CPU.Cache[l] = 0
				}

				s.mem.GPUs[i].Graph = graphSize
				break
			}
		}
	}

	if projectorGPU > 0 && len(s.mem.GPUs[projectorGPU].Weights) > 0 {
		s.mem.GPUs[projectorGPU].Weights[s.totalLayers-1] += projectorWeights
	}

	slog.Debug("memory", "estimate", s.mem)
	s.mem.Log(slog.LevelInfo)

	// The llama engine uses mmap by default
	s.loadRequest.UseMmap = true

	// mmap has issues with partial offloading on metal
	for _, g := range gpus {
		if g.Library == "Metal" &&
			uint64(s.options.NumGPU) > 0 &&
			uint64(s.options.NumGPU) < s.totalLayers {
			s.options.UseMMap = new(bool)
			*s.options.UseMMap = false
		}
	}

	// Windows CUDA should not use mmap for best performance
	// Linux  with a model larger than free space, mmap leads to thrashing
	// For CPU loads we want the memory to be allocated, not FS cache
	totalSize, _ := s.MemorySize()
	if (runtime.GOOS == "windows" && len(gpus) > 0 && gpus[0].Library == "CUDA" && s.options.UseMMap == nil) ||
		(runtime.GOOS == "linux" && systemInfo.FreeMemory < totalSize && s.options.UseMMap == nil) ||
		(len(gpus) == 0 && s.options.UseMMap == nil) ||
		(len(gpus) > 0 && gpus[0].Library == "Vulkan" && s.options.UseMMap == nil) ||
		(s.options.UseMMap != nil && !*s.options.UseMMap) {
		s.loadRequest.UseMmap = false
	}

	if err := s.waitUntilRunnerLaunched(ctx); err != nil {
		return nil, err
	}

	s.loadRequest.GPULayers = gpuLayers
	resp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)
	if err != nil {
		return nil, err
	}

	if !resp.Success {
		return nil, errors.New("failed to allocate memory for model")
	}

	// The llama engine does its memory allocations together with model loading, so we
	// need to wait until it is done to ensure that we have accurate memory data before
	// loading the next model.
	return uniqueDeviceIDs(s.loadRequest.GPULayers), s.WaitUntilRunning(ctx)
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
func (s *llmServer) createLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, error) {
	if memory == nil {
		memory = &ml.BackendMemory{CPU: ml.DeviceMemory{
			Weights: make([]uint64, s.totalLayers),
			Cache:   make([]uint64, s.totalLayers),
		}}
	}
	gpuLayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)

	// Protect the output layer: force it onto the strongest-compute GPU
	// so compute-bound inference isn't bottlenecked by PCIe traffic.
	gpuLayers = protectOutputLayer(gpuLayers, systemGPUs, layers)

	// Redistribute FFN-heavy layers from weaker to stronger GPUs for better compute balance
	if envComputeBoost := envconfig.SchedComputeBoost(); envComputeBoost > 1.0 {
		gpuLayers = redistributeHeavyLayers(gpuLayers, systemGPUs, memory, layers, envComputeBoost)
	}

	err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, layers)
	if err != nil {
		return nil, err
	}
	return gpuLayers, nil
}

func (s *llmServer) buildLayout(systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, []uint64) {
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
	return gpuLayers, layers
}

// verifyLayout ensures that we don't exceed limits, such as requirements about partial offloading or system memory
func (s *llmServer) verifyLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, gpuLayers ml.GPULayersList, layers []uint64) error {
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
		if len(systemGPUs) > 0 && gpuLayers.Sum() < len(layers) && (s.options.NumGPU < 0 || gpuLayers.Sum() < s.options.NumGPU) {
			slog.Info("model requires more gpu memory than is currently available, evicting a model to make space", "loaded layers", gpuLayers.Sum())
			return ErrLoadRequiredFull
		}

		if cpuSize > systemInfo.FreeMemory {
			slog.Info("model requires more system memory than is currently available, evicting a model to make space", "required", cpuSize, "free", systemInfo.FreeMemory)
			return fmt.Errorf("model requires more system memory than is currently available %w", ErrLoadRequiredFull)
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

	if len(systemGPUs) > 0 && gpuLayers.Sum() == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
	}

	return nil
}

// assignLayers packs the maximum number of layers onto the smallest set of GPUs and comes up with a layer assignment
func assignLayers(layers []uint64, gpus []ml.DeviceInfo, requireFull bool, requestedLayers int, lastUsedGPU int) (gpuLayers ml.GPULayersList) {
	// If the user is manually overriding parameters, treat all GPUs equally so they split according to VRAM
	if requestedLayers >= 0 || envconfig.SchedSpread() {
		for i := range gpus {
			gpus[i].Integrated = false
		}
	}

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
// the max number of layers. Distributes capacity proportional to compute power
// across GPUs (capped by actual VRAM), so both cards finish their work at the same time.
// This prevents the stronger GPU from idling while the weaker GPU is still processing.
func findBestFit(layers []uint64, gpus []ml.DeviceInfo, requestedLayers int, forceRequest bool) (gpuLayers ml.GPULayersList) {
	// Compute total compute power across all GPUs.
	// SMCount*ClockMHz is the ideal metric, but the discovery runner subprocess may
	// report identical values for all GPUs (or zeros). Detect this by checking if all
	// GPUs report the same power — real heterogeneous GPUs always differ. Fall back to
	// ComputeMajor which is always correctly populated and encodes generation performance.
	totalRawPower := 0
	allSame := len(gpus) > 1
	for i, g := range gpus {
		totalRawPower += g.SMCount * g.ClockMHz
		if i > 0 && (g.SMCount*g.ClockMHz != gpus[0].SMCount*gpus[0].ClockMHz) {
			allSame = false
		}
	}
	if totalRawPower == 0 || (allSame && len(gpus) > 1) {
		for i := range gpus {
			gpus[i].SMCount = gpus[i].ComputeMajor*100 + gpus[i].ComputeMinor
			gpus[i].ClockMHz = 1
		}
		totalRawPower = 0
		for _, g := range gpus {
			totalRawPower += g.SMCount * g.ClockMHz
		}
		slog.Info("SMCount/ClockMHz not usable, falling back to compute capability",
			"totalRawPower", totalRawPower)
	}

	envComputeBoost := envconfig.SchedComputeBoost()
	if envComputeBoost < 1.0 {
		envComputeBoost = 1.0
	} else if envComputeBoost > 2.0 {
		envComputeBoost = 2.0
	}

	// Compute per-GPU capacity multiplier based on compute power share.
	// At boost=1.0: each GPU gets capacity proportional to its raw compute share.
	// At boost>1.0: stronger GPUs get progressively more capacity, with diminishing
	// returns on weaker GPUs, to accelerate heterogeneous splits.
	// powerShare * boost + (1 - powerShare) = boost * powerShare + 1 - powerShare
	// At extremes: powerShare~1 => ~boost, powerShare~0 => ~1
	computeCapacity := make([]float64, len(gpus))
	adjustedMaxPower := 0.0
	for i, g := range gpus {
		rawPower := float64(g.SMCount * g.ClockMHz)
		powerShare := 0.0
		if totalRawPower > 0 {
			powerShare = rawPower / float64(totalRawPower)
		}
		computeCapacity[i] = powerShare*envComputeBoost + (1 - powerShare)
		if math.IsNaN(computeCapacity[i]) || math.IsInf(computeCapacity[i], 0) {
			computeCapacity[i] = 1.0
		}
		adjustedMaxPower += rawPower * computeCapacity[i]
	}

	slog.Info("scheduling layers across GPUs", "num_gpus", len(gpus), "boost", envComputeBoost,
		"capacity_multiplier", fmt.Sprintf("%v", computeCapacity))

	for _, gl := range ml.ByPerformance(gpus) {
		for i := range gl {
			// Scale FreeMemory by compute capacity multiplier so stronger GPUs can absorb more layers
			gl[i].FreeMemory = uint64(float64(gpus[i].FreeMemory) * computeCapacity[i])
			adjustedPower := float64(gpus[i].SMCount*gpus[i].ClockMHz) * computeCapacity[i]
			capGB := float64(gl[i].FreeMemory) / float64(format.GibiByte)
			powerGB := float64(totalRawPower) * adjustedPower / adjustedMaxPower / format.GibiByte
			slog.Debug(
				"layer allocation", "gpu", gpus[i].Name, "id", gpus[i].ID,
				"raw_power_ratio", fmt.Sprintf("%.3f", float64(gpus[i].SMCount*gpus[i].ClockMHz)/float64(totalRawPower)),
				"compute_cap", fmt.Sprintf("%.3f", computeCapacity[i]),
				"adjusted_gb", fmt.Sprintf("%.1f", capGB),
				"ideal_gb_power", fmt.Sprintf("%.1f", powerGB),
			)
		}

		var high float32 = 1
		var low float32 = 0

		// If we need to fulfill the requested number of layers, pretend we have almost infinite VRAM
		if requestedLayers >= 0 && forceRequest {
			high = 1000
		}

		bestAssignments := greedyFit(layers, gl, high, requestedLayers)
		maxNumGPU := bestAssignments.Sum()

		for high-low > 1e-6 {
			mid := (low + high) / 2
			assignments := greedyFit(layers, gl, mid, requestedLayers)
			if assignments.Sum() == maxNumGPU {
				high = mid
				bestAssignments = assignments
			} else {
				low = mid
			}
		}

		layers = layers[:len(layers)-bestAssignments.Sum()]
		requestedLayers -= bestAssignments.Sum()
		gpuLayers = append(bestAssignments, gpuLayers...)
	}

	return gpuLayers
}

// redistributeHeavyLayers moves FFN-heavy layers from weaker GPUs to stronger GPUs
// after findBestFit to correct for uneven compute distribution. For models where FFN
// dominates compute (87%+ for qwen3), this can significantly improve pipeline balance.
func redistributeHeavyLayers(gpuLayers ml.GPULayersList, gpus []ml.DeviceInfo, memory *ml.BackendMemory, layers []uint64, envComputeBoost float64) ml.GPULayersList {
	if len(gpuLayers) < 2 || envComputeBoost <= 1.0 || len(gpus) == 0 {
		return gpuLayers
	}

	// Compute total compute weight per GPU (layers * layer_size as proxy for FFN compute)
	computeWeightOnGPU := make([]float64, len(gpus))
	gpuIdxMap := make(map[ml.DeviceID]int)
	for i, g := range gpus {
		gpuIdxMap[g.DeviceID] = i
	}
	for _, gl := range gpuLayers {
		idx, ok := gpuIdxMap[gl.DeviceID]
		if !ok || idx < 0 {
			continue
		}
		for _, layerIdx := range gl.Layers {
			computeWeightOnGPU[idx] += float64(layers[layerIdx])
		}
	}

	// Add the output layer's compute cost (lm_head matmul is typically the most expensive operation)
	outputLayerIdx := len(layers) - 1
	for _, gl := range gpuLayers {
		for _, layerIdx := range gl.Layers {
			if layerIdx == outputLayerIdx {
				idx, ok := gpuIdxMap[gl.DeviceID]
				if ok && idx >= 0 {
					// Weight the output layer more heavily since it's a dense matmul
					// over the full vocab, not a standard transformer layer
					computeWeightOnGPU[idx] += float64(layers[outputLayerIdx]) * 2
				}
				break
			}
		}
	}

	// Find strongest GPU by compute power (SMCount * ClockMHz)
	// Fall back to ComputeMajor*ComputeMinor if SMCount not reported
	strongestIdx := 0
	strongestPower := 0.0
	for i, g := range gpus {
		power := float64(g.SMCount * g.ClockMHz)
		if power == 0 {
			power = float64(g.ComputeMajor*100+g.ComputeMinor) * 1
		}
		if power > strongestPower {
			strongestPower = power
			strongestIdx = i
		}
	}

	// Find weakest GPU (not the strongest)
	weakestIdx := -1
	weakestWeight := float64(1e18)
	for _, gl := range gpuLayers {
		idx, ok := gpuIdxMap[gl.DeviceID]
		if !ok || idx == strongestIdx {
			continue
		}
		weight := computeWeightOnGPU[idx]
		if weight < weakestWeight {
			weakestWeight = weight
			weakestIdx = idx
		}
	}

	if weakestIdx < 0 {
		return gpuLayers
	}

	// Compute the target imbalance based on actual GPU compute power ratio.
	// For heterogeneous GPUs, the stronger GPU should hold proportionally more layers.
	strongestRawPower := float64(gpus[strongestIdx].SMCount * gpus[strongestIdx].ClockMHz)
	weakestRawPower := float64(gpus[weakestIdx].SMCount * gpus[weakestIdx].ClockMHz)
	if strongestRawPower == 0 || weakestRawPower == 0 {
		strongestRawPower = float64(gpus[strongestIdx].ComputeMajor*100+gpus[strongestIdx].ComputeMinor)
		weakestRawPower = float64(gpus[weakestIdx].ComputeMajor*100+gpus[weakestIdx].ComputeMinor)
	}
	targetImbalance := strongestRawPower / (weakestRawPower + 1)

	// Check if compute imbalance exists: strongest has significantly less weight than target
	imbalance := computeWeightOnGPU[strongestIdx] / (computeWeightOnGPU[weakestIdx] + 1)
	if imbalance >= targetImbalance*0.9 {
		// Already close to target, skip redistribution
		return gpuLayers
	}

	// Find largest layers on weakest GPU (later layers tend to have larger FFN)
	type layerInfo struct {
		layerIdx int
		size     uint64
	}
	var candidates []layerInfo
	for _, gl := range gpuLayers {
		idx, ok := gpuIdxMap[gl.DeviceID]
		if !ok || idx != weakestIdx {
			continue
		}
		for _, layerIdx := range gl.Layers {
			candidates = append(candidates, layerInfo{layerIdx, layers[layerIdx]})
		}
	}

	// Sort candidates by size descending
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].size > candidates[i].size {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Try moving layers from weakest to strongest
	strongestGlIdx := -1
	for i, gl := range gpuLayers {
		if gl.DeviceID == gpus[strongestIdx].DeviceID {
			strongestGlIdx = i
			break
		}
	}

	// Check strongest GPU actual free memory, accounting for graph and overhead
	strongestFree := gpus[strongestIdx].FreeMemory
	minMem := gpus[strongestIdx].MinimumMemory() + envconfig.GpuOverhead()
	if strongestFree > minMem {
		strongestFree -= minMem
	} else {
		strongestFree = 0
	}
	for j := range memory.GPUs {
		if memory.GPUs[j].DeviceID == gpus[strongestIdx].DeviceID {
			if strongestFree > memory.GPUs[j].Graph {
				strongestFree -= memory.GPUs[j].Graph
			} else {
				strongestFree = 0
			}
			break
		}
	}

	moved := false
	originalWeakestIdx := weakestIdx
	for _, candidate := range candidates {
		// Would strongest still have room?
		if strongestFree < candidate.size {
			continue
		}

		// Move the layer
		newWeakestLayers := make([]int, 0, len(gpuLayers[originalWeakestIdx].Layers)-1)
		for _, l := range gpuLayers[originalWeakestIdx].Layers {
			if l != candidate.layerIdx {
				newWeakestLayers = append(newWeakestLayers, l)
			}
		}
		gpuLayers[originalWeakestIdx].Layers = newWeakestLayers

		gpuLayers[strongestGlIdx].Layers = append(gpuLayers[strongestGlIdx].Layers, candidate.layerIdx)

		// Remove empty layer entry from weakest
		if len(gpuLayers[originalWeakestIdx].Layers) == 0 {
			// Remove weakest from gpuLayers, compact the slice
			var newLayers ml.GPULayersList
			for i, gl := range gpuLayers {
				if i != originalWeakestIdx {
					newLayers = append(newLayers, gl)
				}
			}
			gpuLayers = newLayers
			// Rebuild gpuIdxMap from compacted gpuLayers
			gpuIdxMap = make(map[ml.DeviceID]int)
			for i, gl := range gpuLayers {
				gpuIdxMap[gl.DeviceID] = i
			}
			// Weakest GPU is gone — no more layers to move from it
			break
		}

		strongestFree -= candidate.size
		computeWeightOnGPU[weakestIdx] -= float64(candidate.size)
		computeWeightOnGPU[strongestIdx] += float64(candidate.size)
		moved = true

		slog.Debug("redistributed heavy layer", "from", gpus[weakestIdx].Name, "to", gpus[strongestIdx].Name,
			"layer", candidate.layerIdx, "size", format.HumanBytes2(candidate.size))

		// Stop once compute is roughly balanced
		newImbalance := computeWeightOnGPU[strongestIdx] / (computeWeightOnGPU[weakestIdx] + 1)
		if newImbalance >= targetImbalance*0.9 {
			break
		}
	}

	if moved {
	strongestOnGPU := 0
		for _, gl := range gpuLayers {
			idx, ok := gpuIdxMap[gl.DeviceID]
			if !ok {
				continue
			}
			if idx == strongestIdx {
				strongestOnGPU = len(gl.Layers)
			}
		}
		slog.Info("computed rebalance after redistribution", "strongest_gpu", gpus[strongestIdx].Name,
			"strongest_layers", strongestOnGPU, "gpu", gpus[weakestIdx].Name,
			"before_imbalance", fmt.Sprintf("%.2f", imbalance),
			"after_imbalance", fmt.Sprintf("%.2f", computeWeightOnGPU[strongestIdx]/(computeWeightOnGPU[weakestIdx]+1)))
	}

	return gpuLayers
}

// protectOutputLayer forces the last layer (output layer, most compute-intensive) onto the
// strongest-compute GPU while keeping the rest of the assignment intact. This prevents the
// output layer from being assigned to a weaker GPU where compute-bound inference would
// bottleneck PCIe traffic.
func protectOutputLayer(existing ml.GPULayersList, gpus []ml.DeviceInfo, layers []uint64) ml.GPULayersList {
	// Guard: if no GPUs, return existing assignment unchanged
	if len(gpus) == 0 {
		return existing
	}

	// Find the strongest-compute GPU in the assignment
	strongestGPU := gpus[0]
	strongestTier := ml.ComputeTier(gpus[0].ComputeMajor, gpus[0].ComputeMinor)
	for _, gpu := range gpus {
		tier := ml.ComputeTier(gpu.ComputeMajor, gpu.ComputeMinor)
		// Use SMCount*ClockMHz as tiebreaker, fall back to ComputeMajor*ComputeMinor
		gpuPower := gpu.SMCount * gpu.ClockMHz
		strongestPower := strongestGPU.SMCount * strongestGPU.ClockMHz
		if gpuPower == 0 && strongestPower == 0 {
			gpuPower = gpu.ComputeMajor*100 + gpu.ComputeMinor
			strongestPower = strongestGPU.ComputeMajor*100 + strongestGPU.ComputeMinor
		}
		if tier > strongestTier || (tier == strongestTier && gpuPower > strongestPower) {
			strongestGPU = gpu
			strongestTier = tier
		}
	}

	// Find which GPU currently has the output layer and remove it
	outputLayerIdx := len(layers) - 1
	for i, gl := range existing {
		for _, layerIdx := range gl.Layers {
			if layerIdx == outputLayerIdx {
				// Remove the output layer from this GPU's list
				newLayers := make([]int, 0, len(gl.Layers)-1)
				for _, idx := range gl.Layers {
					if idx != outputLayerIdx {
						newLayers = append(newLayers, idx)
					}
				}
				existing[i].Layers = newLayers
				break
			}
		}
	}

	// Add the output layer to the strongest GPU's list
	for i := range existing {
		if existing[i].DeviceID == strongestGPU.DeviceID {
			existing[i].Layers = append(existing[i].Layers, outputLayerIdx)
			return existing
		}
	}

	// Strongest GPU isn't in the existing assignment — append it as a new GPU
	existing = append(existing, ml.GPULayers{
		DeviceID:     strongestGPU.DeviceID,
		Layers:       []int{outputLayerIdx},
		ComputeMajor: strongestGPU.ComputeMajor,
		ComputeMinor: strongestGPU.ComputeMinor,
	})
	return existing
}

// greedyFit assigns layers incrementally to GPUs, spilling over as each runs out of free space.
// Starts from the strongest GPU (index 0, most VRAM after sort) so it absorbs the heaviest
// layers first. Weaker GPUs receive the remainder. This ensures the most powerful GPU
// gets the most compute-intensive work, reducing pipeline bottleneck.
func greedyFit(layers []uint64, gpus []ml.DeviceInfo, capacity float32, requestedLayers int) (gpuLayers ml.GPULayersList) {
	device := 0
	gpuLayers = ml.GPULayersList{{
		DeviceID:     gpus[device].DeviceID,
		ComputeMajor: gpus[device].ComputeMajor,
		ComputeMinor: gpus[device].ComputeMinor,
	}}
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

			device++
			if device >= len(gpus) {
				return gpuLayers
			}
			gpuLayers = append(ml.GPULayersList{{
				DeviceID:     gpus[device].DeviceID,
				ComputeMajor: gpus[device].ComputeMajor,
				ComputeMinor: gpus[device].ComputeMinor,
			}}, gpuLayers...)
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
		slog.Error("do load request", "error", err)
		return nil, errors.New("model failed to load, this may be due to resource limitations or an internal error, check ollama server logs for details")
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
>>>>>>> Stashed changes

type ServerStatus int

const (
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

// Request/Response types

const (
	llamaServerStreamInitialBufferSize = 64 * 1024
	// llamaServerStreamMaxBufferSize bounds a single runner response stream line.
	llamaServerStreamMaxBufferSize = 8 * format.MegaByte
)

type MediaKind string

const (
	MediaKindUnknown MediaKind = ""
	MediaKindImage   MediaKind = "image"
	MediaKindAudio   MediaKind = "audio"
)

type MediaData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
	Kind MediaKind
}

type Message struct {
	Role       string
	Content    string
	Thinking   string
	Media      []MediaData
	ToolCalls  []api.ToolCall
	ToolName   string
	ToolCallID string
}

func MessageFromAPI(msg api.Message) Message {
	media := make([]MediaData, len(msg.Images))
	for i, data := range msg.Images {
		media[i] = NewMediaData(i, data)
	}

	return Message{
		Role:       msg.Role,
		Content:    msg.Content,
		Thinking:   msg.Thinking,
		Media:      media,
		ToolCalls:  msg.ToolCalls,
		ToolName:   msg.ToolName,
		ToolCallID: msg.ToolCallID,
	}
}

type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Media   []MediaData
	Options *api.Options

	Shift           bool
	Truncate        bool
	PreservedTokens []string // parser tokens to render as text; ignored by non-llama-server runners
	ToolCallTag     string   // raw generic tool parser tag, if any
	LeadingBOS      string   // textual BOS emitted by Go rendering, if any
	// ThinkingClose holds the strings any of which ends the thinking the
	// response begins with, which Format leaves free; none when the response
	// starts in content.
	ThinkingClose []string

	// Logprobs specifies whether to include log probabilities in the response
	Logprobs bool

	// TopLogprobs specifies the number of most likely alternative tokens to return (0-20)
	TopLogprobs int
}

type ChatRequest struct {
	Messages []api.Message
	Tools    api.Tools
	Format   json.RawMessage
	Options  *api.Options
	Think    *api.ThinkValue
	Shift    bool

	Logprobs    bool
	TopLogprobs int
}

type ChatResponse struct {
	Message               api.Message   `json:"message"`
	DoneReason            DoneReason    `json:"done_reason"`
	Done                  bool          `json:"done"`
	PromptEvalCount       int           `json:"prompt_eval_count"`
	PromptEvalCachedCount *int          `json:"prompt_eval_cached_count,omitempty"`
	PromptEvalDuration    time.Duration `json:"prompt_eval_duration"`
	EvalCount             int           `json:"eval_count"`
	EvalDuration          time.Duration `json:"eval_duration"`
	Logprobs              []Logprob     `json:"logprobs,omitempty"`
}

// DoneReason represents the reason why a completion response is done
type DoneReason int

const (
	DoneReasonStop DoneReason = iota
	DoneReasonLength
	DoneReasonConnectionClosed
)

func (d DoneReason) String() string {
	switch d {
	case DoneReasonLength:
		return "length"
	case DoneReasonStop:
		return "stop"
	default:
		return ""
	}
}

// TokenLogprob represents log probability information for a single token alternative.
type TokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// Logprob contains log probability information for a generated token.
type Logprob struct {
	TokenLogprob
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

type CompletionResponse struct {
	Content               string        `json:"content"`
	DoneReason            DoneReason    `json:"done_reason"`
	Done                  bool          `json:"done"`
	PromptEvalCount       int           `json:"prompt_eval_count"`
	PromptEvalCachedCount *int          `json:"prompt_eval_cached_count,omitempty"`
	PromptEvalDuration    time.Duration `json:"prompt_eval_duration"`
	EvalCount             int           `json:"eval_count"`
	EvalDuration          time.Duration `json:"eval_duration"`

	// Logprobs contains log probability information if requested
	Logprobs []Logprob `json:"logprobs,omitempty"`
}
