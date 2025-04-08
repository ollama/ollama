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
	"github.com/ollama/ollama/model"
)

type LlamaServer interface {
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

	// llamaModel is an instance of the cgo llama.cpp model definition
	// nil if this server is running the new engine
	llamaModel     *llama.Model
	llamaModelLock sync.Mutex

	// textProcessor handles text encoding/decoding for the model in the Ollama engine
	// nil if this server is running the llama.cpp based engine
	textProcessor model.TextProcessor

	estimate    MemoryEstimate
	totalLayers uint64
	// gpuCount     int
	gpus         discover.GpuInfoList // Recorded just before the model loaded, free space will be incorrect
	loadDuration time.Duration        // Record how long it took the model to load
	loadProgress float32

	sem *semaphore.Weighted
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

	ggml, _, err := ggml.Decode(f, maxArraySize)
	return ggml, err
}

// NewLlamaServer will run a server for the given GPUs
// The gpu list must be a single family.
func NewLlamaServer(gpus discover.GpuInfoList, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	systemInfo := discover.GetSystemInfo()
	systemTotalMemory := systemInfo.System.TotalMemory
	systemFreeMemory := systemInfo.System.FreeMemory
	systemSwapFreeMemory := systemInfo.System.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	// If the user wants zero GPU layers, reset the gpu list to be CPU/system ram info
	if opts.NumGPU == 0 {
		gpus = discover.GetCPUInfo()
	}

	estimate := EstimateGPULayers(gpus, f, projectors, opts, numParallel)
	if len(gpus) > 1 || gpus[0].Library != "cpu" {
		switch {
		case gpus[0].Library == "metal" && estimate.VRAMSize > systemTotalMemory:
			// disable partial offloading when model is greater than total system memory as this
			// can lead to locking up the system
			opts.NumGPU = 0
		case gpus[0].Library != "metal" && estimate.Layers == 0:
			// Don't bother loading into the GPU if no layers can fit
			gpus = discover.GetCPUInfo()
		case opts.NumGPU < 0 && estimate.Layers > 0 && gpus[0].Library != "cpu":
			opts.NumGPU = estimate.Layers
		}
	}

	// On linux and windows, over-allocating CPU memory will almost always result in an error
	// Darwin has fully dynamic swap so has no direct concept of free swap space
	if runtime.GOOS != "darwin" {
		systemMemoryRequired := estimate.TotalSize - estimate.VRAMSize
		available := systemFreeMemory + systemSwapFreeMemory
		if systemMemoryRequired > available {
			slog.Warn("model request too large for system", "requested", format.HumanBytes2(systemMemoryRequired), "available", available, "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "swap", format.HumanBytes2(systemSwapFreeMemory))
			return nil, fmt.Errorf("model requires more system memory (%s) than is available (%s)", format.HumanBytes2(systemMemoryRequired), format.HumanBytes2(available))
		}
	}

	slog.Info("offload", "", estimate)

	params := []string{
		"--model", modelPath,
		"--ctx-size", strconv.Itoa(opts.NumCtx),
		"--batch-size", strconv.Itoa(opts.NumBatch),
	}

	if opts.NumGPU >= 0 {
		params = append(params, "--n-gpu-layers", strconv.Itoa(opts.NumGPU))
	}

	if envconfig.Debug() {
		params = append(params, "--verbose")
	}

	if opts.MainGPU > 0 {
		params = append(params, "--main-gpu", strconv.Itoa(opts.MainGPU))
	}

	if len(adapters) > 0 {
		for _, adapter := range adapters {
			params = append(params, "--lora", adapter)
		}
	}

	defaultThreads := systemInfo.GetOptimalThreadCount()
	if opts.NumThread > 0 {
		params = append(params, "--threads", strconv.Itoa(opts.NumThread))
	} else if defaultThreads > 0 {
		params = append(params, "--threads", strconv.Itoa(defaultThreads))
	}

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
		params = append(params, "--flash-attn")

		// Flash Attention also supports kv cache quantization
		// Enable if the requested and kv cache type is supported by the model
		if kvct != "" && f.SupportsKVCacheType(kvct) {
			params = append(params, "--kv-cache-type", kvct)
		} else {
			slog.Warn("kv cache type not supported by model", "type", kvct)
		}
	} else if kvct != "" && kvct != "f16" {
		slog.Warn("quantized kv cache requested but flash attention disabled", "type", kvct)
	}

	// mmap has issues with partial offloading on metal
	for _, g := range gpus {
		if g.Library == "metal" &&
			uint64(opts.NumGPU) > 0 &&
			uint64(opts.NumGPU) < f.KV().BlockCount()+1 {
			opts.UseMMap = new(bool)
			*opts.UseMMap = false
		}
	}

	// Windows CUDA should not use mmap for best performance
	// Linux  with a model larger than free space, mmap leads to thrashing
	// For CPU loads we want the memory to be allocated, not FS cache
	if (runtime.GOOS == "windows" && gpus[0].Library == "cuda" && opts.UseMMap == nil) ||
		(runtime.GOOS == "linux" && systemFreeMemory < estimate.TotalSize && opts.UseMMap == nil) ||
		(gpus[0].Library == "cpu" && opts.UseMMap == nil) ||
		(opts.UseMMap != nil && !*opts.UseMMap) {
		params = append(params, "--no-mmap")
	}

	if opts.UseMLock {
		params = append(params, "--mlock")
	}

	// TODO - NUMA support currently doesn't work properly

	params = append(params, "--parallel", strconv.Itoa(numParallel))

	if estimate.TensorSplit != "" {
		params = append(params, "--tensor-split", estimate.TensorSplit)
	}

	if envconfig.MultiUserCache() {
		params = append(params, "--multiuser-cache")
	}

	libs := make(map[string]string)
	if entries, err := os.ReadDir(discover.LibOllamaPath); err == nil {
		for _, entry := range entries {
			libs[entry.Name()] = filepath.Join(discover.LibOllamaPath, entry.Name())
		}
	}

	lib := gpus[0].RunnerName()
	requested := envconfig.LLMLibrary()
	if libs[requested] != "" {
		slog.Info("using requested gpu library", "requested", requested)
		lib = requested
	}

	var compatible []string
	for k := range libs {
		// exact match first
		if k == lib {
			compatible = append([]string{k}, compatible...)
			continue
		}

		// then match the family (e.g. 'cuda')
		if strings.Split(k, "_")[0] == strings.Split(lib, "_")[0] {
			compatible = append(compatible, k)
		}
	}
	slog.Debug("compatible gpu libraries", "compatible", compatible)
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}

	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	var llamaModel *llama.Model
	var textProcessor model.TextProcessor
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

	if len(projectors) > 0 && llamaModel != nil {
		params = append(params, "--mmproj", projectors[0])
	}

	// iterate through compatible GPU libraries such as 'cuda_v12', 'cuda_v11', 'rocm', etc.
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
		finalParams := []string{"runner"}
		if textProcessor != nil {
			// New engine
			// TODO - if we have failure to load scenarios, add logic to retry with the old runner
			finalParams = append(finalParams, "--ollama-engine")
		}
		finalParams = append(finalParams, params...)
		finalParams = append(finalParams, "--port", strconv.Itoa(port))

		var pathEnv string
		switch runtime.GOOS {
		case "windows":
			pathEnv = "PATH"
		case "darwin":
			pathEnv = "DYLD_LIBRARY_PATH"
		default:
			pathEnv = "LD_LIBRARY_PATH"
		}

		var libraryPaths []string
		if libraryPath, ok := os.LookupEnv(pathEnv); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
		}

		if len(compatible) > 0 {
			c := compatible[0]
			if libpath, ok := libs[c]; ok {
				slog.Debug("adding gpu library", "path", libpath)
				libraryPaths = append(libraryPaths, libpath)
			}
		}

		// Note: we always put the dependency path first
		// since this was the exact version we compiled/linked against
		if gpus[0].DependencyPath != nil {
			slog.Debug("adding gpu dependency paths", "paths", gpus[0].DependencyPath)
			// assume gpus from the same library have the same dependency path
			libraryPaths = append(gpus[0].DependencyPath, libraryPaths...)
		}

		// finally, add the root library path
		libraryPaths = append(libraryPaths, discover.LibOllamaPath)

		s := &llmServer{
			port:          port,
			cmd:           exec.Command(exe, finalParams...),
			status:        NewStatusWriter(os.Stderr),
			options:       opts,
			modelPath:     modelPath,
			llamaModel:    llamaModel,
			textProcessor: textProcessor,
			estimate:      estimate,
			numParallel:   numParallel,
			sem:           semaphore.NewWeighted(int64(numParallel)),
			totalLayers:   f.KV().BlockCount() + 1,
			gpus:          gpus,
			done:          make(chan error, 1),
		}

		s.cmd.Env = os.Environ()
		s.cmd.Stdout = os.Stdout
		s.cmd.Stderr = s.status
		s.cmd.SysProcAttr = LlamaServerSysProcAttr

		envWorkarounds := [][2]string{}
		for _, gpu := range gpus {
			envWorkarounds = append(envWorkarounds, gpu.EnvWorkarounds...)
		}
		visibleDevicesEnv, visibleDevicesEnvVal := gpus.GetVisibleDevicesEnv()
		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		// Update or add the path and visible devices variable with our adjusted version
		pathNeeded := true
		devicesNeeded := visibleDevicesEnv != ""
		for i := range s.cmd.Env {
			cmp := strings.SplitN(s.cmd.Env[i], "=", 2)
			if strings.EqualFold(cmp[0], pathEnv) {
				s.cmd.Env[i] = pathEnv + "=" + pathEnvVal
				pathNeeded = false
			} else if devicesNeeded && strings.EqualFold(cmp[0], visibleDevicesEnv) {
				s.cmd.Env[i] = visibleDevicesEnv + "=" + visibleDevicesEnvVal
				devicesNeeded = false
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
		if devicesNeeded {
			s.cmd.Env = append(s.cmd.Env, visibleDevicesEnv+"="+visibleDevicesEnvVal)
		}

		slog.Info("starting llama server", "cmd", s.cmd)
		if envconfig.Debug() {
			filteredEnv := []string{}
			for _, ev := range s.cmd.Env {
				if strings.HasPrefix(ev, "CUDA_") ||
					strings.HasPrefix(ev, "ROCR_") ||
					strings.HasPrefix(ev, "ROCM_") ||
					strings.HasPrefix(ev, "HIP_") ||
					strings.HasPrefix(ev, "GPU_") ||
					strings.HasPrefix(ev, "HSA_") ||
					strings.HasPrefix(ev, "GGML_") ||
					strings.HasPrefix(ev, "PATH=") ||
					strings.HasPrefix(ev, "LD_LIBRARY_PATH=") ||
					strings.HasPrefix(ev, "DYLD_LIBRARY_PATH=") {
					filteredEnv = append(filteredEnv, ev)
				}
			}
			// Log at debug as the environment is inherited and might contain sensitive information
			slog.Debug("subprocess", "environment", filteredEnv)
		}

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

		return s, nil
	}
}

type ServerStatus int

const ( // iota is reset to 0
	ServerStatusReady ServerStatus = iota
	ServerStatusNoSlotsAvailable
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
	case ServerStatusReady, ServerStatusNoSlotsAvailable:
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
	start := time.Now()
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
			s.loadDuration = time.Since(start)
			slog.Info(fmt.Sprintf("llama runner started in %0.2f seconds", s.loadDuration.Seconds()))
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

var grammarJSON = `
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

const maxBufferSize = 512 * format.KiloByte

type ImageData struct {
	Data          []byte `json:"data"`
	ID            int    `json:"id"`
	AspectRatioID int    `json:"aspect_ratio_id"`
}

type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Images  []ImageData
	Options *api.Options

	Grammar string // set before sending the request to the subprocess
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
		return fmt.Errorf("POST predict: %v", err)
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
		slog.Debug("stopping llama server")
		if err := s.cmd.Process.Kill(); err != nil {
			return err
		}
		// if ProcessState is already populated, Wait already completed, no need to wait again
		if s.cmd.ProcessState == nil {
			slog.Debug("waiting for llama server to exit")
			<-s.done
		}

		slog.Debug("llama server stopped")
	}

	return nil
}

func (s *llmServer) EstimatedVRAM() uint64 {
	return s.estimate.VRAMSize
}

func (s *llmServer) EstimatedTotal() uint64 {
	return s.estimate.TotalSize
}

func (s *llmServer) EstimatedVRAMByGPU(gpuID string) uint64 {
	for i, gpu := range s.gpus {
		if gpu.ID == gpuID {
			if i < len(s.estimate.GPUSizes) {
				return s.estimate.GPUSizes[i]
			}
		}
	}
	return 0
}
