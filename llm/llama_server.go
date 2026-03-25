// llama_server.go wraps the llama-server binary as a subprocess
//
// Ollama renders prompts and parses tool calls in Go (using the
// renderers in model/renderers/ and parsers in model/parsers/). The rendered
// prompt is sent as raw text to llama-server's /completion endpoint. This
// preserves Ollama's template rendering, tool call extraction,
// thinking/reasoning support, and context truncation.
//
// For structured output, JSON schemas are passed directly to llama-server via
// its json_schema field (avoiding the CGO SchemaToGrammar dependency). Raw BNF
// grammars are passed via the grammar field.
//
// llama-server auto-detects GPU layers (-ngl), thread count (-t), and flash
// attention (--flash-attn).
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

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

// llamaServerRunner wraps an upstream llama-server process and implements the LlamaServer interface.
// It communicates with llama-server over HTTP.
type llamaServerRunner struct {
	port      int
	cmd       *exec.Cmd
	done      chan struct{}
	doneErr   error
	memTotal  uint64 // actual total buffer size parsed from llama-server logs (bytes)
	memGPU    uint64 // actual GPU buffer size parsed from llama-server logs (bytes)
	status    *StatusWriter
	options   api.Options
	modelPath string

	// Per-device VRAM tracking, populated from llama-server log parsing.
	// Keys are device names from llama-server output (e.g., "CUDA0", "ROCm0", "MTL0").
	vramByDevice map[string]uint64

	// GPU layer offload counts, parsed from "offloaded N/M layers to GPU" log line.
	offloadedLayers int
	offloadedTotal  int

	// System-reported free VRAM per device at model load time, parsed from
	// "using device CUDA0 ... - 15221 MiB free" log lines. This reflects
	// real system state including external VRAM consumers (on platforms where
	// the GPU driver reports accurately). Keys match vramByDevice (e.g., "CUDA0").
	systemFreeAtLoad map[string]uint64

	// gpus is the list of GPU devices assigned to this runner at creation time,
	// used to map DeviceIDs to device names for VRAMByGPU lookups.
	gpus []ml.DeviceInfo

	ggml        *ggml.GGML
	totalLayers uint64
	loadStart   time.Time

	sem *semaphore.Weighted
}

func (s *llamaServerRunner) ModelPath() string {
	return s.modelPath
}

func (s *llamaServerRunner) Pid() int {
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return 0
}

func (s *llamaServerRunner) GetPort() int {
	return s.port
}

func (s *llamaServerRunner) HasExited() bool {
	return s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0
}

func (s *llamaServerRunner) ContextLength() int {
	return s.options.NumCtx
}

// FindLlamaServer locates the llama-server binary in lib/ollama/.
// There is a single binary that dynamically loads GPU backends at runtime.
func FindLlamaServer() (string, error) {
	suffix := "llama-server"
	if runtime.GOOS == "windows" {
		suffix += ".exe"
	}

	// Deduplicate candidates while preserving order
	seen := map[string]bool{}
	var candidates []string
	add := func(dir string) {
		path := filepath.Join(dir, suffix)
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	// 1. lib/ollama/ (distribution layout)
	add(ml.LibOllamaPath)

	// 2. Dev build paths (cmake install destination)
	exe, err := os.Executable()
	if err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		add(filepath.Join(filepath.Dir(exe), "build", "lib", "ollama"))
	}
	if cwd, err := os.Getwd(); err == nil {
		add(filepath.Join(cwd, "build", "lib", "ollama"))
	}

	// 3. Dev build paths (cmake build output, before install)
	// Prefer platform-specific static builds (darwin) over dynamic CPU builds
	addGlob := func(base string) {
		matches, _ := filepath.Glob(filepath.Join(base, "build", "llama-server-*", "bin"))
		slices.SortFunc(matches, func(a, b string) int {
			aIsPlatform := strings.Contains(a, "llama-server-darwin") || strings.Contains(a, "llama-server-cuda") || strings.Contains(a, "llama-server-rocm")
			bIsPlatform := strings.Contains(b, "llama-server-darwin") || strings.Contains(b, "llama-server-cuda") || strings.Contains(b, "llama-server-rocm")
			if aIsPlatform && !bIsPlatform {
				return -1
			}
			if !aIsPlatform && bIsPlatform {
				return 1
			}
			return strings.Compare(a, b)
		})
		for _, m := range matches {
			add(m)
		}
	}
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		addGlob(filepath.Dir(exe))
	}
	if cwd, err := os.Getwd(); err == nil {
		addGlob(cwd)
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("llama-server binary not found (checked: %s). Run 'cmake -S llama/server --preset cpu && cmake --build --preset cpu' first", strings.Join(candidates, ", "))
}

// startLlamaServer spawns the upstream llama-server process with appropriate CLI flags.
func startLlamaServer(
	modelPath string,
	projectors []string,
	adapters []string,
	opts api.Options,
	numParallel int,
	kvCacheType string,
	embedding bool,
	gpuLibs []string,
	extraEnvs map[string]string,
	out io.Writer,
) (cmd *exec.Cmd, port int, err error) {
	exe, err := FindLlamaServer()
	if err != nil {
		return nil, 0, err
	}

	// Allocate a port
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
		port = rand.Intn(65535-49152) + 49152
	}

	// Build CLI flags — minimal set, let llama-server auto-detect the rest
	params := []string{
		"--model", modelPath,
		"--port", strconv.Itoa(port),
		"--host", "127.0.0.1",
		"--no-webui",
		"--offline",
		"-c", strconv.Itoa(opts.NumCtx * numParallel),
		"-np", strconv.Itoa(numParallel),
	}

	// Multimodal projectors
	if len(projectors) > 0 {
		params = append(params, "--mmproj", projectors[0])
	}

	// LoRA adapters
	for _, adapter := range adapters {
		params = append(params, "--lora", adapter)
	}

	// UseMmap
	if opts.UseMMap != nil && !*opts.UseMMap {
		params = append(params, "--no-mmap")
	}

	// KV cache type
	if kvCacheType != "" {
		params = append(params, "--cache-type-k", kvCacheType, "--cache-type-v", kvCacheType)
	}

	// Batch size — match the old engine default (512) instead of
	// llama-server's default (2048) to avoid generation regressions
	if embedding {
		// Embedding mode — set batch size to context size so large inputs fit
		params = append(params, "--embedding")
		params = append(params, "-b", strconv.Itoa(opts.NumCtx*numParallel))
		params = append(params, "-ub", strconv.Itoa(opts.NumCtx*numParallel))
	} else if opts.NumBatch > 0 {
		params = append(params, "-b", strconv.Itoa(opts.NumBatch))
	}

	// GPU layer offloading — only pass if user explicitly set it (non-default).
	// Default behavior: let llama-server auto-detect via -ngl auto.
	if opts.NumGPU > 0 {
		params = append(params, "-ngl", strconv.Itoa(opts.NumGPU))
	} else if opts.NumGPU == 0 {
		// Explicit 0 means CPU only
		params = append(params, "-ngl", "0")
	}
	// NumGPU == -1 (default): don't pass -ngl, let llama-server auto-detect

	// Thread count — only pass if user explicitly set it.
	// Default behavior: let llama-server auto-detect.
	if opts.NumThread > 0 {
		params = append(params, "-t", strconv.Itoa(opts.NumThread))
	}

	// Main GPU selection for multi-GPU systems
	if opts.MainGPU > 0 {
		params = append(params, "-mg", strconv.Itoa(opts.MainGPU))
	}

	// Context shift: enable for small contexts (<8k) where users are more
	// likely to hit overflow on long prompts, matching the old CGO engine's
	// behavior. For 8k+ contexts, disable shifting and let llama-server
	// return a clean 400 error — context shifting at large sizes silently
	// degrades quality because the prompt template and system prompt get
	// evicted (n_keep only preserves a few initial tokens).
	if opts.NumCtx > 0 && opts.NumCtx < 8192 {
		params = append(params, "--context-shift")
		if opts.NumKeep > 0 {
			params = append(params, "--keep", strconv.Itoa(opts.NumKeep))
		}
	}

	// Set up library paths for GPU backend discovery
	var pathEnv string
	switch runtime.GOOS {
	case "windows":
		pathEnv = "PATH"
	case "darwin":
		pathEnv = "DYLD_LIBRARY_PATH"
	default:
		pathEnv = "LD_LIBRARY_PATH"
	}

	// Library path ordering:
	// 1. llama-server's own directory (lib/ollama/) — for ggml-base, ggml-cpu, libllama
	// 2. GPU variant directories (lib/ollama/cuda_v12/) — for cublas, cudart, GPU backend
	//
	// llama-server scans its own directory for CPU backends but not subdirectories.
	// We use GGML_BACKEND_PATH to point it at the specific GPU backend .so file.
	llamaDir := filepath.Dir(exe)
	libraryPaths := []string{llamaDir}
	for _, dir := range gpuLibs {
		if dir == ml.LibOllamaPath {
			continue
		}
		// Check for GPU backend .so in the variant directory
		entries, _ := filepath.Glob(filepath.Join(dir, "libggml-*"))
		if len(entries) == 0 {
			entries, _ = filepath.Glob(filepath.Join(dir, "ggml-*.dll"))
		}
		if len(entries) > 0 {
			if extraEnvs == nil {
				extraEnvs = make(map[string]string)
			}
			extraEnvs["GGML_BACKEND_PATH"] = entries[0]
		}
		libraryPaths = append(libraryPaths, dir)
	}
	if libraryPath, ok := os.LookupEnv(pathEnv); ok {
		libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
	}

	cmd = exec.Command(exe, params...)
	cmd.Env = os.Environ()

	if out != nil {
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return nil, 0, fmt.Errorf("failed to spawn llama-server stdout pipe: %w", err)
		}
		stderr, err := cmd.StderrPipe()
		if err != nil {
			return nil, 0, fmt.Errorf("failed to spawn llama-server stderr pipe: %w", err)
		}
		go func() {
			io.Copy(out, stdout) //nolint:errcheck
		}()
		go func() {
			io.Copy(out, stderr) //nolint:errcheck
		}()
	}
	cmd.SysProcAttr = LlamaServerSysProcAttr

	pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

	// Set environment variables
	pathNeeded := true
	extraEnvsDone := map[string]bool{}
	for k := range extraEnvs {
		extraEnvsDone[k] = false
	}
	for i := range cmd.Env {
		cmp := strings.SplitN(cmd.Env[i], "=", 2)
		if strings.EqualFold(cmp[0], pathEnv) {
			cmd.Env[i] = pathEnv + "=" + pathEnvVal
			pathNeeded = false
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
	for k, done := range extraEnvsDone {
		if !done {
			cmd.Env = append(cmd.Env, k+"="+extraEnvs[k])
		}
	}

	slog.Info("starting llama-server", "cmd", cmd)
	slog.Debug("subprocess", "", filteredEnv(cmd.Env))

	if err = cmd.Start(); err != nil {
		return nil, 0, err
	}
	return cmd, port, nil
}

// NewLlamaServerRunner creates a new llama-server runner that wraps the upstream llama-server binary.
func NewLlamaServerRunner(
	gpus []ml.DeviceInfo,
	modelPath string,
	f *ggml.GGML,
	adapters, projectors []string,
	opts api.Options,
	numParallel int,
	kvCacheType string,
) (LlamaServer, error) {
	// Check if this is an embedding model
	_, isEmbedding := f.KV()[fmt.Sprintf("%s.pooling_type", f.KV().Architecture())]

	gpuLibs := ml.LibraryPaths(gpus)
	status := NewStatusWriter(os.Stderr)

	// memWriter wraps the status writer and parses buffer size lines from llama-server logs
	memWriter := &memoryParsingWriter{inner: status}

	cmd, port, err := startLlamaServer(
		modelPath,
		projectors,
		adapters,
		opts,
		numParallel,
		kvCacheType,
		isEmbedding,
		gpuLibs,
		ml.GetDevicesEnv(gpus, false),
		memWriter,
	)

	s := &llamaServerRunner{
		port:             port,
		cmd:              cmd,
		status:           status,
		options:          opts,
		modelPath:        modelPath,
		vramByDevice:     make(map[string]uint64),
		systemFreeAtLoad: make(map[string]uint64),
		gpus:             gpus,
		ggml:             f,
		totalLayers:      f.KV().BlockCount() + 1,
		loadStart:        time.Now(),
		sem:              semaphore.NewWeighted(int64(numParallel)),
		done:             make(chan struct{}),
	}
	// Point the memory parsing writer at this runner so values are updated as logs stream in
	memWriter.runner = s

	if err != nil {
		var msg string
		if s.status != nil && s.status.LastError() != "" {
			msg = s.status.LastError()
		}
		return nil, fmt.Errorf("error starting llama-server: %v %s", err, msg)
	}

	// Reap subprocess when it exits
	go func() {
		err := s.cmd.Wait()
		if err != nil && s.status != nil && s.status.LastError() != "" {
			slog.Error("llama-server terminated", "error", err)
			s.doneErr = errors.New(s.status.LastError())
		} else {
			s.doneErr = err
		}
		close(s.done)
	}()

	return s, nil
}

// Load waits for llama-server to finish loading the model. lama-server loads
// the model at startup and auto-detects GPU layers, so this just waits for
// health to report ready.
func (s *llamaServerRunner) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	slog.Info("loading model via llama-server", "model", s.modelPath)

	if err := s.WaitUntilRunning(ctx); err != nil {
		return nil, err
	}

	// Verify that buffer size parsing captured GPU allocations.
	// If parsing failed (e.g., llama-server log format changed), warn so the
	// issue is visible in logs when users report problems.
	if len(s.gpus) > 0 && len(s.vramByDevice) == 0 {
		slog.Warn("llama-server VRAM tracking: no per-device buffer sizes were parsed from "+
			"llama-server logs. VRAM accounting will be inaccurate. This may indicate a "+
			"change in llama-server's log format — check for 'buffer size' lines in the output.",
			"model", s.modelPath, "gpus", len(s.gpus))
	}

	// Return device IDs for all GPUs since llama-server manages layer placement itself
	deviceIDs := make([]ml.DeviceID, len(gpus))
	for i, g := range gpus {
		deviceIDs[i] = g.DeviceID
	}

	return deviceIDs, nil
}

// getServerStatus checks llama-server's /health endpoint.
// llama-server returns {"status":"ok"}, {"status":"loading model"}, or {"status":"error"}.
func (s *llamaServerRunner) getServerStatus(ctx context.Context) (ServerStatus, error) {
	if s.cmd.ProcessState != nil {
		msg := ""
		if s.status != nil && s.status.LastError() != "" {
			msg = s.status.LastError()
		}
		if s.cmd.ProcessState.ExitCode() == -1 {
			slog.Warn("llama-server process no longer running", "sys", s.cmd.ProcessState.Sys(), "string", s.cmd.ProcessState)
		}
		return ServerStatusError, fmt.Errorf("llama-server process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return ServerStatusError, fmt.Errorf("error creating health request: %v", err)
	}

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
		return ServerStatusError, fmt.Errorf("read health response: %w", err)
	}

	// llama-server returns {"status":"ok"}, {"status":"loading model"}, {"status":"error", ...}
	var result struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return ServerStatusError, fmt.Errorf("health unmarshal: %w", err)
	}

	switch result.Status {
	case "ok":
		return ServerStatusReady, nil
	case "loading model":
		return ServerStatusLoadingModel, nil
	case "no slot available":
		return ServerStatusNoSlotsAvailable, nil
	default:
		return ServerStatusError, fmt.Errorf("llama-server error: %s", string(body))
	}
}

func (s *llamaServerRunner) getServerStatusRetry(ctx context.Context) (ServerStatus, error) {
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

func (s *llamaServerRunner) Ping(ctx context.Context) error {
	_, err := s.getServerStatus(ctx)
	if err != nil {
		slog.Debug("llama-server unhealthy", "error", err)
	}
	return err
}

func (s *llamaServerRunner) WaitUntilRunning(ctx context.Context) error {
	stallDuration := envconfig.LoadTimeout()
	stallTimer := time.Now().Add(stallDuration)

	slog.Info("waiting for llama-server to start responding")
	var lastStatus ServerStatus = -1

	for {
		select {
		case <-ctx.Done():
			slog.Warn("client connection closed before llama-server finished loading, aborting load")
			return fmt.Errorf("timed out waiting for llama-server to start: %w", ctx.Err())
		case <-s.done:
			if s.status != nil && s.status.LastError() != "" {
				return fmt.Errorf("llama-server process has terminated: %s", s.status.LastError())
			}
			if s.doneErr != nil {
				return fmt.Errorf("llama-server process has terminated: %w", s.doneErr)
			}
			if s.cmd != nil && s.cmd.ProcessState != nil {
				return fmt.Errorf("llama-server process has terminated with exit code %d", s.cmd.ProcessState.ExitCode())
			}
			return errors.New("llama-server process has terminated")
		default:
		}

		if time.Now().After(stallTimer) {
			msg := ""
			if s.status != nil && s.status.LastError() != "" {
				msg = s.status.LastError()
			}
			return fmt.Errorf("timed out waiting for llama-server to start - %s", msg)
		}

		if s.cmd.ProcessState != nil {
			msg := ""
			if s.status != nil && s.status.LastError() != "" {
				msg = s.status.LastError()
			}
			return fmt.Errorf("llama-server process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
		}

		pollCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		status, _ := s.getServerStatus(pollCtx)
		cancel()

		if lastStatus != status && status != ServerStatusReady {
			slog.Info("waiting for llama-server to become available", "status", status)
		}

		switch status {
		case ServerStatusReady:
			slog.Info(fmt.Sprintf("llama-server started in %0.2f seconds", time.Since(s.loadStart).Seconds()))
			return nil
		default:
			lastStatus = status
			// Reset stall timer on progress
			stallTimer = time.Now().Add(stallDuration)
			time.Sleep(time.Millisecond * 250)
		}
	}
}

// ShouldRetryWithMetalTensorDisabled reports whether a startup/discovery
// failure matches the Metal tensor API crash path. Discovery records the
// successful override on the device so regular model loads inherit it.
func ShouldRetryWithMetalTensorDisabled(err error, status *StatusWriter) bool {
	if runtime.GOOS != "darwin" {
		return false
	}

	var msg strings.Builder
	if err != nil {
		msg.WriteString(strings.ToLower(err.Error()))
	}
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

// llamaServerCompletionRequest is the request format for llama-server's POST /completion endpoint.
type llamaServerCompletionRequest struct {
	Prompt          any             `json:"prompt"`
	Stream          bool            `json:"stream"`
	CachePrompt     bool            `json:"cache_prompt"`
	NPredict        int             `json:"n_predict,omitempty"`
	NKeep           int             `json:"n_keep,omitempty"`
	Temperature     float32         `json:"temperature"`
	TopK            int             `json:"top_k"`
	TopP            float32         `json:"top_p"`
	MinP            float32         `json:"min_p"`
	Stop            []string        `json:"stop,omitempty"`
	RepeatPenalty   float32         `json:"repeat_penalty"`
	RepeatLastN     int             `json:"repeat_last_n,omitempty"`
	FreqPenalty     float32         `json:"frequency_penalty"`
	PresPenalty     float32         `json:"presence_penalty"`
	TypicalP        float32         `json:"typical_p,omitempty"`
	Seed            int             `json:"seed"`
	Grammar         string          `json:"grammar,omitempty"`
	JsonSchema      json.RawMessage `json:"json_schema,omitempty"`
	NProbs          int             `json:"n_probs,omitempty"`
	Samplers        []string        `json:"samplers,omitempty"`
	PreservedTokens []string        `json:"preserved_tokens,omitempty"`
}

// optimizedSamplerOrder mirrors llama-server's default sampler chain but moves
// "penalties" after "top_k". The upstream default runs penalties first, which
// iterates and does a hashmap lookup over the entire vocabulary (~128k tokens
// for modern models) every generated token — a measured 28-30% throughput hit
// on small models with the Ollama default repeat_penalty=1.1. Running penalties
// after top_k truncates that work to ~40 tokens with no behavioral change since
// every sampler here is commutative with top_k for the tokens that survive.
//
// See llama.cpp common/common.h COMMON_SAMPLER_TYPE_* for the canonical default.
var optimizedSamplerOrder = []string{
	"dry",
	"top_n_sigma",
	"top_k",
	"penalties",
	"typical_p",
	"top_p",
	"min_p",
	"xtc",
	"temperature",
}

// llamaServerMultimodalPrompt is used when images are present.
// llama-server's /completion endpoint accepts this as the "prompt" field.
type llamaServerMultimodalPrompt struct {
	PromptString   string   `json:"prompt_string"`
	MultimodalData []string `json:"multimodal_data"`
}

// llamaServerCompletionResponse is the response format from llama-server's /completion endpoint.
type llamaServerCompletionResponse struct {
	Content  string `json:"content"`
	Stop     bool   `json:"stop"`
	StopType string `json:"stop_type"`
	Timings  struct {
		PromptN   int     `json:"prompt_n"`
		PromptMS  float64 `json:"prompt_ms"`
		PredictN  int     `json:"predicted_n"`
		PredictMS float64 `json:"predicted_ms"`
	} `json:"timings"`
	CompletionProbabilities []llamaServerTokenProb `json:"completion_probabilities"`
}

type llamaServerTokenProb struct {
	Token       string                 `json:"token"`
	Logprob     float64                `json:"logprob"`
	Prob        float64                `json:"prob"`
	TopLogprobs []llamaServerTokenProb `json:"top_logprobs"`
	TopProbs    []llamaServerTokenProb `json:"top_probs"`
}

func (s *llamaServerRunner) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	slog.Debug("llama-server completion request", "images", len(req.Images), "prompt_len", len(req.Prompt))

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		}
		return err
	}
	defer s.sem.Release(1)

	if req.Options.NumPredict < 0 || req.Options.NumPredict > 10*s.options.NumCtx {
		req.Options.NumPredict = 10 * s.options.NumCtx
	}

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	// Build the llama-server request
	lsReq := llamaServerCompletionRequest{
		Prompt:          req.Prompt,
		Stream:          true,
		CachePrompt:     req.Shift,
		NPredict:        req.Options.NumPredict,
		NKeep:           req.Options.NumKeep,
		Temperature:     req.Options.Temperature,
		TopK:            req.Options.TopK,
		TopP:            req.Options.TopP,
		MinP:            req.Options.MinP,
		Stop:            req.Options.Stop,
		RepeatPenalty:   req.Options.RepeatPenalty,
		RepeatLastN:     req.Options.RepeatLastN,
		FreqPenalty:     req.Options.FrequencyPenalty,
		PresPenalty:     req.Options.PresencePenalty,
		TypicalP:        req.Options.TypicalP,
		Seed:            req.Options.Seed,
		Samplers:        optimizedSamplerOrder,
		PreservedTokens: req.PreservedTokens,
	}

	if req.Logprobs {
		lsReq.NProbs = max(req.TopLogprobs, 1)
	}

	// Handle format: pass JSON schema directly to llama-server, or use grammar
	if len(req.Format) > 0 {
		switch string(req.Format) {
		case `null`, `""`:
			// not set
		case `"json"`:
			lsReq.Grammar = grammarJSON
		default:
			if req.Format[0] == '{' {
				lsReq.JsonSchema = req.Format
			} else {
				return fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", req.Format)
			}
		}
	} else if req.Grammar != "" {
		lsReq.Grammar = req.Grammar
	}

	// Convert images: replace [img-N] markers with <__media__> and
	// package image data as base64 in a multimodal prompt object
	if len(req.Images) > 0 {
		promptStr := lsReq.Prompt.(string)
		var imageData []string
		for _, img := range req.Images {
			marker := fmt.Sprintf("[img-%d]", img.ID)
			promptStr = strings.Replace(promptStr, marker, "<__media__>", 1)
			imageData = append(imageData, base64.StdEncoding.EncodeToString(img.Data))
		}
		lsReq.Prompt = llamaServerMultimodalPrompt{
			PromptString:   promptStr,
			MultimodalData: imageData,
		}
	}

	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return fmt.Errorf("failed to marshal completion request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating completion request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		slog.Error("llama-server completion error", "error", err)
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check ollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llama-server error response: %w", err)
		}

		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(bodyBytes))}
	}

	// Parse SSE stream from llama-server
	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(buf, maxBufferSize)

	var lastToken string
	var tokenRepeat int

	for scanner.Scan() {
		select {
		case <-ctx.Done():
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

			var lsResp llamaServerCompletionResponse
			if err := json.Unmarshal(evt, &lsResp); err != nil {
				return fmt.Errorf("error unmarshalling llama-server response: %v", err)
			}

			// Token repeat detection
			switch {
			case strings.TrimSpace(lsResp.Content) == lastToken:
				tokenRepeat++
			default:
				lastToken = strings.TrimSpace(lsResp.Content)
				tokenRepeat = 0
			}
			if tokenRepeat > 30 {
				slog.Debug("prediction aborted, token repeat limit reached")
				return ctx.Err()
			}

			if lsResp.Content != "" && !lsResp.Stop {
				resp := CompletionResponse{
					Content: lsResp.Content,
				}
				resp.Logprobs = convertLogprobs(lsResp.CompletionProbabilities, req.TopLogprobs > 0)
				fn(resp)
			}

			if lsResp.Stop {
				doneReason := DoneReasonStop
				if lsResp.StopType == "limit" {
					doneReason = DoneReasonLength
				}

				fn(CompletionResponse{
					Content:            lsResp.Content,
					Done:               true,
					DoneReason:         doneReason,
					PromptEvalCount:    lsResp.Timings.PromptN,
					PromptEvalDuration: time.Duration(lsResp.Timings.PromptMS * float64(time.Millisecond)),
					EvalCount:          lsResp.Timings.PredictN,
					EvalDuration:       time.Duration(lsResp.Timings.PredictMS * float64(time.Millisecond)),
				})
				return nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			var msg string
			if s.status != nil && s.status.LastError() != "" {
				msg = s.status.LastError()
			} else {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}
		return fmt.Errorf("error reading llama-server response: %v", err)
	}

	return nil
}

// convertLogprobs converts llama-server's completion_probabilities to Ollama's Logprob format.
// includeTop controls whether top alternatives are included in the output.
func convertLogprobs(probs []llamaServerTokenProb, includeTop bool) []Logprob {
	if len(probs) == 0 {
		return nil
	}
	result := make([]Logprob, len(probs))
	for i, p := range probs {
		// llama-server uses "logprob" for log-probs mode, "prob" for sampling-probs mode
		logprob := p.Logprob
		if logprob == 0 && p.Prob != 0 {
			logprob = p.Prob // Use whichever is set
		}
		result[i] = Logprob{
			TokenLogprob: TokenLogprob{
				Token:   p.Token,
				Logprob: logprob,
			},
		}

		if !includeTop {
			continue
		}

		// Convert top logprobs (could be top_logprobs or top_probs depending on mode)
		topProbs := p.TopLogprobs
		if len(topProbs) == 0 {
			topProbs = p.TopProbs
		}
		for _, tp := range topProbs {
			tl := tp.Logprob
			if tl == 0 && tp.Prob != 0 {
				tl = tp.Prob
			}
			result[i].TopLogprobs = append(result[i].TopLogprobs, TokenLogprob{
				Token:   tp.Token,
				Logprob: tl,
			})
		}
	}
	return result
}

func (s *llamaServerRunner) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return nil, 0, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return nil, 0, err
	} else if status != ServerStatusReady {
		return nil, 0, fmt.Errorf("unexpected server status: %s", status)
	}

	// Use "input" field (not "content") to get the OAI-compatible response format
	// which includes tokens_evaluated for prompt token counting
	data, err := json.Marshal(map[string]string{"input": input})
	if err != nil {
		return nil, 0, fmt.Errorf("error marshaling embed data: %w", err)
	}

	// Use /v1/embeddings (OAI-compatible) to get tokens_evaluated in the response
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/embeddings", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, 0, fmt.Errorf("error creating embed request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, 0, fmt.Errorf("do embedding request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("error reading embed response: %w", err)
	}

	if resp.StatusCode >= 400 {
		statusCode, errMsg := normalizeEmbeddingError(resp.StatusCode, body)
		return nil, 0, api.StatusError{StatusCode: statusCode, ErrorMessage: errMsg}
	}

	// With "input" field, llama-server returns OAI-compatible format:
	//   {"data": [{"embedding": [0.1, ...], "tokens_evaluated": N}], "usage": {"prompt_tokens": N}}
	// With "content" field, it returns:
	//   [{"embedding": [[0.1, ...]], "index": 0}]
	var oaiResp struct {
		Data []struct {
			Embedding       json.RawMessage `json:"embedding"`
			TokensEvaluated int             `json:"tokens_evaluated"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &oaiResp); err == nil && len(oaiResp.Data) > 0 {
		var embedding []float32
		if err := json.Unmarshal(oaiResp.Data[0].Embedding, &embedding); err != nil {
			return nil, 0, fmt.Errorf("unmarshal embedding values: %w", err)
		}
		promptTokens := oaiResp.Usage.PromptTokens
		if promptTokens == 0 {
			promptTokens = oaiResp.Data[0].TokensEvaluated
		}
		return embedding, promptTokens, nil
	}

	// Fallback: non-OAI array format [{"embedding": [[0.1, ...]], "index": 0}]
	var results []struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(body, &results); err != nil {
		return nil, 0, fmt.Errorf("unmarshal embedding response: %w", err)
	}
	if len(results) == 0 {
		return nil, 0, fmt.Errorf("empty embedding response")
	}

	var embedding []float32
	if err := json.Unmarshal(results[0].Embedding, &embedding); err != nil {
		var nested [][]float32
		if err2 := json.Unmarshal(results[0].Embedding, &nested); err2 != nil {
			return nil, 0, fmt.Errorf("unmarshal embedding values: %w (also tried nested: %w)", err, err2)
		}
		if len(nested) > 0 {
			embedding = nested[0]
		}
	}

	return embedding, 0, nil
}

func normalizeEmbeddingError(statusCode int, body []byte) (int, string) {
	raw := strings.TrimSpace(string(body))
	errMsg := extractLlamaServerErrorMessage(body)
	if errMsg == "" {
		errMsg = raw
	}

	if isEmbeddingInputLimitError(errMsg) || isEmbeddingInputLimitError(raw) {
		return http.StatusBadRequest, "the input length exceeds the context length"
	}

	return statusCode, errMsg
}

func extractLlamaServerErrorMessage(body []byte) string {
	var resp struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Error) == 0 {
		return ""
	}

	var msg string
	if err := json.Unmarshal(resp.Error, &msg); err == nil {
		return strings.TrimSpace(msg)
	}

	var nested struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(resp.Error, &nested); err == nil {
		return strings.TrimSpace(nested.Message)
	}

	return ""
}

func isEmbeddingInputLimitError(errMsg string) bool {
	msg := strings.ToLower(errMsg)
	return strings.Contains(msg, "too large") ||
		strings.Contains(msg, "context size") ||
		strings.Contains(msg, "context length") ||
		strings.Contains(msg, "physical batch size") ||
		strings.Contains(msg, "exceeds the available context")
}

// Tokenize calls llama-server's /tokenize endpoint.
func (s *llamaServerRunner) Tokenize(ctx context.Context, content string) ([]int, error) {
	data, err := json.Marshal(map[string]string{"content": content})
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/tokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("tokenize error: %s", body)
	}

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return result.Tokens, nil
}

// Detokenize calls llama-server's /detokenize endpoint.
func (s *llamaServerRunner) Detokenize(ctx context.Context, tokens []int) (string, error) {
	data, err := json.Marshal(map[string][]int{"tokens": tokens})
	if err != nil {
		return "", err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/detokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("detokenize error: %s", body)
	}

	var result struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	return result.Content, nil
}

func (s *llamaServerRunner) Close() error {
	if s.cmd != nil {
		slog.Debug("stopping llama-server", "pid", s.Pid())
		if err := s.cmd.Process.Kill(); err != nil {
			return err
		}
		if s.cmd.ProcessState == nil {
			slog.Debug("waiting for llama-server to exit", "pid", s.Pid())
			<-s.done
		}
		slog.Debug("llama-server stopped", "pid", s.Pid())
	}
	return nil
}

// GetDeviceInfos returns device info for GPUs used by this runner, with FreeMemory
// updated to reflect actual usage. Uses the minimum of:
//   - Our accounting: TotalMemory minus tracked VRAM allocations
//   - System-reported: free VRAM from llama-server at load time minus our allocations
//
// The min-of-two approach handles both our own usage (accurate) and external
// consumers (system-reported, may be optimistic on some platforms).
func (s *llamaServerRunner) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if len(s.gpus) == 0 {
		return nil
	}
	infos := make([]ml.DeviceInfo, len(s.gpus))
	for i, gpu := range s.gpus {
		infos[i] = gpu
		used := s.vramByDevice[gpu.Name]

		// Our accounting: total minus what we allocated
		var accountedFree uint64
		if used < gpu.TotalMemory {
			accountedFree = gpu.TotalMemory - used
		}

		// System-reported: what the GPU said was free at load time, minus what
		// we've allocated since. This captures external consumers on platforms
		// where the driver reports accurately.
		systemFree := accountedFree // default to our accounting
		if sysFree, ok := s.systemFreeAtLoad[gpu.Name]; ok {
			if used < sysFree {
				systemFree = sysFree - used
			} else {
				systemFree = 0
			}
		}

		// Take the minimum — never optimistic
		infos[i].FreeMemory = min(accountedFree, systemFree)
	}
	return infos
}

// MemorySize returns total and GPU memory usage parsed from llama-server's
// post-load log output (e.g., "Metal model buffer size = 1234.56 MiB").
// Falls back to model file size if the log hasn't been parsed yet.
func (s *llamaServerRunner) MemorySize() (total, vram uint64) {
	if s.memTotal > 0 {
		return s.memTotal, s.memGPU
	}
	// Fallback: use model file size as a rough proxy
	slog.Debug("llama-server buffer sizes not available, falling back to file size estimate", "model", s.modelPath)
	if info, err := os.Stat(s.modelPath); err == nil {
		total = uint64(info.Size())
		vram = total
	}
	return total, vram
}

// FullyOffloaded returns true if all model layers are on GPU.
func (s *llamaServerRunner) FullyOffloaded() bool {
	return s.offloadedTotal > 0 && s.offloadedLayers == s.offloadedTotal
}

// PredictServerVRAM estimates VRAM usage for a model without spawning llama-server.
// Uses model file size as a proxy for weights plus a rough KV cache estimate.
// This is intentionally conservative — it overestimates to avoid VRAM contention.
func PredictServerVRAM(modelPath string, f *ggml.GGML, numCtx int) uint64 {
	var weights uint64
	if info, err := os.Stat(modelPath); err == nil {
		weights = uint64(info.Size())
	}

	// KV cache: 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (f16)
	layers := f.KV().BlockCount()
	kvHeads := f.KV().HeadCountKVMin()
	if kvHeads == 0 {
		kvHeads = 1
	}
	headDim := uint64(0)
	if f.KV().HeadCountMax() > 0 {
		headDim = f.KV().EmbeddingLength() / f.KV().HeadCountMax()
	}
	kvCache := 2 * layers * kvHeads * headDim * uint64(numCtx) * 2

	return weights + kvCache
}

// memoryParsingWriter wraps an io.Writer and parses llama-server log output
// for buffer size lines. It updates the runner's per-device VRAM tracking.
//
// Parsed line formats (all backends):
//
//	CUDA0 model buffer size =   852.89 MiB
//	CUDA0 KV buffer size =  1920.00 MiB
//	CUDA0 compute buffer size =   378.04 MiB
//	CPU_Mapped model buffer size =   308.23 MiB
//	CUDA_Host compute buffer size =   268.05 MiB
//	MTL0_Mapped model buffer size =  1918.35 MiB
//	ROCm0 model buffer size =  1918.35 MiB
type memoryParsingWriter struct {
	inner  io.Writer
	runner *llamaServerRunner
}

// offloadRegex matches: "offloaded 29/29 layers to GPU"
var offloadRegex = regexp.MustCompile(`offloaded (\d+)/(\d+) layers to GPU`)

// deviceFreeRegex matches per-device free VRAM reported at model load time:
//
//	using device CUDA0 (NVIDIA GeForce RTX 4060 Ti) (0000:01:00.0) - 15221 MiB free
//	using device MTL0 (Apple M5 Max) (unknown id) - 110100 MiB free
//	using device ROCm0 (AMD Radeon RX 6800) (0000:06:00.0) - 16196 MiB free
var deviceFreeRegex = regexp.MustCompile(`using device (\S+)\s+\(.*\)\s+-\s+(\d+)\s+MiB free`)

// bufferSizeRegex matches all buffer size lines from llama-server:
// model buffers, KV cache buffers, compute buffers, and output buffers.
var bufferSizeRegex = regexp.MustCompile(`(\S+)\s+(?:model |KV |compute |output )?buffer size\s*=\s*([\d.]+)\s*MiB`)

// isGPUBuffer returns true if the backend buffer name represents GPU memory.
// CPU, BLAS, and host-pinned buffers (*_Host) are not GPU memory.
// Device-mapped buffers (e.g., MTL0_Mapped) ARE GPU memory — they're model
// weights in device-accessible memory. Only CPU_Mapped is CPU memory.
func isGPUBuffer(name string) bool {
	if name == "CPU" || name == "BLAS" || strings.HasPrefix(name, "CPU_") {
		return false
	}
	if strings.HasSuffix(name, "_Host") {
		return false
	}
	return true
}

// deviceName returns the base device name for per-device VRAM tracking.
// Strips suffixes like _Mapped, _REPACK so that e.g. "MTL0_Mapped" is
// tracked under "MTL0" alongside "MTL0 KV buffer" and "MTL0 compute buffer".
func deviceName(backendName string) string {
	for _, suffix := range []string{"_Mapped", "_REPACK", "_Private"} {
		if strings.HasSuffix(backendName, suffix) {
			return strings.TrimSuffix(backendName, suffix)
		}
	}
	return backendName
}

func (w *memoryParsingWriter) Write(b []byte) (int, error) {
	if w.runner != nil {
		if match := offloadRegex.FindSubmatch(b); match != nil {
			w.runner.offloadedLayers, _ = strconv.Atoi(string(match[1]))
			w.runner.offloadedTotal, _ = strconv.Atoi(string(match[2]))
		}
		if match := deviceFreeRegex.FindSubmatch(b); match != nil {
			devName := string(match[1])
			if mib, err := strconv.ParseUint(string(match[2]), 10, 64); err == nil {
				w.runner.systemFreeAtLoad[devName] = mib * 1024 * 1024
			}
		}
		for _, match := range bufferSizeRegex.FindAllSubmatch(b, -1) {
			backendName := string(match[1])
			if mib, err := strconv.ParseFloat(string(match[2]), 64); err == nil {
				bytes := uint64(mib * 1024 * 1024)
				w.runner.memTotal += bytes
				if isGPUBuffer(backendName) {
					w.runner.memGPU += bytes
					w.runner.vramByDevice[deviceName(backendName)] += bytes
				}
			}
		}
	}
	return w.inner.Write(b)
}

// VRAMByGPU returns the VRAM used by this runner on the specified device.
// The values are parsed from llama-server's buffer size log output during model load
// (model tensors + KV cache + compute buffers).
func (s *llamaServerRunner) VRAMByGPU(id ml.DeviceID) uint64 {
	// Map DeviceID to the log device name used by llama-server.
	// Discovery stores the device name (e.g., "CUDA0", "ROCm0", "MTL0") from
	// --list-devices stdout, which matches the buffer log prefix.
	for _, gpu := range s.gpus {
		if gpu.DeviceID == id {
			return s.vramByDevice[gpu.Name]
		}
	}
	return 0
}
