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
	"strconv"
	"strings"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

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

// findLlamaServer locates the llama-server binary in the lib/ollama directory.
func findLlamaServer() (string, error) {
	exe, err := os.Executable()
	if err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
	}

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

	add(ml.LibOllamaPath)
	if exe != "" {
		add(filepath.Join(filepath.Dir(exe), "build", "lib", "ollama"))
	}
	if cwd, err := os.Getwd(); err == nil {
		add(filepath.Join(cwd, "build", "lib", "ollama"))
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("llama-server binary not found (checked: %s). Run 'cmake --build build --target llama-server' first", strings.Join(candidates, ", "))
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
	exe, err := findLlamaServer()
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

	// Embedding mode — set batch size to context size so large inputs fit
	if embedding {
		params = append(params, "--embedding")
		params = append(params, "-b", strconv.Itoa(opts.NumCtx*numParallel))
		params = append(params, "-ub", strconv.Itoa(opts.NumCtx*numParallel))
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

	libraryPaths := append([]string{}, gpuLibs...)
	// Also include the directory containing llama-server for backend discovery
	libraryPaths = append(libraryPaths, filepath.Dir(exe))
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
		ml.GetVisibleDevicesEnv(gpus, false),
		memWriter,
	)

	s := &llamaServerRunner{
		port:        port,
		cmd:         cmd,
		status:      status,
		options:     opts,
		modelPath:   modelPath,
		ggml:        f,
		totalLayers: f.KV().BlockCount() + 1,
		loadStart:   time.Now(),
		sem:         semaphore.NewWeighted(int64(numParallel)),
		done:        make(chan struct{}),
	}
	// Point the memory parsing writer at this runner so values are updated as logs stream in
	memWriter.runner = s

	if err != nil {
		var msg string
		if s.status != nil && s.status.LastErrMsg != "" {
			msg = s.status.LastErrMsg
		}
		return nil, fmt.Errorf("error starting llama-server: %v %s", err, msg)
	}

	// Reap subprocess when it exits
	go func() {
		err := s.cmd.Wait()
		if err != nil && s.status != nil && s.status.LastErrMsg != "" {
			slog.Error("llama-server terminated", "error", err)
			s.doneErr = errors.New(s.status.LastErrMsg)
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
		if s.status != nil && s.status.LastErrMsg != "" {
			msg = s.status.LastErrMsg
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
			return fmt.Errorf("llama-server process has terminated: %w", s.doneErr)
		default:
		}

		if time.Now().After(stallTimer) {
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			return fmt.Errorf("timed out waiting for llama-server to start - %s", msg)
		}

		if s.cmd.ProcessState != nil {
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
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

// llamaServerCompletionRequest is the request format for llama-server's POST /completion endpoint.
type llamaServerCompletionRequest struct {
	Prompt        string             `json:"prompt"`
	Stream        bool               `json:"stream"`
	CachePrompt   bool               `json:"cache_prompt"`
	NPredict      int                `json:"n_predict,omitempty"`
	Temperature   float32            `json:"temperature"`
	TopK          int                `json:"top_k"`
	TopP          float32            `json:"top_p"`
	MinP          float32            `json:"min_p"`
	Stop          []string           `json:"stop,omitempty"`
	RepeatPenalty float32            `json:"repeat_penalty"`
	FreqPenalty   float32            `json:"frequency_penalty"`
	PresPenalty   float32            `json:"presence_penalty"`
	Seed          int                `json:"seed"`
	Grammar       string             `json:"grammar,omitempty"`
	JsonSchema    json.RawMessage    `json:"json_schema,omitempty"`
	ImageData     []llamaServerImage `json:"image_data,omitempty"`
	NProbs        int                `json:"n_probs,omitempty"`
}

type llamaServerImage struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
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
		Prompt:        req.Prompt,
		Stream:        true,
		CachePrompt:   req.Shift,
		NPredict:      req.Options.NumPredict,
		Temperature:   req.Options.Temperature,
		TopK:          req.Options.TopK,
		TopP:          req.Options.TopP,
		MinP:          req.Options.MinP,
		Stop:          req.Options.Stop,
		RepeatPenalty: req.Options.RepeatPenalty,
		FreqPenalty:   req.Options.FrequencyPenalty,
		PresPenalty:   req.Options.PresencePenalty,
		Seed:          req.Options.Seed,
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

	// Convert images
	for _, img := range req.Images {
		lsReq.ImageData = append(lsReq.ImageData, llamaServerImage{
			Data: img.Data,
			ID:   img.ID,
		})
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
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
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
		errMsg := string(body)
		statusCode := resp.StatusCode
		// llama-server returns 500 for "input too large" but the embed handler
		// expects 400 to trigger truncation retry. Normalize the error.
		if strings.Contains(errMsg, "too large") || strings.Contains(errMsg, "context size") {
			statusCode = http.StatusBadRequest
			errMsg = "the input length exceeds the context length"
		}
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

func (s *llamaServerRunner) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	// llama-server doesn't expose device info via API
	return nil
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

// WillUseLlamaServer returns true if a model with the given GGUF metadata
// will be served by the llama-server runner (not the ollama engine).
func WillUseLlamaServer(f *ggml.GGML) bool {
	forceOff := os.Getenv("OLLAMA_NEW_ENGINE") == "0" || os.Getenv("OLLAMA_NEW_ENGINE") == "false"
	if forceOff {
		return true
	}
	return !envconfig.NewEngine() && !f.KV().OllamaEngineRequired()
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
// for buffer size lines like "Metal model buffer size = 1234.56 MiB".
// It updates the runner's memTotal and memGPU fields with actual allocations.
type memoryParsingWriter struct {
	inner  io.Writer
	runner *llamaServerRunner
}

// bufferSizeRegex matches: "BACKEND_NAME model buffer size = 1234.56 MiB"
// The backend name is right-padded to 12 chars in the format string, so we
// trim spaces. Examples:
//
//	Metal model buffer size =  1234.56 MiB
//	CUDA0 model buffer size =   567.89 MiB
//	CPU model buffer size =    56.78 MiB
//	CUDA_Host model buffer size =   123.45 MiB
var bufferSizeRegex = regexp.MustCompile(`(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB`)

// isGPUBuffer returns true if the backend buffer name represents GPU memory.
// CPU, BLAS, and host-pinned buffers (*_Host, *_Mapped) are not GPU memory.
// Metal_Private is GPU memory (private device buffers on Metal).
func isGPUBuffer(name string) bool {
	if name == "CPU" || name == "BLAS" {
		return false
	}
	if strings.HasSuffix(name, "_Host") || strings.HasSuffix(name, "_Mapped") {
		return false
	}
	return true
}

func (w *memoryParsingWriter) Write(b []byte) (int, error) {
	if w.runner != nil {
		for _, match := range bufferSizeRegex.FindAllSubmatch(b, -1) {
			backendName := string(match[1])
			if mib, err := strconv.ParseFloat(string(match[2]), 64); err == nil {
				bytes := uint64(mib * 1024 * 1024)
				w.runner.memTotal += bytes
				if isGPUBuffer(backendName) {
					w.runner.memGPU += bytes
				}
			}
		}
	}
	return w.inner.Write(b)
}

func (s *llamaServerRunner) VRAMByGPU(id ml.DeviceID) uint64 {
	// No per-GPU tracking — llama-server manages placement internally
	return 0
}

// IsLlamaServerRunner returns true if the given LlamaServer is backed by
// the upstream llama-server binary (as opposed to the ollama engine or MLX).
func IsLlamaServerRunner(s LlamaServer) bool {
	_, ok := s.(*llamaServerRunner)
	return ok
}

// NewStubLlamaServerRunner creates a minimal llamaServerRunner for testing.
// It satisfies the IsLlamaServerRunner check and has safe no-op behavior for Close.
func NewStubLlamaServerRunner(modelPath string) LlamaServer {
	return &llamaServerRunner{
		port:      -1,
		modelPath: modelPath,
		options:   api.Options{},
		done:      make(chan struct{}),
		sem:       semaphore.NewWeighted(1),
	}
}
