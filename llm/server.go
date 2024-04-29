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
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
)

type LlamaServer interface {
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Embedding(ctx context.Context, prompt string) ([]float64, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	EstimatedVRAM() uint64
}

// llmServer is an instance of the llama.cpp server
type llmServer struct {
	port    int
	cmd     *exec.Cmd
	done    chan error // Channel to signal when the process exits
	status  *StatusWriter
	options api.Options

	// TODO - this should be broken down by GPU
	estimatedVRAM uint64 // Estimated usage of VRAM by the loaded model

	sem *semaphore.Weighted
}

func LoadModel(model string) (*GGML, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, _, err := DecodeGGML(f)
	return ggml, err
}

// NewLlamaServer will run a server for the given GPUs
// The gpu list must be a single family.
func NewLlamaServer(gpus gpu.GpuInfoList, model string, ggml *GGML, adapters, projectors []string, opts api.Options) (LlamaServer, error) {
	var err error
	if opts.NumCtx > int(ggml.KV().ContextLength()) {
		slog.Warn("requested context length is greater than the model's training context window size", "requested", opts.NumCtx, "training size", ggml.KV().ContextLength())
	}

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	cpuRunner := ""
	var estimatedVRAM uint64
	var systemMemory uint64
	if (len(gpus) == 1 && gpus[0].Library == "cpu") || opts.NumGPU == 0 {

		// TODO evaluate system memory to see if we should block the load, or force an unload of another CPU runner

		cpuRunner = serverForCpu()
	} else {
		if gpus[0].Library == "metal" {
			memInfo, err := gpu.GetCPUMem()
			if err != nil {
				slog.Error("failed to lookup system memory", "error", err)
			} else {
				systemMemory = memInfo.TotalMemory
				slog.Debug("system memory", "total", format.HumanBytes2(systemMemory))
			}
		}
		var layers int
		layers, estimatedVRAM = EstimateGPULayers(gpus, ggml, projectors, opts)

		if gpus[0].Library == "metal" && estimatedVRAM > systemMemory {
			// disable partial offloading when model is greater than total system memory as this
			// can lead to locking up the system
			opts.NumGPU = 0
		} else if opts.NumGPU < 0 && layers > 0 && gpus[0].Library != "cpu" {
			opts.NumGPU = layers
		}
	}

	// Loop through potential servers
	finalErr := fmt.Errorf("no suitable llama servers found")

	if len(adapters) > 1 {
		return nil, errors.New("ollama supports only one lora adapter, but multiple were provided")
	}

	availableServers := availableServers()
	var servers []string
	if cpuRunner != "" {
		servers = []string{cpuRunner}
	} else {
		servers = serversForGpu(gpus[0]) // All GPUs in the list are matching Library and Variant
	}
	demandLib := strings.Trim(os.Getenv("OLLAMA_LLM_LIBRARY"), "\"' ")
	if demandLib != "" {
		serverPath := availableServers[demandLib]
		if serverPath == "" {
			slog.Info(fmt.Sprintf("Invalid OLLAMA_LLM_LIBRARY %s - not found", demandLib))
		} else {
			slog.Info("user override", "OLLAMA_LLM_LIBRARY", demandLib, "path", serverPath)
			servers = []string{demandLib}
		}
	}

	if len(servers) == 0 {
		return nil, fmt.Errorf("no servers found for %v", gpus)
	}

	params := []string{
		"--model", model,
		"--ctx-size", fmt.Sprintf("%d", opts.NumCtx),
		"--batch-size", fmt.Sprintf("%d", opts.NumBatch),
		"--embedding",
	}
	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		params = append(params, "--log-format", "json")
	} else {
		params = append(params, "--log-disable")
	}

	if opts.NumGPU >= 0 {
		params = append(params, "--n-gpu-layers", fmt.Sprintf("%d", opts.NumGPU))
	}

	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		params = append(params, "--verbose")
	}

	if opts.MainGPU > 0 {
		params = append(params, "--main-gpu", fmt.Sprintf("%d", opts.MainGPU))
	}

	if len(adapters) > 0 {
		// TODO: applying multiple adapters is not supported by the llama.cpp server yet
		params = append(params, "--lora", adapters[0])
	}

	if len(projectors) > 0 {
		// TODO: applying multiple projectors is not supported by the llama.cpp server yet
		params = append(params, "--mmproj", projectors[0])
	}

	if opts.NumThread > 0 {
		params = append(params, "--threads", fmt.Sprintf("%d", opts.NumThread))
	}

	if !opts.F16KV {
		params = append(params, "--memory-f32")
	}

	if opts.UseMLock {
		params = append(params, "--mlock")
	}

	if !opts.UseMMap {
		params = append(params, "--no-mmap")
	}

	if opts.UseNUMA {
		params = append(params, "--numa")
	}

	// "--cont-batching", // TODO - doesn't seem to have any noticeable perf change for multiple requests
	numParallel := 1
	if onp := os.Getenv("OLLAMA_NUM_PARALLEL"); onp != "" {
		numParallel, err = strconv.Atoi(onp)
		if err != nil || numParallel <= 0 {
			err = fmt.Errorf("invalid OLLAMA_NUM_PARALLEL=%s must be greater than zero - %w", onp, err)
			slog.Error("misconfiguration", "error", err)
			return nil, err
		}
	}
	params = append(params, "--parallel", fmt.Sprintf("%d", numParallel))

	for i := 0; i < len(servers); i++ {
		dir := availableServers[servers[i]]
		if dir == "" {
			// Shouldn't happen
			finalErr = fmt.Errorf("[%d] server %s not listed in available servers %v", i, servers[i], availableServers)
			slog.Error("sever list inconsistent", "error", finalErr)
			continue
		}

		// Find an availableServers  port, retry on each iterration in case the failure was a port conflict race
		port := 0
		if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
			var l *net.TCPListener
			if l, err = net.ListenTCP("tcp", a); err == nil {
				port = l.Addr().(*net.TCPAddr).Port
				l.Close()
			}
		}
		if port == 0 {
			slog.Debug("ResolveTCPAddr failed ", "error", err)
			port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
		}
		finalParams := append(params, "--port", strconv.Itoa(port))

		pathEnv := "LD_LIBRARY_PATH"
		if runtime.GOOS == "windows" {
			pathEnv = "PATH"
		}
		// append the server directory to LD_LIBRARY_PATH/PATH
		libraryPaths := []string{dir}

		if libraryPath, ok := os.LookupEnv(pathEnv); ok {
			// Append our runner directory to the path
			// This will favor system libraries over our bundled library dependencies
			libraryPaths = append(filepath.SplitList(libraryPath), libraryPaths...)
		}

		// Note: we always put the dependency path first
		// since this was the exact version we verified for AMD GPUs
		// and we favor what the user had in their path
		if gpus[0].DependencyPath != "" {
			// TODO refine for multi-gpu support
			libraryPaths = append([]string{gpus[0].DependencyPath}, libraryPaths...)
		}

		server := filepath.Join(dir, "ollama_llama_server")
		if runtime.GOOS == "windows" {
			server = server + ".exe"
		}

		// Detect tmp cleaners wiping out the file
		_, err := os.Stat(server)
		if errors.Is(err, os.ErrNotExist) {
			slog.Warn("llama server disappeared, reinitializing payloads", "path", server, "error", err)
			err = Init()
			if err != nil {
				slog.Warn("failed to reinitialize payloads", "error", err)
				return nil, err
			}
		}

		s := &llmServer{
			port:          port,
			cmd:           exec.Command(server, finalParams...),
			status:        NewStatusWriter(os.Stderr),
			options:       opts,
			estimatedVRAM: estimatedVRAM,
			sem:           semaphore.NewWeighted(int64(numParallel)),
		}

		libEnv := fmt.Sprintf("%s=%s", pathEnv, strings.Join(libraryPaths, string(filepath.ListSeparator)))
		s.cmd.Env = append(os.Environ(), libEnv)
		s.cmd.Stdout = os.Stdout
		s.cmd.Stderr = s.status

		// TODO - multiple GPU selection logic...
		key, val := gpu.GpuInfoList(gpus).GetVisibleDevicesEnv()
		if key != "" {
			s.cmd.Env = append(s.cmd.Env, key+"="+val)
		}

		slog.Info("starting llama server", "cmd", s.cmd.String())
		// Log at debug as the environment is inherited and might contain sensitive information
		slog.Debug("subprocess", "environment", s.cmd.Env)

		if err = s.cmd.Start(); err != nil {
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			err = fmt.Errorf("error starting the external llama server: %v %s", err, msg)
			finalErr = err
			continue
		}

		// reap subprocess when it exits
		go func() {
			// Exit status managed via getServerStatus
			_ = s.cmd.Wait()
		}()

		// TODO - make sure this is all wired up correctly
		// if err = s.WaitUntilRunning(); err != nil {
		// 	slog.Error("error starting llama server", "server", servers[i], "error", err)
		// 	s.Close()
		// 	finalErr = err
		// 	continue
		// }
		return s, nil
	}

	slog.Error("unable to load any llama server", "error", finalErr)
	return nil, finalErr
}

func projectorMemoryRequirements(filename string) uint64 {
	file, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer file.Close()

	ggml, _, err := DecodeGGML(file)
	if err != nil {
		return 0
	}

	var mem uint64
	for _, layer := range ggml.Tensors().Layers() {
		mem += layer.size()
	}

	return mem
}

type ServerStatus int

const ( // iota is reset to 0
	ServerStatusReady ServerStatus = iota
	ServerStatusNoSlotsAvaialble
	ServerStatusLoadingModel
	ServerStatusNotResponding
	ServerStatusError
)

func (s ServerStatus) ToString() string {
	switch s {
	case ServerStatusReady:
		return "llm server ready"
	case ServerStatusNoSlotsAvaialble:
		return "llm busy - no slots available"
	case ServerStatusLoadingModel:
		return "llm server loading model"
	case ServerStatusNotResponding:
		return "llm server not responding"
	default:
		return "llm server error"
	}
}

type ServerStatusResp struct {
	Status          string `json:"status"`
	SlotsIdle       int    `json:"slots_idle"`
	SlotsProcessing int    `json:"slots_processing"`
	Error           string `json:"error"`
}

func (s *llmServer) getServerStatus(ctx context.Context) (ServerStatus, error) {
	// Fail fast if its exited
	if s.cmd.ProcessState != nil {
		msg := ""
		if s.status != nil && s.status.LastErrMsg != "" {
			msg = s.status.LastErrMsg
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
			return ServerStatusNotResponding, fmt.Errorf("server not responding")
		}
		return ServerStatusError, fmt.Errorf("health resp: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServerStatusError, fmt.Errorf("read health request: %w", err)
	}

	var status ServerStatusResp
	if err := json.Unmarshal(body, &status); err != nil {
		return ServerStatusError, fmt.Errorf("health unmarshal encode response: %w", err)
	}

	switch status.Status {
	case "ok":
		return ServerStatusReady, nil
	case "no slot available":
		return ServerStatusNoSlotsAvaialble, nil
	case "loading model":
		return ServerStatusLoadingModel, nil
	default:
		return ServerStatusError, fmt.Errorf("server error: %+v", status)
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
	// TODO we need to wire up a better way to detect hangs during model load and startup of the server
	expiresAt := time.Now().Add(10 * time.Minute) // be generous with timeout, large models can take a while to load
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	slog.Info("waiting for llama runner to start responding")
	var lastStatus ServerStatus = -1
	for {
		select {
		case <-ctx.Done():
			slog.Info("context expired before server started")
			return fmt.Errorf("timed out waiting for llama runner to start: %w", ctx.Err())
		case err := <-s.done:
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			return fmt.Errorf("llama runner process has terminated: %v %s", err, msg)
		case <-ticker.C:
			if time.Now().After(expiresAt) {
				// timeout
				msg := ""
				if s.status != nil && s.status.LastErrMsg != "" {
					msg = s.status.LastErrMsg
				}
				return fmt.Errorf("timed out waiting for llama runner to start: %s", msg)
			}
			if s.cmd.ProcessState != nil {
				msg := ""
				if s.status != nil && s.status.LastErrMsg != "" {
					msg = s.status.LastErrMsg
				}
				return fmt.Errorf("llama runner process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
			}

			c, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
			defer cancel()
			status, err := s.getServerStatus(c)
			if err != nil && lastStatus != status {
				slog.Debug("server not yet available", "error", err)
				lastStatus = status
				continue
			}

			switch status {
			case ServerStatusLoadingModel:
				// TODO - this state never seems to happen with the current server.cpp code (bug?)
				// it doesn't respond to the health endpoint until after the model is loaded
				slog.Debug("loading model")
			case ServerStatusReady:
				slog.Debug(fmt.Sprintf("llama runner started in %f seconds", time.Since(start).Seconds()))
				return nil
			}
		}
	}
}

const jsonGrammar = `
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
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

const maxBufferSize = 512 * format.KiloByte
const maxRetries = 3

type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

type completion struct {
	Content string `json:"content"`
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Stop    bool   `json:"stop"`

	Timings struct {
		PredictedN  int     `json:"predicted_n"`
		PredictedMS float64 `json:"predicted_ms"`
		PromptN     int     `json:"prompt_n"`
		PromptMS    float64 `json:"prompt_ms"`
	}
}

type CompletionRequest struct {
	Prompt  string
	Format  string
	Images  []ImageData
	Options api.Options
}

type CompletionResponse struct {
	Content            string
	Done               bool
	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration
}

func (s *llmServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		slog.Error("Failed to acquire semaphore", "error", err)
		return err
	}
	defer s.sem.Release(1)

	// only allow maximum 10 "context shifts" to avoid infinite generation
	if req.Options.NumPredict < 0 || req.Options.NumPredict > 10*s.options.NumCtx {
		req.Options.NumPredict = 10 * s.options.NumCtx
		slog.Debug("setting token limit to 10x num_ctx", "num_ctx", s.options.NumCtx, "num_predict", req.Options.NumPredict)
	}

	request := map[string]any{
		"prompt":            req.Prompt,
		"stream":            true,
		"n_predict":         req.Options.NumPredict,
		"n_keep":            req.Options.NumKeep,
		"main_gpu":          req.Options.MainGPU,
		"temperature":       req.Options.Temperature,
		"top_k":             req.Options.TopK,
		"top_p":             req.Options.TopP,
		"tfs_z":             req.Options.TFSZ,
		"typical_p":         req.Options.TypicalP,
		"repeat_last_n":     req.Options.RepeatLastN,
		"repeat_penalty":    req.Options.RepeatPenalty,
		"presence_penalty":  req.Options.PresencePenalty,
		"frequency_penalty": req.Options.FrequencyPenalty,
		"mirostat":          req.Options.Mirostat,
		"mirostat_tau":      req.Options.MirostatTau,
		"mirostat_eta":      req.Options.MirostatEta,
		"penalize_nl":       req.Options.PenalizeNewline,
		"seed":              req.Options.Seed,
		"stop":              req.Options.Stop,
		"image_data":        req.Images,
		"cache_prompt":      true,
	}

	// Make sure the server is ready
	status, err := s.getServerStatus(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status.ToString())
	}

	if req.Format == "json" {
		request["grammar"] = jsonGrammar
		if !strings.Contains(strings.ToLower(req.Prompt), "json") {
			slog.Warn("Prompt does not specify that the LLM should response in JSON, but JSON format is expected. For best results specify that JSON is expected in the system prompt.")
		}
	}

	retryDelay := 100 * time.Microsecond
	for retries := 0; retries < maxRetries; retries++ {
		if retries > 0 {
			time.Sleep(retryDelay) // wait before retrying
			retryDelay *= 2        // exponential backoff
		}

		// Handling JSON marshaling with special characters unescaped.
		buffer := &bytes.Buffer{}
		enc := json.NewEncoder(buffer)
		enc.SetEscapeHTML(false)

		if err := enc.Encode(request); err != nil {
			return fmt.Errorf("failed to marshal data: %v", err)
		}

		endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
		if err != nil {
			return fmt.Errorf("error creating POST request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return fmt.Errorf("POST predict: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode >= 400 {
			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				return fmt.Errorf("failed reading llm error response: %w", err)
			}
			log.Printf("llm predict error: %s", bodyBytes)
			return fmt.Errorf("%s", bodyBytes)
		}

		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, maxBufferSize)
		scanner.Buffer(buf, maxBufferSize)

		retryNeeded := false
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

				// try again on slot unavailable
				if bytes.Contains(line, []byte("slot unavailable")) {
					retryNeeded = true
					break
				}

				evt, ok := bytes.CutPrefix(line, []byte("data: "))
				if !ok {
					return fmt.Errorf("error parsing llm response stream: %s", line)
				}

				var c completion
				if err := json.Unmarshal(evt, &c); err != nil {
					return fmt.Errorf("error unmarshaling llm prediction response: %v", err)
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

				if c.Stop {
					fn(CompletionResponse{
						Done:               true,
						PromptEvalCount:    c.Timings.PromptN,
						PromptEvalDuration: parseDurationMs(c.Timings.PromptMS),
						EvalCount:          c.Timings.PredictedN,
						EvalDuration:       parseDurationMs(c.Timings.PredictedMS),
					})
					return nil
				}
			}
		}

		if err := scanner.Err(); err != nil {
			if strings.Contains(err.Error(), "unexpected EOF") {
				s.Close()
				msg := ""
				if s.status != nil && s.status.LastErrMsg != "" {
					msg = s.status.LastErrMsg
				}

				return fmt.Errorf("an unknown error was encountered while running the model %s", msg)
			}
			return fmt.Errorf("error reading llm response: %v", err)
		}

		if !retryNeeded {
			return nil // success
		}
	}

	// should never reach here ideally
	return fmt.Errorf("max retries exceeded")
}

type EmbeddingRequest struct {
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (s *llmServer) Embedding(ctx context.Context, prompt string) ([]float64, error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		slog.Error("Failed to acquire semaphore", "error", err)
		return nil, err
	}
	defer s.sem.Release(1)
	// Make sure the server is ready
	status, err := s.getServerStatus(ctx)
	if err != nil {
		return nil, err
	} else if status != ServerStatusReady {
		return nil, fmt.Errorf("unexpected server status: %s", status.ToString())
	}

	data, err := json.Marshal(TokenizeRequest{Content: prompt})
	if err != nil {
		return nil, fmt.Errorf("error marshaling embed data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/embedding", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("error creating embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do embedding request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading embed response: %w", err)
	}

	if resp.StatusCode >= 400 {
		log.Printf("llm encode error: %s", body)
		return nil, fmt.Errorf("%s", body)
	}

	var embedding EmbeddingResponse
	if err := json.Unmarshal(body, &embedding); err != nil {
		return nil, fmt.Errorf("unmarshal tokenize response: %w", err)
	}

	return embedding.Embedding, nil
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func (s *llmServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	// Make sure the server is ready
	status, err := s.getServerStatus(ctx)
	if err != nil {
		return nil, err
	} else if status != ServerStatusReady && status != ServerStatusNoSlotsAvaialble {
		return nil, fmt.Errorf("unexpected server status: %s", status.ToString())
	}

	data, err := json.Marshal(TokenizeRequest{Content: content})
	if err != nil {
		return nil, fmt.Errorf("marshaling encode data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/tokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do encode request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read encode request: %w", err)
	}

	if resp.StatusCode >= 400 {
		log.Printf("llm encode error: %s", body)
		return nil, fmt.Errorf("%s", body)
	}

	var encoded TokenizeResponse
	if err := json.Unmarshal(body, &encoded); err != nil {
		return nil, fmt.Errorf("unmarshal encode response: %w", err)
	}

	return encoded.Tokens, nil
}

type DetokenizeRequest struct {
	Tokens []int `json:"tokens"`
}

type DetokenizeResponse struct {
	Content string `json:"content"`
}

func (s *llmServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	// Make sure the server is ready
	status, err := s.getServerStatus(ctx)
	if err != nil {
		return "", err
	} else if status != ServerStatusReady && status != ServerStatusNoSlotsAvaialble {
		return "", fmt.Errorf("unexpected server status: %s", status.ToString())
	}

	data, err := json.Marshal(DetokenizeRequest{Tokens: tokens})
	if err != nil {
		return "", fmt.Errorf("marshaling decode data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/detokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return "", fmt.Errorf("decode request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("do decode request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read decode request: %w", err)
	}

	if resp.StatusCode >= 400 {
		log.Printf("llm decode error: %s", body)
		return "", fmt.Errorf("%s", body)
	}

	var decoded DetokenizeResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return "", fmt.Errorf("unmarshal encode response: %w", err)
	}

	return decoded.Content, nil
}

func (s *llmServer) Close() error {
	if s.cmd != nil {
		slog.Debug("stopping llama server")
		if err := s.cmd.Process.Kill(); err != nil {
			return err
		}

		_ = s.cmd.Wait()

		slog.Debug("llama server stopped")
	}

	return nil
}

func (s *llmServer) EstimatedVRAM() uint64 {
	return s.estimatedVRAM
}

func parseDurationMs(ms float64) time.Duration {
	dur, err := time.ParseDuration(fmt.Sprintf("%fms", ms))
	if err != nil {
		panic(err)
	}

	return dur
}
