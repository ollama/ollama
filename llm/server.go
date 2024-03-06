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
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"github.com/jmorganca/ollama/gpu"
)

// LlamaServer is an instance of the llama.cpp server
type LlamaServer struct {
	port int
	cmd  *exec.Cmd
	done chan error // Channel to signal when the process exits
}

func NewLlamaServer(model string, adapters, projectors []string, opts api.Options) (*LlamaServer, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, err := DecodeGGML(f)
	if err != nil {
		return nil, err
	}

	if opts.NumCtx > int(ggml.NumCtx()) {
		slog.Warn(fmt.Sprintf("requested context length is greater than model's max context length (%d > %d), using %d instead", opts.NumCtx, ggml.NumCtx(), ggml.NumCtx()))
		opts.NumCtx = int(ggml.NumCtx())
	}

	if opts.NumCtx < 4 {
		opts.NumCtx = 4
	}

	vram, _ := gpu.CheckVRAM()
	size := ggml.Size

	// fp16 k,v matrices require = n_ctx * n_layer * n_embd / n_head * n_head_kv * 2 bytes each * 2 key and value
	kv := 2 * 2 * int64(opts.NumCtx) * int64(ggml.NumLayers()) * int64(ggml.NumEmbed()) * int64(ggml.NumHeadKv()) / int64(ggml.NumHead())

	// this amount is the overhead + tensors in memory
	// TODO: get this from the llama.cpp's graph calculations instead of
	// estimating it's 1/6 * kv_cache_size * num_gqa
	graph := int64(ggml.NumGQA()) * kv / 6

	info := gpu.GetGPUInfo()
	switch runtime.GOOS {
	case "darwin":
		if opts.NumGPU == 0 {
			break
		}

		if size+kv+graph > vram {
			slog.Info("not enough vram available, falling back to CPU only")
			info.Library = "cpu"
			info.Variant = gpu.GetCPUVariant()
			opts.NumGPU = 0
			break
		}

		// TODO: implement layer splitting on macOS after better memory estimations
		opts.NumGPU = 999
	default:
		if info.Library == "cpu" {
			slog.Info("GPU not available, falling back to CPU")
			opts.NumGPU = 0
			break
		}

		// don't use GPU at all if no layers are loaded
		if opts.NumGPU == 0 {
			info.Library = "cpu"
			info.Variant = gpu.GetCPUVariant()
			break
		}

		// user-defined GPU count
		if opts.NumGPU != -1 {
			break
		}

		// the "main" GPU needs the most memory and determines the limit
		// of how many layers can be loaded. It needs to fit:
		// 1. the full compute graph allocation for all devices (graph)
		// 2. the proportional kv cache for all devices (kv * % layers)
		// 3. the proportional model (size * % layers / # devices)
		// This estimates the number of layers
		maxlayers := int64(ggml.NumLayers()) + 1
		devices := int64(info.DeviceCount)
		avg := vram / devices
		layers := maxlayers * (avg - graph) / (kv + size/devices)
		if layers > maxlayers {
			layers = maxlayers
		}

		// 1 + 2 must fit on the main gpu
		min := graph + kv*layers/maxlayers
		if layers <= 0 || min > avg {
			slog.Info("not enough vram available, falling back to CPU only")
			info.Library = "cpu"
			info.Variant = gpu.GetCPUVariant()
			opts.NumGPU = 0
			break
		}

		opts.NumGPU = int(layers)
	}

	if len(adapters) > 1 {
		return nil, errors.New("ollama supports only one lora adapter, but multiple were provided")
	}

	available := available()
	servers := serversForGpu(info)

	if len(servers) == 0 {
		return nil, fmt.Errorf("no servers found for %v", info)
	}

	dir := available[servers[0]]

	// TODO: let user override with OLLAMA_LLM_LIBRARY
	slog.Info("using server", "server", servers[0])

	params := []string{
		"--model", model,
		"--ctx-size", fmt.Sprintf("%d", opts.NumCtx),
		"--batch-size", fmt.Sprintf("%d", opts.NumBatch),
		"--embedding",
		"--log-disable",
	}

	if opts.NumGPU > 0 {
		params = append(params, "--n-gpu-layers", fmt.Sprintf("%d", opts.NumGPU))
	}

	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		fmt.Println("adding verbose")
		params = append(params, "--verbose")
	}

	if opts.MainGPU > 0 {
		params = append(params, "--main-gpu", fmt.Sprintf("%d", opts.MainGPU))
	}

	if opts.RopeFrequencyBase > 0 {
		params = append(params, "--rope-freq-base", fmt.Sprintf("%f", opts.RopeFrequencyBase))
	}

	if opts.RopeFrequencyScale > 0 {
		params = append(params, "--rope-freq-scale", fmt.Sprintf("%f", opts.RopeFrequencyScale))
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

	port := rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
	params = append(params, "--port", strconv.Itoa(port))

	slog.Info("starting llama server", "params", params)

	// append the server directory to LD_LIBRARY_PATH
	var libraryPaths []string
	if libraryPath, ok := os.LookupEnv("LD_LIBRARY_PATH"); ok {
		libraryPaths = append(libraryPaths, libraryPath)
	}
	libraryPaths = append(libraryPaths, dir)

	server := filepath.Join(dir, "ollama_llama_server")
	if runtime.GOOS == "windows" {
		server = server + ".exe"
	}

	s := &LlamaServer{
		port: port,
		cmd:  exec.Command(server, params...),
	}

	s.cmd.Env = append(os.Environ(), fmt.Sprintf("LD_LIBRARY_PATH=%s", strings.Join(libraryPaths, ":")))
	s.cmd.Stdout = os.Stdout
	s.cmd.Stderr = os.Stderr

	slog.Info("starting llama", "cmd", s.cmd.String())

	if err := s.cmd.Start(); err != nil {
		return nil, fmt.Errorf("error starting the external llama server: %v", err)
	}

	go func() {
		s.done <- s.cmd.Wait() // This will capture the exit error
	}()

	if err := s.waitUntilRunning(); err != nil {
		log.Printf("error starting llama server: %v", err)
		s.Close()

		return nil, err
	}

	return s, nil
}

func (s *LlamaServer) Wait() <-chan error {
	return s.done
}

func (s *LlamaServer) ping(ctx context.Context) error {
	resp, err := http.Head(fmt.Sprintf("http://127.0.0.1:%d", s.port))
	if err != nil {
		return fmt.Errorf("ping resp: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected ping status: %s", resp.Status)
	}
	return nil
}

func (s *LlamaServer) waitUntilRunning() error {
	start := time.Now()
	expiresAt := time.Now().Add(3 * time.Minute) // be generous with timeout, large models can take a while to load
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	slog.Info("waiting for llama runner to start responding")
	for {
		select {
		case err := <-s.done:
			return fmt.Errorf("llama runner process has terminated: %v", err)
		case <-ticker.C:
			if time.Now().After(expiresAt) {
				// timeout
				return fmt.Errorf("timed out waiting for llama runner to start")
			}

			if err := s.ping(context.Background()); err == nil {
				// success
				log.Printf("llama runner started in %f seconds", time.Since(start).Seconds())
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

func (s *LlamaServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
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
	}

	if req.Format == "json" {
		request["grammar"] = jsonGrammar
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
				return fmt.Errorf("llama runner exited, you may not have enough available memory to run this model")
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

func (s *LlamaServer) Embedding(ctx context.Context, prompt string) ([]float64, error) {
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

func (s *LlamaServer) Tokenize(ctx context.Context, content string) ([]int, error) {
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

func (s *LlamaServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
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

func (s *LlamaServer) Close() error {
	if s.cmd != nil {
		return s.cmd.Process.Kill()
	}

	return nil
}

func parseDurationMs(ms float64) time.Duration {
	dur, err := time.ParseDuration(fmt.Sprintf("%fms", ms))
	if err != nil {
		panic(err)
	}

	return dur
}
