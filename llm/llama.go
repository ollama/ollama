package llm

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/*/build/*/bin/*
var llamaCppEmbed embed.FS

func cudaVersion() int {
	// first try nvcc, it gives the most accurate version if available
	cmd := exec.Command("nvcc", "--version")
	output, err := cmd.CombinedOutput()
	if err == nil {
		// regex to match the CUDA version line in nvcc --version output
		re := regexp.MustCompile(`release (\d+\.\d+),`)
		matches := re.FindStringSubmatch(string(output))
		if len(matches) >= 2 {
			cudaVersion := matches[1]
			cudaVersionParts := strings.Split(cudaVersion, ".")
			cudaMajorVersion, err := strconv.Atoi(cudaVersionParts[0])
			if err == nil {
				return cudaMajorVersion
			}
		}
	}

	// fallback to nvidia-smi
	cmd = exec.Command("nvidia-smi")
	output, err = cmd.CombinedOutput()
	if err != nil {
		return -1
	}

	re := regexp.MustCompile(`CUDA Version: (\d+\.\d+)`)
	matches := re.FindStringSubmatch(string(output))
	if len(matches) < 2 {
		return -1
	}

	cudaVersion := matches[1]
	cudaVersionParts := strings.Split(cudaVersion, ".")
	cudaMajorVersion, err := strconv.Atoi(cudaVersionParts[0])
	if err != nil {
		return -1
	}
	return cudaMajorVersion
}

type ModelRunner struct {
	Path string // path to the model runner executable
}

func chooseRunners(runnerType string) []ModelRunner {
	buildPath := path.Join("llama.cpp", runnerType, "build")
	var runners []string

	// set the runners based on the OS
	// IMPORTANT: the order of the runners in the array is the priority order
	switch runtime.GOOS {
	case "darwin":
		runners = []string{
			path.Join(buildPath, "metal", "bin", "server"),
			path.Join(buildPath, "cpu", "bin", "server"),
		}
	case "linux":
		cuda := cudaVersion()
		if cuda == 11 {
			// prioritize CUDA 11 runner
			runners = []string{
				path.Join(buildPath, "cuda-11", "bin", "server"),
				path.Join(buildPath, "cuda-12", "bin", "server"),
				path.Join(buildPath, "cpu", "bin", "server"),
			}
		} else {
			runners = []string{
				path.Join(buildPath, "cuda-12", "bin", "server"),
				path.Join(buildPath, "cuda-11", "bin", "server"),
				path.Join(buildPath, "cpu", "bin", "server"),
			}
		}
	case "windows":
		// TODO: select windows GPU runner here when available
		runners = []string{
			path.Join(buildPath, "cpu", "bin", "Release", "server.exe"),
		}
	default:
		log.Printf("unknown OS, running on CPU: %s", runtime.GOOS)
		runners = []string{
			path.Join(buildPath, "cpu", "bin", "server"),
		}
	}

	// copy the files locally to run the llama.cpp server
	tmpDir, err := os.MkdirTemp("", "llama-*")
	if err != nil {
		log.Fatalf("load llama runner: failed to create temp dir: %v", err)
	}
	runnerAvailable := false // if no runner files are found in the embed, this flag will cause a fast fail
	for _, r := range runners {
		// find all the files in the runner's bin directory
		files, err := fs.Glob(llamaCppEmbed, filepath.Join(filepath.Dir(r), "*"))
		if err != nil {
			// this is expected, ollama may be compiled without all runners packed in
			log.Printf("%s runner not found: %v", r, err)
			continue
		}
		runnerAvailable = true

		for _, f := range files {
			srcFile, err := llamaCppEmbed.Open(f)
			if err != nil {
				log.Fatalf("read llama runner %s: %v", f, err)
			}
			defer srcFile.Close()

			// create the directory in case it does not exist
			destPath := filepath.Join(tmpDir, filepath.Dir(f))
			if err := os.MkdirAll(destPath, 0o755); err != nil {
				log.Fatalf("create runner temp dir %s: %v", filepath.Dir(f), err)
			}
			destFile, err := os.OpenFile(filepath.Join(destPath, filepath.Base(f)), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				log.Fatalf("write llama runner %s: %v", f, err)
			}
			defer destFile.Close()

			if _, err := io.Copy(destFile, srcFile); err != nil {
				log.Fatalf("copy llama runner %s: %v", f, err)
			}
		}
	}
	if !runnerAvailable {
		log.Fatalf("%s runner not found", runnerType)
	}

	// return the runners to try in priority order
	localRunnersByPriority := []ModelRunner{}
	for _, r := range runners {
		localRunnersByPriority = append(localRunnersByPriority, ModelRunner{Path: path.Join(tmpDir, r)})
	}

	return localRunnersByPriority
}

type llamaModel struct {
	hyperparameters llamaHyperparameters
}

func (llm *llamaModel) ModelFamily() string {
	return "llama"
}

func llamaModelType(numLayer uint32) string {
	switch numLayer {
	case 26:
		return "3B"
	case 32:
		return "7B"
	case 40:
		return "13B"
	case 48:
		return "34B"
	case 60:
		return "30B"
	case 80:
		return "65B"
	default:
		return "Unknown"
	}
}

func (llm *llamaModel) ModelType() string {
	return llamaModelType(llm.hyperparameters.NumLayer)
}

func (llm *llamaModel) FileType() string {
	return fileType(llm.hyperparameters.FileType)
}

type llamaHyperparameters struct {
	// NumVocab is the size of the model's vocabulary.
	NumVocab uint32

	// NumEmbd is the size of the model's embedding layer.
	NumEmbd uint32
	NumMult uint32
	NumHead uint32

	// NumLayer is the number of layers in the model.
	NumLayer uint32
	NumRot   uint32

	// FileType describes the quantization level of the model, e.g. Q4_0, Q5_K, etc.
	FileType uint32
}

type Running struct {
	Port   int
	Cmd    *exec.Cmd
	Cancel context.CancelFunc
}

type llama struct {
	api.Options
	Running
}

var errNoGPU = errors.New("nvidia-smi command failed")

// CheckVRAM returns the available VRAM in MiB on Linux machines with NVIDIA GPUs
func CheckVRAM() (int, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	err := cmd.Run()
	if err != nil {
		return 0, errNoGPU
	}

	var total int
	scanner := bufio.NewScanner(&stdout)
	for scanner.Scan() {
		line := scanner.Text()
		vram, err := strconv.Atoi(line)
		if err != nil {
			return 0, fmt.Errorf("failed to parse available VRAM: %v", err)
		}

		total += vram
	}

	return total, nil
}

func NumGPU(opts api.Options) int {
	if opts.NumGPU != -1 {
		return opts.NumGPU
	}
	n := 1 // default to enable metal on macOS
	if runtime.GOOS == "linux" {
		vram, err := CheckVRAM()
		if err != nil {
			if err.Error() != "nvidia-smi command failed" {
				log.Print(err.Error())
			}
			// nvidia driver not installed or no nvidia GPU found
			return 0
		}
		// TODO: this is a very rough heuristic, better would be to calculate this based on number of layers and context size
		switch {
		case vram < 500:
			log.Printf("WARNING: Low VRAM detected, disabling GPU")
			n = 0
		case vram < 1000:
			n = 4
		case vram < 2000:
			n = 8
		case vram < 4000:
			n = 12
		case vram < 8000:
			n = 16
		case vram < 12000:
			n = 24
		case vram < 16000:
			n = 32
		default:
			n = 48
		}
		log.Printf("%d MB VRAM available, loading %d GPU layers", vram, n)
	}
	return n
}

func newLlama(model string, adapters []string, runners []ModelRunner, opts api.Options) (*llama, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	if len(adapters) > 1 {
		return nil, errors.New("ollama supports only one lora adapter, but multiple were provided")
	}

	params := []string{
		"--model", model,
		"--ctx-size", fmt.Sprintf("%d", opts.NumCtx),
		"--rope-freq-base", fmt.Sprintf("%f", opts.RopeFrequencyBase),
		"--rope-freq-scale", fmt.Sprintf("%f", opts.RopeFrequencyScale),
		"--batch-size", fmt.Sprintf("%d", opts.NumBatch),
		"--n-gpu-layers", fmt.Sprintf("%d", NumGPU(opts)),
		"--embedding",
	}

	if opts.NumGQA > 0 {
		params = append(params, "--gqa", fmt.Sprintf("%d", opts.NumGQA))
	}

	if len(adapters) > 0 {
		// TODO: applying multiple adapters is not supported by the llama.cpp server yet
		params = append(params, "--lora", adapters[0])
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

	// start the llama.cpp server with a retry in case the port is already in use
	for _, runner := range runners {
		if _, err := os.Stat(runner.Path); err != nil {
			log.Printf("llama runner not found: %v", err)
			continue
		}

		port := rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
		ctx, cancel := context.WithCancel(context.Background())
		cmd := exec.CommandContext(
			ctx,
			runner.Path,
			append(params, "--port", strconv.Itoa(port))...,
		)
		cmd.Env = append(os.Environ(), fmt.Sprintf("LD_LIBRARY_PATH=%s", filepath.Dir(runner.Path)))
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr

		llm := &llama{Options: opts, Running: Running{Port: port, Cmd: cmd, Cancel: cancel}}

		log.Print("starting llama runner")
		if err := llm.Cmd.Start(); err != nil {
			log.Printf("error starting the external llama runner: %v", err)
			continue
		}

		// monitor the command, it is blocking, so if it exits we need to capture that
		go func() {
			err := llm.Cmd.Wait() // this will block until the command exits
			if err != nil {
				log.Printf("llama runner exited with error: %v", err)
			} else {
				log.Printf("llama runner exited")
			}
		}()

		if err := waitForServer(llm); err != nil {
			log.Printf("error starting llama runner: %v", err)
			llm.Close()
			// try again
			continue
		}

		// server started successfully
		return llm, nil
	}

	return nil, fmt.Errorf("failed to start a llama runner")
}

func waitForServer(llm *llama) error {
	// wait for the server to start responding
	start := time.Now()
	expiresAt := time.Now().Add(2 * time.Minute) // be generous with timeout, large models can take a while to load
	ticker := time.NewTicker(200 * time.Millisecond)

	log.Print("waiting for llama runner to start responding")
	for range ticker.C {
		if time.Now().After(expiresAt) {
			return fmt.Errorf("llama runner did not start within alloted time, retrying")
		}

		// check if the server process has terminated
		if llm.Cmd.ProcessState != nil && llm.Cmd.ProcessState.Exited() {
			return fmt.Errorf("llama runner process has terminated")
		}

		if err := llm.Ping(context.Background()); err == nil {
			break
		}
	}

	log.Printf("llama runner started in %f seconds", time.Since(start).Seconds())
	return nil
}

func (llm *llama) Close() {
	llm.Cancel()
}

func (llm *llama) SetOptions(opts api.Options) {
	llm.Options = opts
}

type GenerationSettings struct {
	FrequencyPenalty float64       `json:"frequency_penalty"`
	IgnoreEOS        bool          `json:"ignore_eos"`
	LogitBias        []interface{} `json:"logit_bias"`
	Mirostat         int           `json:"mirostat"`
	MirostatEta      float64       `json:"mirostat_eta"`
	MirostatTau      float64       `json:"mirostat_tau"`
	Model            string        `json:"model"`
	NCtx             int           `json:"n_ctx"`
	NKeep            int           `json:"n_keep"`
	NPredict         int           `json:"n_predict"`
	NProbs           int           `json:"n_probs"`
	PenalizeNl       bool          `json:"penalize_nl"`
	PresencePenalty  float64       `json:"presence_penalty"`
	RepeatLastN      int           `json:"repeat_last_n"`
	RepeatPenalty    float64       `json:"repeat_penalty"`
	Seed             uint32        `json:"seed"`
	Stop             []string      `json:"stop"`
	Stream           bool          `json:"stream"`
	Temp             float64       `json:"temp"`
	TfsZ             float64       `json:"tfs_z"`
	TopK             int           `json:"top_k"`
	TopP             float64       `json:"top_p"`
	TypicalP         float64       `json:"typical_p"`
}

type Timings struct {
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
}

type Prediction struct {
	Content string `json:"content"`
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Stop    bool   `json:"stop"`

	Timings `json:"timings"`
}

type PredictRequest struct {
	Stream           bool            `json:"stream"`
	NPredict         int             `json:"n_predict,omitempty"`
	TopK             int             `json:"top_k,omitempty"`
	TopP             float32         `json:"top_p,omitempty"`
	TfsZ             float32         `json:"tfs_z,omitempty"`
	TypicalP         float32         `json:"typical_p,omitempty"`
	RepeatLastN      int             `json:"repeat_last_n,omitempty"`
	Temperature      float32         `json:"temperature,omitempty"`
	RepeatPenalty    float32         `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32         `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32         `json:"frequency_penalty,omitempty"`
	Mirostat         int             `json:"mirostat,omitempty"`
	MirostatTau      float32         `json:"mirostat_tau,omitempty"`
	MirostatEta      float32         `json:"mirostat_eta,omitempty"`
	PenalizeNl       bool            `json:"penalize_nl,omitempty"`
	NKeep            int             `json:"n_keep,omitempty"`
	Seed             int             `json:"seed,omitempty"`
	Prompt           string          `json:"prompt,omitempty"`
	NProbs           int             `json:"n_probs,omitempty"`
	LogitBias        map[int]float32 `json:"logit_bias,omitempty"`
	IgnoreEos        bool            `json:"ignore_eos,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
}

func (llm *llama) Predict(ctx context.Context, prevContext []int, prompt string, fn func(api.GenerateResponse)) error {
	prevConvo, err := llm.Decode(ctx, prevContext)
	if err != nil {
		return err
	}

	var nextContext strings.Builder
	nextContext.WriteString(prevConvo)
	nextContext.WriteString(prompt)

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", llm.Port)
	predReq := PredictRequest{
		Prompt:           nextContext.String(),
		Stream:           true,
		NPredict:         llm.NumPredict,
		NKeep:            llm.NumKeep,
		Temperature:      llm.Temperature,
		TopK:             llm.TopK,
		TopP:             llm.TopP,
		TfsZ:             llm.TFSZ,
		TypicalP:         llm.TypicalP,
		RepeatLastN:      llm.RepeatLastN,
		RepeatPenalty:    llm.RepeatPenalty,
		PresencePenalty:  llm.PresencePenalty,
		FrequencyPenalty: llm.FrequencyPenalty,
		Mirostat:         llm.Mirostat,
		MirostatTau:      llm.MirostatTau,
		MirostatEta:      llm.MirostatEta,
		PenalizeNl:       llm.PenalizeNewline,
		Stop:             llm.Stop,
	}
	data, err := json.Marshal(predReq)
	if err != nil {
		return fmt.Errorf("error marshaling data: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBuffer(data))
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
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			// This handles the request cancellation
			return ctx.Err()
		default:
			line := scanner.Text()
			if line == "" {
				continue
			}

			// Read data from the server-side event stream
			if strings.HasPrefix(line, "data: ") {
				evt := line[6:]
				var p Prediction
				if err := json.Unmarshal([]byte(evt), &p); err != nil {
					return fmt.Errorf("error unmarshaling llm prediction response: %v", err)
				}

				if p.Content != "" {
					fn(api.GenerateResponse{Response: p.Content})
					nextContext.WriteString(p.Content)
				}

				if p.Stop {
					embd, err := llm.Encode(ctx, nextContext.String())
					if err != nil {
						return fmt.Errorf("encoding context: %v", err)
					}

					fn(api.GenerateResponse{
						Done:               true,
						Context:            embd,
						PromptEvalCount:    p.PromptN,
						PromptEvalDuration: parseDurationMs(p.PromptMS),
						EvalCount:          p.PredictedN,
						EvalDuration:       parseDurationMs(p.PredictedMS),
					})

					return nil
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading llm response: %v", err)
	}

	return nil
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func (llm *llama) Encode(ctx context.Context, prompt string) ([]int, error) {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/tokenize", llm.Port)
	data, err := json.Marshal(TokenizeRequest{Content: prompt})
	if err != nil {
		return nil, fmt.Errorf("marshaling encode data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBuffer(data))
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

func (llm *llama) Decode(ctx context.Context, tokens []int) (string, error) {
	if len(tokens) == 0 {
		return "", nil
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/detokenize", llm.Port)
	data, err := json.Marshal(DetokenizeRequest{Tokens: tokens})
	if err != nil {
		return "", fmt.Errorf("marshaling decode data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBuffer(data))
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

	// decoded content contains a leading whitespace
	decoded.Content, _ = strings.CutPrefix(decoded.Content, "")

	return decoded.Content, nil
}

type EmbeddingRequest struct {
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (llm *llama) Embedding(ctx context.Context, input string) ([]float64, error) {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/embedding", llm.Port)
	data, err := json.Marshal(TokenizeRequest{Content: input})
	if err != nil {
		return nil, fmt.Errorf("error marshaling embed data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("error creating embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("POST embedding: %w", err)
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

// Ping checks that the server subprocess is still running and responding to requests
func (llm *llama) Ping(ctx context.Context) error {
	resp, err := http.Head(fmt.Sprintf("http://127.0.0.1:%d", llm.Port))
	if err != nil {
		return fmt.Errorf("ping resp: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected ping status: %s", resp.Status)
	}
	return nil
}
