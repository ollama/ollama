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
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/jmorganca/ollama/api"
)

//go:embed llama.cpp/*/build/*/bin/*
var llamaCppEmbed embed.FS

type ModelRunner struct {
	Path string // path to the model runner executable
}

func chooseRunners(workDir, runnerType string) []ModelRunner {
	buildPath := path.Join("llama.cpp", runnerType, "build")
	var runners []string

	// set the runners based on the OS
	// IMPORTANT: the order of the runners in the array is the priority order
	switch runtime.GOOS {
	case "darwin":
		runners = []string{
			path.Join(buildPath, "metal", "bin", "ollama-runner"),
			path.Join(buildPath, "cpu", "bin", "ollama-runner"),
		}
	case "linux":
		runners = []string{
			path.Join(buildPath, "cuda", "bin", "ollama-runner"),
			path.Join(buildPath, "cpu", "bin", "ollama-runner"),
		}
	case "windows":
		// TODO: select windows GPU runner here when available
		runners = []string{
			path.Join(buildPath, "cpu", "bin", "Release", "ollama-runner.exe"),
		}
	default:
		log.Printf("unknown OS, running on CPU: %s", runtime.GOOS)
		runners = []string{
			path.Join(buildPath, "cpu", "bin", "ollama-runner"),
		}
	}

	runnerAvailable := false // if no runner files are found in the embed, this flag will cause a fast fail
	for _, r := range runners {
		// find all the files in the runner's bin directory
		files, err := fs.Glob(llamaCppEmbed, path.Join(path.Dir(r), "*"))
		if err != nil {
			// this is expected, ollama may be compiled without all runners packed in
			log.Printf("%s runner not found: %v", r, err)
			continue
		}

		for _, f := range files {
			runnerAvailable = true

			srcFile, err := llamaCppEmbed.Open(f)
			if err != nil {
				log.Fatalf("read llama runner %s: %v", f, err)
			}
			defer srcFile.Close()

			// create the directory in case it does not exist, filepath.Dir() converts the file path to the OS's format
			destPath := filepath.Join(workDir, filepath.Dir(f))
			if err := os.MkdirAll(destPath, 0o755); err != nil {
				log.Fatalf("create runner temp dir %s: %v", filepath.Dir(f), err)
			}

			// create the path to the destination file, filepath.Base() converts the file path to the OS's format
			destFile := filepath.Join(destPath, filepath.Base(f))

			_, err = os.Stat(destFile)
			switch {
			case errors.Is(err, os.ErrNotExist):
				destFile, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
				if err != nil {
					log.Fatalf("write llama runner %s: %v", f, err)
				}
				defer destFile.Close()

				if _, err := io.Copy(destFile, srcFile); err != nil {
					log.Fatalf("copy llama runner %s: %v", f, err)
				}
			case err != nil:
				log.Fatalf("stat llama runner %s: %v", f, err)
			}
		}
	}
	if !runnerAvailable {
		log.Fatalf("%s runner not found", runnerType)
	}

	// return the runners to try in priority order
	localRunnersByPriority := []ModelRunner{}
	for _, r := range runners {
		// clean the ModelRunner paths so that they match the OS we are running on
		localRunnersByPriority = append(localRunnersByPriority, ModelRunner{Path: filepath.Clean(path.Join(workDir, r))})
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
		return "unknown"
	}
}

func (llm *llamaModel) ModelType() string {
	return llamaModelType(llm.hyperparameters.NumLayer)
}

func (llm *llamaModel) FileType() string {
	return fileType(llm.hyperparameters.FileType)
}

func (llm *llamaModel) NumLayers() int64 {
	return int64(llm.hyperparameters.NumLayer)
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
func CheckVRAM() (int64, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	err := cmd.Run()
	if err != nil {
		return 0, errNoGPU
	}

	var free int64
	scanner := bufio.NewScanner(&stdout)
	for scanner.Scan() {
		line := scanner.Text()
		vram, err := strconv.ParseInt(strings.TrimSpace(line), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("failed to parse available VRAM: %v", err)
		}

		free += vram
	}

	return free, nil
}

func NumGPU(numLayer, fileSizeBytes int64, opts api.Options) int {
	if opts.NumGPU != -1 {
		return opts.NumGPU
	}
	if runtime.GOOS == "linux" {
		vramMib, err := CheckVRAM()
		if err != nil {
			if err.Error() != "nvidia-smi command failed" {
				log.Print(err.Error())
			}
			// nvidia driver not installed or no nvidia GPU found
			return 0
		}

		freeVramBytes := int64(vramMib) * 1024 * 1024 // 1 MiB = 1024^2 bytes

		// Calculate bytes per layer
		// TODO: this is a rough heuristic, better would be to calculate this based on number of layers and context size
		bytesPerLayer := fileSizeBytes / numLayer

		// max number of layers we can fit in VRAM, subtract 5% to prevent consuming all available VRAM and running out of memory
		layers := int(freeVramBytes/bytesPerLayer) * 95 / 100
		log.Printf("%d MiB VRAM available, loading up to %d GPU layers", vramMib, layers)

		return layers
	}
	// default to enable metal on macOS
	return 1
}

func newLlama(model string, adapters []string, runners []ModelRunner, numLayers int64, opts api.Options) (*llama, error) {
	fileInfo, err := os.Stat(model)
	if err != nil {
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
		"--n-gpu-layers", fmt.Sprintf("%d", NumGPU(numLayers, fileInfo.Size(), opts)),
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
	// signal the sub-process to terminate
	llm.Cancel()

	// wait for the command to exit to prevent race conditions with the next run
	if err := llm.Cmd.Wait(); err != nil {
		log.Printf("llama runner exited: %v", err)
	}
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
	Prompt           string   `json:"prompt"`
	Stream           bool     `json:"stream"`
	NPredict         int      `json:"n_predict"`
	NKeep            int      `json:"n_keep"`
	Temperature      float32  `json:"temperature"`
	TopK             int      `json:"top_k"`
	TopP             float32  `json:"top_p"`
	TfsZ             float32  `json:"tfs_z"`
	TypicalP         float32  `json:"typical_p"`
	RepeatLastN      int      `json:"repeat_last_n"`
	RepeatPenalty    float32  `json:"repeat_penalty"`
	PresencePenalty  float32  `json:"presence_penalty"`
	FrequencyPenalty float32  `json:"frequency_penalty"`
	Mirostat         int      `json:"mirostat"`
	MirostatTau      float32  `json:"mirostat_tau"`
	MirostatEta      float32  `json:"mirostat_eta"`
	PenalizeNl       bool     `json:"penalize_nl"`
	Seed             int      `json:"seed"`
	Stop             []string `json:"stop,omitempty"`
}

const maxBufferSize = 512 * 1024 // 512KB

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
		Seed:             llm.Seed,
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
	// increase the buffer size to avoid running out of space
	buf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(buf, maxBufferSize)
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
