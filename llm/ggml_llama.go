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
	"sync"
	"time"

	"github.com/jmorganca/ollama/api"
)

const ModelFamilyLlama ModelFamily = "llama"

//go:embed llama.cpp/ggml/build/*/bin/*
var llamaCppEmbed embed.FS

var (
	ggmlGPU = path.Join("llama.cpp", "ggml", "build", "gpu", "bin")
	ggmlCPU = path.Join("llama.cpp", "ggml", "build", "cpu", "bin")
)

var (
	ggmlInit       sync.Once
	ggmlRunnerPath string
)

func osPath(llamaPath string) string {
	if runtime.GOOS == "windows" {
		return path.Join(llamaPath, "Release")
	}
	return llamaPath
}

func initGGML() {
	ggmlInit.Do(func() {
		tmpDir, err := os.MkdirTemp("", "llama-*")
		if err != nil {
			log.Fatalf("llama.cpp: failed to create temp dir: %v", err)
		}

		llamaPath := osPath(ggmlGPU)
		if _, err := fs.Stat(llamaCppEmbed, llamaPath); err != nil {
			llamaPath = osPath(ggmlCPU)
			if _, err := fs.Stat(llamaCppEmbed, llamaPath); err != nil {
				log.Fatalf("llama.cpp executable not found")
			}
		}

		files := []string{"server"}
		switch runtime.GOOS {
		case "windows":
			files = []string{"server.exe"}
		case "darwin":
			files = append(files, "ggml-metal.metal")
		}

		for _, f := range files {
			srcPath := path.Join(llamaPath, f)
			destPath := filepath.Join(tmpDir, f)

			srcFile, err := llamaCppEmbed.Open(srcPath)
			if err != nil {
				log.Fatalf("read llama.cpp %s: %v", f, err)
			}
			defer srcFile.Close()

			destFile, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				log.Fatalf("write llama.cpp %s: %v", f, err)
			}
			defer destFile.Close()

			if _, err := io.Copy(destFile, srcFile); err != nil {
				log.Fatalf("copy llama.cpp %s: %v", f, err)
			}
		}

		ggmlRunnerPath = filepath.Join(tmpDir, "server")
		if runtime.GOOS == "windows" {
			ggmlRunnerPath = filepath.Join(tmpDir, "server.exe")
		}
	})
}

type ModelRunner struct {
	Path string // path to the model runner executable
}

func ggmlRunner() ModelRunner {
	initGGML()
	return ModelRunner{Path: ggmlRunnerPath}
}

type llamaModel struct {
	hyperparameters llamaHyperparameters
}

func (llm *llamaModel) ModelFamily() ModelFamily {
	return ModelFamilyLlama
}

func (llm *llamaModel) ModelType() ModelType {
	switch llm.hyperparameters.NumLayer {
	case 26:
		return ModelType3B
	case 32:
		return ModelType7B
	case 40:
		return ModelType13B
	case 48:
		return ModelType34B
	case 60:
		return ModelType30B
	case 80:
		return ModelType65B
	}

	// TODO: find a better default
	return ModelType7B
}

func (llm *llamaModel) FileType() FileType {
	return llm.hyperparameters.FileType
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
	FileType llamaFileType
}

type llamaFileType uint32

const (
	llamaFileTypeF32 llamaFileType = iota
	llamaFileTypeF16
	llamaFileTypeQ4_0
	llamaFileTypeQ4_1
	llamaFileTypeQ4_1_F16
	llamaFileTypeQ8_0 llamaFileType = iota + 2
	llamaFileTypeQ5_0
	llamaFileTypeQ5_1
	llamaFileTypeQ2_K
	llamaFileTypeQ3_K_S
	llamaFileTypeQ3_K_M
	llamaFileTypeQ3_K_L
	llamaFileTypeQ4_K_S
	llamaFileTypeQ4_K_M
	llamaFileTypeQ5_K_S
	llamaFileTypeQ5_K_M
	llamaFileTypeQ6_K
)

func (ft llamaFileType) String() string {
	switch ft {
	case llamaFileTypeF32:
		return "F32"
	case llamaFileTypeF16:
		return "F16"
	case llamaFileTypeQ4_0:
		return "Q4_0"
	case llamaFileTypeQ4_1:
		return "Q4_1"
	case llamaFileTypeQ4_1_F16:
		return "Q4_1_F16"
	case llamaFileTypeQ8_0:
		return "Q8_0"
	case llamaFileTypeQ5_0:
		return "Q5_0"
	case llamaFileTypeQ5_1:
		return "Q5_1"
	case llamaFileTypeQ2_K:
		return "Q2_K"
	case llamaFileTypeQ3_K_S:
		return "Q3_K_S"
	case llamaFileTypeQ3_K_M:
		return "Q3_K_M"
	case llamaFileTypeQ3_K_L:
		return "Q3_K_L"
	case llamaFileTypeQ4_K_S:
		return "Q4_K_S"
	case llamaFileTypeQ4_K_M:
		return "Q4_K_M"
	case llamaFileTypeQ5_K_S:
		return "Q5_K_S"
	case llamaFileTypeQ5_K_M:
		return "Q5_K_M"
	case llamaFileTypeQ6_K:
		return "Q6_K"
	default:
		return "Unknown"
	}
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

func newLlama(model string, adapters []string, runner ModelRunner, opts api.Options) (*llama, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	if _, err := os.Stat(runner.Path); err != nil {
		return nil, err
	}

	if len(adapters) > 1 {
		return nil, errors.New("ollama supports only one lora adapter, but multiple were provided")
	}

	params := []string{
		"--model", model,
		"--ctx-size", fmt.Sprintf("%d", opts.NumCtx),
		"--gqa", fmt.Sprintf("%d", opts.NumGQA),
		"--rope-freq-base", fmt.Sprintf("%f", opts.RopeFrequencyBase),
		"--rope-freq-scale", fmt.Sprintf("%f", opts.RopeFrequencyScale),
		"--batch-size", fmt.Sprintf("%d", opts.NumBatch),
		"--n-gpu-layers", fmt.Sprintf("%d", opts.NumGPU),
		"--embedding",
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
	for try := 0; try < 3; try++ {
		port := rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
		ctx, cancel := context.WithCancel(context.Background())
		cmd := exec.CommandContext(
			ctx,
			runner.Path,
			append(params, "--port", strconv.Itoa(port))...,
		)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr

		llm := &llama{Options: opts, Running: Running{Port: port, Cmd: cmd, Cancel: cancel}}

		if err := waitForServer(llm); err != nil {
			log.Printf("error starting llama.cpp server: %v", err)
			llm.Close()
			// try again
			continue
		}
		// server started successfully
		return llm, nil
	}

	return nil, fmt.Errorf("max retry exceeded starting llama.cpp")
}

func waitForServer(llm *llama) error {
	log.Print("starting llama.cpp server")
	var stderr bytes.Buffer
	llm.Cmd.Stderr = &stderr
	err := llm.Cmd.Start()
	if err != nil {
		return fmt.Errorf("error starting the external llama.cpp server: %w", err)
	}

	exitChan := make(chan error, 1)

	// the server is a long running process, watch for it exiting to keep track of something going wrong
	go func() {
		err := llm.Cmd.Wait()
		log.Print(stderr.String())
		exitChan <- err
	}()

	// wait for the server to start responding
	start := time.Now()
	expiresAt := time.Now().Add(30 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)

	log.Print("waiting for llama.cpp server to start responding")

	for {
		select {
		case <-ticker.C:
			if time.Now().After(expiresAt) {
				return fmt.Errorf("llama.cpp server did not start responding within 30 seconds, retrying")
			}
			if err := llm.Ping(context.Background()); err == nil {
				log.Printf("llama.cpp server started in %f seconds", time.Since(start).Seconds())
				return nil
			}
		case err := <-exitChan:
			return fmt.Errorf("llama.cpp server exited unexpectedly: %w", err)
		}
	}
}

func (llm *llama) Close() {
	llm.Running.Cmd.Cancel()
}

func (llm *llama) SetOptions(opts api.Options) {
	llm.Options = opts
}

type Prediction struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`
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
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedN          int     `json:"predicted_n"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptN             int     `json:"prompt_n"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
}

type PredictComplete struct {
	Content            string             `json:"content"`
	GenerationSettings GenerationSettings `json:"generation_settings"`
	Model              string             `json:"model"`
	Prompt             string             `json:"prompt"`
	Stop               bool               `json:"stop"`
	StoppedEOS         bool               `json:"stopped_eos"`
	StoppedLimit       bool               `json:"stopped_limit"`
	StoppedWord        bool               `json:"stopped_word"`
	StoppingWord       string             `json:"stopping_word"`
	Timings            Timings            `json:"timings"`
	TokensCached       int                `json:"tokens_cached"`
	TokensEvaluated    int                `json:"tokens_evaluated"`
	TokensPredicted    int                `json:"tokens_predicted"`
	Truncated          bool               `json:"truncated"`
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

func (llm *llama) Predict(ctx context.Context, predictCtx []int, prompt string, fn func(api.GenerateResponse)) error {
	// we need to find the trimmed prompt context before predicting so that we can return it to the client
	trimmedPrompt, err := llm.marshalPrompt(ctx, predictCtx, prompt)
	if err != nil {
		return fmt.Errorf("marshaling prompt: %v", err)
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", llm.Port)
	predReq := PredictRequest{
		Prompt:           trimmedPrompt,
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
	genCtx := trimmedPrompt // start with the trimmed prompt
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
				var complete PredictComplete
				if err := json.Unmarshal([]byte(evt), &complete); err != nil {
					return fmt.Errorf("error unmarshaling llm complete response: %v", err)
				}

				if complete.Timings.PredictedMS > 0 {
					genCtx += complete.Content
					embd, err := llm.Encode(ctx, genCtx)
					if err != nil {
						return fmt.Errorf("encoding context: %v", err)
					}
					fn(api.GenerateResponse{
						Done:               true,
						Context:            embd,
						PromptEvalCount:    int(complete.Timings.PromptN),
						PromptEvalDuration: parseDurationMs(float64(complete.Timings.PromptMS)),
						EvalCount:          int(complete.Timings.PredictedN),
						EvalDuration:       parseDurationMs(float64(complete.Timings.PredictedMS)),
					})
					return nil
				}

				var pred Prediction
				if err := json.Unmarshal([]byte(evt), &pred); err != nil {
					return fmt.Errorf("error unmarshaling llm prediction response: %v", err)
				}
				genCtx += pred.Content
				fn(api.GenerateResponse{Response: pred.Content})
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading llm response: %v", err)
	}

	return nil
}

func (llm *llama) marshalPrompt(ctx context.Context, pCtx []int, prompt string) (string, error) {
	pEncode, err := llm.Encode(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("encoding prompt context: %w", err)
	}
	tokens := append(pCtx, pEncode...)
	if llm.NumKeep < 0 {
		llm.NumKeep = len(tokens)
	}

	// min(llm.NumCtx - 4, llm.NumKeep)
	if llm.NumCtx-4 < llm.NumKeep {
		llm.NumKeep = llm.NumCtx - 4
	}

	if len(tokens) >= llm.NumCtx {
		// truncate input
		numLeft := (llm.NumCtx - llm.NumKeep) / 2
		truncated := tokens[:llm.NumKeep]
		erasedBlocks := (len(tokens) - llm.NumKeep - numLeft - 1) / numLeft
		truncated = append(truncated, tokens[llm.NumKeep+erasedBlocks*numLeft:]...)
		tokens = truncated
		log.Printf("input truncated: num_ctx=%d num_keep=%d num_left=%d num_tokens=%d", llm.NumCtx, llm.NumKeep, numLeft, len(truncated))
	}

	return llm.Decode(ctx, tokens)
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
	resp, err := http.Head(fmt.Sprintf("http://127.0.0.1:%d", llm.Running.Port))
	if err != nil {
		return fmt.Errorf("ping resp: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected ping status: %s", resp.Status)
	}
	return nil
}
