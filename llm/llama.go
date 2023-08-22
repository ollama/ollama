package llm

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/jmorganca/ollama/api"
)

const ModelFamilyLlama ModelFamily = "llama"

//go:embed llama_cpp
var llamaBin []byte
var _ embed.FS // appease go

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
	Port int
	Cmd  *exec.Cmd
}

type llama struct {
	api.Options
	Running
}

func isPortAvailable(port int) bool {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return false
	}
	listener.Close()
	return true
}

func getRandomPort() (int, error) {
	const (
		minPort  = 1024
		maxPort  = 49151
		attempts = 100 // Number of attempts to find an available port
	)
	for i := 0; i < attempts; i++ {
		port := minPort + rand.Intn(maxPort-minPort+1)
		if isPortAvailable(port) {
			return port, nil
		}
	}
	return -1, fmt.Errorf("could not find an available port")
}

func newLlama(model string, adapters []string, opts api.Options) (*llama, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	// TODO: if GOOS == LINUX, run from memory
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	path := filepath.Join(home, ".ollama", "runners", "llama_cpp")
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(path, llamaBin, 0o755); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}

	port, err := getRandomPort()
	if err != nil {
		return nil, fmt.Errorf("load llm: %w", err)
	}

	params := []string{
		"--model", model,
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", opts.NumCtx),
		// "--threads", fmt.Sprintf("%d", opts.NumThread),
		// "--gqa", fmt.Sprintf("%d", opts.NumGQA),
		// "--rms-norm-eps", fmt.Sprintf("%f", opts.Temperature),
		// "--rope-freq-base", fmt.Sprintf("%f", opts.RopeFrequencyBase),
		// "--rope-freq-scale", fmt.Sprintf("%f", opts.RopeFrequencyScale),
		// "--batch-size", fmt.Sprintf("%d", opts.NumBatch),
		"--embedding",
	}
	// if !opts.F16KV {
	// 	params = append(params, "--memory-f32")
	// }
	// if opts.UseMLock {
	// 	params = append(params, "--mlock")
	// }
	// if !opts.UseMMap {
	// 	params = append(params, "--no-mmap")
	// }
	// if opts.UseNUMA {
	// 	params = append(params, "--numa")
	// }

	// TODO: LoRA adapters
	cmd := exec.Command(
		path,
		params...,
	)
	err = cmd.Start()
	if err != nil {
		return nil, fmt.Errorf("error starting the external server: %w", err)
	}

	// wait for llama.cpp to come up
	deadline := time.Now().Add(10 * time.Second)

	for time.Now().Before(deadline) {
		if !isPortAvailable(port) {
			// TODO: warm up the model
			return &llama{Options: opts, Running: Running{Port: port, Cmd: cmd}}, nil
		}
		time.Sleep(time.Millisecond)
	}

	return nil, fmt.Errorf("timed out waiting for llama.cpp at port %d", port)
}

func (llm *llama) Close() {
	llm.Running.Cmd.Process.Kill()
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
	Grammar          string        `json:"grammar"`
	IgnoreEOS        bool          `json:"ignore_eos"`
	LogitBias        []interface{} `json:"logit_bias"` // TODO
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
	Stop             []interface{} `json:"stop"` // TODO
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
	Grammar          string          `json:"grammar,omitempty"`
	NProbs           int             `json:"n_probs,omitempty"`
	LogitBias        map[int]float32 `json:"logit_bias,omitempty"`
	IgnoreEos        bool            `json:"ignore_eos,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
}

// TODO: client closing should release nested lock
func (llm *llama) Predict(ctx []int, prompt string, fn func(api.GenerateResponse)) error {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", llm.Port)
	predReq := PredictRequest{
		Prompt:           prompt,
		Stream:           true,
		NPredict:         llm.NumPredict,
		TopK:             llm.TopK,
		TopP:             llm.TopP,
		TfsZ:             llm.TFSZ,
		TypicalP:         llm.TypicalP,
		RepeatLastN:      llm.RepeatLastN,
		Temperature:      llm.Temperature,
		RepeatPenalty:    llm.RepeatPenalty,
		PresencePenalty:  llm.PresencePenalty,
		FrequencyPenalty: llm.FrequencyPenalty,
		Mirostat:         llm.Mirostat,
		MirostatTau:      llm.MirostatTau,
		MirostatEta:      llm.MirostatEta,
		PenalizeNl:       llm.PenalizeNewline,
		Grammar:          llm.Grammar,
		Stop:             llm.Stop,
	}
	data, err := json.Marshal(predReq)
	if err != nil {
		return fmt.Errorf("error marshaling data: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewBuffer(data))
	if err != nil {
		return fmt.Errorf("error creating POST request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST predict: %v", err)
	}
	defer resp.Body.Close()

	reader := bufio.NewReader(resp.Body)
	fullResponse := ""
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error reading llm response: %v", err)
		}
		if line == "\n" {
			continue
		}

		// Read data from the server-side event stream
		if len(line) > 6 && line[:6] == "data: " {
			evt := line[6:]
			var complete PredictComplete
			if err := json.Unmarshal([]byte(evt), &complete); err != nil {
				return fmt.Errorf("error unmarshaling llm complete response: %v", err)
			}

			if complete.Timings.PredictedMS > 0 {
				// this is a complete response
				embd := llm.Encode(fullResponse)
				fn(api.GenerateResponse{
					Done:        true,
					Context:     embd,
					SampleCount: int(complete.Timings.PredictedN), // TODO
					// SampleDuration:     parseDurationMs(float64(complete.Timings.PredictedMS)),
					PromptEvalCount:    int(complete.TokensEvaluated),
					PromptEvalDuration: parseDurationMs(float64(complete.Timings.PredictedMS)),
					// EvalCount:          int(complete.TokensEvaluated),
					EvalDuration: parseDurationMs(float64(complete.Timings.PredictedMS)),
				})
				break
			}

			var pred Prediction
			if err := json.Unmarshal([]byte(evt), &pred); err != nil {
				return fmt.Errorf("error unmarshaling llm prediction response: %v", err)
			}
			fullResponse += pred.Content
			fn(api.GenerateResponse{Response: pred.Content})
		}
	}

	return false, false
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func (llm *llama) Encode(prompt string) []int {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/tokenize", llm.Port)
	data, err := json.Marshal(TokenizeRequest{Content: prompt})
	if err != nil {
		// TODO: should this be fatal?
		log.Printf("Error marshaling encode data: %v", err)
		return nil
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewBuffer(data))
	if err != nil {
		log.Printf("error creating POST request: %v", err)
		return nil
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("POST encode: %v", err)
		return nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("error reading tokenize response: %s", err)
		return nil
	}

	var encoded TokenizeResponse
	if err := json.Unmarshal(body, &encoded); err != nil {
		log.Fatalf("error unmarshaling llm tokenize response: %v", err)
	}

	return encoded.Tokens
}

type EmbeddingRequest struct {
	Content string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (llm *llama) Embedding(input string) ([]float64, error) {
	if !llm.EmbeddingOnly {
		return nil, errors.New("llama: embedding not enabled")
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/embedding", llm.Port)
	data, err := json.Marshal(TokenizeRequest{Content: input})
	if err != nil {
		return nil, fmt.Errorf("error marshaling embed data: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewBuffer(data))
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

	var embedding EmbeddingResponse
	if err := json.Unmarshal(body, &embedding); err != nil {
		log.Fatalf("error unmarshaling llm tokenize response: %v", err)
	}

	return embedding.Embedding, nil
}
