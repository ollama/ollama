package api

import (
	"fmt"
	"net/http"
	"strings"
)

type Error struct {
	Code    int32  `json:"code"`
	Message string `json:"message"`
}

func (e Error) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("%d %v", e.Code, strings.ToLower(http.StatusText(int(e.Code))))
	}
	return e.Message
}

type PullRequest struct {
	Model string `json:"model"`
}

type PullProgress struct {
	Total     int64   `json:"total"`
	Completed int64   `json:"completed"`
	Percent   float64 `json:"percent"`
}

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`

	ModelOptions   `json:"model_opts,omitempty"`
	PredictOptions `json:"predict_opts,omitempty"`
}

type ModelOptions struct {
	ContextSize int    `json:"context_size,omitempty"`
	Seed        int    `json:"seed,omitempty"`
	NBatch      int    `json:"n_batch,omitempty"`
	F16Memory   bool   `json:"memory_f16,omitempty"`
	MLock       bool   `json:"mlock,omitempty"`
	MMap        bool   `json:"mmap,omitempty"`
	VocabOnly   bool   `json:"vocab_only,omitempty"`
	LowVRAM     bool   `json:"low_vram,omitempty"`
	Embeddings  bool   `json:"embeddings,omitempty"`
	NUMA        bool   `json:"numa,omitempty"`
	NGPULayers  int    `json:"gpu_layers,omitempty"`
	MainGPU     string `json:"main_gpu,omitempty"`
	TensorSplit string `json:"tensor_split,omitempty"`
}

type PredictOptions struct {
	Seed        int     `json:"seed,omitempty"`
	Threads     int     `json:"threads,omitempty"`
	Tokens      int     `json:"tokens,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	Repeat      int     `json:"repeat,omitempty"`
	Batch       int     `json:"batch,omitempty"`
	NKeep       int     `json:"nkeep,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	Temperature float64 `json:"temp,omitempty"`
	Penalty     float64 `json:"penalty,omitempty"`
	F16KV       bool
	DebugMode   bool
	StopPrompts []string
	IgnoreEOS   bool `json:"ignore_eos,omitempty"`

	TailFreeSamplingZ float64 `json:"tfs_z,omitempty"`
	TypicalP          float64 `json:"typical_p,omitempty"`
	FrequencyPenalty  float64 `json:"freq_penalty,omitempty"`
	PresencePenalty   float64 `json:"pres_penalty,omitempty"`
	Mirostat          int     `json:"mirostat,omitempty"`
	MirostatETA       float64 `json:"mirostat_lr,omitempty"`
	MirostatTAU       float64 `json:"mirostat_ent,omitempty"`
	PenalizeNL        bool    `json:"penalize_nl,omitempty"`
	LogitBias         string  `json:"logit_bias,omitempty"`

	PathPromptCache string
	MLock           bool `json:"mlock,omitempty"`
	MMap            bool `json:"mmap,omitempty"`
	PromptCacheAll  bool
	PromptCacheRO   bool
	MainGPU         string
	TensorSplit     string
}

var DefaultModelOptions ModelOptions = ModelOptions{
	ContextSize: 128,
	Seed:        0,
	F16Memory:   true,
	MLock:       false,
	Embeddings:  true,
	MMap:        true,
	LowVRAM:     false,
}

var DefaultPredictOptions PredictOptions = PredictOptions{
	Seed:              -1,
	Threads:           -1,
	Tokens:            512,
	Penalty:           1.1,
	Repeat:            64,
	Batch:             512,
	NKeep:             64,
	TopK:              90,
	TopP:              0.86,
	TailFreeSamplingZ: 1.0,
	TypicalP:          1.0,
	Temperature:       0.8,
	FrequencyPenalty:  0.0,
	PresencePenalty:   0.0,
	Mirostat:          0,
	MirostatTAU:       5.0,
	MirostatETA:       0.1,
	MMap:              true,
	StopPrompts:       []string{"llama"},
}

type GenerateResponse struct {
	Response string `json:"response"`
}
