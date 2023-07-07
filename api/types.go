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

	ModelOptions   `json:"model_opts"`
	PredictOptions `json:"predict_opts"`
}

type ModelOptions struct {
	ContextSize int    `json:"context_size"`
	Seed        int    `json:"seed"`
	NBatch      int    `json:"n_batch"`
	F16Memory   bool   `json:"memory_f16"`
	MLock       bool   `json:"mlock"`
	MMap        bool   `json:"mmap"`
	VocabOnly   bool   `json:"vocab_only"`
	LowVRAM     bool   `json:"low_vram"`
	Embeddings  bool   `json:"embeddings"`
	NUMA        bool   `json:"numa"`
	NGPULayers  int    `json:"gpu_layers"`
	MainGPU     string `json:"main_gpu"`
	TensorSplit string `json:"tensor_split"`
}

type PredictOptions struct {
	Seed        int     `json:"seed"`
	Threads     int     `json:"threads"`
	Tokens      int     `json:"tokens"`
	TopK        int     `json:"top_k"`
	Repeat      int     `json:"repeat"`
	Batch       int     `json:"batch"`
	NKeep       int     `json:"nkeep"`
	TopP        float64 `json:"top_p"`
	Temperature float64 `json:"temp"`
	Penalty     float64 `json:"penalty"`
	F16KV       bool
	DebugMode   bool
	StopPrompts []string
	IgnoreEOS   bool `json:"ignore_eos"`

	TailFreeSamplingZ float64 `json:"tfs_z"`
	TypicalP          float64 `json:"typical_p"`
	FrequencyPenalty  float64 `json:"freq_penalty"`
	PresencePenalty   float64 `json:"pres_penalty"`
	Mirostat          int     `json:"mirostat"`
	MirostatETA       float64 `json:"mirostat_lr"`
	MirostatTAU       float64 `json:"mirostat_ent"`
	PenalizeNL        bool    `json:"penalize_nl"`
	LogitBias         string  `json:"logit_bias"`

	PathPromptCache string
	MLock           bool `json:"mlock"`
	MMap            bool `json:"mmap"`
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
