package api

import (
	"fmt"
	"os"
	"runtime"
	"time"
)

type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error"`
}

func (e StatusError) Error() string {
	switch {
	case e.Status != "" && e.ErrorMessage != "":
		return fmt.Sprintf("%s: %s", e.Status, e.ErrorMessage)
	case e.Status != "":
		return e.Status
	case e.ErrorMessage != "":
		return e.ErrorMessage
	default:
		// this should not happen
		return "something went wrong, please see the ollama server logs for details"
	}
}

type GenerateRequest struct {
	SessionID int64  `json:"session_id"`
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	Context   []int  `json:"context,omitempty"`

	Options `json:"options"`
}

type CreateRequest struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

type DeleteRequest struct {
	Name string `json:"name"`
}

type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

type PullRequest struct {
	Name     string `json:"name"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
}

type ProgressResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int    `json:"total,omitempty"`
	Completed int    `json:"completed,omitempty"`
}

type PushRequest struct {
	Name     string `json:"name"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username"`
	Password string `json:"password"`
}

type ListResponse struct {
	Models []ListResponseModel `json:"models"`
}

type ListResponseModel struct {
	Name       string    `json:"name"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int       `json:"size"`
}

type GenerateResponse struct {
	SessionID int64     `json:"session_id"`
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response,omitempty"`

	Done    bool  `json:"done"`
	Context []int `json:"context,omitempty"`

	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func (r *GenerateResponse) Summary() {
	if r.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", r.TotalDuration)
	}

	if r.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", r.LoadDuration)
	}

	if r.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", r.PromptEvalCount)
	}

	if r.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", r.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(r.PromptEvalCount)/r.PromptEvalDuration.Seconds())
	}

	if r.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", r.EvalCount)
	}

	if r.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", r.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(r.EvalCount)/r.EvalDuration.Seconds())
	}
}

type Options struct {
	Seed int `json:"seed,omitempty"`

	// Backend options
	UseNUMA bool `json:"numa,omitempty"`

	// Model options
	NumCtx        int  `json:"num_ctx,omitempty"`
	NumBatch      int  `json:"num_batch,omitempty"`
	NumGPU        int  `json:"num_gpu,omitempty"`
	MainGPU       int  `json:"main_gpu,omitempty"`
	LowVRAM       bool `json:"low_vram,omitempty"`
	F16KV         bool `json:"f16_kv,omitempty"`
	LogitsAll     bool `json:"logits_all,omitempty"`
	VocabOnly     bool `json:"vocab_only,omitempty"`
	UseMMap       bool `json:"use_mmap,omitempty"`
	UseMLock      bool `json:"use_mlock,omitempty"`
	EmbeddingOnly bool `json:"embedding_only,omitempty"`

	// Predict options
	RepeatLastN      int     `json:"repeat_last_n,omitempty"`
	RepeatPenalty    float32 `json:"repeat_penalty,omitempty"`
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32 `json:"presence_penalty,omitempty"`
	Temperature      float32 `json:"temperature,omitempty"`
	TopK             int     `json:"top_k,omitempty"`
	TopP             float32 `json:"top_p,omitempty"`
	TFSZ             float32 `json:"tfs_z,omitempty"`
	TypicalP         float32 `json:"typical_p,omitempty"`
	Mirostat         int     `json:"mirostat,omitempty"`
	MirostatTau      float32 `json:"mirostat_tau,omitempty"`
	MirostatEta      float32 `json:"mirostat_eta,omitempty"`

	NumThread int `json:"num_thread,omitempty"`
}

func DefaultOptions() Options {
	return Options{
		Seed: -1,

		UseNUMA: false,

		NumCtx:   2048,
		NumBatch: 512,
		NumGPU:   1,
		LowVRAM:  false,
		F16KV:    true,
		UseMMap:  true,
		UseMLock: false,

		RepeatLastN:      512,
		RepeatPenalty:    1.1,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TFSZ:             1.0,
		TypicalP:         1.0,
		Mirostat:         0,
		MirostatTau:      5.0,
		MirostatEta:      0.1,

		NumThread: runtime.NumCPU(),
	}
}
