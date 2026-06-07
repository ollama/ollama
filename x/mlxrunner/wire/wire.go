// Package wire defines the HTTP wire types exchanged between the MLX runner
// subprocess and its callers (the ollama server's mlxrunner.Client, and the
// cmd/bench profiling driver). It is deliberately free of any cgo/MLX
// dependency so lightweight tools can import the request/response shapes
// without pulling in the MLX runtime. The runner package re-exports these as
// type aliases, so this package is the single source of truth.
package wire

import (
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// CompletionRequest is the JSON body for POST /v1/completions. Fields use Go
// names on the wire (the runner decodes with stdlib JSON and no struct tags).
type CompletionRequest struct {
	Prompt      string
	Options     api.Options
	Logprobs    bool
	TopLogprobs int

	// IgnoreEOS disables stop-token handling so generation runs for the full
	// requested num_predict. Used by profiling/benchmark drivers to get an
	// exact, attributable number of decode passes. The ollama server never
	// sets it, so production behavior is unchanged.
	IgnoreEOS bool
}

// CompletionResponse is one JSONL record streamed from /v1/completions.
type CompletionResponse struct {
	Content    string
	Done       bool
	DoneReason int

	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration

	Logprobs []llm.Logprob

	Error *api.StatusError
}
