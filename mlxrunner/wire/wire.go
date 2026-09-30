// Package wire defines the HTTP wire types exchanged between the MLX runner
// subprocess and its callers (the ollama server's mlxrunner.Client, and the
// cmd/bench driver). It is deliberately free of any cgo/MLX dependency so
// lightweight tools can import the request/response shapes without pulling in
// the MLX runtime. The runner package re-exports these as type aliases, so this
// package is the single source of truth.
package wire

import (
	"encoding/json"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// CompletionRequest is the JSON body for POST /v1/completions. Fields use Go
// names on the wire (the runner decodes with stdlib JSON and no struct tags).
type CompletionRequest struct {
	Prompt      string
	Media       []llm.MediaData
	Format      json.RawMessage
	Options     api.Options
	Logprobs    bool
	TopLogprobs int

	// IgnoreEOS disables stop-token handling so generation runs for the full
	// requested num_predict. Used by profiling/benchmark drivers to get an
	// exact, attributable number of decode passes. The ollama server never
	// sets it, so production behavior is unchanged.
	IgnoreEOS bool

	// Stats asks for Stats on the final response. Used by benchmark drivers;
	// the ollama server never sets it.
	Stats bool
}

// Stats describes how the runner served one request. Sequence fields are
// attributable to this request alone. Runner fields are whole-runner readings
// taken when the final response is sent, before the request's prefix-cache
// bookkeeping runs, so they reflect other sequences as well once requests
// run concurrently.
type Stats struct {
	// Sequence.
	MatchedTokens int // prompt tokens the prefix cache matched, before holding one back to seed decode
	DraftTokens   int // speculative tokens proposed
	AcceptedDraft int // speculative tokens accepted

	// Runner.
	ActiveBytes int64 // MLX active memory
	PeakBytes   int64 // MLX peak memory since the runner last reset it
	CacheBytes  int64 // MLX buffer cache
	ColdBytes   int64 // prefix-cache snapshot storage
	ColdLimit   int64 // bound the prefix cache evicts snapshot storage to
	ColdEvicted int64 // snapshot bytes evicted since the runner started
}

// CompletionResponse is one JSONL record streamed from /v1/completions.
type CompletionResponse struct {
	Content    string
	Done       bool
	DoneReason int

	PromptEvalCount       int
	PromptEvalCachedCount *int
	PromptEvalDuration    time.Duration
	EvalCount             int
	EvalDuration          time.Duration

	Logprobs []llm.Logprob

	Stats *Stats `json:",omitempty"`

	Error *api.StatusError
}
