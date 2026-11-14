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

// PrefillSnapshotInterval is the periodic restore-point spacing in tokens.
// Cache scenarios use the same spacing to check mid-prompt prefix reuse.
const PrefillSnapshotInterval = 8192

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

	// PrefillOnly evaluates the prompt and returns without generating. Used by
	// profiling drivers; the ollama server never sets it.
	PrefillOnly bool

	// Stats asks for Stats on the final response. Used by benchmark drivers;
	// the ollama server never sets it.
	Stats bool
}

// Stats describes how the runner served one request. Sequence fields are
// attributable to this request alone, as are the cold-cache readings, which come
// from the request's own slot. The MLX memory readings are whole-runner values
// taken when the final response is sent, so they include other sequences once
// requests run concurrently.
type Stats struct {
	// Sequence.
	MatchedTokens int // prompt tokens the prefix cache matched, before holding one back to seed decode
	DraftTokens   int // speculative tokens proposed
	AcceptedDraft int // speculative tokens accepted

	// Per-slot snapshot storage.
	ColdBytes   int64 // prefix-cache snapshot storage
	ColdLimit   int64 // bound this slot's prefix cache evicts snapshot storage to
	ColdEvicted int64 // snapshot bytes this slot has evicted

	// Runner.
	ActiveBytes int64 // MLX active memory
	PeakBytes   int64 // MLX peak memory during this request
	CacheBytes  int64 // MLX buffer cache
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
