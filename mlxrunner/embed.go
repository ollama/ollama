package mlxrunner

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
)

// EmbeddingRequest is a short-lived embedding request carried on the
// runner's EmbedRequests channel from the HTTP handler to the runner
// goroutine (same lifecycle shape as completions' Request/Requests).
type EmbeddingRequest struct {
	Content        string
	Dimensions     int
	Media          [][]byte
	EmbedResponses chan EmbedResponse
	Ctx            context.Context //nolint:containedctx // Queued requests carry caller cancellation to the runner.
}

// EmbedResponse carries one embedding, its prompt token count, or an error.
type EmbedResponse struct {
	Embedding       []float32
	PromptEvalCount int
	Error           *api.StatusError
}

// embedWireRequest / embedWireResponse are the HTTP wire shapes, matching the
// old ollamarunner's llm.EmbeddingRequest/EmbeddingResponse keys, plus the
// ollama-side extension for matryoshka dimension selection.
type embedWireRequest struct {
	Content    string   `json:"content"`
	Dimensions int      `json:"dimensions,omitempty"`
	Media      []string `json:"media,omitempty"`
}

type embedWireResponse struct {
	Embedding       []float32 `json:"embedding"`
	PromptEvalCount int       `json:"prompt_eval_count"`
}

// handleEmbed serves POST /v1/embeddings. Requests flow to the runner
// goroutine via EmbedRequests, which serializes with completions on the
// single consumer of the MLX thread (the same lifecycle as the text
// pipeline).
func (r *Runner) handleEmbed(w http.ResponseWriter, req *http.Request) {
	if r.Model == nil {
		http.Error(w, "model not loaded", http.StatusServiceUnavailable)
		return
	}
	if _, ok := r.Model.(interface{ EmbeddingDim() int }); !ok {
		// Mirrors the old ollamarunner's pooling_type gate.
		http.Error(w, "this model does not support embeddings", http.StatusNotImplemented)
		return
	}

	var wire embedWireRequest
	if err := json.NewDecoder(req.Body).Decode(&wire); err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}
	// An item with neither text nor media has nothing to embed.
	if wire.Content == "" && len(wire.Media) == 0 {
		http.Error(w, "empty content", http.StatusBadRequest)
		return
	}

	// Matryoshka dimension validation against the model's declared set.
	if wire.Dimensions > 0 {
		em := r.Model.(interface {
			EmbeddingDim() int
			EmbeddingDimensions() []int
		})
		valid := em.EmbeddingDimensions()
		if len(valid) > 0 {
			ok := false
			for _, d := range valid {
				if wire.Dimensions == d {
					ok = true
					break
				}
			}
			if !ok {
				http.Error(w, fmt.Sprintf("dimensions %d not supported; valid values are %v", wire.Dimensions, valid), http.StatusBadRequest)
				return
			}
		}
		if wire.Dimensions >= em.EmbeddingDim() {
			http.Error(w, fmt.Sprintf("dimensions %d exceeds the model's embedding dimension %d", wire.Dimensions, em.EmbeddingDim()), http.StatusBadRequest)
			return
		}
	}

	request := EmbeddingRequest{
		Content:        wire.Content,
		Dimensions:     wire.Dimensions,
		EmbedResponses: make(chan EmbedResponse, 1),
		Ctx:            req.Context(),
	}

	if len(wire.Media) > 0 {
		request.Media = make([][]byte, len(wire.Media))
		for i, entry := range wire.Media {
			raw, err := base64.StdEncoding.DecodeString(entry)
			if err != nil {
				http.Error(w, fmt.Sprintf("media[%d]: %v", i, err), http.StatusBadRequest)
				return
			}
			request.Media[i] = raw
		}
	}

	select {
	case <-req.Context().Done():
		return
	case r.EmbedRequests <- request:
	}

	var resp EmbedResponse
	select {
	case <-req.Context().Done():
		return
	case resp = <-request.EmbedResponses:
	}

	if resp.Error != nil {
		http.Error(w, resp.Error.ErrorMessage, resp.Error.StatusCode)
		return
	}

	if err := json.NewEncoder(w).Encode(embedWireResponse{
		Embedding:       resp.Embedding,
		PromptEvalCount: resp.PromptEvalCount,
	}); err != nil {
		slog.Error("failed to encode embedding response", "error", err)
	}
}

// embedBudgetFraction reserves headroom for transient MLX buffers the
// estimate does not model.
const embedBudgetFraction = 0.5

// embedFallbackBudgetBytes is the budget when the device's recommended
// working-set size is unavailable (no GPU).
const embedFallbackBudgetBytes = 1 << 30

// PredictedEmbedAllocationBytes estimates the forward's dominant
// allocations: the attention mask (L*L float32, worst case), the hidden
// output [1, L, D] float32, and the token row.
func PredictedEmbedAllocationBytes(tokens int, hidden int32) int64 {
	if tokens < 0 {
		tokens = 0
	}
	L := int64(tokens)
	const f32 = 4
	mask := L * L * f32
	hiddenOut := L * int64(hidden) * f32
	tokenRow := L * f32
	return mask + hiddenOut + tokenRow
}

// EmbedCapacityBudget is the byte budget a single embed request's
// predicted allocations must fit within.
func EmbedCapacityBudget() int64 {
	budget := int64(embedFallbackBudgetBytes)
	if limit, err := mlx.MaxRecommendedWorkingSetSize(); err == nil && limit > 0 {
		budget = int64(limit)
	}
	return int64(float64(budget) * embedBudgetFraction)
}

// embedHiddenSize is the model's hidden width for the allocation estimate;
// 0 when the model doesn't expose it (the hidden term is minor — the mask
// dominates).
func embedHiddenSize(m any) int32 {
	type hiddenSizer interface{ HiddenSize() int32 }
	if h, ok := m.(hiddenSizer); ok {
		return h.HiddenSize()
	}
	return 0
}

// runEmbed executes one embedding request on the runner goroutine (the MLX
// thread): tokenize (+ media expansion if Media entries ride along) → single
// bidirectional forward → masked mean pool → L2. Media kind sniffing happens
// here because the runner has already dropped the original base64 framing;
// deeper validation is the model's PrepareMedia.
func (r *Runner) runEmbed(ctx context.Context, request EmbeddingRequest) error {
	defer close(request.EmbedResponses)

	fail := func(code int, err error) error {
		request.EmbedResponses <- EmbedResponse{Error: &api.StatusError{StatusCode: code, ErrorMessage: err.Error()}}
		return nil
	}

	// EmbedMedia is the narrow surface gemma4embedding (and any future
	// multimodal embedding arch) exposes to the runner. Text-only models
	// don't implement it and media requests 501 here.
	type embedMedia interface {
		PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error)
		EncodeMedia(prepared *model.PreparedRequest) ([]batch.MediaItem, error)
		MediaTokenStrings() (boi, eoi, image, boa, eoa, audio string)
		SupportsImages() bool
		SupportsAudio() bool
	}

	var tokens []int32
	var prepared *model.PreparedRequest

	if len(request.Media) == 0 {
		// Embedding encoders trained with the HF TemplateProcessing
		// post-processor (not honored by the Go tokenizer) expect explicit
		// BOS + EOS around the text. Append both unconditionally when the
		// vocab defines them.
		tokens = r.Tokenizer.Encode(request.Content, true)
		if len(tokens) == 0 {
			return fail(http.StatusBadRequest, errors.New("empty token sequence"))
		}
		if eos := r.Tokenizer.EOS(); eos >= 0 {
			tokens = append(tokens, eos)
		}
	} else {
		em, ok := r.Model.(embedMedia)
		if !ok {
			return fail(http.StatusNotImplemented, errors.New("media embedding not supported on this model"))
		}

		// Mirror sentence-transformers: it never rewrites the caller's text;
		// it appends one placeholder per media entry and feeds
		// caller-text+appended-placeholders to the model. The bracketed
		// expansions the model sees (BOI...EOI with the soft-token run) are
		// produced by PrepareMedia per segment, not by rewriting the text.
		var segments []model.Segment
		if tt := r.Tokenizer.Encode(request.Content, false); len(tt) > 0 {
			segments = append(segments, model.Segment{Tokens: tt})
		}
		for i, raw := range request.Media {
			var kind string
			if looksLikeImage(raw) {
				kind = "image"
			} else if looksLikeAudio(raw) {
				kind = "audio"
			} else {
				return fail(http.StatusBadRequest, fmt.Errorf("media[%d]: not a PNG/JPEG/GIF/WebP or wav/ogg blob", i))
			}
			switch kind {
			case "image":
				if !em.SupportsImages() {
					return fail(http.StatusBadRequest, errors.New("model has no vision tower but media contains an image"))
				}
			case "audio":
				if !em.SupportsAudio() {
					return fail(http.StatusBadRequest, errors.New("model has no audio tower but media contains audio"))
				}
			}
			segments = append(segments, model.Segment{Kind: kind, Data: raw})
		}

		pr, err := em.PrepareMedia(segments)
		if err != nil {
			return fail(http.StatusBadRequest, err)
		}
		prepared = pr
		// BOS in front, EOS at end, matching the no-media path.
		bos := r.Tokenizer.BOS()
		if bos >= 0 {
			tokens = append([]int32{bos}, prepared.Tokens...)
		} else {
			tokens = prepared.Tokens
		}
		if eos := r.Tokenizer.EOS(); eos >= 0 {
			tokens = append(tokens, eos)
		}
	}

	if maxLen := r.Model.MaxContextLength(); maxLen > 0 && len(tokens) > maxLen {
		// 413 distinguishes "input too long" from other 400s so the server's
		// truncate-and-retry path can tell a real length problem apart from
		// unrelated validation (media sniffing, marker counts, etc.).
		return fail(http.StatusRequestEntityTooLarge,
			fmt.Errorf("the input length (%d tokens) exceeds the context length (%d)", len(tokens), maxLen))
	}

	// Pre-flight: reject before MLX panics at metal::malloc on a forward
	// that cannot fit. 413, like the token-count gate, so the server's
	// truncate retry applies.
	if budget := EmbedCapacityBudget(); budget > 0 {
		if need := PredictedEmbedAllocationBytes(len(tokens), embedHiddenSize(r.Model)); need > budget {
			return fail(http.StatusRequestEntityTooLarge,
				fmt.Errorf("embedding request would allocate %d bytes (budget %d); reduce input length or parallel load", need, budget))
		}
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Per-request hygiene, mirroring the completion pipeline (see
	// TextGenerationPipeline): without the scope + ClearCache, freed
	// buffers accumulate across embeds until metal::malloc panics.
	// EncodeMedia runs inside the scope: its feature graphs are lazy
	// and consumed by Forward, so they must share one scope lifetime.
	mlx.ResetPeakMemory()
	mlx.Scoped(func() {
		var media []batch.MediaItem
		if prepared != nil {
			em := r.Model.(embedMedia)
			mi, err := em.EncodeMedia(prepared)
			if err != nil {
				fail(http.StatusBadRequest, err)
				return
			}
			// Re-base media Pos onto the final stream when a BOS was prepended
			// after PrepareMedia computed them against a BOS-less token list.
			if r.Tokenizer.BOS() >= 0 {
				for i := range mi {
					mi[i].Pos++
				}
			}
			media = mi
		}
		inputIDs := mlx.FromValues(tokens, 1, len(tokens))
		hidden, _ := r.Model.Forward(&batch.Batch{
			InputIDs:     inputIDs,
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tokens))},
			Media:        media,
		}, nil)

		L := int32(hidden.Dim(1))
		D := int32(hidden.Dim(2))
		rows := mlx.Reshape(hidden, L, D)
		// The embedding is plain floats by now, so it escapes the
		// scope as CPU data; every GPU intermediate is freed at exit.
		embedding := meanPoolAndNormalize(rows, len(tokens))
		request.EmbedResponses <- EmbedResponse{Embedding: embedding, PromptEvalCount: len(tokens)}
	})
	mlx.ClearCache()
	return nil
}

// looksLikeImage / looksLikeAudio sniff the container headers the runner
// sees from embedding clients. The model does deeper format validation in
// PrepareMedia; these exist to reject obviously-wrong blobs with an
// actionable message before the tower runs. gemma4's own path has no
// exported sniffer; if that changes, drop these.
func looksLikeImage(b []byte) bool {
	if len(b) < 4 {
		return false
	}
	switch {
	case b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G':
		return true
	case b[0] == 0xff && b[1] == 0xd8:
		return true
	case b[0] == 'G' && b[1] == 'I' && b[2] == 'F':
		return true
	case len(b) >= 12 && string(b[0:4]) == "RIFF" && string(b[8:12]) == "WEBP":
		return true
	}
	return false
}

func looksLikeAudio(b []byte) bool {
	if len(b) < 4 {
		return false
	}
	switch {
	case string(b[0:4]) == "RIFF" && len(b) >= 12 && string(b[8:12]) == "WAVE":
		return true
	case string(b[0:4]) == "OggS":
		return true
	}
	return false
}

// meanPoolAndNormalize implements the sentence-transformers module chain for
// this model (1_Pooling mean + 2_Normalize L2): masked mean over token
// embeddings, then L2 normalization. hidden is [L, D] for a single padded
// row; seqLen is the row's real token count.
func meanPoolAndNormalize(hidden *mlx.Array, seqLen int) []float32 {
	L := hidden.Dim(0)
	D := hidden.Dim(1)

	maskVals := make([]float32, L)
	for i := range maskVals {
		if i < seqLen {
			maskVals[i] = 1
		}
	}
	mask := mlx.FromValues(maskVals, L, 1)

	// Mean over real tokens only (compute in fp32).
	h32 := hidden.AsType(mlx.DTypeFloat32)
	sum := h32.Multiply(mask).SumAxis(0, false)
	count := mlx.FromValue(float32(seqLen))
	pooled := sum.Divide(count)

	// L2 normalize.
	sq := pooled.Multiply(pooled)
	norm := sq.SumAxis(0, false).Sqrt()
	normed := pooled.Divide(mlx.Maximum(norm, mlx.FromValue(float32(1e-12))))

	mlx.Eval(normed)
	vals := normed.Floats()
	if len(vals) != D {
		// Shouldn't happen; Floats returns the full flattened buffer.
		return vals
	}
	return vals
}
