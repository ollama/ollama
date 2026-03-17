package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func prefillChunkSize() int {
	return 2 << 10
}

var imgTagRe = regexp.MustCompile(`\[img-(\d+)\]`)

func (r *Runner) TextGenerationPipeline(request Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}
	mlx.ResetPeakMemory()
	ctx := request.Ctx
	var (
		sample, logprobs         *mlx.Array
		nextSample, nextLogprobs *mlx.Array
	)

	defer func() {
		if request.Sampler != nil {
			request.Sampler.Free()
		}
		mlx.Unpin(sample, logprobs)
		mlx.Unpin(nextSample, nextLogprobs)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.log()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))
	}()

	// Check for multimodal model with images.
	mmModel, isMultimodal := r.Model.(base.MultimodalModel)
	hasImages := isMultimodal && len(request.Images) > 0

	// Tokenize prompt. For multimodal models with images, parse [img-N]
	// tags and build a token sequence with placeholder tokens.
	var inputs []int32
	var mmSegments []base.MultimodalSegment

	if hasImages {
		var err error
		inputs, mmSegments, err = r.tokenizeWithImages(request.Prompt, request.Images, mmModel)
		if err != nil {
			return err
		}
	} else {
		inputs = r.Tokenizer.Encode(request.Prompt, true)
	}

	if len(inputs) == 0 {
		return errors.New("empty prompt")
	}

	if len(inputs) >= r.contextLength {
		return api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: fmt.Sprintf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(inputs), r.contextLength),
		}
	}

	// Cap generation to stay within the model's context length.
	maxGenerate := r.contextLength - len(inputs)
	if request.Options.MaxTokens <= 0 {
		request.Options.MaxTokens = maxGenerate
	} else {
		request.Options.MaxTokens = min(request.Options.MaxTokens, maxGenerate)
	}

	request.Sampler.ResetHistory(inputs)

	session := r.cache.begin(r.Model, inputs)
	defer session.close()
	caches := session.caches
	tokens := session.remaining
	prefillChunk := prefillChunkSize()

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = appendCacheState(state, c)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}

	now := time.Now()
	total, processed := len(tokens), 0

	if hasImages && len(mmSegments) > 0 {
		// Check if remaining tokens contain any multimodal placeholders.
		// If not, the KV cache already covers them from a prior turn
		// and we can skip encoding entirely.
		offset := len(inputs) - len(tokens)
		adjustedSegments := make([]base.MultimodalSegment, 0, len(mmSegments))
		for _, seg := range mmSegments {
			if seg.StartPos >= offset && seg.StartPos+seg.NumTokens <= offset+len(tokens) {
				adjusted := seg
				adjusted.StartPos -= offset
				adjustedSegments = append(adjustedSegments, adjusted)
			}
		}

		if len(adjustedSegments) > 0 {
			n, err := mmModel.Prefill(tokens, adjustedSegments, caches, prefillChunk, materializeCaches)
			if err != nil {
				return err
			}
			processed = n
			slog.Info("Multimodal prefill complete", "processed", processed, "total", total, "segments", len(adjustedSegments))
		} else {
			slog.Info("Multimodal tokens cached from prior turn, skipping encoding")
		}
	}

	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return err
		}

		n := min(prefillChunk, total-processed-1)
		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		mlx.ClearCache()
	}

	step := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		sample := request.Sampler.Sample(logprobs)

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		mlx.AsyncEval(sample, logprobs)

		return sample, logprobs
	}

	sample, logprobs = step(mlx.FromValues(tokens[processed:], total-processed))

	var b bytes.Buffer

	final := CompletionResponse{Done: true, PromptEvalCount: len(inputs), EvalCount: request.Options.MaxTokens, DoneReason: 1}
	for i := range request.Options.MaxTokens {
		if err := ctx.Err(); err != nil {
			return err
		}

		request.Sampler.AppendToken(sample)
		nextSample, nextLogprobs = step(sample)

		if i == 0 {
			mlx.Eval(sample)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		session.outputs = append(session.outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.DoneReason = 0
			final.EvalCount = i
			break
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- CompletionResponse{
			Content: r.Decode(output, &b),
		}:
		}

		mlx.Unpin(sample, logprobs)
		sample, logprobs = nextSample, nextLogprobs
		nextSample, nextLogprobs = nil, nil

		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalDuration = time.Since(now)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

// tokenizeWithImages parses [img-N] tags from the prompt and builds a token
// sequence with multimodal placeholder tokens at the right positions.
// Preprocessing is delegated to the model via EncodeMultimodal.
func (r *Runner) tokenizeWithImages(prompt string, images []imageData, vm base.MultimodalModel) ([]int32, []base.MultimodalSegment, error) {
	// Split prompt on [img-N] tags.
	parts := imgTagRe.Split(prompt, -1)
	matches := imgTagRe.FindAllStringSubmatch(prompt, -1)

	// Encode each input via the model's EncodeMultimodal. The model
	// determines the modality from the data and returns the appropriate
	// placeholder token ID (e.g., image_token_id vs audio_token_id).
	encoded := make(map[int]base.EncodedMultimodal)

	for _, img := range images {
		enc, err := vm.EncodeMultimodal(img.Data)
		if err != nil {
			return nil, nil, fmt.Errorf("encode multimodal input %d: %w", img.ID, err)
		}
		encoded[img.ID] = enc
		slog.Info("Multimodal input encoded", "id", img.ID, "tokens", enc.NumTokens)
	}

	// Build token sequence: tokenize text parts, insert placeholder runs.
	var allTokens []int32
	var segments []base.MultimodalSegment

	for i, part := range parts {
		if part != "" {
			tokens := r.Tokenizer.Encode(part, i == 0)
			allTokens = append(allTokens, tokens...)
		} else if i == 0 {
			tokens := r.Tokenizer.Encode("", true)
			allTokens = append(allTokens, tokens...)
		}

		if i < len(matches) {
			imgID, _ := strconv.Atoi(matches[i][1])
			enc, ok := encoded[imgID]
			if !ok {
				return nil, nil, fmt.Errorf("multimodal input %d referenced in prompt but not provided", imgID)
			}

			allTokens = append(allTokens, enc.PrefixTokens...)
			startPos := len(allTokens)
			for range enc.NumTokens {
				allTokens = append(allTokens, enc.PlaceholderID)
			}
			allTokens = append(allTokens, enc.SuffixTokens...)
			segments = append(segments, base.MultimodalSegment{
				Data:      enc.Data,
				ID:        imgID,
				StartPos:  startPos,
				NumTokens: enc.NumTokens,
			})
		}
	}

	return allTokens, segments, nil
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
