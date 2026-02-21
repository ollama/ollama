//go:build mlx

package mlxrunner

import (
	"bytes"
	"errors"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func prefillChunkSize(lowMemoryDecode bool) int {
	if v := os.Getenv("OLLAMA_MLX_PREFILL_CHUNK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}

	if lowMemoryDecode {
		// Recurrent/no-prompt-cache path favors lower peak memory over prefill throughput.
		// Keep this conservative to avoid transient prefill spikes and allocator thrash.
		return 32
	}
	return 2 << 10
}

func mlxDebugMemoryEnabled() bool {
	return os.Getenv("OLLAMA_MLX_DEBUG_MEMORY") != ""
}

func finalizeRequestCaches(usePromptCache bool, insertCache func(), freeCaches func(), logMemory func(string, int)) {
	if usePromptCache {
		insertCache()
		logMemory("request_done_cached", -1)
		return
	}
	freeCaches()
	logMemory("request_done_freed", -1)
}

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

	inputs := r.Tokenizer.Encode(request.Prompt, true)

	usePromptCache := true
	if m, ok := r.Model.(interface{ DisablePromptCache() bool }); ok && m.DisablePromptCache() {
		usePromptCache = false
	}
	lowMemoryDecode := !usePromptCache
	prefillChunk := prefillChunkSize(lowMemoryDecode)

	var caches []cache.Cache
	var tokens []int32
	if usePromptCache {
		caches, tokens = r.FindNearestCache(inputs)
	} else {
		tokens = inputs
	}

	if len(caches) == 0 {
		if cacheFactory, ok := r.Model.(interface{ NewCaches() []cache.Cache }); ok {
			caches = cacheFactory.NewCaches()
		} else {
			caches = make([]cache.Cache, r.Model.NumLayers())
			for i := range caches {
				caches[i] = cache.NewKVCache()
			}
		}
	}

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.Materialize()...)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}
	freeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.Materialize()...)
		}
		if len(state) == 0 {
			return
		}
		// Non-prompt-cache requests allocate fresh caches every generation.
		// Explicitly free cache roots so graph chains are reclaimed promptly.
		mlx.Free(state...)
		mlx.ClearCache()
	}
	debugMemory := mlxDebugMemoryEnabled()
	logMemory := func(phase string, token int) {
		if !debugMemory {
			return
		}
		if token >= 0 {
			slog.Info("MLX memory", "phase", phase, "token", token, "memory", mlx.Memory{})
			return
		}
		slog.Info("MLX memory", "phase", phase, "memory", mlx.Memory{})
	}
	logMemory("prefill_start", -1)

	total, processed := len(tokens), 0
	slog.Info("Prompt processing progress", "processed", processed, "total", total)
	for total-processed > 1 {
		n := min(prefillChunk, total-processed-1)
		temp := r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		materializeCaches()
		mlx.Free(temp)
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		mlx.ClearCache()
	}
	logMemory("prefill_done", -1)

	step := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		return request.Sample(logprobs), logprobs
	}

	sample, logprobs := step(mlx.FromValues(tokens[processed:], total-processed))
	if !lowMemoryDecode {
		mlx.AsyncEval(sample, logprobs)
	} else {
		// Materialize cache updates to prevent transform graph growth.
		materializeCaches()
	}
	logMemory("decode_init", -1)

	var b bytes.Buffer

	now := time.Now()
	final := Response{Done: true, PromptTokens: total, CompletionTokens: request.Options.MaxTokens, DoneReason: 1}
	outputs := make([]int32, 0, request.Options.MaxTokens)
	for i := range request.Options.MaxTokens {
		if i == 0 {
			slog.Info("Prompt processing progress", "processed", total, "total", total)
			mlx.Eval(sample)
			logMemory("decode_first_eval", i)
			final.PromptTokensDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		outputs = append(outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.Token = int(output)
			final.DoneReason = 0
			final.CompletionTokens = i
			mlx.Free(sample, logprobs)
			break
		}

		request.Responses <- Response{
			Text:  r.Decode(output, &b),
			Token: int(output),
		}

		// For recurrent linear-attention models, avoid async prefetch to reduce
		// peak memory and clear allocator cache every token.
		if lowMemoryDecode {
			mlx.Free(sample, logprobs)
			if i+1 >= request.Options.MaxTokens {
				break
			}
			mlx.ClearCache()
			sample, logprobs = step(mlx.FromValues([]int32{output}, 1))
			// Materialize cache updates to avoid unbounded transform chains.
			materializeCaches()
			if i%32 == 0 {
				logMemory("decode_lowmem_step", i)
			}
			continue
		}

		nextSample, nextLogprobs := step(sample)
		mlx.AsyncEval(nextSample, nextLogprobs)

		mlx.Free(sample, logprobs)
		if i%256 == 0 {
			mlx.ClearCache()
		}
		if i%64 == 0 {
			logMemory("decode_async_step", i)
		}

		sample, logprobs = nextSample, nextLogprobs
	}
	final.CompletionTokensDuration = time.Since(now)
	request.Responses <- final
	finalizeRequestCaches(usePromptCache,
		func() { r.InsertCache(append(inputs, outputs...), caches) },
		freeCaches,
		logMemory,
	)
	return nil
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
