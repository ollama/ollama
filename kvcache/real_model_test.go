package kvcache_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	_ "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/models/llama"
	"github.com/ollama/ollama/model/input"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/tokenizer"
)

const modelPath = "/usr/share/ollama/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"

func loadModel(t *testing.T) (model.Model, tokenizer.Tokenizer) {
	t.Helper()

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found: %s", modelPath)
	}

	m, err := model.New(modelPath, ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	tok, ok := m.(tokenizer.Tokenizer)
	if !ok {
		t.Fatal("model does not implement tokenizer.Tokenizer")
	}

	return m, tok
}

type cacheSetter interface {
	SetCache(kvcache.Cache)
}

func runPagedInference(t *testing.T, m model.Model, tokens []int32, warmup bool) time.Duration {
	t.Helper()

	capacity := len(tokens) + 256
	paged := kvcache.NewPagedCache(nil)
	paged.SetBlockSize(16)

	base, ok := m.(cacheSetter)
	if !ok {
		t.Fatal("model does not support SetCache")
	}
	base.SetCache(paged)

	paged.Init(m.Backend(), ml.DTypeF16, 1, capacity, 1)

	// Pre-allocate blocks for the full sequence (uses dummy context)
	preCtx := m.Backend().NewContext().Input()
	if err := paged.StartForward(preCtx, input.Batch{
		Sequences: []int{0},
		Positions: []int32{int32(len(tokens) - 1)},
	}, false); err != nil {
		t.Fatalf("pre-alloc failed: %v", err)
	}
	preCtx.Close()

	var totalDuration time.Duration
	tokenCount := 0

	for pos, tokenID := range tokens {
		ctx := m.Backend().NewContext().Input()

		inputTensor := ctx.FromInts([]int32{tokenID}, 1)
		batch := input.Batch{
			Inputs:    inputTensor,
			Sequences: []int{0},
			Positions: []int32{int32(pos)},
		}

		if err := paged.StartForward(ctx, batch, false); err != nil {
			ctx.Close()
			t.Fatalf("startForward pos %d: %v", pos, err)
		}

		start := time.Now()
		_, err := m.Forward(ctx, batch)
		if err != nil {
			ctx.Close()
			t.Fatalf("forward pos %d: %v", pos, err)
		}
		elapsed := time.Since(start)

		ctx.Close()

		if !warmup {
			totalDuration += elapsed
			tokenCount++
		}
	}

	if warmup {
		return 0
	}

	return totalDuration
}

func TestRealModelInference_PagedCache(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real model test in short mode")
	}

	m, tok := loadModel(t)

	t.Logf("Model loaded, backend: %T", m.Backend())
	if v, ok := m.(interface{ NumLayers() int }); ok {
		t.Logf("NumLayers: %d", v.NumLayers())
	}

	testPrompt := "Hello, my name is"
	tokensRaw, err := tok.Encode(testPrompt, true)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	t.Logf("Prompt: %q -> %d tokens: %v", testPrompt, len(tokensRaw), tokensRaw)

	seqLen := 32
	if len(tokensRaw) < seqLen {
		padding := 32 - len(tokensRaw)
		for range padding {
			tokensRaw = append(tokensRaw, 352)
		}
	}
	tokens := tokensRaw[:seqLen]

	fmt.Printf("\n=== Real Model Inference (Mistral-7B-Instruct-v0.3) ===\n")
	fmt.Printf("Prompt: %q\n", testPrompt)
	fmt.Printf("Tokens: %d\n\n", seqLen)

	t.Log("=== Warmup ===")
	runSingleTokenForward(t, m, tokens, true)

	t.Log("=== CausalCache (per-token) ===")
	dCausal := runSingleTokenForward(t, m, tokens, false)
	fmt.Printf("CausalCache (per-token): %.2f t/s (per-token latency: %.2f ms)\n",
		float64(seqLen)/dCausal.Seconds(),
		float64(dCausal.Milliseconds())/float64(seqLen))

	t.Log("=== PagedCache (per-token) ===")
	dPaged := runPagedInference(t, m, tokens, true)
	dPaged = runPagedInference(t, m, tokens, false)
	fmt.Printf("PagedCache   (per-token): %.2f t/s (per-token latency: %.2f ms)\n",
		float64(seqLen)/dPaged.Seconds(),
		float64(dPaged.Milliseconds())/float64(seqLen))

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("PagedCache / CausalCache ratio: %.1f%%\n",
		100.0*float64(dCausal)/float64(dPaged))
}

func runSingleTokenForward(t *testing.T, m model.Model, tokens []int32, warmup bool) time.Duration {
	t.Helper()

	cg, ok := m.(cacheGetter)
	if !ok {
		t.Fatal("model does not implement GetCache")
	}
	cache := cg.GetCache()

	capacity := len(tokens) + 256
	cache.Init(m.Backend(), ml.DTypeF16, 1, capacity, 1)

	// Pre-allocate blocks
	preCtx := m.Backend().NewContext().Input()
	if err := cache.StartForward(preCtx, input.Batch{
		Sequences: []int{0},
		Positions: []int32{int32(len(tokens) - 1)},
	}, false); err != nil {
		preCtx.Close()
		t.Fatalf("pre-alloc failed: %v", err)
	}
	preCtx.Close()

	var totalDuration time.Duration
	tokenCount := 0

	for pos, tokenID := range tokens {
		ctx := m.Backend().NewContext().Input()

		inputTensor := ctx.FromInts([]int32{tokenID}, 1)
		batch := input.Batch{
			Inputs:    inputTensor,
			Sequences: []int{0},
			Positions: []int32{int32(pos)},
		}

		if err := cache.StartForward(ctx, batch, false); err != nil {
			ctx.Close()
			t.Fatalf("startForward pos %d: %v", pos, err)
		}

		start := time.Now()
		_, err := m.Forward(ctx, batch)
		if err != nil {
			ctx.Close()
			t.Fatalf("forward pos %d: %v", pos, err)
		}
		elapsed := time.Since(start)

		ctx.Close()

		if !warmup {
			totalDuration += elapsed
			tokenCount++
		}
	}

	if warmup {
		return 0
	}

	return totalDuration
}

type cacheGetter interface {
	GetCache() kvcache.Cache
}
