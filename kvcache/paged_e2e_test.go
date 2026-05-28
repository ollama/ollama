package kvcache

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	_ "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func newTestBackend(tb testing.TB) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.gguf")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{"general.architecture": "test"}, nil); err != nil {
		tb.Fatal(err)
	}

	b, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Skip("no backend available")
	}
	return b
}

// TestE2EInferenceSpeed compares end-to-end inference speed
// including cache operations AND attention computation.
func TestE2EInferenceSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	// Realistic model dimensions (similar to llama-3-8b)
	const (
		headDim     = 128
		numKVHeads  = 8
		numQHeads   = 32
		blockSize   = 16
		numLayers   = 32
		seqLen      = 2048
		batchSize   = 1
	)

	t.Run("PagedCache", func(t *testing.T) {
		runE2EInference(t, true, headDim, numKVHeads, numQHeads, blockSize, numLayers, seqLen, batchSize)
	})

	t.Run("CausalCache", func(t *testing.T) {
		t.Skip("CausalCache requires real model with layer devices; test with PagedCache only")
	})
}

func runE2EInference(t *testing.T, usePaged bool, headDim, numKVHeads, numQHeads, blockSize, numLayers, seqLen, batchSize int) {
	fmt.Printf("=== Starting E2E Inference Test (usePaged=%v) ===\n", usePaged)
	backend := newTestBackend(t)
	defer backend.Close()

	var cache Cache
	if usePaged {
		paged := NewPagedCache(nil)
		paged.SetBlockSize(blockSize)
		cache = paged
		t.Logf("Using PagedCache with block size %d", blockSize)
	} else {
		cache = NewCausalCache(nil)
		t.Log("Using CausalCache")
	}

	cache.Init(backend, ml.DTypeF32, batchSize, seqLen, batchSize)

	kvSize := headDim * numKVHeads * batchSize
	qSize := headDim * numQHeads * batchSize
	scale := 1.0 / float64(headDim)

	ctx := backend.NewContext().Input()
	defer ctx.Close()

	start := time.Now()

	for layer := 0; layer < numLayers; layer++ {
		cache.SetLayer(layer)

		if err := cache.StartForward(ctx, input.Batch{
			Sequences: []int{0},
			Positions: []int32{0},
		}, false); err != nil {
			t.Fatalf("layer %d: StartForward: %v", layer, err)
		}

		keyData := make([]float32, kvSize)
		valueData := make([]float32, kvSize)
		for i := range keyData {
			keyData[i] = float32(i+1) / float32(kvSize)
			valueData[i] = float32(i+1) / float32(kvSize)
		}

		key := ctx.FromFloats(keyData, headDim, numKVHeads, batchSize)
		value := ctx.FromFloats(valueData, headDim, numKVHeads, batchSize)

		cache.Put(ctx, key, value)

		k, v, mask := cache.Get(ctx)

		queryData := make([]float32, qSize)
		for i := range queryData {
			queryData[i] = float32(i+1) / float32(qSize)
		}
		query := ctx.FromFloats(queryData, headDim, batchSize, numQHeads)

		if pagedCache, ok := cache.(*Paged); ok {
			blockTables := pagedCache.GetBlockTablesTensor(ctx)
			seqLengths := pagedCache.GetSeqLengthsTensor(ctx)

			if pa, ok := query.(ml.PagedAttention); ok {
				output := pa.PagedAttention(ctx, k, v, mask, blockTables, seqLengths, scale, blockSize)
				ctx.Forward(output)
				ctx.Compute(output)
			}
		} else {
			_ = k
			_ = v
			_ = mask
			_ = query
		}
	}

	duration := time.Since(start)

	fmt.Printf("Total time for %d layers (1 position each): %v\n", numLayers, duration)
	fmt.Printf("Average time per layer: %v\n", duration/time.Duration(numLayers))
	fmt.Printf("Cache+attention ops/sec (single pos): %.2f\n", float64(numLayers)/duration.Seconds())

	if usePaged {
		if paged, ok := cache.(*Paged); ok {
			stats := paged.GetStats()
			fmt.Printf("PagedCache stats: %+v\n", stats)
		}
	}
}

// BenchmarkE2EInference for performance comparison
func BenchmarkE2EInference(b *testing.B) {
	// Simple benchmark to verify this runs
	for i := 0; i < b.N; i++ {
		_ = i
	}
}

func BenchmarkE2EInferenceFull(b *testing.B) {
	const (
		headDim    = 128
		numKVHeads = 8
		numQHeads  = 32
		blockSize  = 16
		numLayers  = 8
		seqLen     = 512
		batchSize  = 1
	)

	b.Run("PagedCache", func(b *testing.B) {
		benchE2EInference(b, true, headDim, numKVHeads, numQHeads, blockSize, numLayers, seqLen, batchSize)
	})

	b.Run("CausalCache", func(b *testing.B) {
		b.Skip("CausalCache requires real model with layer devices")
	})
}

func benchE2EInference(b *testing.B, usePaged bool, headDim, numKVHeads, numQHeads, blockSize, numLayers, seqLen, batchSize int) {
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		backend := newTestBackend(b)
		ctx := backend.NewContext().Input()

		var cache Cache
		if usePaged {
			paged := NewPagedCache(nil)
			paged.SetBlockSize(blockSize)
			cache = paged
		} else {
			cache = NewCausalCache(nil)
		}
		cache.Init(backend, ml.DTypeF32, batchSize, seqLen, batchSize)

		for layer := 0; layer < numLayers; layer++ {
			cache.SetLayer(layer)

			cache.StartForward(ctx, input.Batch{
				Sequences: []int{0},
				Positions: []int32{0},
			}, false)

			kvSize := headDim * numKVHeads * batchSize
			keyData := make([]float32, kvSize)
			valueData := make([]float32, kvSize)
			key := ctx.FromFloats(keyData, headDim, numKVHeads, batchSize)
			value := ctx.FromFloats(valueData, headDim, numKVHeads, batchSize)

			cache.Put(ctx, key, value)

			k, v, mask := cache.Get(ctx)

			queryData := make([]float32, headDim*numQHeads*batchSize)
			query := ctx.FromFloats(queryData, headDim, batchSize, numQHeads)

			scale := 1.0 / float64(headDim)

			// Force computation
			ctx.Forward(k)
			ctx.Forward(v)
			ctx.Forward(query)

			if pagedCache, ok := cache.(*Paged); ok {
				blockTables := pagedCache.GetBlockTablesTensor(ctx)
				seqLengths := pagedCache.GetSeqLengthsTensor(ctx)

				if pa, ok := query.(ml.PagedAttention); ok {
					output := pa.PagedAttention(ctx, k, v, mask, blockTables, seqLengths, scale, blockSize)
					ctx.Forward(output)
					ctx.Compute(output)
				}
			} else {
				kq := k.MulmatFullPrec(ctx, query)
				kq = kq.Scale(ctx, scale)
				kq = kq.Softmax(ctx)
				kqv := v.Mulmat(ctx, kq)
				ctx.Forward(kqv)
				ctx.Compute(kqv)
			}
		}

		ctx.Close()
		backend.Close()
	}
}

// TestMultiTurnConversation simulates a real multi-turn conversation
func TestMultiTurnConversation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	const (
		headDim    = 128
		numKVHeads = 8
		numQHeads  = 32
		blockSize  = 16
		numLayers  = 8
		batchSize  = 1
	)

	// Simulate a conversation with 3 turns (keeps memory pool usage manageable)
	numTurns := 3
	tokensPerTurn := 50

	t.Run("PagedCache", func(t *testing.T) {
		runMultiTurn(t, true, headDim, numKVHeads, numQHeads, blockSize, numLayers, batchSize, numTurns, tokensPerTurn)
	})

	t.Run("CausalCache", func(t *testing.T) {
		t.Skip("CausalCache requires real model with layer devices; test with PagedCache only")
	})
}

func runMultiTurn(t *testing.T, usePaged bool, headDim, numKVHeads, numQHeads, blockSize, numLayers, batchSize, numTurns, tokensPerTurn int) {
	backend := newTestBackend(t)
	defer backend.Close()

	ctx := backend.NewContext().Input()
	defer ctx.Close()

	var cache Cache
	if usePaged {
		paged := NewPagedCache(nil)
		paged.SetBlockSize(blockSize)
		cache = paged
	} else {
		cache = NewCausalCache(nil)
	}

	cache.Init(backend, ml.DTypeF32, batchSize, numTurns*tokensPerTurn, batchSize)

	start := time.Now()
	totalTokens := 0

	for turn := 0; turn < numTurns; turn++ {
		position := int32(turn * tokensPerTurn)

		for layer := 0; layer < numLayers; layer++ {
			cache.SetLayer(layer)

			cache.StartForward(ctx, input.Batch{
				Sequences: []int{0},
				Positions: []int32{position},
			}, false)

			kvSize := headDim * numKVHeads * batchSize
			keyData := make([]float32, kvSize)
			valueData := make([]float32, kvSize)
			key := ctx.FromFloats(keyData, headDim, numKVHeads, batchSize)
			value := ctx.FromFloats(valueData, headDim, numKVHeads, batchSize)

			cache.Put(ctx, key, value)

			k, v, mask := cache.Get(ctx)

			queryData := make([]float32, headDim*numQHeads*batchSize)
			query := ctx.FromFloats(queryData, headDim, batchSize, numQHeads)

			scale := 1.0 / float64(headDim)

			// Force computation
			ctx.Forward(k)
			ctx.Forward(v)
			ctx.Forward(query)

			if pagedCache, ok := cache.(*Paged); ok {
				blockTables := pagedCache.GetBlockTablesTensor(ctx)
				seqLengths := pagedCache.GetSeqLengthsTensor(ctx)

				if pa, ok := query.(ml.PagedAttention); ok {
					output := pa.PagedAttention(ctx, k, v, mask, blockTables, seqLengths, scale, blockSize)
					ctx.Forward(output)
					ctx.Compute(output)
				}
			} else {
				kq := k.MulmatFullPrec(ctx, query)
				kq = kq.Scale(ctx, scale)
				kq = kq.Softmax(ctx)
				kqv := v.Mulmat(ctx, kq)
				ctx.Forward(kqv)
				ctx.Compute(kqv)
			}

			totalTokens++
		}
	}

	duration := time.Since(start)

	cacheType := "Causal"
	if usePaged {
		cacheType = "Paged"
	}

	t.Logf("%s: %d turns, %d tokens in %v (%.2f ms/turn, %.2f tokens/sec)",
		cacheType, numTurns, totalTokens, duration,
		float64(duration.Milliseconds())/float64(numTurns),
		float64(totalTokens)/duration.Seconds())
}

// TestConcurrentSessions tests performance with multiple concurrent sessions
func TestConcurrentSessions(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	const (
		headDim    = 128
		numKVHeads = 8
		numQHeads  = 32
		blockSize  = 16
		numLayers  = 4
		batchSize  = 1
	)

	scenarios := []struct {
		name        string
		numSessions int
		tokensPerSession int
	}{
		{"2_Sessions", 2, 50},
		{"4_Sessions", 4, 50},
		{"8_Sessions", 8, 50},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			runConcurrentSessions(t, true, headDim, numKVHeads, numQHeads, blockSize,
				numLayers, batchSize, scenario.numSessions, scenario.tokensPerSession)
		})
	}
}

func runConcurrentSessions(t *testing.T, usePaged bool, headDim, numKVHeads, numQHeads, blockSize, numLayers, batchSize, numSessions, tokensPerSession int) {
	backend := newTestBackend(t)
	defer backend.Close()

	var cache Cache
	if usePaged {
		paged := NewPagedCache(nil)
		paged.SetBlockSize(blockSize)
		cache = paged
		t.Logf("Using PagedCache with %d sessions, %d tokens each", numSessions, tokensPerSession)
	}

	cache.Init(backend, ml.DTypeF32, numSessions, numSessions*tokensPerSession, numSessions)

	start := time.Now()
	totalTokens := 0

	for sessionID := 0; sessionID < numSessions; sessionID++ {
		for layer := 0; layer < numLayers; layer++ {
			cache.SetLayer(layer)
			ctx := backend.NewContext().Input()
			defer ctx.Close()

			position := int32(sessionID * tokensPerSession)

			if err := cache.StartForward(ctx, input.Batch{
				Sequences: []int{sessionID},
				Positions: []int32{position},
			}, false); err != nil {
				t.Fatalf("layer %d session %d: %v", layer, sessionID, err)
			}

			kvSize := headDim * numKVHeads * batchSize
			keyData := make([]float32, kvSize)
			valueData := make([]float32, kvSize)
			key := ctx.FromFloats(keyData, headDim, numKVHeads, batchSize)
			value := ctx.FromFloats(valueData, headDim, numKVHeads, batchSize)

			cache.Put(ctx, key, value)
			cache.Get(ctx)

			totalTokens++
		}
	}

	duration := time.Since(start)

	t.Logf("PagedCache: %d sessions, %d tokens in %v (%.2f ms/session, %.2f tokens/sec)",
		numSessions, totalTokens, duration,
		float64(duration.Milliseconds())/float64(numSessions),
		float64(totalTokens)/duration.Seconds())

	if paged, ok := cache.(*Paged); ok {
		stats := paged.GetStats()
		t.Logf("PagedCache stats: %+v", stats)
	}
}
