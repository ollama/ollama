package kvcache

import (
	"fmt"
	"testing"
	"time"

	_ "github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// ModelConfig represents different model architectures
type ModelConfig struct {
	Name       string
	HeadDim    int
	NumKVHeads int
	NumQHeads  int
	NumLayers  int
	BlockSize  int
	SeqLen     int
	BatchSize  int
}

// Common model configurations
var modelConfigs = []ModelConfig{
	{
		Name:       "TinyLlama",  // ~1B params
		HeadDim:    64,
		NumKVHeads: 4,
		NumQHeads:   4,
		NumLayers:  22,
		BlockSize:  16,
		SeqLen:     2048,
		BatchSize:  1,
	},
	{
		Name:       "Llama-3-8B",  // ~8B params
		HeadDim:    128,
		NumKVHeads: 8,
		NumQHeads:   32,
		NumLayers:  32,
		BlockSize:  16,
		SeqLen:     8192,
		BatchSize:  1,
	},
	{
		Name:       "Llama-3-70B", // ~70B params
		HeadDim:    128,
		NumKVHeads: 8,
		NumQHeads:   64,
		NumLayers:  80,
		BlockSize:  16,
		SeqLen:     8192,
		BatchSize:  1,
	},
	{
		Name:       "Mistral-7B",  // ~7B params
		HeadDim:    128,
		NumKVHeads: 8,
		NumQHeads:   32,
		NumLayers:  32,
		BlockSize:  16,
		SeqLen:     32768,
		BatchSize:  1,
	},
	{
		Name:       "Mixtral-8x7B", // MoE ~47B params
		HeadDim:    128,
		NumKVHeads: 8,
		NumQHeads:   32,
		NumLayers:  32,
		BlockSize:  16,
		SeqLen:     32768,
		BatchSize:  1,
	},
	{
		Name:       "Qwen-72B",     // ~72B params
		HeadDim:    128,
		NumKVHeads: 8,
		NumQHeads:   64,
		NumLayers:  80,
		BlockSize:  16,
		SeqLen:     8192,
		BatchSize:  1,
	},
}

// TestModelConfigurations tests PagedCache with various model architectures
func TestModelConfigurations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model config test in short mode")
	}

	for _, config := range modelConfigs {
		t.Run(config.Name, func(t *testing.T) {
			runModelConfigTest(t, config)
		})
	}
}

func runModelConfigTest(t *testing.T, config ModelConfig) {
	fmt.Printf("\n=== Testing %s ===\n", config.Name)
	fmt.Printf("HeadDim: %d, KVHeads: %d, QHeads: %d, Layers: %d, SeqLen: %d\n",
		config.HeadDim, config.NumKVHeads, config.NumQHeads, config.NumLayers, config.SeqLen)

	backend := newTestBackend(t)
	defer backend.Close()

	paged := NewPagedCache(nil)
	paged.SetBlockSize(config.BlockSize)

	testTokens := minInt(config.SeqLen, 16)
	numBlocks := (testTokens / config.BlockSize) + 20
	fmt.Printf("Allocating %d blocks for %d tokens (blockSize=%d)\n", numBlocks, testTokens, config.BlockSize)

	paged.Init(backend, ml.DTypeF32, config.BatchSize, testTokens, config.BatchSize)

	testLayers := minInt(config.NumLayers, 8)

	kvSize := config.HeadDim * config.NumKVHeads * config.BatchSize
	qSize := config.HeadDim * config.NumQHeads * config.BatchSize
	scale := 1.0 / float64(config.HeadDim)

	start := time.Now()
	totalOps := 0

	for layer := 0; layer < testLayers; layer++ {
		paged.SetLayer(layer)
		ctx := backend.NewContext().Input()

		if err := paged.StartForward(ctx, input.Batch{
			Sequences: []int{0},
			Positions: []int32{int32(testTokens - 1)},
		}, false); err != nil {
			t.Fatalf("layer %d pre-alloc: %v", layer, err)
		}

		for pos := 0; pos < testTokens; pos++ {
			if err := paged.StartForward(ctx, input.Batch{
				Sequences: []int{0},
				Positions: []int32{int32(pos)},
			}, false); err != nil {
				t.Fatalf("layer %d pos %d: %v", layer, pos, err)
			}

			keyData := make([]float32, kvSize)
			valueData := make([]float32, kvSize)
			for i := range keyData {
				keyData[i] = float32(i+1) / float32(kvSize)
				valueData[i] = float32(i+1) / float32(kvSize)
			}
			key := ctx.FromFloats(keyData, config.HeadDim, config.NumKVHeads, config.BatchSize)
			value := ctx.FromFloats(valueData, config.HeadDim, config.NumKVHeads, config.BatchSize)

			paged.Put(ctx, key, value)

			k, v, mask := paged.Get(ctx)

			queryData := make([]float32, qSize)
			for i := range queryData {
				queryData[i] = float32(i+1) / float32(qSize)
			}
			query := ctx.FromFloats(queryData, config.HeadDim, config.BatchSize, config.NumQHeads)

			if pa, ok := query.(ml.PagedAttention); ok {
				blockTables := paged.GetBlockTablesTensor(ctx)
				seqLengths := paged.GetSeqLengthsTensor(ctx)
				output := pa.PagedAttention(ctx, k, v, mask, blockTables, seqLengths, scale, config.BlockSize)
				ctx.Forward(output)
				ctx.Compute(output)
			} else {
				_ = k
				_ = v
				_ = mask
			}

			totalOps++
		}

		ctx.Close()
	}

	duration := time.Since(start)

	fmt.Printf("Results for %s:\n", config.Name)
	fmt.Printf("  Processed: %d layers x %d positions = %d cache+attention ops\n", testLayers, testTokens, totalOps)
	fmt.Printf("  Time: %v\n", duration)
	fmt.Printf("  Latency: %.2f ms/op\n", float64(duration.Milliseconds())/float64(totalOps))
	fmt.Printf("  ops/sec: %.2f\n", float64(totalOps)/duration.Seconds())

	stats := paged.GetStats()
	fmt.Printf("  Memory: %d blocks allocated, %d free (%.1f%% utilization)\n",
		stats.AllocatedBlocks, stats.FreeBlocks, stats.Utilization*100)
}

// TestBlockSizeImpact tests different block sizes for the same model
func TestBlockSizeImpact(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping block size test in short mode")
	}

	baseConfig := ModelConfig{
		Name:       "Llama-3-8B",
		HeadDim:    128,
		NumKVHeads: 8,
		NumQHeads:   32,
		NumLayers:  4, // Reduced for testing
		SeqLen:     2048,
		BatchSize:  1,
	}

	blockSizes := []int{8, 16, 32, 64, 128}

	for _, blockSize := range blockSizes {
		t.Run(fmt.Sprintf("BlockSize_%d", blockSize), func(t *testing.T) {
			config := baseConfig
			config.BlockSize = blockSize
			runBlockSizeTest(t, config, blockSize)
		})
	}
}

func runBlockSizeTest(t *testing.T, config ModelConfig, blockSize int) {
	backend := newTestBackend(t)
	defer backend.Close()

	paged := NewPagedCache(nil)
	paged.SetBlockSize(blockSize)
	paged.Init(backend, ml.DTypeF32, config.BatchSize, config.SeqLen, config.BatchSize)

	start := time.Now()
	testTokens := 256

	for layer := 0; layer < config.NumLayers; layer++ {
		paged.SetLayer(layer)
		ctx := backend.NewContext().Input()

		for pos := 0; pos < testTokens; pos++ {
			paged.StartForward(ctx, input.Batch{
				Sequences: []int{0},
				Positions: []int32{int32(pos)},
			}, false)

			kvSize := config.HeadDim * config.NumKVHeads
			keyData := make([]float32, kvSize)
			valueData := make([]float32, kvSize)
			key := ctx.FromFloats(keyData, config.HeadDim, config.NumKVHeads, 1)
			value := ctx.FromFloats(valueData, config.HeadDim, config.NumKVHeads, 1)

			paged.Put(ctx, key, value)
			paged.Get(ctx)
		}

		ctx.Close()
	}

	duration := time.Since(start)
	stats := paged.GetStats()

	fmt.Printf("BlockSize %d: %.2f ms/token, %d blocks, %.1f%% util\n",
		blockSize,
		float64(duration.Milliseconds())/float64(testTokens*config.NumLayers),
		stats.NumBlocks,
		stats.Utilization*100)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
