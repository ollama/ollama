//go:build mlx && cgo

package constrained

import (
	"testing"
	"time"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// JSON grammar in GBNF format (llama.cpp format) for comparison
// Using a simpler grammar that's easier to debug
const jsonGBNF = `
root   ::= value
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws "}" | "{" members "}"
members ::= member ("," member)*
member ::= ws string ws ":" element
array  ::= "[" ws "]" | "[" elements "]"
elements ::= element ("," element)*
element ::= ws value ws
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? ("0" | [1-9][0-9]*) ("." [0-9]+)? ([eE][+-]?[0-9]+)?
ws     ::= [ \t\n\r]*
`

// TestLlamaCppGrammarBasic tests basic llama.cpp grammar functionality
func TestLlamaCppGrammarBasic(t *testing.T) {
	// Create a small vocab with just the tokens we need
	vocabValues := []string{
		"{",       // 0
		"}",       // 1
		"\"key\"", // 2
		":",       // 3
		"\"val\"", // 4
		"true",    // 5
		"false",   // 6
		"null",    // 7
		"123",     // 8
		"[",       // 9
		"]",       // 10
		",",       // 11
		" ",       // 12
		"\n",      // 13
	}

	vocabIds := make([]uint32, len(vocabValues))
	for i := range vocabIds {
		vocabIds[i] = uint32(i)
	}

	t.Logf("Creating grammar with %d tokens", len(vocabValues))
	for i, v := range vocabValues {
		t.Logf("  Token %d: %q", i, v)
	}

	g := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
	if g == nil {
		t.Fatal("Failed to create grammar")
	}
	defer g.Free()
	t.Log("Grammar created successfully")

	// Create token data
	tokens := make([]llama.TokenData, len(vocabValues))
	for i := range tokens {
		tokens[i] = llama.TokenData{ID: int32(i), Logit: 1.0}
	}

	// Apply grammar - this should mask invalid tokens
	g.Apply(tokens)

	t.Log("After Apply() at initial state:")
	for i, tok := range tokens {
		if tok.Logit > -1000 { // Not masked
			t.Logf("  Valid token %d: %q (logit=%.2f)", i, vocabValues[i], tok.Logit)
		}
	}

	// Try accepting "{" (token 0)
	t.Log("\nAccepting token 0 ({)...")
	g.Accept(0)

	// Reset logits and apply again
	for i := range tokens {
		tokens[i].Logit = 1.0
	}
	g.Apply(tokens)

	t.Log("After accepting '{', valid tokens:")
	for i, tok := range tokens {
		if tok.Logit > -1000 {
			t.Logf("  Valid token %d: %q (logit=%.2f)", i, vocabValues[i], tok.Logit)
		}
	}
}

// TestComparisonBenchmarkSmallVocab compares MLX vs llama.cpp with a small vocab
func TestComparisonBenchmarkSmallVocab(t *testing.T) {
	// Small vocab that works with both systems
	vocabValues := []string{
		"{",       // 0
		"}",       // 1
		"\"key\"", // 2
		":",       // 3
		"\"val\"", // 4
		"true",    // 5
		"false",   // 6
		"null",    // 7
		"123",     // 8
		"[",       // 9
		"]",       // 10
		",",       // 11
		" ",       // 12
		"\n",      // 13
	}
	vocabSize := len(vocabValues)

	vocabIds := make([]uint32, vocabSize)
	for i := range vocabIds {
		vocabIds[i] = uint32(i)
	}

	iterations := 10000

	// ===== MLX Engine =====
	mlxEngine, err := NewEngine(vocabValues)
	if err != nil {
		t.Fatalf("Failed to create MLX engine: %v", err)
	}
	defer mlxEngine.Close()

	logits := mlx.Ones(int32(vocabSize))
	mlx.Keep(logits)

	// Warm up MLX
	for i := 0; i < 100; i++ {
		_ = mlxEngine.ApplyMask(logits)
	}
	mlx.Eval(logits)

	// Benchmark MLX (no eval - graph building only)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = mlxEngine.ApplyMask(logits)
	}
	mlxDuration := time.Since(start)
	mlxAvg := mlxDuration / time.Duration(iterations)

	// ===== llama.cpp Grammar =====
	llamaGrammar := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
	if llamaGrammar == nil {
		t.Fatal("Failed to create llama.cpp grammar")
	}
	defer llamaGrammar.Free()

	tokens := make([]llama.TokenData, vocabSize)
	for i := range tokens {
		tokens[i] = llama.TokenData{ID: int32(i), Logit: 1.0}
	}

	// Warm up llama.cpp
	for i := 0; i < 100; i++ {
		for j := range tokens {
			tokens[j].Logit = 1.0
		}
		llamaGrammar.Apply(tokens)
	}

	// Benchmark llama.cpp
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for j := range tokens {
			tokens[j].Logit = 1.0
		}
		llamaGrammar.Apply(tokens)
	}
	llamaDuration := time.Since(start)
	llamaAvg := llamaDuration / time.Duration(iterations)

	// ===== Sequence benchmark =====
	sequence := []int32{0, 2, 3, 4, 1} // {, "key", :, "val", }

	// MLX sequence (no per-token eval)
	start = time.Now()
	for i := 0; i < iterations; i++ {
		mlxEngine.Reset()
		for _, tokenId := range sequence {
			_ = mlxEngine.ApplyMask(logits)
			mlxEngine.AcceptString(vocabValues[tokenId])
		}
	}
	mlxSeqDuration := time.Since(start)
	mlxSeqAvg := mlxSeqDuration / time.Duration(iterations*len(sequence))

	// llama.cpp sequence
	start = time.Now()
	for i := 0; i < iterations; i++ {
		g := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
		if g == nil {
			continue
		}
		for _, tokenId := range sequence {
			for j := range tokens {
				tokens[j].Logit = 1.0
			}
			g.Apply(tokens)
			g.Accept(tokenId)
		}
		g.Free()
	}
	llamaSeqDuration := time.Since(start)
	llamaSeqAvg := llamaSeqDuration / time.Duration(iterations*len(sequence))

	// ===== Print Results =====
	t.Logf("\n")
	t.Logf("╔══════════════════════════════════════════════════════════════╗")
	t.Logf("║     MLX vs llama.cpp COMPARISON (vocab=%d tokens)            ║", vocabSize)
	t.Logf("╠══════════════════════════════════════════════════════════════╣")
	t.Logf("║                                                              ║")
	t.Logf("║  Single Apply (graph building / mask computation):           ║")
	t.Logf("║    MLX (no eval):   %12v                              ║", mlxAvg)
	t.Logf("║    llama.cpp:       %12v                              ║", llamaAvg)
	if mlxAvg < llamaAvg {
		t.Logf("║    MLX is %.1fx FASTER                                       ║", float64(llamaAvg)/float64(mlxAvg))
	} else {
		t.Logf("║    llama.cpp is %.1fx FASTER                                 ║", float64(mlxAvg)/float64(llamaAvg))
	}
	t.Logf("║                                                              ║")
	t.Logf("║  Sequence {\"key\":\"val\"} (5 tokens):                          ║")
	t.Logf("║    MLX (no eval):   %12v/token                        ║", mlxSeqAvg)
	t.Logf("║    llama.cpp:       %12v/token                        ║", llamaSeqAvg)
	if mlxSeqAvg < llamaSeqAvg {
		t.Logf("║    MLX is %.1fx FASTER                                       ║", float64(llamaSeqAvg)/float64(mlxSeqAvg))
	} else {
		t.Logf("║    llama.cpp is %.1fx FASTER                                 ║", float64(mlxSeqAvg)/float64(llamaSeqAvg))
	}
	t.Logf("║                                                              ║")
	t.Logf("║  Throughput:                                                 ║")
	t.Logf("║    MLX:             %8.0f tokens/sec                      ║", float64(time.Second)/float64(mlxSeqAvg))
	t.Logf("║    llama.cpp:       %8.0f tokens/sec                      ║", float64(time.Second)/float64(llamaSeqAvg))
	t.Logf("╚══════════════════════════════════════════════════════════════╝")
}

// createComparisonVocab creates vocabulary data for both systems
func createComparisonVocab(size int) ([]string, []uint32, []string) {
	vocab := createBenchVocabN(size)

	// Create vocab IDs and values for llama.cpp
	vocabIds := make([]uint32, size)
	vocabValues := make([]string, size)
	for i := 0; i < size; i++ {
		vocabIds[i] = uint32(i)
		vocabValues[i] = vocab[i]
	}

	return vocab, vocabIds, vocabValues
}

// ============ Comparison Benchmarks ============

// BenchmarkComparison_MLX benchmarks our MLX engine
func BenchmarkComparison_MLX_128k(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	// Warm up
	for i := 0; i < 10; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}
}

// BenchmarkComparison_LlamaCpp benchmarks llama.cpp grammar
func BenchmarkComparison_LlamaCpp_128k(b *testing.B) {
	_, vocabIds, vocabValues := createComparisonVocab(128000)

	// Create llama.cpp grammar
	g := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
	if g == nil {
		b.Skip("Failed to create llama.cpp grammar")
		return
	}
	defer g.Free()

	// Create token data
	tokens := make([]llama.TokenData, 128000)
	for i := range tokens {
		tokens[i] = llama.TokenData{ID: int32(i), Logit: 1.0}
	}

	// Warm up
	for i := 0; i < 10; i++ {
		// Reset logits
		for j := range tokens {
			tokens[j].Logit = 1.0
		}
		g.Apply(tokens)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset logits
		for j := range tokens {
			tokens[j].Logit = 1.0
		}
		g.Apply(tokens)
	}
}

// BenchmarkComparison_MLX_Sequence benchmarks MLX with full sequence
func BenchmarkComparison_MLX_Sequence(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Reset()
		for _, token := range sequence {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
			e.AcceptString(token)
		}
	}
}

// BenchmarkComparison_LlamaCpp_Sequence benchmarks llama.cpp with full sequence
func BenchmarkComparison_LlamaCpp_Sequence(b *testing.B) {
	vocab, vocabIds, vocabValues := createComparisonVocab(128000)

	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}

	// Find token IDs for sequence
	tokenIds := make([]int32, len(sequence))
	for i, tok := range sequence {
		for j, v := range vocab {
			if v == tok {
				tokenIds[i] = int32(j)
				break
			}
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create fresh grammar for each iteration (grammar state changes on accept)
		g := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
		if g == nil {
			b.Skip("Failed to create llama.cpp grammar")
			return
		}

		tokens := make([]llama.TokenData, 128000)
		for j := range tokens {
			tokens[j] = llama.TokenData{ID: int32(j), Logit: 1.0}
		}

		for _, tokenId := range tokenIds {
			// Apply grammar mask
			g.Apply(tokens)
			// Accept token
			g.Accept(tokenId)
			// Reset logits for next iteration
			for j := range tokens {
				tokens[j].Logit = 1.0
			}
		}

		g.Free()
	}
}

// ============ Detailed Comparison Report ============

// TestComparisonReport runs both systems and prints a comparison
func TestComparisonReport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping comparison report in short mode")
	}

	vocabSize := 128000
	iterations := 100

	vocab, vocabIds, vocabValues := createComparisonVocab(vocabSize)

	// ===== MLX Engine =====
	mlxEngine, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("Failed to create MLX engine: %v", err)
	}
	defer mlxEngine.Close()

	logits := mlx.Ones(int32(vocabSize))
	mlx.Keep(logits)

	// Warm up MLX
	for i := 0; i < 50; i++ {
		masked := mlxEngine.ApplyMask(logits)
		mlx.Eval(masked)
	}

	// Benchmark MLX ApplyMask
	start := time.Now()
	for i := 0; i < iterations; i++ {
		masked := mlxEngine.ApplyMask(logits)
		mlx.Eval(masked)
	}
	mlxApplyDuration := time.Since(start)
	mlxAvgApply := mlxApplyDuration / time.Duration(iterations)

	// ===== llama.cpp Grammar =====
	llamaGrammar := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
	if llamaGrammar == nil {
		t.Log("WARNING: Could not create llama.cpp grammar - skipping comparison")
		t.Logf("\n=== MLX Engine Only (vocab=%d) ===", vocabSize)
		t.Logf("MLX ApplyMask:    %v/token", mlxAvgApply)
		return
	}
	defer llamaGrammar.Free()

	tokens := make([]llama.TokenData, vocabSize)
	for i := range tokens {
		tokens[i] = llama.TokenData{ID: int32(i), Logit: 1.0}
	}

	// Warm up llama.cpp
	for i := 0; i < 50; i++ {
		for j := range tokens {
			tokens[j].Logit = 1.0
		}
		llamaGrammar.Apply(tokens)
	}

	// Benchmark llama.cpp Apply
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for j := range tokens {
			tokens[j].Logit = 1.0
		}
		llamaGrammar.Apply(tokens)
	}
	llamaApplyDuration := time.Since(start)
	llamaAvgApply := llamaApplyDuration / time.Duration(iterations)

	// ===== Sequence Comparison =====
	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}

	// MLX sequence
	start = time.Now()
	for i := 0; i < iterations; i++ {
		mlxEngine.Reset()
		for _, token := range sequence {
			masked := mlxEngine.ApplyMask(logits)
			mlx.Eval(masked)
			mlxEngine.AcceptString(token)
		}
	}
	mlxSeqDuration := time.Since(start)
	mlxSeqTokens := iterations * len(sequence)
	mlxSeqAvg := mlxSeqDuration / time.Duration(mlxSeqTokens)

	// llama.cpp sequence (need to recreate grammar each time due to state)
	tokenIds := make([]int32, len(sequence))
	for i, tok := range sequence {
		for j, v := range vocab {
			if v == tok {
				tokenIds[i] = int32(j)
				break
			}
		}
	}

	start = time.Now()
	for i := 0; i < iterations; i++ {
		g := llama.NewGrammar(jsonGBNF, vocabIds, vocabValues, []int32{})
		if g == nil {
			continue
		}
		for _, tokenId := range tokenIds {
			for j := range tokens {
				tokens[j].Logit = 1.0
			}
			g.Apply(tokens)
			g.Accept(tokenId)
		}
		g.Free()
	}
	llamaSeqDuration := time.Since(start)
	llamaSeqTokens := iterations * len(sequence)
	llamaSeqAvg := llamaSeqDuration / time.Duration(llamaSeqTokens)

	// ===== Print Report =====
	t.Logf("\n")
	t.Logf("╔══════════════════════════════════════════════════════════════╗")
	t.Logf("║         CONSTRAINED DECODING BENCHMARK COMPARISON            ║")
	t.Logf("║                   vocab_size = %d                         ║", vocabSize)
	t.Logf("╠══════════════════════════════════════════════════════════════╣")
	t.Logf("║                                                              ║")
	t.Logf("║  Single Mask Application:                                    ║")
	t.Logf("║    MLX Engine:      %12v/token                       ║", mlxAvgApply)
	t.Logf("║    llama.cpp:       %12v/token                       ║", llamaAvgApply)
	t.Logf("║    Speedup:         %.2fx                                    ║", float64(llamaAvgApply)/float64(mlxAvgApply))
	t.Logf("║                                                              ║")
	t.Logf("║  Full Sequence ({\"key\": \"value\"}):                           ║")
	t.Logf("║    MLX Engine:      %12v/token                       ║", mlxSeqAvg)
	t.Logf("║    llama.cpp:       %12v/token                       ║", llamaSeqAvg)
	t.Logf("║    Speedup:         %.2fx                                    ║", float64(llamaSeqAvg)/float64(mlxSeqAvg))
	t.Logf("║                                                              ║")
	t.Logf("║  Throughput:                                                 ║")
	t.Logf("║    MLX:             %.0f tokens/sec                        ║", float64(time.Second)/float64(mlxSeqAvg))
	t.Logf("║    llama.cpp:       %.0f tokens/sec                        ║", float64(time.Second)/float64(llamaSeqAvg))
	t.Logf("║                                                              ║")
	t.Logf("║  Target: <50μs/token                                         ║")
	if mlxAvgApply < 50*time.Microsecond {
		t.Logf("║  MLX Status: ✓ PASS                                          ║")
	} else {
		t.Logf("║  MLX Status: ✗ FAIL (%.1fx over target)                       ║", float64(mlxAvgApply)/(50*float64(time.Microsecond)))
	}
	t.Logf("╚══════════════════════════════════════════════════════════════╝")
}

// ============ Memory Comparison ============

// TestMemoryComparison compares memory usage between systems
func TestMemoryComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory comparison in short mode")
	}

	vocabSize := 128000

	// Calculate MLX memory usage
	pda, _ := GetJSONPDA()
	numTerminals := len(pda.Terminals)

	// Each symbol mask: vocab_size * 4 bytes (float32)
	// Plus negInfMask: vocab_size * 4 bytes
	// Plus threshold scalar: 4 bytes
	mlxMaskMemory := numTerminals * vocabSize * 4
	mlxTotalMemory := mlxMaskMemory + vocabSize*4 + 4

	t.Logf("\n=== Memory Usage Comparison (vocab=%d) ===", vocabSize)
	t.Logf("")
	t.Logf("MLX Engine:")
	t.Logf("  Symbol masks:     %d terminals × %d vocab × 4 bytes = %.2f MB",
		numTerminals, vocabSize, float64(mlxMaskMemory)/(1024*1024))
	t.Logf("  Neg-inf mask:     %d vocab × 4 bytes = %.2f MB",
		vocabSize, float64(vocabSize*4)/(1024*1024))
	t.Logf("  Total:            %.2f MB", float64(mlxTotalMemory)/(1024*1024))
	t.Logf("")
	t.Logf("llama.cpp Grammar:")
	t.Logf("  (Memory usage depends on grammar complexity)")
	t.Logf("  Typically lower than precomputed masks")
	t.Logf("")
	t.Logf("Trade-off:")
	t.Logf("  MLX: Higher memory, faster per-token")
	t.Logf("  llama.cpp: Lower memory, more CPU work per-token")
}

// ============ Scaling Benchmark ============

// TestScalingBenchmark measures how performance scales with vocab size
func TestScalingBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping scaling benchmark in short mode")
	}

	sizes := []int{8000, 16000, 32000, 64000, 128000}
	iterations := 100

	t.Logf("\n=== Scaling Benchmark ===")
	t.Logf("")
	t.Logf("Vocab Size | MLX ApplyMask | Per 1k tokens")
	t.Logf("-----------|---------------|---------------")

	for _, size := range sizes {
		vocab := createBenchVocabN(size)
		e, err := NewEngine(vocab)
		if err != nil {
			t.Fatalf("Failed at size %d: %v", size, err)
		}

		logits := mlx.Ones(int32(size))
		mlx.Keep(logits)

		// Warm up
		for i := 0; i < 20; i++ {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
		}

		// Benchmark
		start := time.Now()
		for i := 0; i < iterations; i++ {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
		}
		duration := time.Since(start)
		avg := duration / time.Duration(iterations)
		per1k := avg * 1000 / time.Duration(size)

		t.Logf("%10d | %13v | %v", size, avg, per1k)

		e.Close()
	}
}
