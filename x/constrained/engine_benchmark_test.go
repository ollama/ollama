//go:build mlx

package constrained

import (
	"fmt"
	"testing"
	"time"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Vocabulary sizes to test (matching real models)
var vocabSizes = []int{
	32000,  // Llama 2
	128000, // Llama 3
	256000, // Large models
}

// createBenchVocabN creates a vocabulary of size n with realistic token distribution
func createBenchVocabN(n int) []string {
	vocab := make([]string, n)

	// JSON structural tokens (first 20)
	jsonTokens := []string{
		"{", "}", "[", "]", ":", ",",
		"true", "false", "null",
		" ", "\n", "\t", "\r",
		"\"", "'",
	}
	for i, t := range jsonTokens {
		if i < n {
			vocab[i] = t
		}
	}

	// String tokens (indices 20-1000)
	stringIdx := 20
	for i := 0; i < 980 && stringIdx+i < n; i++ {
		vocab[stringIdx+i] = fmt.Sprintf("\"token%d\"", i)
	}

	// Number tokens (indices 1000-2000)
	numberIdx := 1000
	for i := 0; i < 1000 && numberIdx+i < n; i++ {
		vocab[numberIdx+i] = fmt.Sprintf("%d", i)
	}

	// Generic tokens (rest)
	for i := 2000; i < n; i++ {
		vocab[i] = fmt.Sprintf("tok%d", i)
	}

	return vocab
}

// ============ Core Performance Benchmarks ============

// BenchmarkApplyMask_32k measures mask application with 32k vocab
func BenchmarkApplyMask_32k(b *testing.B) {
	benchmarkApplyMask(b, 32000)
}

// BenchmarkApplyMask_128k measures mask application with 128k vocab
func BenchmarkApplyMask_128k(b *testing.B) {
	benchmarkApplyMask(b, 128000)
}

// BenchmarkApplyMask_256k measures mask application with 256k vocab
func BenchmarkApplyMask_256k(b *testing.B) {
	benchmarkApplyMask(b, 256000)
}

func benchmarkApplyMask(b *testing.B, vocabSize int) {
	vocab := createBenchVocabN(vocabSize)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(vocabSize))
	mlx.Keep(logits)

	// Warm up
	for i := 0; i < 10; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}

	b.ReportMetric(float64(vocabSize), "vocab_size")
}

// ============ State-Dependent Benchmarks ============

// BenchmarkApplyMaskAfterBrace measures mask after { (STRING or } valid)
func BenchmarkApplyMaskAfterBrace(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	e.AcceptString("{")

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}
}

// BenchmarkApplyMaskMidObject measures mask in middle of object
func BenchmarkApplyMaskMidObject(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	// State: {"key": _value_
	e.AcceptString("{")
	e.AcceptString("\"key\"")
	e.AcceptString(":")

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}
}

// ============ Token Sequence Benchmarks ============

// BenchmarkSequence_SimpleObject benchmarks {"key": "value"}
func BenchmarkSequence_SimpleObject(b *testing.B) {
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

	b.ReportMetric(float64(len(sequence)), "tokens")
}

// BenchmarkSequence_NestedObject benchmarks {"a": {"b": {"c": 1}}}
func BenchmarkSequence_NestedObject(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	sequence := []string{
		"{", "\"a\"", ":", "{", "\"b\"", ":", "{", "\"c\"", ":", "1", "}", "}", "}",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Reset()
		for _, token := range sequence {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
			e.AcceptString(token)
		}
	}

	b.ReportMetric(float64(len(sequence)), "tokens")
}

// BenchmarkSequence_LargeArray benchmarks [1, 2, 3, ..., 100]
func BenchmarkSequence_LargeArray(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	// Build sequence: [1, 2, 3, ..., 50]
	sequence := []string{"["}
	for i := 1; i <= 50; i++ {
		sequence = append(sequence, fmt.Sprintf("%d", i))
		if i < 50 {
			sequence = append(sequence, ",")
		}
	}
	sequence = append(sequence, "]")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Reset()
		for _, token := range sequence {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
			e.AcceptString(token)
		}
	}

	b.ReportMetric(float64(len(sequence)), "tokens")
}

// BenchmarkSequence_MixedTypes benchmarks complex mixed-type object
func BenchmarkSequence_MixedTypes(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(128000))
	mlx.Keep(logits)

	sequence := []string{
		"{",
		"\"name\"", ":", "\"test\"", ",",
		"\"count\"", ":", "42", ",",
		"\"enabled\"", ":", "true", ",",
		"\"data\"", ":", "null", ",",
		"\"items\"", ":", "[", "1", ",", "2", ",", "3", "]",
		"}",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Reset()
		for _, token := range sequence {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
			e.AcceptString(token)
		}
	}

	b.ReportMetric(float64(len(sequence)), "tokens")
}

// ============ Component Benchmarks ============

// BenchmarkValidInputs measures PDA valid input computation
func BenchmarkValidInputs(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = e.runtime.ValidInputs()
	}
}

// BenchmarkStateTransition measures PDA state transition
func BenchmarkStateTransition(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Reset()
		for _, token := range sequence {
			e.AcceptString(token)
		}
	}
}

// BenchmarkClassifyToken measures token classification
func BenchmarkClassifyToken(b *testing.B) {
	tokens := []string{
		"{", "}", "[", "]", ":", ",",
		"\"hello\"", "\"world\"",
		"123", "-45.67", "1e10",
		"true", "false", "null",
		" ", "\n",
		"unknown",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, token := range tokens {
			_ = ClassifyToken(token)
		}
	}
}

// BenchmarkPrecomputeMasks measures one-time mask precomputation
func BenchmarkPrecomputeMasks_32k(b *testing.B) {
	benchmarkPrecomputeMasks(b, 32000)
}

func BenchmarkPrecomputeMasks_128k(b *testing.B) {
	benchmarkPrecomputeMasks(b, 128000)
}

func benchmarkPrecomputeMasks(b *testing.B, vocabSize int) {
	vocab := createBenchVocabN(vocabSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e, _ := NewEngine(vocab)
		e.Close()
	}
}

// ============ Memory Benchmarks ============

func BenchmarkMemoryAllocs_32k(b *testing.B) {
	benchmarkMemoryAllocs(b, 32000)
}

func BenchmarkMemoryAllocs_128k(b *testing.B) {
	benchmarkMemoryAllocs(b, 128000)
}

func benchmarkMemoryAllocs(b *testing.B, vocabSize int) {
	vocab := createBenchVocabN(vocabSize)
	e, err := NewEngine(vocab)
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(vocabSize))
	mlx.Keep(logits)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}
}

// ============ No-Eval Benchmarks (simulating LLM graph integration) ============

// BenchmarkApplyMaskNoEval_128k measures mask generation WITHOUT GPU sync
// This simulates adding mask to LLM compute graph
func BenchmarkApplyMaskNoEval_128k(b *testing.B) {
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
		_ = e.ApplyMask(logits) // No Eval - just build graph
	}
}

// BenchmarkSequenceNoEval simulates real LLM usage - build graph, eval once at end
func BenchmarkSequenceNoEval_SimpleObject(b *testing.B) {
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
		var lastMasked *mlx.Array
		for _, token := range sequence {
			lastMasked = e.ApplyMask(logits) // Build graph only
			e.AcceptString(token)
		}
		mlx.Eval(lastMasked) // Single eval at end
	}

	b.ReportMetric(float64(len(sequence)), "tokens")
}

// TestNoEvalThroughput compares with-eval vs no-eval performance
func TestNoEvalThroughput(t *testing.T) {
	vocabSize := 128000
	iterations := 1000

	vocab := createBenchVocabN(vocabSize)
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(vocabSize))
	mlx.Keep(logits)

	// Warm up
	for i := 0; i < 100; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}

	// Test 1: With Eval (current approach)
	start := time.Now()
	for i := 0; i < iterations; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}
	withEvalDuration := time.Since(start)
	withEvalAvg := withEvalDuration / time.Duration(iterations)

	// Test 2: Without Eval (graph building only)
	start = time.Now()
	var lastMasked *mlx.Array
	for i := 0; i < iterations; i++ {
		lastMasked = e.ApplyMask(logits)
	}
	mlx.Eval(lastMasked) // Single eval at end
	noEvalDuration := time.Since(start)
	noEvalAvg := noEvalDuration / time.Duration(iterations)

	// Test 3: Sequence without per-token eval
	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}
	start = time.Now()
	for i := 0; i < iterations; i++ {
		e.Reset()
		for _, token := range sequence {
			lastMasked = e.ApplyMask(logits)
			e.AcceptString(token)
		}
		mlx.Eval(lastMasked)
	}
	seqNoEvalDuration := time.Since(start)
	seqTokens := iterations * len(sequence)
	seqNoEvalAvg := seqNoEvalDuration / time.Duration(seqTokens)

	t.Logf("\n")
	t.Logf("╔══════════════════════════════════════════════════════════════╗")
	t.Logf("║     WITH-EVAL vs NO-EVAL COMPARISON (vocab=%d)          ║", vocabSize)
	t.Logf("╠══════════════════════════════════════════════════════════════╣")
	t.Logf("║                                                              ║")
	t.Logf("║  Single ApplyMask:                                           ║")
	t.Logf("║    With Eval():     %12v/token                       ║", withEvalAvg)
	t.Logf("║    Without Eval():  %12v/token                       ║", noEvalAvg)
	t.Logf("║    Speedup:         %.1fx                                    ║", float64(withEvalAvg)/float64(noEvalAvg))
	t.Logf("║                                                              ║")
	t.Logf("║  Sequence (5 tokens, eval once at end):                      ║")
	t.Logf("║    Per-token avg:   %12v/token                       ║", seqNoEvalAvg)
	t.Logf("║    Throughput:      %.0f tokens/sec                        ║", float64(time.Second)/float64(seqNoEvalAvg))
	t.Logf("║                                                              ║")
	t.Logf("║  Target: <50μs/token                                         ║")
	if noEvalAvg < 50*time.Microsecond {
		t.Logf("║  Status: ✓ PASS (graph-only: %.1fx under target)             ║", 50*float64(time.Microsecond)/float64(noEvalAvg))
	} else {
		t.Logf("║  Status: Graph-only: %.1fx over target                       ║", float64(noEvalAvg)/(50*float64(time.Microsecond)))
	}
	t.Logf("╚══════════════════════════════════════════════════════════════╝")
}

// ============ Throughput Test ============

// TestThroughputReport runs a throughput test and prints detailed metrics
func TestThroughputReport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping throughput report in short mode")
	}

	vocabSize := 128000
	iterations := 1000

	vocab := createBenchVocabN(vocabSize)
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	logits := mlx.Ones(int32(vocabSize))
	mlx.Keep(logits)

	// Warm up
	for i := 0; i < 100; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}

	// Measure ApplyMask latency
	start := time.Now()
	for i := 0; i < iterations; i++ {
		masked := e.ApplyMask(logits)
		mlx.Eval(masked)
	}
	applyMaskDuration := time.Since(start)
	avgApplyMask := applyMaskDuration / time.Duration(iterations)

	// Measure full sequence generation
	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}
	start = time.Now()
	for i := 0; i < iterations; i++ {
		e.Reset()
		for _, token := range sequence {
			masked := e.ApplyMask(logits)
			mlx.Eval(masked)
			e.AcceptString(token)
		}
	}
	fullSequenceDuration := time.Since(start)
	tokensGenerated := iterations * len(sequence)
	avgPerToken := fullSequenceDuration / time.Duration(tokensGenerated)
	tokensPerSecond := float64(tokensGenerated) / fullSequenceDuration.Seconds()

	t.Logf("\n=== Throughput Report (vocab=%d) ===", vocabSize)
	t.Logf("ApplyMask latency:    %v/token", avgApplyMask)
	t.Logf("Full loop latency:    %v/token", avgPerToken)
	t.Logf("Throughput:           %.0f tokens/sec", tokensPerSecond)
	t.Logf("Tokens generated:     %d", tokensGenerated)
	t.Logf("")
	t.Logf("Target: <50μs/token")
	if avgApplyMask < 50*time.Microsecond {
		t.Logf("Status: ✓ PASS")
	} else {
		t.Logf("Status: ✗ FAIL (%.1fx over target)", float64(avgApplyMask)/(50*float64(time.Microsecond)))
	}
}
