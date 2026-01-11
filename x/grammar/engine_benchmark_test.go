//go:build mlx

package grammar

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// newBenchEngine creates a JSON engine for benchmarks
func newBenchEngine(b *testing.B, vocab []string) *Engine {
	b.Helper()
	grammar, err := JSONGrammar()
	if err != nil {
		b.Fatalf("failed to create JSON grammar: %v", err)
	}
	e, err := NewEngine(grammar, vocab)
	if err != nil {
		b.Fatalf("failed to create engine: %v", err)
	}
	return e
}

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
	e := newBenchEngine(b, vocab)
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

// ============ state-Dependent Benchmarks ============

// BenchmarkApplyMaskAfterBrace measures mask after { (STRING or } valid)
func BenchmarkApplyMaskAfterBrace(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
	defer e.Close()

	// state: {"key": _value_
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
	e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
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

// BenchmarkValidInputs measures pda valid input computation
func BenchmarkValidInputs(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e := newBenchEngine(b, vocab)
	defer e.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = e.validTerminals()
	}
}

// BenchmarkStateTransition measures pda state transition
func BenchmarkStateTransition(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e := newBenchEngine(b, vocab)
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

// BenchmarkConstrainedGrammar_128k benchmarks x/grammar (graph only, no eval).
func BenchmarkConstrainedGrammar_128k(b *testing.B) {
	vocab := createBenchVocabN(128000)
	e := newBenchEngine(b, vocab)
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
		_ = e.ApplyMask(logits) // Graph only, no eval
	}
}

// BenchmarkNewEngine measures one-time engine initialization.
func BenchmarkNewEngine_32k(b *testing.B) {
	benchmarkNewEngine(b, 32000)
}

func BenchmarkNewEngine_128k(b *testing.B) {
	benchmarkNewEngine(b, 128000)
}

func benchmarkNewEngine(b *testing.B, vocabSize int) {
	vocab := createBenchVocabN(vocabSize)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
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
	e := newBenchEngine(b, vocab)
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
