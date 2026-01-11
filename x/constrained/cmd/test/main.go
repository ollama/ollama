//go:build mlx

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/ollama/ollama/x/constrained"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

var (
	benchmark   = flag.Bool("benchmark", false, "Run performance benchmarks")
	stress      = flag.Bool("stress", false, "Run stress test with many tokens")
	iterations  = flag.Int("iterations", 1000, "Number of benchmark iterations")
	vocabSize   = flag.Int("vocab-size", 128000, "Vocabulary size for tests")
	warmup      = flag.Int("warmup", 100, "Warmup iterations before benchmark")
	percentiles = flag.Bool("percentiles", false, "Show latency percentiles")
)

func main() {
	flag.Parse()

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║      MLX Constrained Decoding - Integration Test Suite       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// System info
	fmt.Printf("System: %s/%s, CPUs: %d\n", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Printf("Vocab Size: %d\n", *vocabSize)
	fmt.Println()

	// Create test vocabulary
	vocab := createVocab(*vocabSize)
	fmt.Printf("Created vocabulary with %d tokens\n", len(vocab))

	// Create engine
	start := time.Now()
	engine, err := constrained.NewEngine(vocab)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()
	fmt.Printf("Engine created in %v\n", time.Since(start))
	fmt.Println()

	// Run functional tests
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                      FUNCTIONAL TESTS                          ")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	passed, failed := runFunctionalTests(engine, vocab)
	fmt.Printf("\nResult: %d passed, %d failed\n", passed, failed)

	if failed > 0 {
		os.Exit(1)
	}

	if *benchmark {
		fmt.Println()
		fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		fmt.Println("                    PERFORMANCE BENCHMARKS                      ")
		fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		runBenchmarks(engine, vocab)
	}

	if *stress {
		fmt.Println()
		fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		fmt.Println("                        STRESS TEST                             ")
		fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		runStressTest(engine, vocab)
	}

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    ALL TESTS PASSED                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
}

func createVocab(size int) []string {
	vocab := make([]string, size)

	// Standard JSON tokens
	jsonTokens := []string{
		"{", "}", "[", "]", ":", ",",
		"true", "false", "null",
		" ", "\n", "\t", "\r",
	}
	for i, t := range jsonTokens {
		if i < size {
			vocab[i] = t
		}
	}

	// String tokens (various keys and values)
	stringTokens := []string{
		"\"name\"", "\"value\"", "\"id\"", "\"type\"", "\"data\"",
		"\"items\"", "\"count\"", "\"enabled\"", "\"description\"",
		"\"hello\"", "\"world\"", "\"test\"", "\"foo\"", "\"bar\"",
		"\"key\"", "\"status\"", "\"message\"", "\"error\"", "\"result\"",
	}
	for i, s := range stringTokens {
		if 15+i < size {
			vocab[15+i] = s
		}
	}

	// Number tokens
	for i := 0; i < 100 && 40+i < size; i++ {
		vocab[40+i] = fmt.Sprintf("%d", i)
	}

	// Fill rest with generic tokens
	for i := 150; i < size; i++ {
		vocab[i] = fmt.Sprintf("tok%d", i)
	}

	return vocab
}

func runFunctionalTests(engine *constrained.Engine, vocab []string) (passed, failed int) {
	tests := []struct {
		name     string
		sequence []string
		valid    bool
	}{
		{"empty object", []string{"{", "}"}, true},
		{"simple object", []string{"{", "\"name\"", ":", "\"value\"", "}"}, true},
		{"object with number", []string{"{", "\"count\"", ":", "42", "}"}, true},
		{"object with true", []string{"{", "\"enabled\"", ":", "true", "}"}, true},
		{"object with false", []string{"{", "\"enabled\"", ":", "false", "}"}, true},
		{"object with null", []string{"{", "\"data\"", ":", "null", "}"}, true},
		{"empty array", []string{"[", "]"}, true},
		{"array of numbers", []string{"[", "1", ",", "2", ",", "3", "]"}, true},
		{"array of mixed", []string{"[", "1", ",", "\"str\"", ",", "true", "]"}, true},
		{"nested object", []string{"{", "\"data\"", ":", "{", "\"id\"", ":", "1", "}", "}"}, true},
		{"deeply nested", []string{"{", "\"a\"", ":", "{", "\"b\"", ":", "{", "\"c\"", ":", "1", "}", "}", "}"}, true},
		{"object with array", []string{"{", "\"items\"", ":", "[", "1", ",", "2", "]", "}"}, true},
		{"multiple properties", []string{"{", "\"name\"", ":", "\"test\"", ",", "\"count\"", ":", "42", "}"}, true},
		{"string primitive", []string{"\"hello\""}, true},
		{"number primitive", []string{"42"}, true},
		{"true primitive", []string{"true"}, true},
		{"false primitive", []string{"false"}, true},
		{"null primitive", []string{"null"}, true},
	}

	for _, test := range tests {
		engine.Reset()
		success := true

		for _, token := range test.sequence {
			if !engine.AcceptString(token) {
				success = false
				break
			}
		}

		if success && !engine.IsComplete() {
			success = false
		}

		if success == test.valid {
			fmt.Printf("  ✓ %s\n", test.name)
			passed++
		} else {
			fmt.Printf("  ✗ %s (expected valid=%v, got valid=%v)\n", test.name, test.valid, success)
			failed++
		}
	}

	return passed, failed
}

func runBenchmarks(engine *constrained.Engine, vocab []string) {
	logits := mlx.Ones(int32(len(vocab)))
	mlx.Keep(logits)

	// Warm up
	fmt.Printf("\nWarming up (%d iterations)...\n", *warmup)
	for i := 0; i < *warmup; i++ {
		masked := engine.ApplyMask(logits)
		mlx.Eval(masked)
	}

	// 1. ApplyMask latency
	fmt.Printf("\n1. ApplyMask Latency (%d iterations)\n", *iterations)
	latencies := make([]time.Duration, *iterations)

	for i := 0; i < *iterations; i++ {
		start := time.Now()
		masked := engine.ApplyMask(logits)
		mlx.Eval(masked)
		latencies[i] = time.Since(start)
	}

	printLatencyStats("   ApplyMask", latencies)

	// 2. Full sequence benchmark
	sequences := []struct {
		name   string
		tokens []string
	}{
		{"Simple Object", []string{"{", "\"key\"", ":", "\"value\"", "}"}},
		{"Nested Object", []string{"{", "\"a\"", ":", "{", "\"b\"", ":", "1", "}", "}"}},
		{"Array", []string{"[", "1", ",", "2", ",", "3", ",", "4", ",", "5", "]"}},
		{"Mixed", []string{"{", "\"str\"", ":", "\"val\"", ",", "\"num\"", ":", "42", ",", "\"arr\"", ":", "[", "1", "]", "}"}},
	}

	fmt.Printf("\n2. Sequence Benchmarks (%d iterations each)\n", *iterations/10)
	for _, seq := range sequences {
		seqLatencies := make([]time.Duration, *iterations/10)

		for i := 0; i < *iterations/10; i++ {
			engine.Reset()
			start := time.Now()
			for _, token := range seq.tokens {
				masked := engine.ApplyMask(logits)
				mlx.Eval(masked)
				engine.AcceptString(token)
			}
			seqLatencies[i] = time.Since(start)
		}

		avgPerToken := average(seqLatencies) / time.Duration(len(seq.tokens))
		fmt.Printf("   %s (%d tokens): %v total, %v/token\n",
			seq.name, len(seq.tokens), average(seqLatencies), avgPerToken)
	}

	// 3. State transition benchmark
	fmt.Printf("\n3. PDA State Transition (%d iterations)\n", *iterations)
	sequence := []string{"{", "\"key\"", ":", "\"value\"", "}"}

	start := time.Now()
	for i := 0; i < *iterations; i++ {
		engine.Reset()
		for _, token := range sequence {
			engine.AcceptString(token)
		}
	}
	transitionDuration := time.Since(start)
	transitionsPerSec := float64(*iterations*len(sequence)) / transitionDuration.Seconds()
	fmt.Printf("   Transitions: %.0f/sec, %v per transition\n",
		transitionsPerSec, transitionDuration/time.Duration(*iterations*len(sequence)))

	// 4. Memory report
	fmt.Printf("\n4. Memory Usage\n")
	pda, _ := constrained.GetJSONPDA()
	numTerminals := len(pda.Terminals)
	maskMemory := numTerminals * len(vocab) * 4
	fmt.Printf("   Symbol masks: %d × %d × 4 = %.2f MB\n",
		numTerminals, len(vocab), float64(maskMemory)/(1024*1024))

	// 5. Summary
	fmt.Printf("\n5. Summary\n")
	avgLatency := average(latencies)
	fmt.Printf("   Average ApplyMask latency: %v\n", avgLatency)
	fmt.Printf("   Target: <50μs\n")
	if avgLatency < 50*time.Microsecond {
		fmt.Printf("   Status: ✓ PASS (%.1fx under target)\n", 50*float64(time.Microsecond)/float64(avgLatency))
	} else {
		fmt.Printf("   Status: ✗ FAIL (%.1fx over target)\n", float64(avgLatency)/(50*float64(time.Microsecond)))
	}
}

func runStressTest(engine *constrained.Engine, vocab []string) {
	logits := mlx.Ones(int32(len(vocab)))
	mlx.Keep(logits)

	// Generate a large array: [1, 2, 3, ..., 500]
	fmt.Println("\nGenerating large array [1, 2, ..., 500]...")
	sequence := []string{"["}
	for i := 1; i <= 500; i++ {
		sequence = append(sequence, fmt.Sprintf("%d", i%100)) // Use modulo to stay in vocab
		if i < 500 {
			sequence = append(sequence, ",")
		}
	}
	sequence = append(sequence, "]")
	fmt.Printf("Sequence length: %d tokens\n", len(sequence))

	// Run stress test
	iterations := 10
	latencies := make([]time.Duration, 0, len(sequence)*iterations)

	for iter := 0; iter < iterations; iter++ {
		engine.Reset()
		for _, token := range sequence {
			start := time.Now()
			masked := engine.ApplyMask(logits)
			mlx.Eval(masked)
			latencies = append(latencies, time.Since(start))

			if !engine.AcceptString(token) {
				fmt.Printf("Failed to accept token: %s\n", token)
				return
			}
		}
	}

	if !engine.IsComplete() {
		fmt.Println("ERROR: Not in accepting state after sequence")
		return
	}

	fmt.Println("\nStress Test Results:")
	printLatencyStats("   Per-token", latencies)

	totalTokens := len(sequence) * iterations
	totalTime := sum(latencies)
	fmt.Printf("   Total tokens: %d\n", totalTokens)
	fmt.Printf("   Total time: %v\n", totalTime)
	fmt.Printf("   Throughput: %.0f tokens/sec\n", float64(totalTokens)/totalTime.Seconds())

	// Validate JSON
	fmt.Printf("\nValidating generated JSON structure...")
	var result interface{}
	// Build JSON string
	jsonStr := ""
	for _, token := range sequence {
		jsonStr += token
	}
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		fmt.Printf(" FAIL: %v\n", err)
	} else {
		fmt.Printf(" OK\n")
	}
}

func printLatencyStats(prefix string, latencies []time.Duration) {
	if len(latencies) == 0 {
		return
	}

	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	avg := average(latencies)
	min := sorted[0]
	max := sorted[len(sorted)-1]
	p50 := sorted[len(sorted)/2]
	p95 := sorted[int(float64(len(sorted))*0.95)]
	p99 := sorted[int(float64(len(sorted))*0.99)]

	fmt.Printf("%s: avg=%v, min=%v, max=%v\n", prefix, avg, min, max)
	if *percentiles {
		fmt.Printf("%s  p50=%v, p95=%v, p99=%v\n", prefix, p50, p95, p99)
	}
}

func average(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	var total time.Duration
	for _, l := range latencies {
		total += l
	}
	return total / time.Duration(len(latencies))
}

func sum(latencies []time.Duration) time.Duration {
	var total time.Duration
	for _, l := range latencies {
		total += l
	}
	return total
}

// Unused but needed to avoid import error
var _ = math.Inf
