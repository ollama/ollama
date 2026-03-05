//go:build mlx

package mlx

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestMain initializes MLX before running tests.
// If MLX libraries are not available, tests are skipped.
func TestMain(m *testing.M) {
	// Change to repo root so ./build/lib/ollama/ path works
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..", "..")
	if err := os.Chdir(repoRoot); err != nil {
		fmt.Printf("Failed to change to repo root: %v\n", err)
		os.Exit(1)
	}

	if err := InitMLX(); err != nil {
		fmt.Printf("Skipping MLX tests: %v\n", err)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// TestBasicCleanup verifies non-kept arrays are freed and kept arrays survive.
func TestBasicCleanup(t *testing.T) {
	weight := NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	intermediate := NewArrayFloat32([]float32{1, 1}, []int32{1, 2})
	result := Matmul(intermediate, weight)
	Keep(result)

	// Before eval: intermediate should be valid
	if !intermediate.Valid() {
		t.Fatal("intermediate should be valid before Eval")
	}

	Eval(result)

	// After eval: intermediate should be freed
	if intermediate.Valid() {
		t.Fatal("intermediate should be freed after Eval")
	}

	// Result should have correct values
	data := result.Data()
	if data[0] != 4 || data[1] != 6 {
		t.Errorf("expected [4, 6], got %v", data)
	}

	// Weight should survive
	if !weight.Valid() {
		t.Error("weight was freed")
	}
}

// TestKeptSurvives verifies kept arrays are not freed.
func TestKeptSurvives(t *testing.T) {
	a := NewArrayFloat32([]float32{1, 2}, []int32{2})
	b := NewArrayFloat32([]float32{3, 4}, []int32{2})
	result := Add(a, b)
	Keep(result)

	Eval(result)

	if !result.Valid() {
		t.Error("kept result was freed")
	}

	data := result.Data()
	if data[0] != 4 || data[1] != 6 {
		t.Errorf("expected [4, 6], got %v", data)
	}
}

// TestEvalAutoKeeps verifies Eval automatically keeps its outputs.
func TestEvalAutoKeeps(t *testing.T) {
	a := NewArrayFloat32([]float32{1, 2}, []int32{2})
	b := NewArrayFloat32([]float32{3, 4}, []int32{2})
	result := Add(a, b)

	// Don't call Keep(result) - Eval should auto-keep it
	Eval(result)

	// Result should survive (auto-kept by Eval)
	if !result.Valid() {
		t.Error("Eval output was freed - should be auto-kept")
	}

	// Inputs should be freed (not kept)
	if a.Valid() {
		t.Error("input 'a' should be freed")
	}
	if b.Valid() {
		t.Error("input 'b' should be freed")
	}

	// Verify data is correct
	data := result.Data()
	if data[0] != 4 || data[1] != 6 {
		t.Errorf("expected [4, 6], got %v", data)
	}
}

// TestWeightsSurvive verifies kept arrays survive multiple Eval cycles.
func TestWeightsSurvive(t *testing.T) {
	weight := NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	for i := 0; i < 5; i++ {
		x := NewArrayFloat32([]float32{1, 1}, []int32{1, 2})
		result := Matmul(x, weight)
		Keep(result)
		Eval(result)
	}

	if !weight.Valid() {
		t.Error("weight was freed after multiple iterations")
	}
}

// TestAsyncEvalCleanup verifies AsyncEval cleans up and dispatches.
func TestAsyncEvalCleanup(t *testing.T) {
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2}) // Identity matrix
	Keep(weight)
	weight.Eval()

	// First async step
	x1 := NewArrayFloat32([]float32{1, 2}, []int32{1, 2})
	result1 := Matmul(x1, weight)
	Keep(result1)
	AsyncEval(result1)

	// Second async step
	x2 := NewArrayFloat32([]float32{3, 4}, []int32{1, 2})
	result2 := Matmul(x2, weight)
	Keep(result2)
	AsyncEval(result2)

	// Sync and verify results
	result1.Eval()
	d1 := result1.Data()
	if d1[0] != 1 || d1[1] != 2 {
		t.Errorf("result1: expected [1, 2], got %v", d1)
	}

	result2.Eval()
	d2 := result2.Data()
	if d2[0] != 3 || d2[1] != 4 {
		t.Errorf("result2: expected [3, 4], got %v", d2)
	}

	if !weight.Valid() {
		t.Error("weight was freed during async")
	}
}

// TestMultiOutput verifies multiple kept arrays survive.
func TestMultiOutput(t *testing.T) {
	a := NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{2, 2})
	sum := Add(a, a)
	prod := Mul(a, a)
	Keep(sum, prod)

	Eval(sum, prod)

	// Both kept arrays should be valid
	if !sum.Valid() || !prod.Valid() {
		t.Error("kept arrays should survive cleanup")
	}

	// Verify values
	sumData := sum.Data()
	prodData := prod.Data()
	if sumData[0] != 2 || prodData[0] != 1 {
		t.Errorf("unexpected results: sum=%v prod=%v", sumData, prodData)
	}
}

// TestChaining verifies output from one step can be used in next.
func TestChaining(t *testing.T) {
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	// First step
	x := NewArrayFloat32([]float32{1, 2}, []int32{1, 2})
	out1 := Matmul(x, weight)
	Keep(out1)
	AsyncEval(out1)

	// Second step uses output of first
	out2 := Add(out1, out1)
	Keep(out2)
	Eval(out2)

	// out1 should survive (was kept)
	if !out1.Valid() {
		t.Error("out1 was freed but used by second step")
	}

	// Final result should be correct
	data := out2.Data()
	if data[0] != 2 || data[1] != 4 {
		t.Errorf("expected [2, 4], got %v", data)
	}
}

// TestGenerationLoop simulates the LLM generation pattern with cache.
func TestGenerationLoop(t *testing.T) {
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	// Simulate cache - starts as zeros
	cache := NewArrayFloat32([]float32{0, 0}, []int32{1, 2})
	Keep(cache)
	cache.Eval()

	var lastToken *Array

	// Simulate 5 generation steps
	for step := 0; step < 5; step++ {
		oldCache := cache

		// Simulate forward pass
		input := NewArrayFloat32([]float32{float32(step + 1), float32(step + 2)}, []int32{1, 2})
		output := Matmul(input, weight)

		// Simulate cache update
		newCache := Add(output, cache)

		// Mark what survives
		Keep(output, newCache)

		if step < 4 {
			AsyncEval(output, newCache)
		} else {
			Eval(output, newCache)
		}

		// Free old cache, update references
		oldCache.Free()
		lastToken = output
		cache = newCache
	}

	// Token output should be valid
	if !lastToken.Valid() {
		t.Error("token output was freed")
	}

	// Cache should be valid
	if !cache.Valid() {
		t.Error("cache was freed")
	}

	// Weight should survive all iterations
	if !weight.Valid() {
		t.Error("weight was freed")
	}
}

// BenchmarkCleanupOnly isolates cleanup cost without MLX ops.
func BenchmarkCleanupOnly(b *testing.B) {
	// Pre-create weight
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create 100 arrays - minimal ops
		arrays := make([]*Array, 100)
		for j := range arrays {
			arrays[j] = NewArrayFloat32([]float32{1, 2}, []int32{1, 2})
		}
		Keep(arrays[0])
		Eval() // Just cleanup
	}
}

// BenchmarkNewArrayOnly measures array creation overhead.
func BenchmarkNewArrayOnly(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{2, 2})
	}
}

// BenchmarkCGOCallOverhead measures raw CGO call cost.
func BenchmarkCGOCallOverhead(b *testing.B) {
	arr := NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{2, 2})
	Keep(arr)
	arr.Eval()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = arr.Ndim() // Simple CGO call
	}
}

// BenchmarkCleanup_50 measures cleanup with 50 arrays.
func BenchmarkCleanup_50(b *testing.B) {
	benchCleanup(b, 50)
}

// BenchmarkCleanup_500 measures cleanup with 500 arrays (LLM scale).
func BenchmarkCleanup_500(b *testing.B) {
	benchCleanup(b, 500)
}

// BenchmarkCleanup_1000 measures cleanup with 1000 arrays.
func BenchmarkCleanup_1000(b *testing.B) {
	benchCleanup(b, 1000)
}

func benchCleanup(b *testing.B, numArrays int) {
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := NewArrayFloat32([]float32{1, 2}, []int32{1, 2})
		for j := 0; j < numArrays; j++ {
			x = Add(x, x)
		}
		result := Matmul(x, weight)
		Keep(result)
		Eval(result)
	}
}

// BenchmarkGenerationLoop_10 simulates 10 token generation steps.
func BenchmarkGenerationLoop_10(b *testing.B) {
	benchGenerationLoop(b, 10)
}

// BenchmarkGenerationLoop_100 simulates 100 token generation steps.
func BenchmarkGenerationLoop_100(b *testing.B) {
	benchGenerationLoop(b, 100)
}

func benchGenerationLoop(b *testing.B, steps int) {
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := NewArrayFloat32([]float32{0, 0}, []int32{1, 2})
		Keep(cache)
		cache.Eval()

		for step := 0; step < steps; step++ {
			oldCache := cache
			input := NewArrayFloat32([]float32{1, 2}, []int32{1, 2})
			output := Matmul(input, weight)
			newCache := Add(output, cache)
			Keep(output, newCache)

			if step < steps-1 {
				AsyncEval(output, newCache)
			} else {
				Eval(output, newCache)
			}
			oldCache.Free()
			cache = newCache
		}
	}
}

// BenchmarkLLMForward simulates a realistic LLM forward pass with ~500 ops.
func BenchmarkLLMForward(b *testing.B) {
	// Simulate weights for 32 layers
	numLayers := 32
	weights := make([]*Array, numLayers*4) // q, k, v, o per layer
	for i := range weights {
		weights[i] = NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	}
	Keep(weights...)
	Eval(weights...)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := NewArrayFloat32([]float32{1, 2}, []int32{1, 2})

		// Simulate 32 transformer layers
		for layer := 0; layer < numLayers; layer++ {
			// Attention block (simplified)
			q := Matmul(x, weights[layer*4])
			k := Matmul(x, weights[layer*4+1])
			v := Matmul(x, weights[layer*4+2])
			attn := Matmul(Softmax(Matmul(q, Transpose(k, 1, 0)), -1), v)
			attnOut := Matmul(attn, weights[layer*4+3])

			// Residual + layernorm (simplified)
			x = Add(x, attnOut)
			x = RMSNormNoWeight(x, 1e-5)

			// FFN (simplified as single matmul)
			ffn := Matmul(x, weights[layer*4])
			ffn = SiLU(ffn)
			x = Add(x, ffn)
		}
		Keep(x)
		Eval(x)
	}
}

// ============ Compile Tests ============

// gelu implements GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
func gelu(x *Array) *Array {
	sqrt2 := NewScalarArray(1.4142135623730951)
	half := NewScalarArray(0.5)
	one := NewScalarArray(1.0)
	scaled := Div(x, sqrt2)
	erfd := Erf(scaled)
	return Mul(Mul(x, half), Add(one, erfd))
}

// TestCompileBasic verifies compiled function produces correct output.
func TestCompileBasic(t *testing.T) {
	x := NewArrayFloat32([]float32{-1, 0, 1, 2}, []int32{4})
	Keep(x)
	x.Eval()

	// Uncompiled
	expected := gelu(x)
	Keep(expected)
	Eval(expected)

	// Compiled
	compiled := Compile(func(inputs []*Array) []*Array {
		return []*Array{gelu(inputs[0])}
	})
	defer compiled.Free()

	result := compiled.Call(x)[0]
	Keep(result)
	Eval(result)

	// Compare with tolerance
	expData := expected.Data()
	resData := result.Data()
	for i := range expData {
		diff := expData[i] - resData[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-5 {
			t.Errorf("mismatch at %d: expected %f, got %f (diff=%e)", i, expData[i], resData[i], diff)
		}
	}
}

// TestCompileMultipleInputs verifies compiled function with multiple inputs.
func TestCompileMultipleInputs(t *testing.T) {
	a := NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{4})
	b := NewArrayFloat32([]float32{5, 6, 7, 8}, []int32{4})
	Keep(a, b)
	Eval(a, b)

	compiled := Compile(func(inputs []*Array) []*Array {
		sum := Add(inputs[0], inputs[1])
		prod := Mul(inputs[0], inputs[1])
		return []*Array{sum, prod}
	})
	defer compiled.Free()

	outputs := compiled.Call(a, b)
	Keep(outputs...)
	Eval(outputs...)

	sumData := outputs[0].Data()
	prodData := outputs[1].Data()
	if sumData[0] != 6 || prodData[0] != 5 {
		t.Errorf("unexpected: sum[0]=%f, prod[0]=%f", sumData[0], prodData[0])
	}
}

// TestCompileReuse verifies compiled function can be called multiple times.
func TestCompileReuse(t *testing.T) {
	compiled := Compile(func(inputs []*Array) []*Array {
		return []*Array{Add(inputs[0], inputs[0])}
	})
	defer compiled.Free()

	for i := 0; i < 5; i++ {
		x := NewArrayFloat32([]float32{float32(i)}, []int32{1})
		Keep(x)
		x.Eval()
		result := compiled.Call(x)[0]
		Keep(result)
		Eval(result)
		data := result.Data()
		expected := float32(i * 2)
		if data[0] != expected {
			t.Errorf("iteration %d: expected %f, got %f", i, expected, data[0])
		}
	}
}

// BenchmarkGELUUncompiled benchmarks uncompiled GELU.
func BenchmarkGELUUncompiled(b *testing.B) {
	x := RandomNormal([]int32{1000, 1024}, 42)
	Keep(x)
	x.Eval()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y := x
		for j := 0; j < 10; j++ {
			y = gelu(y)
		}
		Keep(y)
		Eval(y)
	}
}

// BenchmarkGELUCompiled benchmarks compiled GELU.
func BenchmarkGELUCompiled(b *testing.B) {
	x := RandomNormal([]int32{1000, 1024}, 42)
	Keep(x)
	x.Eval()

	compiled := Compile(func(inputs []*Array) []*Array {
		y := inputs[0]
		for j := 0; j < 10; j++ {
			y = gelu(y)
		}
		return []*Array{y}
	})
	defer compiled.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := compiled.Call(x)
		Keep(result[0])
		Eval(result[0])
	}
}

// TestCompileNoMemoryLeak verifies compiled functions don't leak memory.
func TestCompileNoMemoryLeak(t *testing.T) {
	x := RandomNormal([]int32{100, 100}, 42)
	Keep(x)
	x.Eval()

	compiled := Compile(func(inputs []*Array) []*Array {
		y := inputs[0]
		for j := 0; j < 5; j++ {
			y = gelu(y)
		}
		return []*Array{y}
	})
	defer compiled.Free()

	// Warmup to establish baseline
	for i := 0; i < 10; i++ {
		result := compiled.Call(x)
		Keep(result[0])
		Eval(result[0])
		result[0].Free()
	}

	MetalResetPeakMemory()
	initialMem := MetalGetActiveMemory()

	for i := 0; i < 100; i++ {
		result := compiled.Call(x)
		Keep(result[0])
		Eval(result[0])
		result[0].Free()
	}

	Eval() // Final cleanup

	finalMem := MetalGetActiveMemory()
	peakMem := MetalGetPeakMemory()

	// Memory should not grow significantly (allow 10MB slack for caching)
	growth := int64(finalMem) - int64(initialMem)
	if growth > 10*1024*1024 {
		t.Errorf("memory grew by %d bytes over 100 iterations", growth)
	}
	t.Logf("memory: initial=%dMB, final=%dMB, peak=%dMB, growth=%dKB",
		initialMem/(1<<20), finalMem/(1<<20), peakMem/(1<<20), growth/1024)
}

// TestCompileWithRandomState verifies compiled function can capture and update random state.
func TestCompileWithRandomState(t *testing.T) {
	// Simulate logits for sampling
	logits := NewArrayFloat32([]float32{0.1, 0.2, 0.3, 0.4}, []int32{1, 4})
	Keep(logits)
	logits.Eval()

	// Initial random key
	key := RandomKey(42)
	Keep(key)

	// Compile a sampling function that splits the key
	compiled := Compile(func(inputs []*Array) []*Array {
		logits := inputs[0]
		keyIn := inputs[1]

		// Split key: one for sampling, one for next iteration
		key1, key2 := RandomSplit(keyIn)

		// Sample from logits
		sample := RandomCategoricalWithKey(logits, key2, -1, 1)

		return []*Array{sample, key1}
	})
	defer compiled.Free()

	// Run multiple sampling steps
	samples := make([]int32, 10)
	for i := 0; i < 10; i++ {
		outputs := compiled.Call(logits, key)
		Keep(outputs...)
		Eval(outputs...)
		samples[i] = outputs[0].ItemInt32()
		key.Free()
		key = outputs[1]
	}

	// Verify we got valid samples (0-3)
	for i, s := range samples {
		if s < 0 || s > 3 {
			t.Errorf("sample %d out of range: %d", i, s)
		}
	}
	t.Logf("samples: %v", samples)

	// Verify samples aren't all the same (randomness works)
	allSame := true
	for i := 1; i < len(samples); i++ {
		if samples[i] != samples[0] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("all samples are the same - random state may not be updating")
	}
}

// swiGLU implements the GPT-OSS custom SwiGLU activation.
func swiGLU(gate, up *Array, alpha, limit float32) *Array {
	gateClipped := ClipScalar(gate, 0, limit, false, true)
	upClipped := ClipScalar(up, -limit, limit, true, true)
	gluScaled := MulScalar(gateClipped, alpha)
	sig := Sigmoid(gluScaled)
	outGlu := Mul(gateClipped, sig)
	return Mul(outGlu, AddScalar(upClipped, 1.0))
}

// TestCompileSwiGLU verifies compiled SwiGLU produces correct output.
func TestCompileSwiGLU(t *testing.T) {
	gate := NewArrayFloat32([]float32{-1, 0, 1, 2, 5, 10}, []int32{6})
	up := NewArrayFloat32([]float32{-5, -1, 0, 1, 5, 10}, []int32{6})
	Keep(gate, up)
	Eval(gate, up)

	const alpha float32 = 1.702
	const limit float32 = 7.0

	// Uncompiled
	expected := swiGLU(gate, up, alpha, limit)
	Keep(expected)
	Eval(expected)

	// Compiled
	compiled := Compile(func(inputs []*Array) []*Array {
		return []*Array{swiGLU(inputs[0], inputs[1], alpha, limit)}
	})
	defer compiled.Free()

	result := compiled.Call(gate, up)[0]
	Keep(result)
	Eval(result)

	// Compare
	expData := expected.Data()
	resData := result.Data()
	for i := range expData {
		diff := expData[i] - resData[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-5 {
			t.Errorf("mismatch at %d: expected %f, got %f", i, expData[i], resData[i])
		}
	}
	t.Logf("SwiGLU results: %v", resData)
}

// BenchmarkSwiGLUUncompiled benchmarks uncompiled SwiGLU.
func BenchmarkSwiGLUUncompiled(b *testing.B) {
	gate := RandomNormal([]int32{1, 2880}, 42)
	up := RandomNormal([]int32{1, 2880}, 43)
	Keep(gate, up)
	Eval(gate, up)

	const alpha float32 = 1.702
	const limit float32 = 7.0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := swiGLU(gate, up, alpha, limit)
		Keep(result)
		Eval(result)
	}
}

// BenchmarkSwiGLUCompiled benchmarks compiled SwiGLU.
func BenchmarkSwiGLUCompiled(b *testing.B) {
	gate := RandomNormal([]int32{1, 2880}, 42)
	up := RandomNormal([]int32{1, 2880}, 43)
	Keep(gate, up)
	Eval(gate, up)

	const alpha float32 = 1.702
	const limit float32 = 7.0

	compiled := Compile(func(inputs []*Array) []*Array {
		return []*Array{swiGLU(inputs[0], inputs[1], alpha, limit)}
	})
	defer compiled.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := compiled.Call(gate, up)
		Keep(result[0])
		Eval(result[0])
	}
}

// BenchmarkSwiGLU10xUncompiled benchmarks 10 chained SwiGLU ops uncompiled.
func BenchmarkSwiGLU10xUncompiled(b *testing.B) {
	x := RandomNormal([]int32{1, 2880}, 42)
	Keep(x)
	x.Eval()

	const alpha float32 = 1.702
	const limit float32 = 7.0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y := x
		for j := 0; j < 10; j++ {
			y = swiGLU(y, y, alpha, limit)
		}
		Keep(y)
		Eval(y)
	}
}

// BenchmarkSwiGLU10xCompiled benchmarks 10 chained SwiGLU ops compiled.
func BenchmarkSwiGLU10xCompiled(b *testing.B) {
	x := RandomNormal([]int32{1, 2880}, 42)
	Keep(x)
	x.Eval()

	const alpha float32 = 1.702
	const limit float32 = 7.0

	compiled := Compile(func(inputs []*Array) []*Array {
		y := inputs[0]
		for j := 0; j < 10; j++ {
			y = swiGLU(y, y, alpha, limit)
		}
		return []*Array{y}
	})
	defer compiled.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := compiled.Call(x)
		Keep(result[0])
		Eval(result[0])
	}
}

// ============ Sampler Benchmarks ============

// sampleTopK implements top-k sampling
func sampleTopK(logits, key *Array, k int) (*Array, *Array) {
	neg := Neg(logits)
	indices := Argpartition(neg, k-1, -1)
	topK := Slice(indices, []int32{0}, []int32{int32(k)})
	values := TakeAlongAxis(logits, topK, -1)
	key1, key2 := RandomSplit(key)
	sampled := RandomCategoricalWithKey(values, key2, -1, 1)
	return Take(topK, sampled, -1), key1
}

// sampleTopP implements top-p (nucleus) sampling
func sampleTopP(logits, key *Array, p float32, vocabSize int32) (*Array, *Array) {
	sorted := Argsort(Neg(logits), -1)
	sortedLogits := TakeAlongAxis(logits, sorted, -1)
	probs := Softmax(sortedLogits, -1)
	cumProbs := Cumsum(probs, -1)
	mask := LessScalar(cumProbs, p)
	negInf := FullDtype(float32(-1e9), logits.Dtype(), vocabSize)
	masked := Where(mask, sortedLogits, negInf)
	key1, key2 := RandomSplit(key)
	sampled := RandomCategoricalWithKey(masked, key2, -1, 1)
	return Take(sorted, sampled, -1), key1
}

// BenchmarkSampleTopKUncompiled benchmarks uncompiled top-k sampling.
func BenchmarkSampleTopKUncompiled(b *testing.B) {
	vocabSize := int32(32000)
	logits := RandomNormal([]int32{vocabSize}, 42)
	key := RandomKey(42)
	Keep(logits, key)
	Eval(logits, key)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var token *Array
		token, key = sampleTopK(logits, key, 40)
		Keep(token, key)
		Eval(token)
	}
}

// BenchmarkSampleTopKCompiled benchmarks compiled top-k sampling.
func BenchmarkSampleTopKCompiled(b *testing.B) {
	vocabSize := int32(32000)
	logits := RandomNormal([]int32{vocabSize}, 42)
	key := RandomKey(42)
	Keep(logits, key)
	Eval(logits, key)

	compiled := Compile(func(inputs []*Array) []*Array {
		token, newKey := sampleTopK(inputs[0], inputs[1], 40)
		return []*Array{token, newKey}
	})
	defer compiled.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs := compiled.Call(logits, key)
		Keep(outputs...)
		Eval(outputs[0])
		key = outputs[1]
	}
}

// BenchmarkSampleTopPUncompiled benchmarks uncompiled top-p sampling.
func BenchmarkSampleTopPUncompiled(b *testing.B) {
	vocabSize := int32(32000)
	logits := RandomNormal([]int32{vocabSize}, 42)
	key := RandomKey(42)
	Keep(logits, key)
	Eval(logits, key)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var token *Array
		token, key = sampleTopP(logits, key, 0.9, vocabSize)
		Keep(token, key)
		Eval(token)
	}
}

// BenchmarkSampleTopPCompiled benchmarks compiled top-p sampling.
func BenchmarkSampleTopPCompiled(b *testing.B) {
	vocabSize := int32(32000)
	logits := RandomNormal([]int32{vocabSize}, 42)
	key := RandomKey(42)
	Keep(logits, key)
	Eval(logits, key)

	compiled := Compile(func(inputs []*Array) []*Array {
		token, newKey := sampleTopP(inputs[0], inputs[1], 0.9, vocabSize)
		return []*Array{token, newKey}
	})
	defer compiled.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		outputs := compiled.Call(logits, key)
		Keep(outputs...)
		Eval(outputs[0])
		key = outputs[1]
	}
}

// TestCompiledSamplerMemoryStable verifies compiled samplers don't leak memory.
func TestCompiledSamplerMemoryStable(t *testing.T) {
	vocabSize := int32(32000)
	logits := RandomNormal([]int32{vocabSize}, 42)
	key := RandomKey(42)
	Keep(logits, key)
	Eval(logits, key)

	compiledTopK := Compile(func(inputs []*Array) []*Array {
		token, newKey := sampleTopK(inputs[0], inputs[1], 40)
		return []*Array{token, newKey}
	})
	defer compiledTopK.Free()

	compiledTopP := Compile(func(inputs []*Array) []*Array {
		token, newKey := sampleTopP(inputs[0], inputs[1], 0.9, vocabSize)
		return []*Array{token, newKey}
	})
	defer compiledTopP.Free()

	// Warmup
	for i := 0; i < 10; i++ {
		out := compiledTopK.Call(logits, key)
		Keep(out...)
		Eval(out[0])
		out[0].Free()
		key = out[1]
	}

	MetalResetPeakMemory()
	initialMem := MetalGetActiveMemory()

	// Run 500 iterations of each sampler
	for i := 0; i < 500; i++ {
		// TopK
		out := compiledTopK.Call(logits, key)
		Keep(out...)
		Eval(out[0])
		out[0].Free()
		key = out[1]

		// TopP
		out = compiledTopP.Call(logits, key)
		Keep(out...)
		Eval(out[0])
		out[0].Free()
		key = out[1]
	}

	Eval() // Final cleanup

	finalMem := MetalGetActiveMemory()
	peakMem := MetalGetPeakMemory()

	growth := int64(finalMem) - int64(initialMem)
	t.Logf("memory: initial=%dMB, final=%dMB, peak=%dMB, growth=%dKB",
		initialMem/(1<<20), finalMem/(1<<20), peakMem/(1<<20), growth/1024)

	// Memory should stay bounded (allow 20MB for caching overhead)
	if growth > 20*1024*1024 {
		t.Errorf("memory grew by %d bytes over 1000 sampler calls - possible leak!", growth)
	}
}

// BenchmarkSimpleOps measures simple ops with cleanup
func BenchmarkSimpleOps(b *testing.B) {
	weight := NewArrayFloat32([]float32{1, 0, 0, 1}, []int32{2, 2})
	Keep(weight)
	weight.Eval()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := NewArrayFloat32([]float32{1, 2}, []int32{1, 2})
		result := Matmul(x, weight)
		Keep(result)
		AsyncEval(result)
		result.Eval()
	}
}

// BenchmarkLayerLike measures layer-like ops (~15 ops)
func BenchmarkLayerLike(b *testing.B) {
	hidden := int32(256)
	w := Ones(hidden, hidden)
	Keep(w)
	w.Eval()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Ones(1, hidden)
		// Simulate attention-like ops with proper shapes
		h := Matmul(x, w)                  // [1, 256] @ [256, 256] = [1, 256]
		h = Add(h, Matmul(h, w))           // residual
		h = Mul(h, Sigmoid(Matmul(h, w)))  // gating
		h = Matmul(h, w)                   // output projection
		h = Add(x, RMSNormNoWeight(h, 1e-5)) // residual + norm
		Keep(h)
		AsyncEval(h)
		Eval(h)
	}
}

// BenchmarkManyOps measures with increasing op counts
func BenchmarkManyOps(b *testing.B) {
	w := Ones(64, 64)
	Keep(w)
	w.Eval()

	for _, numOps := range []int{10, 50, 100, 500, 1000} {
		b.Run(fmt.Sprintf("ops_%d", numOps), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				x := Ones(1, 64)
				for j := 0; j < numOps; j++ {
					x = Add(x, Matmul(x, w))
				}
				Keep(x)
				AsyncEval(x)
				Eval(x)
			}
		})
	}
}

// BenchmarkLLMScale measures at LLM-realistic scale (~1348 arrays)
func BenchmarkLLMScale(b *testing.B) {
	// Simulate Qwen-like model: 24 layers, each with ~56 ops = 1344 arrays
	numLayers := 24
	opsPerLayer := 56

	// Create weights
	hidden := int32(64)
	weights := make([]*Array, numLayers*4)
	for i := range weights {
		weights[i] = Ones(hidden, hidden)
	}
	Keep(weights...)
	Eval(weights...)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := Ones(1, hidden)

		for layer := 0; layer < numLayers; layer++ {
			for op := 0; op < opsPerLayer/4; op++ {
				x = Add(x, Matmul(x, weights[layer*4]))
				x = Mul(x, Sigmoid(x))
			}
		}
		Keep(x)
		AsyncEval(x)
		Eval(x)
	}
}

// BenchmarkArrayFreeLoop measures the cost of freeing N arrays
func BenchmarkArrayFreeLoop(b *testing.B) {
	for _, count := range []int{100, 500, 1000, 1500} {
		b.Run(fmt.Sprintf("arrays_%d", count), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				arrays := make([]*Array, count)
				for j := 0; j < count; j++ {
					arrays[j] = NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{2, 2})
				}
				b.StartTimer()

				// Cleanup all arrays
				Eval()
			}
		})
	}
}

// BenchmarkCleanupIsolated measures just cleanup time
func BenchmarkCleanupIsolated(b *testing.B) {
	w := NewArrayFloat32([]float32{1}, []int32{1, 1})
	Keep(w)
	w.Eval()

	for _, count := range []int{100, 500, 1000, 1500} {
		b.Run(fmt.Sprintf("arrays_%d", count), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				x := NewArrayFloat32([]float32{1}, []int32{1})
				for j := 0; j < count; j++ {
					x = Add(x, x)
				}
				Keep(x)
				b.StartTimer()
				Eval() // Just cleanup
			}
		})
	}
}

// TestMemoryStable verifies that cleanup doesn't cause unbounded memory growth.
func TestMemoryStable(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory test in short mode")
	}

	// Create realistic-sized arrays (like KV cache)
	batchSize := int32(1)
	numHeads := int32(8)
	seqLen := int32(256)
	headDim := int32(64)
	cacheShape := []int32{batchSize, numHeads, seqLen, headDim}
	cacheSize := batchSize * numHeads * seqLen * headDim * 4 // float32 = 4 bytes

	// Initial cache
	keys := Zeros(cacheShape, DtypeFloat32)
	values := Zeros(cacheShape, DtypeFloat32)
	Keep(keys, values)
	Eval(keys, values)

	// Warmup
	for i := 0; i < 5; i++ {
		oldKeys, oldValues := keys, values

		newKeys := Add(keys, keys)
		newValues := Add(values, values)
		Keep(newKeys, newValues)
		Eval(newKeys, newValues)

		oldKeys.Free()
		oldValues.Free()
		keys, values = newKeys, newValues
	}

	MetalResetPeakMemory()
	initialMem := MetalGetActiveMemory()

	// Run 100 steps
	for step := 0; step < 100; step++ {
		oldKeys, oldValues := keys, values

		newKeys := Add(keys, keys)
		newValues := Add(values, values)
		Keep(newKeys, newValues)
		Eval(newKeys, newValues)

		oldKeys.Free()
		oldValues.Free()
		keys, values = newKeys, newValues
	}

	Eval() // Final cleanup

	finalMem := MetalGetActiveMemory()
	peakMem := MetalGetPeakMemory()

	growth := int64(finalMem) - int64(initialMem)
	expectedMaxGrowth := int64(cacheSize * 4 * 10)

	t.Logf("cache size: %d bytes", cacheSize*2)
	t.Logf("memory: initial=%dMB, final=%dMB, peak=%dMB, growth=%dKB",
		initialMem/(1<<20), finalMem/(1<<20), peakMem/(1<<20), growth/1024)

	if growth > expectedMaxGrowth {
		t.Errorf("memory grew by %d bytes over 100 steps (expected max %d) - possible leak",
			growth, expectedMaxGrowth)
	}
}
