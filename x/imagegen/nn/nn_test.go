//go:build mlx

package nn

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
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

	if err := mlx.InitMLX(); err != nil {
		fmt.Printf("Skipping nn tests: %v\n", err)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// TestLinearNoBias verifies Linear without bias computes x @ w.T correctly.
func TestLinearNoBias(t *testing.T) {
	// Weight: [out=2, in=3] -> transposed at forward time
	weight := mlx.NewArrayFloat32([]float32{
		1, 2, 3, // row 0
		4, 5, 6, // row 1
	}, []int32{2, 3})
	mlx.Eval(weight)

	linear := NewLinear(weight, nil)

	// Input: [1, 3]
	x := mlx.NewArrayFloat32([]float32{1, 1, 1}, []int32{1, 3})
	mlx.Eval(x)

	out := linear.Forward(x)
	mlx.Eval(out)

	// Expected: [1,1,1] @ [[1,4],[2,5],[3,6]] = [6, 15]
	data := out.Data()
	if len(data) != 2 || data[0] != 6 || data[1] != 15 {
		t.Errorf("expected [6, 15], got %v", data)
	}
}

// TestLinearWithBias verifies Linear with bias computes x @ w.T + b correctly.
func TestLinearWithBias(t *testing.T) {
	weight := mlx.NewArrayFloat32([]float32{
		1, 2, 3,
		4, 5, 6,
	}, []int32{2, 3})
	bias := mlx.NewArrayFloat32([]float32{10, 20}, []int32{2})
	mlx.Eval(weight, bias)

	linear := NewLinear(weight, bias)

	x := mlx.NewArrayFloat32([]float32{1, 1, 1}, []int32{1, 3})
	mlx.Eval(x)

	out := linear.Forward(x)
	mlx.Eval(out)

	// Expected: [6, 15] + [10, 20] = [16, 35]
	data := out.Data()
	if len(data) != 2 || data[0] != 16 || data[1] != 35 {
		t.Errorf("expected [16, 35], got %v", data)
	}
}

// TestLinearBatched verifies Linear works with batched input.
func TestLinearBatched(t *testing.T) {
	weight := mlx.NewArrayFloat32([]float32{
		1, 0,
		0, 1,
	}, []int32{2, 2}) // Identity
	mlx.Eval(weight)

	linear := NewLinear(weight, nil)

	// Batch of 3 inputs
	x := mlx.NewArrayFloat32([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, []int32{3, 2})
	mlx.Eval(x)

	out := linear.Forward(x)
	mlx.Eval(out)

	// Identity should return same values
	data := out.Data()
	expected := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("at %d: expected %f, got %f", i, v, data[i])
		}
	}
}

// TestRMSNorm verifies RMSNorm computation.
func TestRMSNorm(t *testing.T) {
	weight := mlx.NewArrayFloat32([]float32{1, 1, 1, 1}, []int32{4})
	mlx.Eval(weight)

	norm := NewRMSNorm(weight, 1e-5)

	// Input with known RMS
	x := mlx.NewArrayFloat32([]float32{2, 2, 2, 2}, []int32{1, 4})
	mlx.Eval(x)

	out := norm.Forward(x, 0) // eps=0 uses stored Eps
	mlx.Eval(out)

	// RMS of [2,2,2,2] = 2, so normalized = [1,1,1,1]
	data := out.Data()
	for i, v := range data {
		if math.Abs(float64(v-1.0)) > 1e-4 {
			t.Errorf("at %d: expected ~1.0, got %f", i, v)
		}
	}
}

// TestRMSNormWithScale verifies RMSNorm applies weight scaling.
func TestRMSNormWithScale(t *testing.T) {
	weight := mlx.NewArrayFloat32([]float32{2, 2, 2, 2}, []int32{4})
	mlx.Eval(weight)

	norm := NewRMSNorm(weight, 1e-5)

	x := mlx.NewArrayFloat32([]float32{2, 2, 2, 2}, []int32{1, 4})
	mlx.Eval(x)

	out := norm.Forward(x, 0) // eps=0 uses stored Eps
	mlx.Eval(out)

	// Normalized [1,1,1,1] * weight [2,2,2,2] = [2,2,2,2]
	data := out.Data()
	for i, v := range data {
		if math.Abs(float64(v-2.0)) > 1e-4 {
			t.Errorf("at %d: expected ~2.0, got %f", i, v)
		}
	}
}

// TestEmbedding verifies embedding lookup.
func TestEmbedding(t *testing.T) {
	// Embedding table: 4 tokens, dim 3
	weight := mlx.NewArrayFloat32([]float32{
		0, 0, 0, // token 0
		1, 1, 1, // token 1
		2, 2, 2, // token 2
		3, 3, 3, // token 3
	}, []int32{4, 3})
	mlx.Eval(weight)

	emb := NewEmbedding(weight)

	// Look up tokens [1, 3, 0]
	indices := mlx.NewArrayInt32([]int32{1, 3, 0}, []int32{3})
	mlx.Eval(indices)

	out := emb.Forward(indices)
	mlx.Eval(out)

	data := out.Data()
	expected := []float32{1, 1, 1, 3, 3, 3, 0, 0, 0}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("at %d: expected %f, got %f", i, v, data[i])
		}
	}
}

// TestRepeatKV verifies K/V repetition for GQA.
func TestRepeatKV(t *testing.T) {
	// [B=1, num_kv_heads=2, S=2, head_dim=2]
	x := mlx.NewArrayFloat32([]float32{
		// head 0
		1, 2, // pos 0
		3, 4, // pos 1
		// head 1
		5, 6, // pos 0
		7, 8, // pos 1
	}, []int32{1, 2, 2, 2})
	mlx.Eval(x)

	// Repeat factor 2: 2 kv heads -> 4 heads
	out := RepeatKV(x, 2)
	mlx.Eval(out)

	shape := out.Shape()
	if shape[0] != 1 || shape[1] != 4 || shape[2] != 2 || shape[3] != 2 {
		t.Errorf("expected shape [1,4,2,2], got %v", shape)
	}

	data := out.Data()
	// After repeat: head0, head0, head1, head1
	expected := []float32{
		1, 2, 3, 4, // head 0 (original)
		1, 2, 3, 4, // head 0 (repeat)
		5, 6, 7, 8, // head 1 (original)
		5, 6, 7, 8, // head 1 (repeat)
	}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("at %d: expected %f, got %f", i, v, data[i])
		}
	}
}

// TestRepeatKVNoOp verifies RepeatKV with factor 1 returns input unchanged.
func TestRepeatKVNoOp(t *testing.T) {
	x := mlx.NewArrayFloat32([]float32{1, 2, 3, 4}, []int32{1, 1, 2, 2})
	mlx.Eval(x)

	out := RepeatKV(x, 1)
	// Should return same pointer
	if out != x {
		t.Error("RepeatKV with factor 1 should return input unchanged")
	}
}

// TestApplyCausalMask verifies causal masking.
func TestApplyCausalMask(t *testing.T) {
	// [B=1, heads=1, S=3, S=3] - all ones
	scores := mlx.Ones(1, 1, 3, 3)
	mlx.Eval(scores)

	out := ApplyCausalMask(scores)
	mlx.Eval(out)

	data := out.Data()
	// Lower triangular should be 1, upper should be -1e9
	// Row 0: [1, -inf, -inf]
	// Row 1: [1, 1, -inf]
	// Row 2: [1, 1, 1]
	if data[0] != 1 || data[1] >= 0 || data[2] >= 0 {
		t.Errorf("row 0 wrong: %v", data[0:3])
	}
	if data[3] != 1 || data[4] != 1 || data[5] >= 0 {
		t.Errorf("row 1 wrong: %v", data[3:6])
	}
	if data[6] != 1 || data[7] != 1 || data[8] != 1 {
		t.Errorf("row 2 wrong: %v", data[6:9])
	}
}

// TestApplyCausalMaskWithOffset verifies causal masking with cache offset.
func TestApplyCausalMaskWithOffset(t *testing.T) {
	// Simulating: cache has 2 tokens, adding 1 new query
	// scores: [B=1, heads=1, queryLen=1, keyLen=3]
	scores := mlx.Ones(1, 1, 1, 3)
	mlx.Eval(scores)

	out := ApplyCausalMaskWithOffset(scores, 2)
	mlx.Eval(out)

	data := out.Data()
	// With offset=2, query at position 2 can attend to all 3 positions
	if data[0] != 1 || data[1] != 1 || data[2] != 1 {
		t.Errorf("expected [1, 1, 1], got %v", data)
	}
}

// TestApplyCausalMaskWithOffsetZero verifies offset=0 falls back to regular causal.
func TestApplyCausalMaskWithOffsetZero(t *testing.T) {
	scores := mlx.Ones(1, 1, 2, 2)
	mlx.Eval(scores)

	out := ApplyCausalMaskWithOffset(scores, 0)
	mlx.Eval(out)

	data := out.Data()
	// Standard causal: [1, -inf], [1, 1]
	if data[0] != 1 || data[1] >= 0 {
		t.Errorf("row 0 wrong: %v", data[0:2])
	}
	if data[2] != 1 || data[3] != 1 {
		t.Errorf("row 1 wrong: %v", data[2:4])
	}
}

// BenchmarkLinearSmall benchmarks small Linear forward pass.
func BenchmarkLinearSmall(b *testing.B) {
	weight := mlx.RandomNormal([]int32{256, 256}, 42)
	mlx.Eval(weight)

	linear := NewLinear(weight, nil)

	x := mlx.RandomNormal([]int32{1, 256}, 43)
	mlx.Eval(x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := linear.Forward(x)
		mlx.Eval(out)
	}
}

// BenchmarkLinearLarge benchmarks larger Linear forward pass.
func BenchmarkLinearLarge(b *testing.B) {
	weight := mlx.RandomNormal([]int32{4096, 4096}, 42)
	mlx.Eval(weight)

	linear := NewLinear(weight, nil)

	x := mlx.RandomNormal([]int32{1, 4096}, 43)
	mlx.Eval(x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := linear.Forward(x)
		mlx.Eval(out)
	}
}

// BenchmarkRMSNorm benchmarks RMSNorm forward pass.
func BenchmarkRMSNorm(b *testing.B) {
	weight := mlx.Ones(4096)
	mlx.Eval(weight)

	norm := NewRMSNorm(weight, 1e-5)

	x := mlx.RandomNormal([]int32{1, 4096}, 42)
	mlx.Eval(x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := norm.Forward(x, 0)
		mlx.Eval(out)
	}
}

// BenchmarkEmbedding benchmarks embedding lookup.
func BenchmarkEmbedding(b *testing.B) {
	// Typical vocab size
	weight := mlx.RandomNormal([]int32{32000, 4096}, 42)
	mlx.Eval(weight)

	emb := NewEmbedding(weight)

	// Single token lookup
	indices := mlx.NewArrayInt32([]int32{1000}, []int32{1})
	mlx.Eval(indices)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := emb.Forward(indices)
		mlx.Eval(out)
	}
}

// BenchmarkRepeatKV benchmarks K/V repetition.
func BenchmarkRepeatKV(b *testing.B) {
	// Typical GQA setup: 8 kv heads -> 32 heads
	x := mlx.RandomNormal([]int32{1, 8, 512, 128}, 42)
	mlx.Eval(x)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := RepeatKV(x, 4)
		mlx.Eval(out)
	}
}
