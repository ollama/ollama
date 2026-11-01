package mlx

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// distinctValues returns shuffled distinct values, so each top k has one answer.
func distinctValues(shape ...int) *Array {
	size := 1
	for _, d := range shape {
		size *= d
	}
	values := make([]float32, size)
	for i, v := range rand.New(rand.NewPCG(1, 2)).Perm(size) {
		values[i] = float32(v)
	}
	return FromValues(values, shape...)
}

func TestTopKIndicesMatchesSingleSort(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		for _, tt := range []struct {
			shape []int
			k     int
		}{
			{[]int{2, 100}, 20},
			{[]int{3, topKChunk}, 20},
			{[]int{2, topKChunk + 1}, 20},
			{[]int{2, 248320}, 20},
			{[]int{1, 248320}, 1},
			{[]int{2, 3, 3 * topKChunk}, 16},
			{[]int{1, 5 * topKChunk}, 1000},
		} {
			a := distinctValues(tt.shape...)
			got, want := TopKIndices(a, tt.k), topKSorted(a, tt.k)
			wantDims := append(slices.Clone(tt.shape[:len(tt.shape)-1]), tt.k)
			if !slices.Equal(got.Dims(), wantDims) {
				t.Errorf("%v k=%d: shape %v, want %v", tt.shape, tt.k, got.Dims(), wantDims)
				continue
			}
			if !slices.Equal(got.Ints(), want.Ints()) {
				t.Errorf("%v k=%d: indices differ from a single sort", tt.shape, tt.k)
			}
		}
	})
}

// A row with fewer finite values than k still returns unique, in-range indices.
func TestTopKIndicesMaskedRow(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		const n, k = 248320, 20
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(math.Inf(-1))
		}
		finite := []int{7, 4096, 100000, 200001, n - 1}
		for i, j := range finite {
			values[j] = float32(i + 1)
		}

		ids := TopKIndices(FromValues(values, 1, n), k).Ints()
		if want := []int32{n - 1, 200001, 100000, 4096, 7}; !slices.Equal(ids[:len(want)], want) {
			t.Errorf("finite entries %v, want %v", ids[:len(want)], want)
		}
		seen := map[int32]bool{}
		for _, id := range ids {
			if id < 0 || id >= n || seen[id] {
				t.Fatalf("index %d repeated or out of range in %v", id, ids)
			}
			seen[id] = true
		}
	})
}

func BenchmarkTopKIndices(b *testing.B) {
	for _, rows := range []int{1, 8} {
		for _, bench := range []struct {
			name string
			fn   func(*Array, int) *Array
		}{
			{"sort", topKSorted},
			{"chunked", TopKIndices},
		} {
			b.Run(fmt.Sprintf("rows%d/%s", rows, bench.name), func(b *testing.B) {
				thread := mlxTestThread(b)
				if err := thread.Do(b.Context(), func() error {
					a := distinctValues(rows, 248320)
					Eval(a)
					defer ClearCache()

					Scoped(func() { Eval(bench.fn(a, 20)) })
					b.ResetTimer()
					for range b.N {
						Scoped(func() { Eval(bench.fn(a, 20)) })
					}
					return nil
				}); err != nil {
					b.Fatal(err)
				}
			})
		}
	}
}
