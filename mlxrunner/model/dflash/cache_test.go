package dflash

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// With zero queries, attention must be the mean over the mathematically
// allowed values. This catches accidental causal masking of proposal rows
// and window-offset mistakes independently of transformer weight arithmetic.
func TestNoncausalWindowAndRollback(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg, a := meanAttention(2, 4)
		kv := cache.NewRotatingKVCache(4)
		defer kv.Free()
		context := make([]float32, 16)
		for i := range 8 {
			context[2*i] = float32(i + 1)
			context[2*i+1] = float32(i + 1)
		}
		forward := func(offset int, contextValues []float32, proposals []float32, want []float32) {
			mlx.Scoped(func() {
				l := len(proposals) / 2
				n := len(contextValues) / 2
				bb := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, l), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(l)}}
				kv.PrepareSnapshots([]int{offset})
				out := a.Forward(mlx.FromValues(proposals, 1, l, 2), mlx.Zeros(mlx.DTypeFloat32, 1, 1, n, 2), mlx.FromValues(contextValues, 1, 1, n, 2), kv, bb, mlx.FromValues([]int32{int32(offset)}, 1), nn.AttentionMask{}, 1, int32(l), cfg)
				mlx.Eval(out)
				// MLX runs float32 matmuls as TF32 on M5-class GPUs by default,
				// up to ~7e-4 here against ~1e-7 with MLX_ENABLE_TF32=0. Mask or
				// rollback errors exceed 0.1% by orders of magnitude.
				for i, v := range out.Floats() {
					if math.Abs(float64(v-want[i])) > math.Max(1e-3, math.Abs(float64(want[i]))*1e-3) {
						t.Fatalf("offset %d index %d: got %g want %g", offset, i, v, want[i])
					}
				}
				snapshots := kv.TakeSnapshots()
				if !kv.Restore(nil, offset) && !kv.Restore(snapshots[0], offset) {
					t.Fatal("could not restore context")
				}
				for _, snapshot := range snapshots {
					if snapshot != nil {
						snapshot.Close()
					}
				}
				if kv.Offset() != offset {
					t.Fatal("proposal rows remained committed")
				}
			})
		}
		// At q=8: context 6,7,8 plus the whole proposal block 9,10 => 8.
		// At q=9: context 7,8 plus block 9,10 => 8.5.
		forward(8, context, []float32{9, 9, 10, 10}, []float32{8, 8, 8.5, 8.5})
		// Discarded proposal values must not contaminate the next context run.
		// The newly committed context row is 20, not the old proposal value 9.
		forward(9, []float32{20, 20}, []float32{30, 30, 40, 40}, []float32{21, 21, 24.5, 24.5})
	})
}

// Use a scalar visibility oracle across every acceptance position, with
// separately committed target context and rejected proposal values far apart.
func TestRepeatedBlockRollback(t *testing.T) {
	for _, width := range []int{2, 5, 8} {
		for _, initial := range []int{1, 7, 8, 9, 17} {
			for accepted := range width {
				mlxtest.RunSubtest(t, fmt.Sprintf("width %d context %d accepted %d", width, initial, accepted), func(t *mlxtest.T) {
					const window = 8
					cfg, a := meanAttention(2, window)
					kv := cache.NewRotatingKVCache(window)
					defer kv.Free()
					context := []float32{}
					commit := func(n int) {
						// One-token updates deliberately wrap the physical ring.
						for range n {
							i := len(context)
							v := float32((i*17)%31 + 1)
							context = append(context, v)
							b := &batch.Batch{SeqOffsets: []int32{int32(i)}}
							kv.Update(b, mlx.Zeros(mlx.DTypeFloat32, 1, 1, 1, 2), mlx.FromValues([]float32{v, v}, 1, 1, 1, 2))
						}
					}
					commit(initial)
					for round := range 3 {
						offset := len(context)
						kv.PrepareSnapshots([]int{offset})
						proposals := make([]float32, 2*width)
						for i := range width {
							proposals[2*i] = float32(100 + 13*i)
							proposals[2*i+1] = proposals[2*i]
						}
						b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, width), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(width)}}
						out := a.Forward(mlx.FromValues(proposals, 1, width, 2), nil, nil, kv, b, mlx.FromValues([]int32{int32(offset)}, 1), nn.AttentionMask{}, 1, int32(width), cfg)
						mlx.Eval(out)
						if err := visibleMeanError(out.Floats(), context, proposals, offset, window); err != nil {
							t.Fatalf("width=%d initial=%d accepted=%d round=%d %v", width, initial, accepted, round, err)
						}
						snapshots := kv.TakeSnapshots()
						if len(snapshots) != 1 || snapshots[0] == nil {
							t.Fatal("missing rollback snapshot")
						}
						if !kv.Restore(snapshots[0], offset) {
							t.Fatal("rollback failed")
						}
						snapshots[0].Close()
						if kv.Offset() != offset {
							t.Fatal("wrong committed offset")
						}
						commit(accepted + 1)
					}
				})
			}
		}
	}
}

// meanAttention is one head of zero queries over width-wide rows: every output
// row is the mean of the value rows it may see, and values copy their input.
func meanAttention(width int, window int32) (*Config, *Attention) {
	kv, ones := make([]float32, 2*width*width), make([]float32, width)
	for i := range width {
		kv[(width+i)*width+i], ones[i] = 1, 1
	}
	norm := func() *nn.RMSNorm { return nn.NewRMSNorm(mlx.FromValues(ones, width), 1e-6) }
	cfg := &Config{HiddenSize: int32(width), NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: int32(width), RMSNormEps: 1e-6, RopeTheta: 10000, Scale: float32(1 / math.Sqrt(float64(width))), SlidingWindow: window}
	return cfg, &Attention{
		QProj:  &nn.Linear{Weight: mlx.Zeros(mlx.DTypeFloat32, width, width)},
		KVProj: &nn.Linear{Weight: mlx.FromValues(kv, 2*width, width)},
		OProj:  &nn.Linear{Weight: mlx.FromValues(kv[width*width:], width, width)},
		QNorm:  norm(),
		KNorm:  norm(),
	}
}

// visibleMeanError checks two-wide meanAttention rows, each holding one value
// twice, against the mean of every proposal and the context values within
// window of the row's position start+q.
func visibleMeanError(out, context, proposals []float32, start, window int) error {
	rows := len(proposals) / 2
	for q := range rows {
		sum, count := 0.0, 0
		for k, v := range context {
			if start+q-k < window {
				sum += float64(v)
				count++
			}
		}
		for j := range rows {
			sum += float64(proposals[2*j])
			count++
		}
		want := sum / float64(count)
		for d := range 2 {
			if got := float64(out[2*q+d]); math.Abs(got-want) > math.Max(1e-3, math.Abs(want)*2e-3) {
				return fmt.Errorf("q=%d got=%g want=%g", q, got, want)
			}
		}
	}
	return nil
}
