package gemma4embedding

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
)

// TestBidirectionalSlidingWindowMask verifies the band geometry against the
// reference (transformers masking_utils): a pair (q,k) is attended iff
// |q-k| <= window, inclusive. window=3 attends d<=3, blocks d>=4.
// Padding columns are not masked here (SDPA's kLens handles those).
func TestBidirectionalSlidingWindowMask(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		L := 10
		b := &batch.Batch{
			InputIDs:     mlx.FromValues(make([]int32, L), 1, L),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(L)},
		}
		const window = 3 // inclusive max attended distance
		mask := BidirectionalSlidingWindowMask(b, window, mlx.DTypeFloat32)
		if mask.IsZero() {
			t.Fatal("expected array mask")
		}
		arr := mask.AsArray(b, L, mlx.DTypeFloat32)
		mlx.Eval(arr)
		vals := arr.Floats() // [B=1, 1, L, L]
		if len(vals) != L*L {
			t.Fatalf("got %d mask values, want %d", len(vals), L*L)
		}
		for q := range L {
			for k := range L {
				got := vals[q*L+k]
				d := q - k
				if d < 0 {
					d = -d
				}
				wantBlocked := d > window
				isBlocked := math.IsInf(float64(got), -1)
				if wantBlocked != isBlocked {
					t.Errorf("q=%d k=%d (d=%d): blocked=%v want %v (val %v)", q, k, d, isBlocked, wantBlocked, got)
				}
			}
		}

		// Short batch: window larger than any row's furthest pair -> the
		// mask is all-attending, so skip materializing it.
		short := &batch.Batch{
			InputIDs:     mlx.FromValues(make([]int32, 2), 1, 2),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{2},
		}
		if m := BidirectionalSlidingWindowMask(short, 513, mlx.DTypeFloat32); !m.IsZero() {
			t.Error("expected zero mask for short rows")
		}
		// A row of length window+1 has its furthest pair at exactly the
		// boundary: d == window is attended, so the mask must exist.
		edge := &batch.Batch{
			InputIDs:     mlx.FromValues(make([]int32, window+1), 1, window+1),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{window + 1},
		}
		if m := BidirectionalSlidingWindowMask(edge, window, mlx.DTypeFloat32); m.IsZero() {
			t.Error("expected materialized mask at the d==window boundary")
		}
	})
}

// TestBuildMasksWindow pins the reference window derivation: config
// sliding_window (512) plus one, so a sliding layer attends |q-kv| <= 513
// inclusive — matching transformers' config.sliding_window + 1 overlay.
func TestBuildMasksWindow(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &Model{Cfg: tinyTextConfig()}
		m.Cfg.SlidingWindow = 512
		L := 600
		b := &batch.Batch{
			InputIDs:     mlx.FromValues(make([]int32, L), 1, L),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(L)},
		}
		sliding, _ := m.buildMasks(b)
		if sliding.IsZero() {
			t.Fatal("expected a materialized sliding mask at 600 tokens")
		}
		arr := sliding.AsArray(b, L, mlx.DTypeFloat32)
		mlx.Eval(arr)
		vals := arr.Floats()

		// Reference: attend |q-kv| <= 513.
		probe := func(q, k int) bool {
			v := vals[q*L+k]
			return math.IsInf(float64(v), -1) // blocked?
		}
		if probe(0, 513) {
			t.Error("d=513 must be attended (reference: abs <= sliding_window+1)")
		}
		if !probe(0, 514) {
			t.Error("d=514 must be blocked")
		}
		if probe(0, 512) {
			t.Error("d=512 must be attended")
		}
	})
}
