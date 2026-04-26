//go:build mlx

package sample

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// slotLogits builds a [1, V] logits tensor for a single-slot Sample call.
func slotLogits(values []float32) *mlx.Array {
	return mlx.FromValues(values, 1, len(values))
}

// batchLogits stacks per-row float32 slices of equal length into a [B, V]
// logits tensor.
func batchLogits(rows ...[]float32) *mlx.Array {
	v := len(rows[0])
	flat := make([]float32, 0, len(rows)*v)
	for _, r := range rows {
		if len(r) != v {
			panic("batchLogits: rows must share vocab size")
		}
		flat = append(flat, r...)
	}
	return mlx.FromValues(flat, len(rows), v)
}

// sampleOne runs Sample on a freshly-added single slot and returns the
// sampled token id. Used both for the single-slot options table and as the
// reference oracle for the batched-equivalence test.
func sampleOne(t *testing.T, opts Options, priorTokens []int32, values []float32) int {
	t.Helper()
	s := New(128)
	t.Cleanup(func() {
		s.Free()
		mlx.Sweep()
	})
	s.Add(0, opts, priorTokens)

	got := s.Sample([]int{0}, slotLogits(values)).Token
	mlx.Eval(got)
	return got.Int()
}

// logOf returns log(p) as a float32 so tests can build logits that softmax to
// a chosen probability distribution.
func logOf(p float64) float32 { return float32(math.Log(p)) }

// TestSampleSingleSlotOptions pins the per-slot behavior of each Options
// knob against a concrete expected token. Expected values are worked out by
// hand from the math of each transform, not from a second call into the
// sampler — so a regression in any single transform shows up here.
func TestSampleSingleSlotOptions(t *testing.T) {
	skipIfNoMLX(t)

	cases := []struct {
		name   string
		opts   Options
		priors []int32
		logits []float32
		want   int
	}{
		{
			name:   "presence penalty",
			opts:   Options{RepeatLastN: 1, PresencePenalty: 6},
			priors: []int32{1},
			logits: []float32{0, 5, 4},
			want:   2, // token 1: 5 - 6 = -1, argmax shifts to 2
		},
		{
			name:   "repeat penalty on positive logits",
			opts:   Options{RepeatLastN: 1, RepeatPenalty: 2},
			priors: []int32{1},
			logits: []float32{0, 5, 4},
			want:   2, // token 1 positive → divided: 5/2 = 2.5, argmax shifts to 2
		},
		{
			name:   "repeat penalty on negative logits",
			opts:   Options{RepeatLastN: 1, RepeatPenalty: 4},
			priors: []int32{1},
			logits: []float32{-5, -1, -3},
			want:   2, // token 1 negative → multiplied: -1*4 = -4, argmax shifts to 2
		},
		{
			name:   "frequency penalty",
			opts:   Options{RepeatLastN: 4, FrequencyPenalty: 2},
			priors: []int32{1, 1},
			logits: []float32{0, 5, 4},
			want:   2, // 5 - 2*count(1)=2*2=4 → 1, argmax shifts to 2
		},
		{
			name:   "top-k",
			opts:   Options{Temperature: 1, TopK: 1},
			logits: []float32{1, 5, 4},
			want:   1, // only argmax survives → deterministic even with temperature
		},
		{
			name:   "top-p",
			opts:   Options{Temperature: 1, TopP: 0.4},
			logits: []float32{logOf(0.5), logOf(0.3), logOf(0.2)},
			want:   0, // exclusive cumsum below 0.4 keeps only token 0
		},
		{
			name:   "min-p",
			opts:   Options{Temperature: 1, MinP: 0.7},
			logits: []float32{logOf(0.5), logOf(0.3), logOf(0.2)},
			want:   0, // threshold 0.5*0.7=0.35 drops all but the top token
		},
		{
			name:   "RepeatLastN=0 disables penalties",
			opts:   Options{RepeatLastN: 0, RepeatPenalty: 2, PresencePenalty: 10},
			priors: []int32{1},
			logits: []float32{0, 5, 4},
			want:   1, // 0 = disabled per API contract, argmax unchanged
		},
		{
			name:   "RepeatLastN=-1 resolves to num_ctx",
			opts:   Options{RepeatLastN: -1, PresencePenalty: 6},
			priors: []int32{1},
			logits: []float32{0, 5, 4},
			want:   2, // -1 → num_ctx (128); penalty applies, argmax shifts
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sampleOne(t, tc.opts, tc.priors, tc.logits); got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}

// TestSampleHistoryWindow verifies that penalty history respects the
// RepeatLastN window: priors longer than RepeatLastN are trimmed on Add,
// and once the ring wraps, tokens that rotate out no longer contribute
// to penalties.
func TestSampleHistoryWindow(t *testing.T) {
	skipIfNoMLX(t)

	s := New(128)
	t.Cleanup(func() {
		s.Free()
		mlx.Sweep()
	})

	// RepeatLastN=2 with priors {1, 2, 3}: makeHistoryRow keeps only
	// {2, 3}. Token 1 was trimmed — its penalty is NOT active.
	s.Add(0, Options{RepeatLastN: 2, PresencePenalty: 10}, []int32{1, 2, 3})

	// Step 1: logits favor token 1 (trimmed). If the trim were broken it
	// would be penalized and the argmax would move.
	step1 := s.Sample([]int{0}, slotLogits([]float32{0, 5, 0, 0, 0})).Token
	mlx.Eval(step1)
	if got := step1.Int(); got != 1 {
		t.Fatalf("step 1 = %d, want 1 (token 1 trimmed from priors)", got)
	}
	// After step 1 the ring holds {1, 3}; token 2 has rotated out.

	// Step 2: logits favor token 2 (rotated out). If the ring wrap were
	// wrong, token 2 would still be penalized.
	step2 := s.Sample([]int{0}, slotLogits([]float32{0, 0, 5, 0, 0})).Token
	mlx.Eval(step2)
	if got := step2.Int(); got != 2 {
		t.Fatalf("step 2 = %d, want 2 (token 2 rotated out of ring)", got)
	}
}

// TestBatchSamplingPreservesPerSlotBehavior is the core equivalence test:
// for every representative dispatch branch (uniform, serial on mixed opts,
// serial on partial ring, subset/out-of-order), a batched Sample call must
// produce the same token per row as running the same slot alone.
func TestBatchSamplingPreservesPerSlotBehavior(t *testing.T) {
	skipIfNoMLX(t)

	type slot struct {
		id     int
		opts   Options
		priors []int32
	}

	cases := []struct {
		name   string
		slots  []slot
		sample []int
		rows   [][]float32
	}{
		{
			name: "uniform",
			slots: []slot{
				{10, Options{RepeatLastN: 2, PresencePenalty: 5}, []int32{1, 2}},
				{20, Options{RepeatLastN: 2, PresencePenalty: 5}, []int32{0, 2}},
			},
			sample: []int{10, 20},
			rows:   [][]float32{{0, 5, 4}, {3, 0, 0}},
		},
		{
			name: "serial — mixed opts",
			slots: []slot{
				{1, Options{RepeatLastN: 1, RepeatPenalty: 2}, []int32{1}},
				{2, Options{Temperature: 1, TopK: 1}, nil},
			},
			sample: []int{1, 2},
			rows:   [][]float32{{0, 5, 4, 1}, {2, 1, 5, 3}},
		},
		{
			name: "serial — partial ring",
			slots: []slot{
				{1, Options{RepeatLastN: 4, PresencePenalty: 5}, []int32{1, 1, 1, 1}},
				{2, Options{RepeatLastN: 4, PresencePenalty: 5}, []int32{2}},
			},
			sample: []int{1, 2},
			rows:   [][]float32{{0, 5, 4}, {0, 4, 5}},
		},
		{
			name: "subset out-of-order",
			slots: []slot{
				{10, Options{RepeatLastN: 2, PresencePenalty: 10}, []int32{1, 1}},
				{20, Options{RepeatLastN: 2, PresencePenalty: 10}, []int32{2, 2}},
				{30, Options{RepeatLastN: 2, PresencePenalty: 10}, []int32{3, 3}},
			},
			sample: []int{30, 10},
			rows:   [][]float32{{5, 5, 5, 0, 5, 5}, {5, 0, 5, 5, 0, 5}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Per-slot reference for each sampled seq.
			want := make([]int, len(tc.sample))
			for i, id := range tc.sample {
				var spec slot
				for _, s := range tc.slots {
					if s.id == id {
						spec = s
						break
					}
				}
				want[i] = sampleOne(t, spec.opts, spec.priors, tc.rows[i])
			}

			// Batched call.
			s := New(128)
			t.Cleanup(func() {
				s.Free()
				mlx.Sweep()
			})
			for _, spec := range tc.slots {
				s.Add(spec.id, spec.opts, spec.priors)
			}
			res := s.Sample(tc.sample, batchLogits(tc.rows...))
			mlx.Eval(res.Token)
			got := res.Token.Ints()

			for i, id := range tc.sample {
				if got[i] != want[i] {
					t.Errorf("seq %d: batched = %d, per-slot = %d", id, got[i], want[i])
				}
			}
		})
	}
}

// TestRemoveDoesNotLeakHistory: after Remove, a newly-added slot at the
// recycled row must start from its own priors only — no carryover from
// the removed slot's history.
func TestRemoveDoesNotLeakHistory(t *testing.T) {
	skipIfNoMLX(t)

	opts := Options{RepeatLastN: 1, PresencePenalty: 10}
	s := New(128)
	t.Cleanup(func() {
		s.Free()
		mlx.Sweep()
	})
	s.Add(1, opts, []int32{1})
	s.Add(2, opts, []int32{2})
	s.Remove(1)
	s.Add(3, opts, []int32{0})

	// Slot 2 retains history {2}; slot 3 retains history {0}. With
	// equal logits and PresencePenalty=10 the argmax drops to the first
	// unpenalized token.
	res := s.Sample([]int{2, 3}, batchLogits(
		[]float32{3, 3, 0},
		[]float32{3, 3, 0},
	))
	mlx.Eval(res.Token)
	tokens := res.Token.Ints()
	if tokens[0] != 0 {
		t.Errorf("slot 2 = %d, want 0 (token 2 penalized)", tokens[0])
	}
	if tokens[1] != 1 {
		t.Errorf("slot 3 = %d, want 1 (token 0 penalized, no slot-1 carryover)", tokens[1])
	}
}
