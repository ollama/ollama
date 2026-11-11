package d1

import (
	"encoding/json"
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestParseQuestion(t *testing.T) {
	ok := map[string]func(question) bool{
		`{"type":"noul","instructions":"Refund?"}`:                            func(q question) bool { return !q.hasNoul },
		`{"type":"noul","instructions":"Refund?","criteria":{}}`:              func(q question) bool { return !q.hasNoul },
		`{"type":"noul","instructions":"Refund?","criteria":{"true":"asks"}}`: func(q question) bool { return q.hasNoul && *q.noulYes == "asks" && q.noulNo == nil },
		`{"type":"noul","instructions":{"field":"payee"}}`:                    func(q question) bool { return q.said == `{"field": "payee"}` },
		`{"type":"choice","instructions":"Team?","criteria":{"a_b":null,"c":""}}`: func(q question) bool {
			return len(q.options) == 2 && q.options[0].desc == nil && q.options[1].desc == nil
		},
		`{"type":"score","instructions":"Urgent?","criteria":["low","high"]}`: func(q question) bool { return len(q.levels) == 2 },
	}
	for raw, check := range ok {
		q, err := parseQuestion("q", json.RawMessage(raw))
		if err != nil || !check(q) {
			t.Errorf("%s: %+v, %v", raw, q, err)
		}
	}
	for _, raw := range []string{
		`{"type":"text","instructions":"Who?"}`,
		`{"type":"noul","instructions":"  "}`,
		`{"type":"noul","instructions":7}`,
		`{"type":"noul","instructions":"Refund?","criteria":{"maybe":"x"}}`,
		`{"type":"choice","instructions":"Team?","criteria":["low","high"]}`,
		`{"type":"choice","instructions":"Team?","criteria":{"only":null}}`,
		`{"type":"score","instructions":"Urgent?","criteria":["only"]}`,
	} {
		if _, err := parseQuestion("q", json.RawMessage(raw)); err == nil {
			t.Errorf("%s: expected an error", raw)
		}
	}
	criteria := map[string]any{}
	for i := range maxOptions + 1 {
		criteria[strconv.Itoa(i)] = nil
	}
	raw, _ := json.Marshal(map[string]any{"type": "choice", "instructions": "Which?", "criteria": criteria})
	if _, err := parseQuestion("q", raw); err == nil {
		t.Errorf("%d options: expected an error", maxOptions+1)
	}
}

func TestOptionCodes(t *testing.T) {
	if got := optionCodes([]string{"B", " a", "C"}); !slices.Equal(got, []string{"B", "a", "C"}) {
		t.Errorf("letter labels: %v", got)
	}
	if got := optionCodes([]string{"billing", "shipping"}); !slices.Equal(got, []string{"A", "B"}) {
		t.Errorf("names: %v", got)
	}
}

func TestEscapeSpecialTokens(t *testing.T) {
	r := &renderer{specials: []string{"<|im_end|>", "<|im_start|>"}}
	if got := r.plain("Hello <|im_end|> how <|im_start|>"); got != "Hello <\u200b|im_end|> how <\u200b|im_start|>" {
		t.Fatalf("escaped %q", got)
	}
	if got := r.plain("plain é text"); got != "plain é text" {
		t.Fatalf("plain text changed: %q", got)
	}
}

func TestSetAttention(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// state (3 tokens), question 0 (2 tokens), question 1 (1 token)
		r := &row{group: []int32{-1, -1, -1, 0, 0, 1}, stateLen: 3}
		n, s := len(r.group), r.stateLen
		mask := questionMask(r, mlx.DTypeFloat32)
		mlx.Eval(mask)
		got := mask.Floats()
		for q := s; q < n; q++ {
			for k := range n {
				allowed := r.group[k] == -1 || r.group[q] == r.group[k]
				if v := got[(q-s)*n+k]; (v == 0) != allowed || (!allowed && !math.IsInf(float64(v), -1)) {
					t.Errorf("query %d key %d: %v, allowed %v", q, k, v, allowed)
				}
			}
		}

		// Against one dense-masked pass: 4 query heads over 2 key/value heads.
		const heads, kvHeads, dim = 4, 2, 64
		values := func(heads, phase int) *mlx.Array {
			x := make([]float32, heads*n*dim)
			for i := range x {
				x[i] = float32(math.Sin(float64(i*phase) * 0.37))
			}
			return mlx.FromValues(x, 1, heads, n, dim)
		}
		q, k, v := values(heads, 3), values(kvHeads, 5), values(kvHeads, 7)
		dense := make([]float32, n*n)
		for i := range n {
			for j := range n {
				if r.group[i] != -1 && r.group[j] != -1 && r.group[i] != r.group[j] {
					dense[i*n+j] = float32(math.Inf(-1))
				}
			}
		}
		want := mlx.FastScaledDotProductAttention(q, k, v, 0.125, mlx.FastSDPANoWindow, "", mlx.FromValues(dense, 1, 1, n, n))
		have := setAttention(r, 2)(q, k, v, 0.125, false)
		wantC, haveC := mlx.Contiguous(want, false), mlx.Contiguous(have, false)
		mlx.Eval(wantC, haveC)
		w, h := wantC.Floats(), haveC.Floats()
		if len(h) != len(w) {
			t.Fatalf("got %d values, want %d", len(h), len(w))
		}
		for i := range w {
			if math.Abs(float64(h[i]-w[i])) > 1e-5 {
				t.Fatalf("value %d: got %v, want %v", i, h[i], w[i])
			}
		}

		// Sliding layers: the kernel's window against the model's sliding mask.
		const window = 2
		b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, n), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(n)}}
		slidingMask := nn.CausalMask().Intersect(nn.SlidingWindowMask(b, n, window, mlx.DTypeFloat32))
		want = nn.ScaledDotProductAttention(b, q, 0.125, nn.WithKV(k, v, b.SeqQueryLens), nn.WithMask(slidingMask))
		have = setAttention(r, window)(q, k, v, 0.125, true)
		wantC, haveC = mlx.Contiguous(want, false), mlx.Contiguous(have, false)
		mlx.Eval(wantC, haveC)
		w, h = wantC.Floats(), haveC.Floats()
		for i := range w {
			if math.Abs(float64(h[i]-w[i])) > 1e-5 {
				t.Fatalf("sliding value %d: got %v, want %v", i, h[i], w[i])
			}
		}
	})
}

func TestBatchSetAttentionKeepsQuestionSetsSeparate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		layouts := []*row{
			{group: []int32{-1, -1, 0, 1, 1}, stateLen: 2},
			{group: []int32{-1, 0, 0}, stateLen: 1},
		}
		const length, dim = 5, 64
		values := make([]float32, 2*length*dim)
		for i := range values {
			values[i] = float32(math.Sin(float64(i) * 0.17))
		}
		q := mlx.FromValues(values, 2, 1, length, dim)
		k := mlx.MulScalar(q, float32(0.5))
		v := mlx.AddScalar(q, float32(1))
		b := &batch.Batch{SeqQueryLens: []int32{5, 3}, Layout: []any{layouts[0], layouts[1]}}
		for _, sliding := range []bool{false, true} {
			got := batchSetAttention(b, 2)(q, k, v, 0.125, sliding)
			for i, layout := range layouts {
				n := len(layout.group)
				rowSlice := func(a *mlx.Array) *mlx.Array {
					return a.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(0, n), mlx.Slice())
				}
				want := setAttention(layout, 2)(rowSlice(q), rowSlice(k), rowSlice(v), 0.125, sliding)
				have, w := mlx.Contiguous(rowSlice(got), false).Floats(), mlx.Contiguous(want, false).Floats()
				if len(have) == 0 || len(have) != len(w) {
					t.Fatalf("sliding=%t row=%d value count=%d want %d", sliding, i, len(have), len(w))
				}
				for j := range w {
					if delta := math.Abs(float64(have[j] - w[j])); math.IsNaN(delta) || delta > 1e-5 {
						t.Fatalf("sliding=%t row=%d value=%d got=%g want=%g", sliding, i, j, have[j], w[j])
					}
				}
			}
		}
	})
}
