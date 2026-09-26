package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/systemone"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

// This small stateful model uses real KV and recurrent caches. Its outputs
// depend on both recurrent tensors, all KV values, and absolute positions, so a
// missing restore or a branch that overwrites the prefix changes the answer.
type scoringTestModel struct {
	textOnlyModel
	caches      []cache.Cache
	positions   []int
	lengths     []int
	projections int
	cancel      context.CancelFunc
	cancelAfter int
}

func (m *scoringTestModel) NewCaches() []cache.Cache {
	m.caches = []cache.Cache{cache.NewKVCache(), cache.NewRecurrentCache(1, 1, 1, 1, 1), nil}
	return m.caches
}

func (m *scoringTestModel) Forward(b *batch.Batch, caches []cache.Cache) (*mlx.Array, *mlx.Array) {
	n, offset := b.InputIDs.Dim(1), int(b.SeqOffsets[0])
	m.positions = append(m.positions, offset)
	m.lengths = append(m.lengths, n)
	x := b.InputIDs.AsType(mlx.DTypeFloat32)
	kv := caches[0].(*cache.KVCache).Update(b, x.Reshape(1, 1, n, 1), x.Multiply(x).Reshape(1, 1, n, 1))
	rc := caches[1].(*cache.RecurrentCache)
	history := rc.Get(b, mlx.DTypeFloat32)
	positions := make([]float32, n)
	for i := range positions {
		positions[i] = float32(offset + i + 1)
	}
	conv := history.ConvState().Add(x.SumAxis(1, true).Reshape(1, 1, 1))
	delta := history.DeltaState().Add(x.Multiply(mlx.FromValues(positions, 1, n)).SumAxis(1, true).Reshape(1, 1, 1, 1))
	rc.Put(b, []*mlx.Array{conv}, []*mlx.Array{delta})
	values := mlx.Concatenate([]*mlx.Array{
		kv.V().SumAxis(2, true).Reshape(1), conv.Reshape(1), delta.Reshape(1),
		x.Slice(mlx.Slice(), mlx.Slice(n-1, n)).Reshape(1),
		mlx.FromValues([]float32{float32(offset + n), 1, 2, 3}, 4),
	}, 0).Reshape(1, 1, 8)
	hidden := mlx.Zeros(mlx.DTypeFloat32, 1, n, 8).Add(values)
	if m.cancel != nil && len(m.positions) == m.cancelAfter {
		m.cancel()
	}
	return hidden, hidden
}

func (m *scoringTestModel) Unembed(hidden *mlx.Array) *mlx.Array {
	if hidden.Dim(1) != 1 {
		panic("scoring projected more than the final position")
	}
	m.projections++
	return hidden
}

func TestScoreSharedPrefix(t *testing.T) {
	for _, tt := range []struct {
		name   string
		tokens [][]int32
	}{
		{"branch lengths and repeated restore", [][]int32{{1, 2, 3, 4}, {1, 2, 5}, {1, 2, 3, 6, 7}, {1, 2, 3, 4}}},
		{"identical prompts", [][]int32{{1, 2, 3}, {1, 2, 3}}},
		{"empty prefix", [][]int32{{1, 2}, {2, 1, 3}}},
		{"one prompt", [][]int32{{1, 2, 3, 4}}},
		{"one token", [][]int32{{1}, {1}, {2}}},
		{"chunked prefix", [][]int32{append(slices.Repeat([]int32{1}, prefillChunkSize()+3), 2), append(slices.Repeat([]int32{1}, prefillChunkSize()+3), 3)}},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			m := &scoringTestModel{}
			r := &Runner{Model: m}
			rows := make([]scoreRow, len(tt.tokens))
			for i, tokens := range tt.tokens {
				rows[i] = scoreRow{tokens: tokens, candidates: []int32{2, 0, 1, 3, 4}}
			}
			independent, err := r.scoreRows(context.Background(), rows, 0)
			if err != nil {
				t.Fatal(err)
			}
			prefix := scorePrefixLength(rows)
			// Repeated requests must release cache state and isolate branches.
			for range 3 {
				m.positions, m.lengths, m.projections = nil, nil, 0
				got, err := r.scoreRows(context.Background(), rows, prefix)
				if err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if !slices.Equal(got[i], independent[i]) {
						t.Fatalf("row %d shared %v != independent %v", i, got[i], independent[i])
					}
				}
				for _, c := range m.caches {
					if c != nil && c.Offset() != 0 {
						t.Fatal("request left its caches alive")
					}
					if c != nil {
						for _, state := range c.State() {
							if state != nil {
								t.Fatal("request retained cache arrays")
							}
						}
					}
				}
				if m.projections != len(rows) {
					t.Fatalf("got %d projections for %d rows", m.projections, len(rows))
				}
				total := 0
				for _, n := range m.lengths {
					total += n
				}
				want := prefix
				for _, row := range rows {
					want += len(row.tokens) - prefix
				}
				if total != want {
					t.Fatalf("forwarded %d tokens, want %d", total, want)
				}
			}
		})
	}
}

func TestScoreCancellation(t *testing.T) {
	for _, cancelAfter := range []int{1, 2} {
		mlxtest.Run(t, func(t *mlxtest.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m := &scoringTestModel{cancel: cancel, cancelAfter: cancelAfter}
			r := &Runner{Model: m}
			rows := []scoreRow{{tokens: []int32{1, 2, 3}, candidates: []int32{0}}, {tokens: []int32{1, 2, 4}, candidates: []int32{0}}}
			_, err := r.scoreRows(ctx, rows, 2)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("got %v, want context cancellation", err)
			}
			if len(m.positions) != cancelAfter {
				t.Fatal("cancellation did extra forwards")
			}
			for _, c := range m.caches {
				if c != nil && c.Offset() != 0 {
					t.Fatal("cancelled request retained cache state")
				}
			}
		})
	}
}

func TestScoreValidation(t *testing.T) {
	tok := newTestTokenizer(t, []int32{7})
	r := &Runner{Tokenizer: tok, contextLength: 10}
	for _, input := range []llm.ScoreRequest{
		{},
		{MaxTokens: 10},
		{MaxTokens: 11, Rows: []llm.ScoreRow{{Prompt: "1", Candidates: []string{"2"}}}},
		{MaxTokens: 1, Rows: []llm.ScoreRow{{Prompt: "12", Candidates: []string{"3"}}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "", Candidates: []string{"1"}}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "1"}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "1", Candidates: []string{"23"}}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "1", Candidates: []string{"2", "2"}}}},
	} {
		_, err := r.score(context.Background(), input)
		var status api.StatusError
		if !errors.As(err, &status) || status.StatusCode != 400 {
			t.Fatalf("expected validation error before model access, got %v", err)
		}
	}
}

// Optional full-model check against independently prefilling each question.
func TestNimbleModel(t *testing.T) {
	name := os.Getenv("OLLAMA_NIMBLE_MODEL")
	if name == "" {
		t.Skip("set OLLAMA_NIMBLE_MODEL to an imported Qwen3.5/Nimble model")
	}
	var req systemone.Request
	if err := json.Unmarshal([]byte(`{
		"model":"nimble", "state":"I was charged twice. Please refund the duplicate.",
		"questions":{
			"department":{"type":"choice","instructions":"Which department?","criteria":{"billing":"Payments","technical":"Software"}},
			"refund":{"type":"noul","instructions":"Refund requested?"},
			"urgency":{"type":"score","instructions":"How urgent?","criteria":["Routine","Urgent","Emergency"]}
		}
	}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := systemone.Compile(req)
	if err != nil {
		t.Fatal(err)
	}

	mlxtest.Run(t, func(t *mlxtest.T) {
		r := &Runner{}
		if err := r.Load(name); err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		shared, err := r.score(context.Background(), c.Request)
		if err != nil {
			t.Fatal(err)
		}
		probabilities := func(logits []float32) []float64 {
			p := make([]float64, len(logits))
			peak := slices.Max(logits)
			var sum float64
			for i, x := range logits {
				p[i] = math.Exp(float64(x - peak))
				sum += p[i]
			}
			for i := range p {
				p[i] /= sum
			}
			return p
		}
		for i, row := range c.Request.Rows {
			single, err := r.score(context.Background(), llm.ScoreRequest{Rows: []llm.ScoreRow{row}, MaxTokens: c.Request.MaxTokens})
			if err != nil {
				t.Fatal(err)
			}
			// Different prefill shapes can choose different kernels, and this
			// backbone retains its native precision, as do quantized heads.
			// Check the resulting decisions/probabilities;
			// log exact logits to make numerical differences visible.
			p, q := probabilities(shared.Logits[i]), probabilities(single.Logits[0])
			for j := range p {
				if diff := math.Abs(p[j] - q[j]); diff > 0.01 {
					t.Errorf("row %d candidate %d probability difference=%g", i, j, diff)
				}
			}
			if slices.Index(p, slices.Max(p)) != slices.Index(q, slices.Max(q)) {
				t.Errorf("row %d changed its winning candidate", i)
			}
			t.Logf("row %d: shared=%v independent=%v", i, shared.Logits[i], single.Logits[0])
		}
	})
}
