package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlx/mlxthread"
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
	convPrefix := history.ConvState().Reshape(1, 1).Add(x.Cumsum(1, false, true))
	deltaPrefix := history.DeltaState().Reshape(1, 1).Add(x.Multiply(mlx.FromValues(positions, 1, n)).Cumsum(1, false, true))
	valuePrefix := kv.V().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, offset), mlx.Slice()).SumAxis(2, true).Reshape(1, 1).Add(x.Multiply(x).Cumsum(1, false, true))
	var conv, delta *mlx.Array
	var convStates, deltaStates []*mlx.Array
	for _, end := range append(rc.SnapshotSplits(n), n) {
		prefix := x.Slice(mlx.Slice(), mlx.Slice(0, end))
		conv = history.ConvState().Add(prefix.SumAxis(1, true).Reshape(1, 1, 1))
		delta = history.DeltaState().Add(prefix.Multiply(mlx.FromValues(positions[:end], 1, end)).SumAxis(1, true).Reshape(1, 1, 1, 1))
		convStates = append(convStates, conv)
		deltaStates = append(deltaStates, delta)
	}
	rc.Put(b, convStates, deltaStates)
	hidden := mlx.Stack([]*mlx.Array{
		valuePrefix, convPrefix, deltaPrefix, x,
		mlx.FromValues(positions, 1, n),
		mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, n), 1),
		mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, n), 2),
		mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, n), 3),
	}, -1)
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
		{"prompt is prefix", [][]int32{{1, 2}, {1, 2, 3, 4}}},
		{"empty prefix", [][]int32{{1, 2}, {2, 1, 3}}},
		{"one prompt", [][]int32{{1, 2, 3, 4}}},
		{"one token", [][]int32{{1}, {1}, {2}}},
		{"chunked prefix", [][]int32{append(slices.Repeat([]int32{1}, prefillChunkSize()+3), 2), append(slices.Repeat([]int32{1}, prefillChunkSize()+3), 3)}},
		{"chunked suffix", [][]int32{append([]int32{1, 2}, slices.Repeat([]int32{3}, prefillChunkSize()+3)...), append([]int32{1, 2}, slices.Repeat([]int32{4}, prefillChunkSize()+3)...)}},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			m := &scoringTestModel{}
			r := &Runner{Model: m}
			rows := make([]scoreRow, len(tt.tokens))
			for i, tokens := range tt.tokens {
				rows[i] = scoreRow{tokens: tokens, candidates: []int32{2, 0, 1, 3, 4}}
			}
			independent := make([][]float32, len(rows))
			for i, row := range rows {
				cold := &Runner{Model: &scoringTestModel{}}
				got, _, err := cold.scoreRows(context.Background(), []scoreRow{row}, 0)
				if err != nil {
					t.Fatal(err)
				}
				independent[i] = got[0]
				cold.scoreCache.close()
			}
			defer func() { r.scoreCache.close() }()
			prefix := scorePrefixLength(rows)
			for repeat := range 3 {
				m.positions, m.lengths, m.projections = nil, nil, 0
				got, cached, err := r.scoreRows(context.Background(), rows, prefix)
				if err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if !slices.Equal(got[i], independent[i]) {
						t.Fatalf("repeat %d row %d shared %v != independent %v", repeat, i, got[i], independent[i])
					}
				}
				if m.projections != len(rows) {
					t.Fatalf("got %d projections for %d rows", m.projections, len(rows))
				}
				if repeat > 0 {
					total := 0
					for _, n := range m.lengths {
						total += n
					}
					input := 0
					for _, row := range rows {
						input += len(row.tokens)
					}
					if cached != input-total {
						t.Fatalf("cached=%d, want %d", cached, input-total)
					}
					if total != 0 {
						t.Fatalf("repeat forwarded %d tokens, want zero", total)
					}
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
			defer func() { r.scoreCache.close() }()
			rows := []scoreRow{{tokens: []int32{1, 2, 3}, candidates: []int32{0}}, {tokens: []int32{1, 2, 4}, candidates: []int32{0}}}
			_, _, err := r.scoreRows(ctx, rows, 2)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("got %v, want context cancellation", err)
			}
			if len(m.positions) != cancelAfter {
				t.Fatal("cancellation did extra forwards")
			}

			m.cancel = nil
			if _, _, err := r.scoreRows(context.Background(), rows, 2); err != nil {
				t.Fatalf("request after cancellation: %v", err)
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
		{MaxTokens: 10, Rows: make([]llm.ScoreRow, 65)},
		{Rows: []llm.ScoreRow{{Prompt: "1", Candidates: []string{"2"}}}},
		{MaxTokens: 11, Rows: []llm.ScoreRow{{Prompt: "1", Candidates: []string{"2"}}}},
		{MaxTokens: 1, Rows: []llm.ScoreRow{{Prompt: "12", Candidates: []string{"3"}}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "", Candidates: []string{"1"}}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "1"}}},
		{MaxTokens: 10, Rows: []llm.ScoreRow{{Prompt: "1", Candidates: slices.Repeat([]string{"2"}, 27)}}},
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

func TestScoreHandlerScratchCache(t *testing.T) {
	worker := mlxtest.Worker(t)
	if err := worker.Do(context.Background(), func() error {
		if !mlx.GPUIsAvailable() {
			return errors.New("GPU unavailable")
		}
		mlx.ClearCache()
		return nil
	}); err != nil {
		t.Skipf("MLX GPU not available: %v", err)
	}
	t.Cleanup(func() {
		if err := worker.Do(context.Background(), func() error {
			mlx.ClearCache()
			return nil
		}); err != nil {
			t.Error(err)
		}
	})
	r := &Runner{Tokenizer: newTestTokenizer(t, []int32{7}), contextLength: 512, mlxThread: worker}
	defer worker.Do(context.Background(), func() error { r.Close(); return nil })
	for _, n := range []int{16, 128, 512, 16} {
		for _, cancelled := range []bool{false, true} {
			// Start with reusable scratch from an earlier request.
			// These CPU-filled arrays are freed synchronously into MLX's cache.
			const scratchBytes = 4 << 20
			if err := worker.Do(context.Background(), func() error {
				r.Close()
				mlx.Scoped(func() { mlx.FromValues(make([]int32, scratchBytes/4), scratchBytes/4) })
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			m := &scoringTestModel{}
			if cancelled {
				m.cancel, m.cancelAfter = cancel, 1
			}
			r.Model = m
			input, err := json.Marshal(llm.ScoreRequest{
				MaxTokens: 512,
				Rows:      []llm.ScoreRow{{Prompt: strings.Repeat("1", n), Candidates: []string{"2"}}},
			})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/score", strings.NewReader(string(input)))
			response := httptest.NewRecorder()
			r.scoreHandler(response, req)
			cancel()
			wantStatus := http.StatusOK
			if cancelled {
				wantStatus = http.StatusInternalServerError
			}
			if response.Code != wantStatus {
				t.Fatalf("tokens=%d cancelled=%t: status %d: %s", n, cancelled, response.Code, response.Body)
			}
			memory, err := mlxthread.Call(context.Background(), worker, func() (int, error) {
				memory := mlx.CacheMemory()
				if !cancelled && mlx.MetalIsAvailable() && memory == 0 {
					return 0, errors.New("successful scoring discarded all reusable scratch buffers")
				}
				return memory, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			// Successful requests may reuse scratch; cancelled requests release it.
			if memory > scoreScratchLimit || (cancelled && memory >= scratchBytes) {
				t.Fatalf("tokens=%d cancelled=%t: retained %d scratch bytes", n, cancelled, memory)
			}
		}
	}
}
