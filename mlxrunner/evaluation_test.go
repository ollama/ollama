package mlxrunner

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

// A cacheless forward exposes the row's tokens, layout and media in its output,
// making cross-request wiring and padded evaluations observable without weights.
type rowTestModel struct {
	stubModel
	batches []int
}

func (m *rowTestModel) Forward(b *batch.Batch, _ []cache.Layer) (*mlx.Array, *mlx.Array) {
	m.batches = append(m.batches, b.InputIDs.Dim(0))
	h := b.InputIDs.AsType(mlx.DTypeFloat32).ExpandDims(-1)
	offsets := make([]float32, len(b.SeqQueryLens))
	for i := range offsets {
		offsets[i] = b.Layout[i].(float32)
	}
	h = h.Add(mlx.FromValues(offsets, len(offsets), 1, 1))
	for _, item := range b.Media {
		h = h.SliceUpdate(item.Features.Reshape(1, 1, 1), mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(item.Pos, item.Pos+1), mlx.Slice())
	}
	return h, nil
}

func (m *rowTestModel) PrepareMedia([]model.Segment) (*model.PreparedRequest, error) { return nil, nil }
func (m *rowTestModel) EncodeMedia(_ *model.PreparedItem, a *mlx.Array) *mlx.Array   { return a }

func TestWholeForwardRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &rowTestModel{}
		r := &Runner{Model: m, parallel: 2, Requests: make(chan queuedRequest, 2)}
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
		for _, order := range [][]int{{0, 1}, {1, 0}} {
			result := make([][]float32, 2)
			done := make([]chan error, 2)
			for _, id := range order {
				tokens := [][]int32{{1, 2, 3}, {4}}[id]
				input := &model.PreparedRequest{Tokens: tokens, Layout: float32(10 * (id + 1))}
				var items []mediaItem
				if id == 0 {
					input.Items = []model.PreparedItem{{Range: [2]int{1, 2}, Source: 0, MediaData: []float32{77}, Dims: []int{1, 1}}}
					var err error
					items, err = bindItems(input, []model.Segment{{Kind: "image", Data: []byte("distinct image")}})
					if err != nil {
						t.Fatal(err)
					}
				}
				done[id] = make(chan error, 1)
				r.Requests <- &evaluationRequest{ctx: context.Background(), done: done[id], open: func(slot *batchSlot) (*evaluationState, error) {
					return r.openEvaluation(slot, input, items, func(h *mlx.Array) error { result[id] = h.Floats(); return nil }), nil
				}}
			}
			s.admitPending()
			s.step()
			for id := range result {
				if err := <-done[id]; err != nil {
					t.Fatal(err)
				}
				want := [][]float32{{11, 77, 13}, {24}}[id]
				if !slices.Equal(result[id], want) {
					t.Fatalf("order %v row %d: %v, want %v", order, id, result[id], want)
				}
			}
		}
		if !slices.Equal(m.batches, []int{2, 2}) {
			t.Fatalf("forward batch sizes %v, want [2 2]", m.batches)
		}
	})
}

func TestEvaluationCancellationAndSlotReuse(t *testing.T) {
	testCtx := t.Context()
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &rowTestModel{}
		r := &Runner{Model: m, parallel: 2}
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
		ctx, cancel := context.WithCancel(testCtx)
		defer cancel()
		results := make([][]float32, 3)
		requests := make([]*evaluationRequest, 3)
		for i := range requests {
			requestCtx := testCtx
			if i == 0 {
				requestCtx = ctx
			}
			requests[i] = &evaluationRequest{ctx: requestCtx, done: make(chan error, 1), open: func(slot *batchSlot) (*evaluationState, error) {
				input := &model.PreparedRequest{Tokens: []int32{int32(i + 1)}, Layout: float32(10)}
				return r.openEvaluation(slot, input, nil, func(h *mlx.Array) error {
					results[i] = h.Floats()
					return nil
				}), nil
			}}
		}
		s.admit(requests[0])
		s.admit(requests[1])
		cancel()
		s.step()
		if err := <-requests[0].done; !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled request: %v", err)
		}
		if err := <-requests[1].done; err != nil {
			t.Fatal(err)
		}
		s.admit(requests[2])
		s.step()
		if err := <-requests[2].done; err != nil {
			t.Fatal(err)
		}
		if len(results[0]) != 0 || !slices.Equal(results[1], []float32{12}) || !slices.Equal(results[2], []float32{13}) {
			t.Fatalf("evaluations after cancellation/reuse: %v", results)
		}
		for _, slot := range s.slots {
			if slot.busy() {
				t.Fatal("completed evaluation still owns a slot")
			}
		}
		s.step()
		for _, request := range requests {
			select {
			case err := <-request.done:
				t.Fatalf("duplicate completion: %v", err)
			default:
			}
		}
	})
}

type cachedRowTestModel struct {
	stubModel
	shapes [][2]int
}

func (m *cachedRowTestModel) NewCaches() []cache.Cache {
	return []cache.Cache{cache.NewRecurrentCache(1, 1, 1, 1, 1)}
}

func (m *cachedRowTestModel) Forward(b *batch.Batch, layers []cache.Layer) (*mlx.Array, *mlx.Array) {
	B, L := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	m.shapes = append(m.shapes, [2]int{B, L})
	c := layers[0].(cache.Recurrent)
	h := c.Get(b, mlx.DTypeFloat32)
	x := b.InputIDs.AsType(mlx.DTypeFloat32).Reshape(B, L, 1)
	conv := h.ConvState().Add(x.Cumsum(1, false, true))
	delta := h.DeltaState().Reshape(B, 1, 1).Add(x.Multiply(x).Cumsum(1, false, true))
	var convStates, deltaStates []*mlx.Array
	for _, end := range append(c.SnapshotSplits(L), L) {
		convStates = append(convStates, conv.Slice(mlx.Slice(), mlx.Slice(end-1, end), mlx.Slice()))
		deltaStates = append(deltaStates, delta.Slice(mlx.Slice(), mlx.Slice(end-1, end), mlx.Slice()).Reshape(B, 1, 1, 1))
	}
	c.Put(b, convStates, deltaStates)
	return conv.Add(delta), nil
}

func TestCachedForwardBatchesAndRestores(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &cachedRowTestModel{}
		r := &Runner{Model: m, parallel: 2, Requests: make(chan queuedRequest, 2)}
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
		defer func() {
			for _, slot := range s.slots {
				if slot != nil && slot.scoreCache != nil {
					slot.scoreCache.close()
				}
			}
		}()
		for round, prompts := range [][][]int32{
			{{1, 2, 3, 4, 5}, {6, 7, 8}},
			{{1, 2, 9, 10}, {6, 7, 8, 9}},
			{{1, 2, 3, 4, 5}, {6, 10}},
		} {
			got := make([][]float32, 2)
			done := make([]chan error, 2)
			cached := make([]int, 2)
			for id, tokens := range prompts {
				done[id] = make(chan error, 1)
				r.Requests <- &evaluationRequest{ctx: context.Background(), done: done[id], open: func(slot *batchSlot) (*evaluationState, error) {
					p := r.openEvaluation(slot, &model.PreparedRequest{Tokens: tokens, ScorePrefix: 2 - id}, nil, func(h *mlx.Array) error { got[id] = h.Floats(); return nil })
					cached[id] = p.position
					return p, nil
				}}
			}
			s.admitPending()
			for s.slots[0].busy() || s.slots[1].busy() {
				s.step()
			}
			for id, tokens := range prompts {
				if err := <-done[id]; err != nil {
					t.Fatal(err)
				}
				ref := newScoreTestRunner(&cachedRowTestModel{})
				h, _, err := ref.scoreHiddenRow(context.Background(), &model.PreparedRequest{Tokens: tokens}, nil)
				if err != nil {
					t.Fatal(err)
				}
				want := h.Floats()
				ref.scoreCache.close()
				if !slices.Equal(got[id], want) {
					t.Fatalf("round %d row %d: %v, want %v", round, id, got[id], want)
				}
			}
			if round == 1 && (cached[0] != 2 || cached[1] != 3) {
				t.Fatalf("prefix reuse lost: %v", cached)
			}
		}
		if !slices.Contains(m.shapes, [2]int{2, 3}) {
			t.Fatalf("no shared multi-token forward: %v", m.shapes)
		}
	})
}

func TestCachedForwardChunkBoundaries(t *testing.T) {
	for _, tt := range []struct {
		name   string
		length [2]int
		media  [2][]mediaItem
		rows   int
		tokens int
	}{
		{name: "unequal", length: [2]int{8, 3}, rows: 2, tokens: 3},
		{name: "budget", length: [2]int{4096, 4096}, rows: 2, tokens: 1024},
		{
			name: "successive atomic boundaries", length: [2]int{8, 5}, rows: 2, tokens: 2,
			media: [2][]mediaItem{
				{{pos: 3, length: 4, item: &model.PreparedItem{}}},
				{{pos: 2, length: 3, item: &model.PreparedItem{}}},
			},
		},
		{
			name: "atomic expansion does not fit peer", length: [2]int{8, 5}, rows: 1, tokens: 8,
			media: [2][]mediaItem{{{pos: 0, length: 7, item: &model.PreparedItem{}}}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := &batchScheduler{slots: make([]*batchSlot, 2)}
			for i, n := range tt.length {
				p := &evaluationState{input: &model.PreparedRequest{Tokens: make([]int32, n)}, session: &cacheSession{}, media: &requestMedia{items: tt.media[i]}}
				s.slots[i] = &batchSlot{evaluation: &activeEvaluation{request: &evaluationRequest{ctx: t.Context()}, state: p}}
			}
			rows, n := s.cachedForwardSlots(s.slots[0])
			if len(rows) != tt.rows || n != tt.tokens {
				t.Fatalf("chunk %dx%d, want %dx%d", len(rows), n, tt.rows, tt.tokens)
			}
		})
	}
}
