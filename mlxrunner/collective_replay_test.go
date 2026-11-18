package mlxrunner

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/sample"
)

// Keep token predictions simple while the real recurrent operation and cache
// ownership run through acceptDrafts, including its grammar-error path.
type collectiveRecurrentModel struct {
	fakeMTPModel
	deferredWrites  int
	panicAfterWrite bool
}

func (m *collectiveRecurrentModel) NewCaches() []cache.Cache {
	return []cache.Cache{cache.NewRecurrentCache(2, 3, 2, 32, 32)}
}

func (m *collectiveRecurrentModel) Forward(b *batch.Batch, layers []cache.Layer) (*mlx.Array, *mlx.Array) {
	const packedDim = 128
	B, L := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	ids := b.InputIDs.Ints()
	packed := make([]float32, len(ids)*packedDim)
	predictions := make([]int32, len(ids))
	for i, id := range ids {
		predictions[i] = m.predict[id]
		for j := range packedDim {
			packed[i*packedDim+j] = float32(math.Sin(float64(id+1)*.13+float64(j)*.07) * .2)
		}
	}
	c := layers[0].(cache.Recurrent)
	history := c.Get(b, mlx.DTypeBFloat16)
	splits := c.SnapshotSplits(L)
	options := []nn.RecurrentOption{nn.WithRecurrentHistory(history), nn.WithSnapshotSplits(splits)}
	x := mlx.FromValues(packed, B, L, packedDim).AsType(mlx.DTypeBFloat16)
	ba := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, B, L, 4), .1)
	dt := mlx.Zeros(mlx.DTypeBFloat16, 2)
	a := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 2), .2)
	conv := make([]*mlx.Array, len(splits)+1)
	for i, boundary := range append(splits, L) {
		conv[i] = mlx.AddScalar(history.ConvState(), float32(boundary))
	}
	var y *mlx.Array
	if deferred, ok := c.(cache.RecurrentReplay); ok && B > 1 && L > 1 && len(splits) == L-1 {
		var end *mlx.Array
		var inputs nn.GatedDeltaReplay
		y, end, inputs = nn.GatedDeltaForReplay(b, x, ba, dt, a, options...)
		deferred.PutReplay(b, conv, end, inputs)
		m.deferredWrites++
		if m.panicAfterWrite {
			panic("collective fixture abort")
		}
	} else {
		var states []*mlx.Array
		y, states = nn.GatedDelta(b, x, ba, dt, a, options...)
		c.Put(b, conv, states)
	}
	out := oneHotLogits(predictions).Reshape(B, L, mtpTestVocab)
	out = out.Add(mlx.MulScalar(mlx.Sum(y.AsType(mlx.DTypeFloat32).Reshape(-1), 0, false), 0))
	return out, out
}

func TestCollectiveReplayAbortBeforeResolve(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &collectiveRecurrentModel{fakeMTPModel: fakeMTPModel{tok: tok, predict: map[int32]int32{1: 1}}, panicAfterWrite: true}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		rounds := make([]draftRound, 2)
		positions := []int{0, 0}
		for i := range rounds {
			cs := m.NewCaches()
			defer cs[0].Free()
			r.Sampler.Add(i, sample.Options{Temperature: 0}, nil)
			drafts := []int32{1, 1, 1}
			rounds[i] = draftRound{
				s: &speculationSession{spec: &speculation{r: r, slotID: i, targets: cs}, drafter: nopDrafter{}}, position: &positions[i],
				current:    sample.Result{Token: mlx.FromValues([]int32{1}, 1)},
				candidates: &draftCandidates{tokens: mlx.FromValues(drafts, 1, 3), dist: r.Sampler.Distribution(i, oneHotLogits(drafts), nil)},
			}
		}
		func() {
			defer func() {
				if got := recover(); got != "collective fixture abort" {
					t.Fatalf("abort = %v, want original fixture panic", got)
				}
			}()
			r.acceptDrafts(rounds)
		}()
		for _, round := range rounds {
			owner := round.s.spec.targets[0]
			if owner.Offset() != 0 || len(owner.TakeSnapshots()) != 0 {
				t.Fatal("abort changed prior offset or retained a snapshot schedule")
			}
			mlx.Eval(owner.State()...)
		}
		m.panicAfterWrite = false
		for i, out := range r.acceptDrafts(rounds) {
			if out.err != nil || out.accepted != 3 || positions[i] != 4 {
				t.Fatalf("retry row %d: accepted %d position %d error %v", i, out.accepted, positions[i], out.err)
			}
		}
	})
}

func TestCollectiveReplayAcceptanceContinuation(t *testing.T) {
	for _, depth := range []int{1, 2} {
		mlxtest.RunSubtest(t, fmt.Sprintf("short-depth-%d", depth), func(t *mlxtest.T) {
			tok := newTestTokenizer(t, []int32{6})
			m := &collectiveRecurrentModel{fakeMTPModel: fakeMTPModel{tok: tok, predict: map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6}}}
			r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
			defer r.Sampler.Free()
			owners, references := make([][]cache.Cache, 3), make([][]cache.Cache, 3)
			positions := make([]int, 3)
			for i := range owners {
				owners[i], references[i] = m.NewCaches(), m.NewCaches()
				defer owners[i][0].Free()
				defer references[i][0].Free()
				r.Sampler.Add(i, sample.Options{Temperature: 0}, nil)
			}
			cases := []struct {
				current                int32
				drafts, want, retained []int32
			}{
				{1, []int32{2, 3, 4}, []int32{2, 3, 4, 5}, []int32{1, 2, 3, 4}},
				{2, []int32{3, 7, 7}, []int32{3, 4}, []int32{2, 3}},
				{4, []int32{5, 6, 7}, []int32{5, 6}, []int32{4, 5}},
			}
			for step := range 2 {
				if step == 1 {
					// A short tail remains near a request budget. The first row
					// accepts EOS; the second keeps its draft and samples EOS.
					cases = cases[:2]
					cases[0].current, cases[0].drafts, cases[0].want, cases[0].retained = 5, []int32{6}, []int32{6}, []int32{5}
					cases[1].current, cases[1].drafts, cases[1].want, cases[1].retained = 4, []int32{5}, []int32{5, 6}, []int32{4, 5}
					if depth == 2 {
						cases[0].drafts = append(cases[0].drafts, 7)
						cases[1].drafts = append(cases[1].drafts, 6)
					}
				}
				rounds := make([]draftRound, len(cases))
				for i, tc := range cases {
					rounds[i] = draftRound{
						s: &speculationSession{spec: &speculation{r: r, slotID: i, targets: owners[i]}, drafter: nopDrafter{}}, position: &positions[i],
						current:    sample.Result{Token: mlx.FromValues([]int32{tc.current}, 1)},
						candidates: &draftCandidates{tokens: mlx.FromValues(tc.drafts, 1, len(tc.drafts)), dist: r.Sampler.Distribution(i, oneHotLogits(tc.drafts), nil)},
					}
				}
				outcomes := r.acceptDrafts(rounds)
				for i, tc := range cases {
					if out := outcomes[i]; out.err != nil || !slices.Equal(resultIDs(out.results), tc.want) {
						t.Fatalf("step %d row %d: results %v error %v, want %v", step, i, resultIDs(out.results), out.err, tc.want)
					}
					before := references[i][0].Offset()
					m.Forward(&batch.Batch{InputIDs: mlx.FromValues(tc.retained, 1, len(tc.retained)), SeqOffsets: []int32{int32(before)}, SeqQueryLens: []int32{int32(len(tc.retained))}}, cache.Layers(references[i]))
					if positions[i] != references[i][0].Offset() || owners[i][0].Offset() != positions[i] {
						t.Fatalf("step %d row %d offset mismatch", step, i)
					}
					for j, state := range owners[i][0].State() {
						got := state.AsType(mlx.DTypeFloat32)
						want := references[i][0].State()[j].AsType(mlx.DTypeFloat32)
						mlx.Eval(got, want)
						gv, wv := got.Floats(), want.Floats()
						if !slices.Equal(got.Dims(), want.Dims()) || len(gv) == 0 || len(gv) != len(wv) {
							t.Fatalf("step %d row %d state %d shape %v length %d, want nonempty %v length %d", step, i, j, got.Dims(), len(gv), want.Dims(), len(wv))
						}
						for k := range gv {
							if math.IsNaN(float64(gv[k])) || math.IsInf(float64(gv[k]), 0) || math.IsNaN(float64(wv[k])) || math.IsInf(float64(wv[k]), 0) {
								t.Fatalf("step %d row %d state %d element %d is not finite: got %g, want %g", step, i, j, k, gv[k], wv[k])
							}
							if math.Float32bits(gv[k]) != math.Float32bits(wv[k]) {
								t.Fatalf("step %d row %d state %d element %d = %g, want %g", step, i, j, k, gv[k], wv[k])
							}
						}
					}
					if len(owners[i][0].TakeSnapshots()) != 0 {
						t.Fatal("speculative snapshots were not drained")
					}
				}
			}
			if m.deferredWrites != 1 {
				t.Fatalf("deferred forwards = %d, want one; short drafts should capture states directly", m.deferredWrites)
			}
		})
	}
}
