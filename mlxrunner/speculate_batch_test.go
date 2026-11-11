package mlxrunner

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/sample"
)

func TestAcceptDraftBatch(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{6})
		m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok, predict: map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6}}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		cases := []struct {
			prefix, drafts, want []int32
			current              int32
			accepted, observed   int
			kept                 []float32
		}{
			{[]int32{0}, []int32{2, 3, 4}, []int32{2, 3, 4, 5}, 1, 3, 3, []float32{0, 1, 2, 3, 4}},
			{[]int32{0, 0, 0}, []int32{3, 7, 7}, []int32{3, 4}, 2, 1, 3, []float32{0, 0, 0, 2, 3}},
			{[]int32{0, 0}, []int32{5, 6, 7}, []int32{5, 6}, 4, 2, 2, []float32{0, 0, 4, 5}},
		}
		rounds := make([]draftRound, len(cases))
		positions := make([]int, len(cases))
		for i, tc := range cases {
			cs := m.NewCaches()
			defer cs[0].Free()
			m.Forward(&batch.Batch{InputIDs: mlx.FromValues(tc.prefix, 1, len(tc.prefix)), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(tc.prefix))}}, cache.Layers(cs))
			r.Sampler.Add(i, sample.Options{Temperature: 0}, tc.prefix)
			positions[i] = len(tc.prefix)
			rounds[i] = draftRound{
				s:        &speculationSession{spec: &speculation{r: r, slotID: i, targets: cs}, drafter: nopDrafter{}, layout: []any{i}},
				position: &positions[i], current: sample.Result{Token: mlx.FromValues([]int32{tc.current}, 1)},
				candidates: &draftCandidates{tokens: mlx.FromValues(tc.drafts, 1, len(tc.drafts)), dist: r.Sampler.Distribution(i, oneHotLogits(tc.drafts), nil)},
			}
		}
		m.batches = nil
		outcomes := r.acceptDrafts(rounds)
		if !slices.Equal(m.batches, []int{3}) {
			t.Fatalf("target forwards = %v, want one three-row forward", m.batches)
		}
		if got := m.layouts[len(m.layouts)-1]; len(got) != 3 || got[0] != 0 || got[1] != 1 || got[2] != 2 {
			t.Fatalf("verification layouts = %v", got)
		}
		for i, tc := range cases {
			out := outcomes[i]
			if out.err != nil || !slices.Equal(resultIDs(out.results), tc.want) || out.accepted != tc.accepted || out.observed != tc.observed {
				t.Fatalf("row %d: results %v accepted %d observed %d error %v", i, resultIDs(out.results), out.accepted, out.observed, out.err)
			}
			c := rounds[i].s.spec.targets[0]
			if positions[i] != len(tc.kept) || c.Offset() != len(tc.kept) || !slices.Equal(c.State()[0].Floats(), tc.kept) {
				t.Fatalf("row %d: position %d offset %d cache %v, want %v", i, positions[i], c.Offset(), c.State()[0].Floats(), tc.kept)
			}
			for _, snapshot := range c.TakeSnapshots() {
				if snapshot != nil {
					snapshot.Close()
					t.Errorf("row %d left a speculative snapshot", i)
				}
			}
		}
	})
}

func TestAcceptDraftBatchRejectsMismatchedLengths(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &schedulingModel{}
		r := &Runner{Model: m}
		rounds := make([]draftRound, 2)
		positions := []int{0, 0}
		for i := range rounds {
			cs := m.NewCaches()
			defer cs[0].Free()
			rounds[i] = draftRound{
				s: &speculationSession{spec: &speculation{r: r, slotID: i, targets: cs}}, position: &positions[i],
				current:    sample.Result{Token: mlx.FromValues([]int32{1}, 1)},
				candidates: &draftCandidates{tokens: mlx.Zeros(mlx.DTypeInt32, 1, i+1)},
			}
		}
		for i, out := range r.acceptDrafts(rounds) {
			if out.err == nil || positions[i] != 0 || rounds[i].s.spec.targets[0].Offset() != 0 {
				t.Fatalf("row %d: error %v position %d", i, out.err, positions[i])
			}
		}
		if len(m.batches) != 0 {
			t.Fatalf("incompatible rows reached forward: %v", m.batches)
		}
	})
}

func TestAcceptDraftBatchGrammarFailureIsolated(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok, predict: map[int32]int32{1: 2, 2: 3, 3: 4}}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		// The digit-only fixture cannot produce the opening quote for a string.
		engine, g := testDigitGrammar(t, `{"type":"string"}`)
		r.grammarEngine = engine
		rounds := make([]draftRound, 2)
		positions := []int{0, 0}
		for i := range rounds {
			cs := m.NewCaches()
			defer cs[0].Free()
			r.Sampler.Add(i, sample.Options{Temperature: 0}, nil)
			drafts := []int32{2, 3}
			rounds[i] = draftRound{
				s: &speculationSession{spec: &speculation{r: r, slotID: i, targets: cs}, drafter: nopDrafter{}}, position: &positions[i],
				current:    sample.Result{Token: mlx.FromValues([]int32{1}, 1)},
				candidates: &draftCandidates{tokens: mlx.FromValues(drafts, 1, 2), dist: r.Sampler.Distribution(i, oneHotLogits(drafts), nil)},
			}
		}
		rounds[0].grammar = g
		out := r.acceptDrafts(rounds)
		if out[0].err == nil || len(out[0].results) != 0 || positions[0] != 0 || rounds[0].s.spec.targets[0].Offset() != 1 {
			t.Fatalf("failed row: %+v position %d cache offset %d", out[0], positions[0], rounds[0].s.spec.targets[0].Offset())
		}
		if out[1].err != nil || !slices.Equal(resultIDs(out[1].results), []int32{2, 3, 4}) || positions[1] != 3 {
			t.Fatalf("healthy row: %+v position %d", out[1], positions[1])
		}
		for i, round := range rounds {
			if snapshots := round.s.spec.targets[0].TakeSnapshots(); len(snapshots) != 0 {
				for _, snapshot := range snapshots {
					if snapshot != nil {
						snapshot.Close()
					}
				}
				t.Errorf("row %d left %d snapshot slots", i, len(snapshots))
			}
		}
	})
}

type sampledSchedulingModel struct{ schedulingModel }

func (m *sampledSchedulingModel) Unembed(x *mlx.Array) *mlx.Array {
	// Keep several plausible tokens so sampling and rejection consume real RNG.
	return mlx.MulScalar(x, 0.02)
}

func TestAcceptDraftBatchPreservesSampling(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		type result struct {
			ids           []int32
			accepted, pos int
			state         []float32
		}
		run := func(seed int, shared bool) []result {
			tok := newTestTokenizer(t, []int32{7})
			m := &sampledSchedulingModel{schedulingModel: schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok, predict: map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}}}}
			r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
			defer r.Sampler.Free()
			rounds := make([]draftRound, 2)
			positions := []int{0, 0}
			for i := range rounds {
				cs := m.NewCaches()
				defer cs[0].Free()
				r.Sampler.Add(i, sample.Options{Temperature: 0.8, UseSeed: true, Seed: seed + 37*i, RepeatPenalty: 1.1, RepeatLastN: 8}, []int32{int32(i + 1)})
				drafts := []int32{int32(2 + 3*i), int32(3 + 3*i)}
				rounds[i] = draftRound{
					s: &speculationSession{spec: &speculation{r: r, slotID: i, targets: cs}, drafter: nopDrafter{}}, position: &positions[i],
					current:    sample.Result{Token: mlx.FromValues([]int32{int32(1 + 3*i)}, 1)},
					candidates: &draftCandidates{tokens: mlx.FromValues(drafts, 1, 2), dist: r.Sampler.Distribution(i, mlx.MulScalar(oneHotLogits(drafts), 0.03), nil)},
				}
			}
			var outcomes []draftOutcome
			if shared {
				outcomes = r.acceptDrafts(rounds)
			} else {
				for _, round := range rounds {
					outcomes = append(outcomes, r.acceptDrafts([]draftRound{round})...)
				}
			}
			results := make([]result, len(rounds))
			for i, out := range outcomes {
				if out.err != nil {
					t.Fatal(out.err)
				}
				results[i] = result{resultIDs(out.results), out.accepted, positions[i], rounds[i].s.spec.targets[0].State()[0].Floats()}
			}
			return results
		}
		seen := make(map[int]bool)
		for seed := range 16 {
			want, got := run(seed, false), run(seed, true)
			for i := range want {
				seen[got[i].accepted] = true
				if !slices.Equal(got[i].ids, want[i].ids) || got[i].accepted != want[i].accepted || got[i].pos != want[i].pos || !slices.Equal(got[i].state, want[i].state) {
					t.Fatalf("seed %d row %d: shared %+v, isolated %+v", seed, i, got[i], want[i])
				}
			}
		}
		if len(seen) < 2 {
			t.Fatalf("fixture did not exercise distinct acceptance lengths: %v", seen)
		}
	})
}

// A reader can stop consuming immediately after verification. The remaining row
// must own compact recurrent state even when its next forward never runs.
func TestAcceptDraftBatchMaterializesParkedState(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &recurrentSchedulingModel{fakeMTPModel{tok: tok}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		rounds := make([]draftRound, 4)
		positions := make([]int, len(rounds))
		mlx.DefaultStream().Synchronize()
		before := mlx.ActiveMemory()
		mlx.Scoped(func() {
			for i := range rounds {
				cs := m.NewCaches()
				t.Cleanup(cs[0].Free)
				r.Sampler.Add(i, sample.Options{Temperature: 0}, nil)
				drafts := []int32{1, 1}
				rounds[i] = draftRound{
					s:        &speculationSession{spec: &speculation{r: r, slotID: i, targets: cs}, drafter: nopDrafter{}},
					position: &positions[i], current: sample.Result{Token: mlx.FromValues([]int32{1}, 1)},
					candidates: &draftCandidates{tokens: mlx.FromValues(drafts, 1, len(drafts)), dist: r.Sampler.Distribution(i, oneHotLogits(drafts), nil)},
				}
			}
			for i, out := range r.acceptDrafts(rounds) {
				if out.err != nil || out.accepted != 2 || positions[i] != 3 {
					t.Fatalf("row %d: error %v accepted %d position %d", i, out.err, out.accepted, positions[i])
				}
			}
		})
		for i, round := range rounds[1:] {
			round.s.spec.targets[0].Free()
			r.Sampler.Remove(i + 1)
		}
		mlx.DefaultStream().Synchronize()
		rowBytes := rounds[0].s.spec.targets[0].State()[1].NumBytes()
		if held := mlx.ActiveMemory() - before; held >= 2*rowBytes {
			t.Fatalf("parked verifier row retained %d bytes, want one %d-byte state plus small sampling buffers", held, rowBytes)
		}
	})
}

// A drafter keeps target features between rounds. Keeping one request must not
// retain every peer's features after the shared verification has finished.
type wideDraftFeaturesModel struct{ schedulingModel }

const draftFeatureWidth = 65536

func (m *wideDraftFeaturesModel) Forward(b *batch.Batch, cs []cache.Layer) (*mlx.Array, *mlx.Array) {
	hidden, _ := m.schedulingModel.Forward(b, cs)
	rows, tokens := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	features := mlx.Arange(0, float64(rows*tokens*draftFeatureWidth), 1, mlx.DTypeFloat32).Reshape(rows, tokens, draftFeatureWidth)
	return hidden, features
}

func TestAcceptDraftBatchOwnsRetainedFeatures(t *testing.T) {
	for _, kind := range []string{"mtp", "cached-mtp", "dflash"} {
		t.Run(kind, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				tok := newTestTokenizer(t, []int32{7})
				m := &wideDraftFeaturesModel{schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}}
				r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), cache: &prefixCache{}}
				defer r.Sampler.Free()
				const rows, drafts, survivor = 4, 2, 2
				rounds := make([]draftRound, rows)
				positions := make([]int, rows)
				sessions := make([]draftSession, rows)
				mlx.DefaultStream().Synchronize()
				before := mlx.ActiveMemory()
				mlx.Scoped(func() {
					for i := range rows {
						cs := m.NewCaches()
						t.Cleanup(cs[0].Free)
						spec := &speculation{r: r, slotID: i, targets: cs}
						switch kind {
						case "mtp":
							sessions[i] = newMTPDrafter(spec).open(nil)
						case "cached-mtp":
							spec.draft = &fakeKVDraft{}
							spec.draftKV = []cache.Cache{cache.NewKVCache()}
							t.Cleanup(spec.draftKV[0].Free)
							sessions[i] = newMTPDrafter(spec).open(nil)
						case "dflash":
							draft := &fakeBlockDraft{blockSize: 4, maskToken: 6}
							spec.draft = draft
							spec.draftKV = draft.NewCaches()
							t.Cleanup(spec.draftKV[0].Free)
							sessions[i] = newDFlashDrafter(spec, draft).open(nil)
						}
						t.Cleanup(sessions[i].close)
						r.Sampler.Add(i, sample.Options{Temperature: 0}, nil)
						ids := []int32{1, 1}
						rounds[i] = draftRound{
							s: &speculationSession{spec: spec, drafter: sessions[i]}, position: &positions[i],
							current:    sample.Result{Token: mlx.FromValues([]int32{1}, 1)},
							candidates: &draftCandidates{tokens: mlx.FromValues(ids, 1, drafts), dist: r.Sampler.Distribution(i, oneHotLogits(ids), nil)},
						}
					}
					for i, out := range r.acceptDrafts(rounds) {
						if out.err != nil || out.accepted != drafts || positions[i] != drafts+1 {
							t.Fatalf("row %d: error %v accepted %d position %d", i, out.err, out.accepted, positions[i])
						}
					}
				})
				for i := range rows {
					if i == survivor {
						continue
					}
					sessions[i].close()
					rounds[i].s.spec.targets[0].Free()
					for _, c := range rounds[i].s.spec.draftKV {
						c.Free()
					}
					r.Sampler.Remove(i)
				}
				mlx.DefaultStream().Synchronize()
				// Allow one complete committed row plus small sampler/cache allocations;
				// the retained frontier alone is smaller still.
				rowBytes := (drafts + 1) * draftFeatureWidth * 4
				if held := mlx.ActiveMemory() - before; held >= 2*rowBytes {
					t.Fatalf("one drafter retained %d bytes, want less than two %d-byte feature rows", held, rowBytes)
				}
				// Inspect values only after measuring retention: reading them would force
				// a lazy copy and conceal a missing materialization in acceptDrafts.
				var held *mlx.Array
				var token int
				switch session := sessions[survivor].(type) {
				case *mtpDraftSession:
					held, token = session.frontierHidden, drafts
				case *dflashDraftSession:
					held = session.pendingFeatures[0]
				}
				mlx.Eval(held)
				if got, want := held.Floats()[0], float32((survivor*(drafts+1)+token)*draftFeatureWidth); got != want {
					t.Fatalf("row %d retained feature %v, want %v", survivor, got, want)
				}
			})
		})
	}
}
