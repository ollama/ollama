package mlxrunner

import (
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
)

type fakeSharedTargetDraft struct {
	calls    []blockCall
	previous []int32
}

func (*fakeSharedTargetDraft) LoadWeights(map[string]*mlx.Array) error { return nil }
func (*fakeSharedTargetDraft) NewCaches() []cache.Cache                { return nil }
func (*fakeSharedTargetDraft) SharedTargetBlockParams() (int, int32)   { return 6, 7 }
func (*fakeSharedTargetDraft) Unembed(x *mlx.Array) *mlx.Array         { return x }

func (d *fakeSharedTargetDraft) Forward(b *batch.Batch, targetCaches, draftCaches []cache.Layer) (*mlx.Array, *mlx.Array) {
	target := targetCaches[0].(*fakeRewindableCache)
	if target.Offset() != int(b.SeqOffsets[0]) || len(draftCaches) != 0 {
		panic("shared draft must read the committed target cache without private KV")
	}
	d.calls = append(d.calls, blockCall{offset: b.SeqOffsets[0], ctx: b.Hidden.Argmax(-1, false).AsType(mlx.DTypeInt32).Ints(), block: b.InputIDs.Ints()})
	row := mlx.FromValues([]float32{0, 30, 20, 0, 0, 0, 0, 0}, 1, 1, mtpTestVocab)
	return mlx.BroadcastTo(row, 1, int32(b.InputIDs.Dim(1)), mtpTestVocab), nil
}

func (d *fakeSharedTargetDraft) CorrectHidden(row, previous *mlx.Array) *mlx.Array {
	d.previous = append(d.previous, previous.Int())
	return row
}

type correctedBlockDraft struct{ fakeBlockDraft }

func (*correctedBlockDraft) CorrectHidden(row, _ *mlx.Array) *mlx.Array { return row }

func TestDFlashBlockDraftDoesNotOptIntoSharedTarget(t *testing.T) {
	draft := &correctedBlockDraft{fakeBlockDraft{blockSize: 8}}
	spec := newSpeculation(nil, 0, draft, nil, nil)
	if _, ok := spec.drafter.(*dflashDrafter); !ok {
		t.Fatalf("drafter = %T, want *dflashDrafter", spec.drafter)
	}
	if got := spec.depth.drafterLimit; got != 7 {
		t.Fatalf("DFlash depth limit = %d, want 7", got)
	}
}

func TestSharedTargetDraftAdjustsDepthToYield(t *testing.T) {
	d := newSpeculation(nil, 0, &fakeSharedTargetDraft{}, nil, nil).depth
	if d.drafterLimit != 5 {
		t.Fatalf("controller limit = %d, want the checkpoint's limit of 5", d.drafterLimit)
	}
	for depth := 0; depth <= d.drafterLimit; depth++ {
		d.cost.observe(depth, time.Duration(46+15*depth)*time.Millisecond)
	}
	run := func(rounds int, accept bool) int {
		var drafted int
		for range rounds {
			depth := d.next()
			if depth == 0 {
				continue
			}
			drafted++
			accepted := 0
			if accept {
				accepted = depth
			}
			d.acc.observe(depth, accepted)
		}
		return drafted
	}
	if run(128, true) == 0 || d.selected() != 5 {
		t.Fatalf("high yield selected depth %d, want 5", d.selected())
	}
	if run(64, false) == 0 || d.selected() != 0 {
		t.Fatalf("low yield selected depth %d, want 0", d.selected())
	}
	if run(depthProbeIntervalMax, true) == 0 {
		t.Fatal("no recovery probe within the controller's maximum interval")
	}
	run(256, true)
	if d.selected() != 5 {
		t.Fatalf("recovered yield selected depth %d, want 5", d.selected())
	}
}

func TestSharedTargetDraftSamplesPrefixAndResumesAfterRejection(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		r := mtpTestRunner(t, map[int32]int32{5: 1, 1: 3, 2: 6}, []int32{7}, sampler.Options{
			RepeatLastN: 4, RepeatPenalty: 2,
		})
		caches, _ := newMTPTestCaches(1)
		target := caches[0].(*fakeRewindableCache)
		target.feed([]int32{4, 6})
		r.Sampler.Commit(pipelineSlot, []int32{4, 6, 5})
		draft := &fakeSharedTargetDraft{}
		r.spec = newSpeculation(r, pipelineSlot, draft, caches, draft.NewCaches())
		spec := testSpeculationSession(r, caches)
		defer spec.drafter.close()

		mlx.Scoped(func() {
			spec.drafter.committed(mlx.FromValues([]int32{4, 6}, 1, 2), oneHotLogits([]int32{4, 6}), 0, nil)
		})
		current := sampler.Result{Token: mlx.FromValues([]int32{5}, 1)}
		candidates := spec.drafter.propose(current.Token, 2)
		if got := candidates.tokens.Ints(); !slices.Equal(got, []int32{1, 2}) {
			t.Fatalf("draft = %v, want [1 2] with token 1 penalized in the second row", got)
		}
		if !slices.Equal(draft.previous, []int32{5, 1}) {
			t.Fatalf("correction predecessors = %v, want [5 1]", draft.previous)
		}
		if got := draft.calls[0]; got.offset != 2 || !slices.Equal(got.ctx, []int32{6}) || !slices.Equal(got.block, []int32{5, 7, 7}) {
			t.Fatalf("draft inputs = %+v, want offset 2, feature [6], block [5 7 7]", got)
		}

		position := 2
		results, accepted, observed, err := spec.accept(&position, current, candidates, nil)
		if err != nil || accepted != 1 || observed != 2 || !slices.Equal(resultIDs(results), []int32{1, 3}) {
			t.Fatalf("verify returned %v, accepted %d, observed %d, err %v", resultIDs(results), accepted, observed, err)
		}
		if position != 4 || !slices.Equal(target.tokens, []int32{4, 6, 5, 1}) {
			t.Fatalf("position %d cache %v, want 4 and [4 6 5 1]", position, target.tokens)
		}
		spec.drafter.propose(results[len(results)-1].Token, 1)
		if got := draft.calls[1]; got.offset != 4 || !slices.Equal(got.ctx, []int32{3}) || !slices.Equal(got.block, []int32{3, 7}) {
			t.Fatalf("resumed draft inputs = %+v, want offset 4, feature [3], block [3 7]", got)
		}
	})
}
