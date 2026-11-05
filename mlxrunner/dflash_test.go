package mlxrunner

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
)

// fakeBlockDraft is a block-diffusion draft: one Draft call ingests context
// feature rows and fills a block's mask positions in parallel. It feeds the
// context rows' hot indices and then the block's token ids to its cache
// (advancing the offset like the real model's single write), records each
// call, and scripts block row i as the i-th successor of the anchor under
// predict.
type fakeBlockDraft struct {
	predict     map[int32]int32
	blockSize   int
	maskToken   int32
	draftCaches []cache.Cache
	calls       []blockCall
}

// blockCall is one recorded Draft call: the absolute slot of the first
// context row, the hot index of each context feature row (nil for a
// block-only call), and the block's token ids (nil for a context-only call).
type blockCall struct {
	offset int32
	ctx    []int32
	block  []int32
}

func (d *fakeBlockDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func (d *fakeBlockDraft) NewCaches() []cache.Cache { return d.draftCaches }

func (d *fakeBlockDraft) BlockParams() (int, int32) { return d.blockSize, d.maskToken }

func (d *fakeBlockDraft) Forward(b *batch.Batch, _, draftCaches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	call := blockCall{offset: b.SeqOffsets[0]}

	if b.Hidden != nil {
		mlx.Eval(b.Hidden)
		call.ctx = make([]int32, b.Hidden.Dim(1))
		flat := b.Hidden.Floats()
		for r := range call.ctx {
			call.ctx[r] = -1
			for v := range mtpTestVocab {
				if flat[r*mtpTestVocab+v] != 0 {
					call.ctx[r] = int32(v)
					break
				}
			}
		}
	}
	if b.InputIDs != nil {
		mlx.Eval(b.InputIDs)
		call.block = b.InputIDs.Ints()
	}
	d.calls = append(d.calls, call)

	if rc, ok := draftCaches[0].(*fakeRewindableCache); ok {
		rc.feed(call.ctx)
		rc.feed(call.block)
	}

	if call.block == nil {
		return nil, nil
	}
	// Row i predicts the token at its own position: the anchor row restates
	// the anchor, mask row i the i-th successor of the anchor.
	preds := make([]int32, len(call.block))
	preds[0] = call.block[0]
	for i := 1; i < len(preds); i++ {
		preds[i] = d.predict[preds[i-1]]
	}
	h := oneHotLogits(preds)
	return h, h
}

// Unembed is the identity: the fake's hidden already is its one-hot logits.
func (d *fakeBlockDraft) Unembed(x *mlx.Array) *mlx.Array { return x }

var _ model.BlockDraft = (*fakeBlockDraft)(nil)

// newBlockTestSession wires a runner around a fakeBlockDraft and opens one
// request's drafting session, returning the concrete session for
// internal-state assertions.
func newBlockTestSession(t *mlxtest.T, predict map[int32]int32, blockSize int) (*Runner, *fakeBlockDraft, *dflashDraftSession, []cache.Cache) {
	t.Helper()
	r := mtpTestRunner(t, predict, []int32{7}, sampler.Options{})
	caches, _ := newMTPTestCaches(2) // caches[0] target, caches[1] draft context
	draft := &fakeBlockDraft{predict: predict, blockSize: blockSize, maskToken: 6, draftCaches: caches[1:]}
	r.cache.caches = caches
	r.spec = newSpeculation(r, draft, caches[:1], caches[1:])
	return r, draft, r.spec.drafter.open(nil).(*dflashDraftSession), caches
}

// draftTokensOf reads the draft cache's fed token stream.
func draftTokensOf(caches []cache.Cache) []int32 {
	return caches[1].(*fakeRewindableCache).tokens
}

func TestDFlashCommittedBuffersPastFlushCap(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		_, draft, session, caches := newBlockTestSession(t, nil, 4)

		// One prefill-sized run at the flush cap writes through immediately in a
		// single context-only Draft call.
		n := dflashPendingFlushTokens
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32(i % mtpTestVocab)
		}
		session.committed(mlx.FromValues(ids, 1, n), oneHotLogits(ids), 0, nil)
		if got := len(draft.calls); got != 1 {
			t.Fatalf("draft calls after cap-sized run = %d, want 1", got)
		}
		if got := caches[1].Offset(); got != n {
			t.Fatalf("draft cache offset = %d, want %d", got, n)
		}

		// A run below the cap only buffers; settle writes it through, skipping
		// the leading rows the flush already covered.
		tail := []int32{1, 2, 3}
		session.committed(mlx.FromValues(tail, 1, 3), oneHotLogits(tail), n-1, nil)
		if got := len(draft.calls); got != 1 {
			t.Fatalf("draft calls after buffered run = %d, want 1 (buffered)", got)
		}
		session.settle(nil)
		want := blockCall{offset: int32(n), ctx: []int32{2, 3}}
		if got := draft.calls[1]; got.offset != want.offset || !slices.Equal(got.ctx, want.ctx) || got.block != nil {
			t.Fatalf("settle flush = %+v, want %+v", got, want)
		}
		if got := caches[1].Offset(); got != n+2 {
			t.Fatalf("draft cache offset = %d, want %d (level with reports)", got, n+2)
		}
	})
}

func TestDFlashCommittedGapPanics(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		_, _, session, _ := newBlockTestSession(t, nil, 4)

		session.committed(mlx.FromValues([]int32{1}, 1, 1), oneHotLogits([]int32{1}), 0, nil)
		defer func() {
			if recover() == nil {
				t.Fatalf("committed run past the frontier did not panic")
			}
		}()
		// The frontier is at slot 1; a run starting at 3 leaves slot 1..2 unfed.
		session.committed(mlx.FromValues([]int32{4}, 1, 1), oneHotLogits([]int32{4}), 3, nil)
	})
}

func TestDFlashRestoredPrefixResumes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		r := mtpTestRunner(t, nil, []int32{7}, sampler.Options{})
		caches, _ := newMTPTestCaches(2)
		draft := &fakeBlockDraft{blockSize: 4, maskToken: 6, draftCaches: caches[1:]}
		r.cache.caches = caches
		r.spec = newSpeculation(r, draft, caches[:1], caches[1:])

		// A restored prefix arrives with the draft caches already written.
		restored := []int32{1, 2, 3, 4, 5}
		caches[1].(*fakeRewindableCache).feed(restored)
		session := r.spec.drafter.open(nil).(*dflashDraftSession)
		if session.ctxOffset != len(restored) {
			t.Fatalf("ctxOffset = %d, want %d (synced to restored offset)", session.ctxOffset, len(restored))
		}

		// The resumed prefill's run overlaps the restore point; only the rows
		// past the frontier are buffered and written.
		run := []int32{2, 3, 0, 1}
		session.committed(mlx.FromValues(run, 1, 4), oneHotLogits(run), 3, nil)
		session.settle(nil)
		want := blockCall{offset: 5, ctx: []int32{0, 1}}
		if got := draft.calls[0]; got.offset != want.offset || !slices.Equal(got.ctx, want.ctx) || got.block != nil {
			t.Fatalf("resume flush = %+v, want %+v", got, want)
		}
		if got, wantTok := draftTokensOf(caches), append(restored, 0, 1); !slices.Equal(got, wantTok) {
			t.Fatalf("draft cache = %v, want %v", got, wantTok)
		}
	})
}

func TestDFlashProposeBounds(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5}
		_, draft, session, _ := newBlockTestSession(t, predict, 4)
		current := mlx.FromValues([]int32{1}, 1)

		// Nothing committed yet: no context to draft from.
		if session.propose(current, 4) != nil {
			t.Fatalf("propose with no context did not decline")
		}
		session.committed(mlx.FromValues([]int32{1}, 1, 1), oneHotLogits([]int32{1}), 0, nil)
		if session.propose(current, 0) != nil {
			t.Fatalf("propose with no budget did not decline")
		}

		// The block caps the draft at blockSize-1 mask rows regardless of budget.
		cand := session.propose(current, 10)
		if cand == nil {
			t.Fatalf("propose declined with context and budget")
		}
		mlx.Eval(cand.tokens)
		if got := cand.tokens.Ints(); !slices.Equal(got, []int32{2, 3, 4}) {
			t.Fatalf("draft tokens = %v, want [2 3 4]", got)
		}
		if got, want := draft.calls[0].block, []int32{1, 6, 6, 6}; !slices.Equal(got, want) {
			t.Fatalf("block = %v, want %v (anchor plus blockSize-1 masks)", got, want)
		}
	})
}

func TestDFlashBlockRewoundBeforeContextWrites(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		predict := map[int32]int32{2: 3, 3: 4, 4: 5}
		_, draft, session, caches := newBlockTestSession(t, predict, 4)

		session.committed(mlx.FromValues([]int32{1}, 1, 1), oneHotLogits([]int32{1}), 0, nil)
		if session.propose(mlx.FromValues([]int32{2}, 1), 3) == nil {
			t.Fatalf("propose declined")
		}
		// The proposal's block sits in the caches until the next write.
		if got, want := draftTokensOf(caches), []int32{1, 2, 6, 6, 6}; !slices.Equal(got, want) {
			t.Fatalf("draft cache after propose = %v, want %v", got, want)
		}

		// The next round's report rewinds the block before appending context, so
		// the accepted tokens' rows land at their true slots.
		run := []int32{2, 3, 4}
		session.committed(mlx.FromValues(run, 1, 3), oneHotLogits(run), 1, nil)
		session.settle(nil)
		if got, want := draftTokensOf(caches), []int32{1, 2, 3, 4}; !slices.Equal(got, want) {
			t.Fatalf("draft cache after settle = %v, want %v (block rewound)", got, want)
		}
		if got := caches[1].Offset(); got != 4 {
			t.Fatalf("draft cache offset = %d, want 4 (level with reports)", got)
		}
		want := blockCall{offset: 1, ctx: []int32{2, 3, 4}}
		if got := draft.calls[1]; got.offset != want.offset || !slices.Equal(got.ctx, want.ctx) || got.block != nil {
			t.Fatalf("context flush = %+v, want %+v", got, want)
		}
	})
}

func TestDFlashCloseDrainsOutstandingBlock(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		predict := map[int32]int32{2: 3, 3: 4, 4: 5}
		_, _, session, caches := newBlockTestSession(t, predict, 4)

		session.committed(mlx.FromValues([]int32{1}, 1, 1), oneHotLogits([]int32{1}), 0, nil)
		if session.propose(mlx.FromValues([]int32{2}, 1), 3) == nil {
			t.Fatalf("propose declined")
		}
		// A session that ends with a proposal in flight still leaves the caches
		// level: close rewinds the block even with nothing pending to flush.
		session.close()
		if got, want := draftTokensOf(caches), []int32{1}; !slices.Equal(got, want) {
			t.Fatalf("draft cache after close = %v, want %v", got, want)
		}
	})
}

func TestDecodeBlockDraft(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// The block draft mirrors the target chain, so one proposal round accepts
		// every draft and the bonus token is the EOS.
		const eos int32 = 7
		predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: eos, eos: 0}
		r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
		caches, _ := newMTPTestCaches(2)
		draft := &fakeBlockDraft{predict: predict, blockSize: 3, maskToken: 6, draftCaches: caches[1:]}
		r.cache.caches = caches
		r.spec = newSpeculation(r, draft, caches[:1], caches[1:])
		session, ch := newMTPTestSession(caches)

		req := Request{
			Responses:         ch,
			Tokens:            []int32{1},
			CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
			SamplerOpts:       sampler.Options{},
		}
		spec := r.spec.open(req, nil)
		if spec == nil || !spec.enabled {
			t.Fatalf("open rejected a block-draft request")
		}
		pinDraftLimit(spec, 4)
		d := spec.decoder(mlx.FromValues([]int32{1}, 1), 0, nil)
		if err := r.decode(context.Background(), req, session, d, 0); err != nil {
			t.Fatalf("decode: %v", err)
		}
		d.close()
		spec.close()

		content, final := collectResponses(ch)
		if content != "2345" {
			t.Fatalf("content = %q, want %q", content, "2345")
		}
		if !final.Done || final.DoneReason != 0 {
			t.Fatalf("final = %+v, want Done with EOS reason", final)
		}
		if want := []int32{2, 3, 4, 5, eos}; !slices.Equal(session.outputs, want) {
			t.Fatalf("session outputs = %v, want %v", session.outputs, want)
		}

		// The unprimed drafter parks the first call, so two tokens decode as
		// pipelined plain forwards; the resumed round then validates the current
		// token and blockSize-1 drafts in one fused forward.
		wantForwards := []forwardCall{{offset: 0, n: 1}, {offset: 1, n: 1}, {offset: 2, n: 3}}
		model := r.Model.(*fakeMTPModel)
		if !slices.Equal(model.forwards, wantForwards) {
			t.Fatalf("target forwards = %v, want %v", model.forwards, wantForwards)
		}

		// Ending the parked stretch settles the buffered context through, so the
		// proposal runs block-only; close's flush then writes the accepted rows
		// after rewinding the block.
		wantCalls := []blockCall{
			{offset: 0, ctx: []int32{2, 3}},
			{offset: 2, block: []int32{3, 6, 6}},
			{offset: 2, ctx: []int32{4, 5, 7}},
		}
		if len(draft.calls) != len(wantCalls) {
			t.Fatalf("draft calls = %+v, want %+v", draft.calls, wantCalls)
		}
		for i, want := range wantCalls {
			got := draft.calls[i]
			if got.offset != want.offset || !slices.Equal(got.ctx, want.ctx) || !slices.Equal(got.block, want.block) {
				t.Fatalf("draft call %d = %+v, want %+v", i, got, want)
			}
		}

		// The draft caches end level with the target, holding only context rows.
		if got, want := caches[1].Offset(), caches[0].Offset(); got != want {
			t.Fatalf("draft cache offset = %d, want %d (level with target)", got, want)
		}
		if toks := draftTokensOf(caches); slices.Contains(toks, 6) {
			t.Fatalf("draft cache retains block rows: %v", toks)
		}
	})
}

// This test model's learned score favors token 0 rather than the unary winner.
// The real sampler callback must supply both the choice and its probability.
type selectedBlockDraft struct{ *fakeBlockDraft }

func (d *selectedBlockDraft) Select(_ *mlx.Array, logits *mlx.Array, _ *mlx.Array, choose func(ids, scores *mlx.Array) (tokens, probs *mlx.Array)) (tokens, ids, probs *mlx.Array) {
	n := logits.Dim(1)
	unary := logits.Argmax(-1, false).AsType(mlx.DTypeInt32).Reshape(n, 1)
	ids = unary.Concatenate(1, mlx.Zeros(mlx.DTypeInt32, n, 1))
	scores := mlx.BroadcastTo(mlx.FromValues([]float32{0, 2}, 1, 2), int32(n), 2)
	tokens, probs = choose(ids, scores)
	return tokens.Reshape(1, n), ids.Reshape(1, n, 2), probs.Reshape(1, n, 2)
}

func TestBlockSelectorProposalAndRejection(t *testing.T) {
	for _, temperature := range []float32{0, 1} {
		mlxtest.RunSubtest(t, fmt.Sprintf("temperature %g", temperature), func(t *mlxtest.T) {
			r, draft, session, caches := newBlockTestSession(t, map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 0: 2}, 4)
			defer session.close()
			r.spec.draft = &selectedBlockDraft{draft}
			r.Sampler.Remove(pipelineSlot)
			r.Sampler.Add(pipelineSlot, sampler.Options{Temperature: temperature, UseSeed: true, Seed: 42}, nil)
			session.committed(mlx.FromValues([]int32{1}, 1, 1), oneHotLogits([]int32{1}), 0, nil)
			candidates := session.propose(mlx.FromValues([]int32{1}, 1), 3)
			mlx.Eval(candidates.Arrays()...)
			if candidates.dist.IDs == nil || !slices.Equal(candidates.dist.IDs.Ints(), []int32{2, 0, 3, 0, 4, 0}) {
				t.Fatalf("selector support = %v", candidates.dist.IDs)
			}
			want := float32(0)
			if temperature > 0 {
				want = float32(1 / (1 + math.Exp(2)))
			}
			// Broadcast inputs can leave zero-stride output rows; materialize for the host assertion.
			probs := mlx.Contiguous(candidates.dist.Probs, false)
			mlx.Eval(probs)
			for i, q := range probs.Floats() {
				expected := want
				if i%2 == 1 {
					expected = 1 - want
				}
				if math.Abs(float64(q-expected)) > 1e-6 {
					t.Fatalf("proposal probability %d = %g want %g", i, q, expected)
				}
			}
			if temperature == 0 && !slices.Equal(candidates.tokens.Ints(), []int32{0, 0, 0}) {
				t.Fatalf("selected tokens = %v want [0 0 0]", candidates.tokens.Ints())
			}
			// At temperature 1 a dense target verifies the sparse proposal. Token 2
			// is both the only acceptable draft and the residual, so any draw emits it first.
			position := 0
			spec := testSpeculationSession(r, caches[:1])
			results, accepted, _, err := spec.accept(&position, sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}, candidates, nil)
			if err != nil {
				t.Fatal(err)
			}
			ids := resultIDs(results)
			if len(ids) == 0 || ids[0] != 2 || (temperature == 0 && (accepted != 0 || len(ids) != 1)) {
				t.Fatalf("accepted %d results %v; want target token 2 first", accepted, ids)
			}
		})
	}
}

// A selector draft opens at the fixed DFlash2 depth and keeps it whatever a
// round accepts, without training the depth controller; a plain block draft
// still takes the controller's depth.
func TestSelectorDraftKeepsFixedDepth(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		r, draft, session, _ := newBlockTestSession(t, map[int32]int32{1: 2}, 8)
		session.close()
		req := Request{Tokens: []int32{1, 2, 3}}

		plain := r.spec.open(req, nil)
		if plain.fixedDepth || plain.limit != r.spec.depth.scheduled {
			t.Fatalf("block draft: fixedDepth %v limit %d, want the controller's %d", plain.fixedDepth, plain.limit, r.spec.depth.scheduled)
		}
		plain.close()

		r.spec.draft = &selectedBlockDraft{draft}
		s := r.spec.open(req, nil)
		defer s.close()
		if !s.fixedDepth || s.limit != dflash2Depth {
			t.Fatalf("selector draft: fixedDepth %v limit %d, want %d", s.fixedDepth, s.limit, dflash2Depth)
		}
		scheduled := r.spec.depth.scheduled
		for _, accepted := range []int{0, 3, dflash2Depth} {
			s.beginRound()
			s.endRound(dflash2Depth, accepted, dflash2Depth)
			if s.limit != dflash2Depth {
				t.Fatalf("limit %d after accepting %d, want %d", s.limit, accepted, dflash2Depth)
			}
		}
		if slices.ContainsFunc(r.spec.depth.acc.seen, func(n int) bool { return n != 0 }) || r.spec.depth.scheduled != scheduled {
			t.Fatalf("fixed-depth rounds trained the depth controller: seen %v, scheduled %d -> %d", r.spec.depth.acc.seen, scheduled, r.spec.depth.scheduled)
		}
	})
}
