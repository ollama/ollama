package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

// skipIfNoMLX skips a test that exercises native MLX when the dynamic library
// is unavailable, as on CI runners without an MLX build.
func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// The MTP fakes make hidden state and logits the same tensor (Forward returns
// one-hot logits, Unembed is the identity), so tests fully script target and
// draft predictions.

const mtpTestVocab = 8

// oneHotLogits builds logits with a large value on each listed token id.
func oneHotLogits(tokens []int32) *mlx.Array {
	data := make([]float32, len(tokens)*mtpTestVocab)
	for i, tok := range tokens {
		data[i*mtpTestVocab+int(tok)] = 30
	}
	return mlx.FromValues(data, 1, len(tokens), mtpTestVocab)
}

// fakeMTPModel is a target whose next-token prediction is a fixed function of
// the input token; Forward also feeds ids to the caches so offsets advance.
type fakeMTPModel struct {
	predict map[int32]int32
	tok     *tokenizer.Tokenizer
	// forwards records each Forward call so tests can assert contiguous writes.
	forwards []forwardCall
}

type forwardCall struct {
	offset int32
	n      int32
}

func (m *fakeMTPModel) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	mlx.Eval(b.InputIDs)
	ids := b.InputIDs.Ints()
	m.forwards = append(m.forwards, forwardCall{offset: b.SeqOffsets[0], n: int32(len(ids))})
	for _, c := range caches {
		if rc, ok := c.(*fakeRewindableCache); ok {
			seg := make([]int32, len(ids))
			for i, id := range ids {
				seg[i] = int32(id)
			}
			rc.feed(seg)
		}
	}

	preds := make([]int32, len(ids))
	for i, id := range ids {
		preds[i] = m.predict[int32(id)]
	}
	return oneHotLogits(preds)
}

func (m *fakeMTPModel) Unembed(x *mlx.Array) *mlx.Array { return x }
func (m *fakeMTPModel) NumLayers() int                  { return 1 }
func (m *fakeMTPModel) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *fakeMTPModel) MaxContextLength() int           { return 4096 }
func (m *fakeMTPModel) LoadWeights(map[string]*mlx.Array) error {
	return nil
}

// TokenEmbeddings returns a width-1 embedding holding the token id as a float,
// so a draft can recover which token it is extending from inputs[...,0].
func (m *fakeMTPModel) TokenEmbeddings(inputIDs *mlx.Array) *mlx.Array {
	mlx.Eval(inputIDs)
	ids := inputIDs.Ints()
	data := make([]float32, len(ids))
	for i, id := range ids {
		data[i] = float32(id)
	}
	return mlx.FromValues(data, inputIDs.Dim(0), inputIDs.Dim(1), 1)
}

var (
	_ base.Model             = (*fakeMTPModel)(nil)
	_ base.MTPEmbeddingModel = (*fakeMTPModel)(nil)
)

// fakeMTPDraft extends the token in inputEmbeds through predict; a map (not a
// step counter) keeps drafting consistent regardless of batching.
type fakeMTPDraft struct {
	predict map[int32]int32
	// calls records each Draft call so tests can assert the position convention.
	calls []draftCall
}

type draftCall struct {
	position int32
	from     int32
}

func (d *fakeMTPDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func (d *fakeMTPDraft) Draft(inputEmbeds *mlx.Array, position int32, caches []cache.Cache) (logits, hidden *mlx.Array) {
	mlx.Eval(inputEmbeds)
	prev := int32(inputEmbeds.Floats()[0])
	d.calls = append(d.calls, draftCall{position: position, from: prev})
	return oneHotLogits([]int32{d.predict[prev]}), mlx.Zeros(mlx.DTypeFloat32, 1, 1, mtpTestVocab)
}

var (
	_ base.DraftModel    = (*fakeMTPDraft)(nil)
	_ base.MTPDraftModel = (*fakeMTPDraft)(nil)
)

// newTestTokenizer builds a byte-level BPE tokenizer over single-character
// tokens "0".."7" with the given EOS ids, so Decode(id) yields that digit and
// IsEOS reports membership.
func newTestTokenizer(t *testing.T, eos []int32) *tokenizer.Tokenizer {
	t.Helper()
	vocab := make(map[string]int32, mtpTestVocab)
	for i := range mtpTestVocab {
		vocab[fmt.Sprintf("%d", i)] = int32(i)
	}
	model := map[string]any{
		"type":   "BPE",
		"vocab":  vocab,
		"merges": []string{},
	}
	data, err := json.Marshal(map[string]any{"model": model})
	if err != nil {
		t.Fatalf("marshal tokenizer: %v", err)
	}
	genConfig, err := json.Marshal(map[string]any{"eos_token_id": eos})
	if err != nil {
		t.Fatalf("marshal generation config: %v", err)
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(data, &tokenizer.TokenizerConfig{GenerationConfigJSON: genConfig})
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	return tok
}

// mtpTestRunner wires a Runner with the MTP fakes and a real sampler
// registered with opts.
func mtpTestRunner(t *testing.T, predict map[int32]int32, eos []int32, opts sampler.Options) *Runner {
	t.Helper()
	tok := newTestTokenizer(t, eos)
	r := &Runner{
		Model:     &fakeMTPModel{predict: predict, tok: tok},
		Tokenizer: tok,
		Sampler:   sampler.New(4096),
	}
	r.Sampler.Add(pipelineSlot, opts, nil)
	t.Cleanup(func() { r.Sampler.Remove(pipelineSlot) })
	return r
}

// collectResponses drains a buffered response channel into the concatenated
// streamed content and the captured terminal response.
func collectResponses(ch chan CompletionResponse) (content string, final CompletionResponse) {
	var b strings.Builder
	for {
		select {
		case resp := <-ch:
			if resp.Done {
				return b.String(), resp
			}
			b.WriteString(resp.Content)
		default:
			return b.String(), final
		}
	}
}

// resultIDs reads the token id of each result.
func resultIDs(results []sampler.Result) []int {
	ids := make([]int, 0, len(results))
	for _, res := range results {
		ids = append(ids, res.Token.Int())
	}
	return ids
}

func TestAcceptMTPDraftsGreedyAcceptAll(t *testing.T) {
	skipIfNoMLX(t)
	// Target predicts 1->2->3->4 along the accepted chain; the draft proposed
	// exactly that, so every draft token is accepted and the bonus token is the
	// target's prediction after the last accepted token.
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5}
	r := mtpTestRunner(t, predict, nil, sampler.Options{})

	caches, _ := newMTPTestCaches(1)
	candidates := scriptedCandidates(r, []int32{2, 3, 4})
	baseLogits := oneHotLogits([]int32{2}).Squeeze(1) // target prediction after the seed token 1

	position := caches[0].Offset()

	spec := testSpeculationSession(r, caches)
	current := sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}
	results, accepted, err := spec.accept(&position, current, oneHotLogits([]int32{2}), baseLogits, candidates)
	if err != nil {
		t.Fatalf("accept: %v", err)
	}
	if accepted != 3 {
		t.Fatalf("accepted = %d, want 3", accepted)
	}
	// The run is the accepted drafts followed by the bonus token (5).
	if got := resultIDs(results); !slices.Equal(got, []int{2, 3, 4, 5}) {
		t.Fatalf("results = %v, want [2 3 4 5]", got)
	}
	if position != 3 {
		t.Fatalf("position = %d, want 3", position)
	}
	if got := caches[0].Offset(); got != 3 {
		t.Fatalf("cache offset = %d, want 3 (all drafts kept)", got)
	}
}

func TestAcceptMTPDraftsGreedyMismatch(t *testing.T) {
	skipIfNoMLX(t)
	// Target predicts 1->2->9 but the draft proposed 2 then 7: the second draft
	// token mismatches, so only the first is accepted and the bonus is the
	// target's own prediction (3) at the rejection point.
	predict := map[int32]int32{1: 2, 2: 3, 7: 0}
	r := mtpTestRunner(t, predict, nil, sampler.Options{})

	caches, _ := newMTPTestCaches(1)
	candidates := scriptedCandidates(r, []int32{2, 7})
	baseLogits := oneHotLogits([]int32{2}).Squeeze(1)

	position := caches[0].Offset()

	spec := testSpeculationSession(r, caches)
	current := sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}
	results, accepted, err := spec.accept(&position, current, oneHotLogits([]int32{2}), baseLogits, candidates)
	if err != nil {
		t.Fatalf("accept: %v", err)
	}
	if accepted != 1 {
		t.Fatalf("accepted = %d, want 1", accepted)
	}
	// The run is the one accepted draft (2) followed by the target's own
	// prediction at the rejection point (3).
	if got := resultIDs(results); !slices.Equal(got, []int{2, 3}) {
		t.Fatalf("results = %v, want [2 3]", got)
	}
	if position != 1 {
		t.Fatalf("position = %d, want 1", position)
	}
	if got := caches[0].Offset(); got != 1 {
		t.Fatalf("cache offset = %d, want 1 (rolled back to accepted)", got)
	}
}

func TestAcceptMTPDraftsGreedyEOS(t *testing.T) {
	skipIfNoMLX(t)
	// The second accepted draft token is EOS: it is recorded but stops
	// generation and no bonus token is produced. Both accepted tokens are
	// committed, so the caches hold the EOS's KV.
	const eos int32 = 6
	predict := map[int32]int32{1: 2, 2: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})

	caches, _ := newMTPTestCaches(1)
	candidates := scriptedCandidates(r, []int32{2, eos})
	baseLogits := oneHotLogits([]int32{2}).Squeeze(1)

	position := caches[0].Offset()

	spec := testSpeculationSession(r, caches)
	current := sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}
	results, accepted, err := spec.accept(&position, current, oneHotLogits([]int32{2}), baseLogits, candidates)
	if err != nil {
		t.Fatalf("accept: %v", err)
	}
	if accepted != 2 {
		t.Fatalf("accepted = %d, want 2 (token + EOS)", accepted)
	}
	// The EOS ends generation, so the run is exactly the accepted tokens
	// with no bonus appended.
	if got := resultIDs(results); !slices.Equal(got, []int{2, int(eos)}) {
		t.Fatalf("results = %v, want [2 %d]", got, eos)
	}
	if len(results) != accepted {
		t.Fatalf("results has %d tokens, want exactly the %d accepted with no bonus after EOS", len(results), accepted)
	}
	if position != 2 {
		t.Fatalf("position = %d, want 2 (both accepted tokens committed)", position)
	}
	if got := caches[0].Offset(); got != 2 {
		t.Fatalf("cache offset = %d, want 2 (both accepted tokens committed)", got)
	}
}

func TestRunMTPDecodeGreedy(t *testing.T) {
	skipIfNoMLX(t)
	// The seed token 1 is the last prefill token; its prediction (2) is the
	// first generated token. The decode then walks 2->3->4->EOS. The draft
	// proposes the correct chain so steps are accepted in a single forward.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
	// The draft mirrors the target chain, so every drafted token is accepted.
	draft := &fakeMTPDraft{predict: predict}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(1)
	session, ch := newMTPTestSession(caches)
	position := 1 // one prefill token already processed

	req := Request{
		Responses:         ch,
		Tokens:            []int32{0},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	d := testDecoder(r, req, caches, []int32{1}, position)
	if err := r.decode(context.Background(), req, session, d, 0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	d.close()

	content, final := collectResponses(ch)
	if content != "234" {
		t.Fatalf("content = %q, want %q", content, "234")
	}
	if !final.Done {
		t.Fatalf("final response not marked Done")
	}
	if final.DoneReason != 0 {
		t.Fatalf("DoneReason = %d, want 0 (EOS)", final.DoneReason)
	}
	if got := []int32{2, 3, 4, eos}; !slices.Equal(session.outputs, got) {
		t.Fatalf("session outputs = %v, want %v", session.outputs, got)
	}

	// The target writes the caches contiguously: the seed token at offset 1,
	// the round's current token at offset 2, then the 4-token validation
	// forward at offset 3.
	wantForwards := []forwardCall{{offset: 1, n: 1}, {offset: 2, n: 1}, {offset: 3, n: 4}}
	model := r.Model.(*fakeMTPModel)
	if !slices.Equal(model.forwards, wantForwards) {
		t.Fatalf("target forwards = %v, want %v", model.forwards, wantForwards)
	}

	// Single-position drafting anchors every Draft call in a round at the
	// last committed position (offset 1 — the current token at offset 2 is
	// not yet validated), while the extended token advances along the
	// proposed chain.
	wantDraft := []draftCall{{1, 2}, {1, 3}, {1, 4}, {1, eos}}
	if !slices.Equal(draft.calls, wantDraft) {
		t.Fatalf("draft calls = %v, want %v", draft.calls, wantDraft)
	}
}

func TestRunMTPDecodeSampled(t *testing.T) {
	skipIfNoMLX(t)
	// The same chain at temperature 1: because oneHotLogits uses a large gap,
	// the proposal and target distributions are effectively point masses, so the
	// rejection-sampling accept path that the sampled and greedy paths now share
	// accepts the mirrored draft chain deterministically.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{Temperature: 1, Seed: 42, UseSeed: true})
	r.spec = newSpeculation(r, &fakeMTPDraft{predict: predict})

	caches, _ := newMTPTestCaches(1)
	session, ch := newMTPTestSession(caches)
	position := 1

	req := Request{
		Responses:         ch,
		Tokens:            []int32{0},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{Temperature: 1, Seed: 42, UseSeed: true},
	}
	d := testDecoder(r, req, caches, []int32{1}, position)
	if err := r.decode(context.Background(), req, session, d, 0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	d.close()

	content, final := collectResponses(ch)
	if content != "234" {
		t.Fatalf("content = %q, want %q", content, "234")
	}
	if final.DoneReason != 0 {
		t.Fatalf("DoneReason = %d, want 0 (EOS)", final.DoneReason)
	}
	if got := []int32{2, 3, 4, eos}; !slices.Equal(session.outputs, got) {
		t.Fatalf("session outputs = %v, want %v", session.outputs, got)
	}
}

func TestDecodePlain(t *testing.T) {
	skipIfNoMLX(t)
	// The same chain with no speculationSession: decode's pipelined loop runs,
	// dispatching the forward that produces the next token before the
	// current one is emitted.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})

	caches, _ := newMTPTestCaches(1)
	session, ch := newMTPTestSession(caches)
	position := 1

	req := Request{
		Responses:         ch,
		Tokens:            []int32{0},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	d := testDecoder(r, req, caches, []int32{1}, position)
	if err := r.decode(context.Background(), req, session, d, 0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	d.close()

	content, final := collectResponses(ch)
	if content != "234" {
		t.Fatalf("content = %q, want %q", content, "234")
	}
	if final.DoneReason != 0 || final.EvalCount != 3 {
		t.Fatalf("final DoneReason = %d, EvalCount = %d, want 0 (EOS) and 3", final.DoneReason, final.EvalCount)
	}
	if got := []int32{2, 3, 4, eos}; !slices.Equal(session.outputs, got) {
		t.Fatalf("session outputs = %v, want %v", session.outputs, got)
	}

	// One forward per token: the seed at offset 1, then each sampled token
	// in turn — including the ending EOS, whose forward is already in
	// flight when the token is checked. Every forwarded token is recorded,
	// so the caches rest exactly at the recorded path.
	wantForwards := []forwardCall{{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}
	model := r.Model.(*fakeMTPModel)
	if !slices.Equal(model.forwards, wantForwards) {
		t.Fatalf("target forwards = %v, want %v", model.forwards, wantForwards)
	}
}

func TestDecodeCancelledMidStream(t *testing.T) {
	skipIfNoMLX(t)
	// Cancelling while accepted drafts stream must leave the session
	// consistent: every token committed to the caches is recorded in
	// session.outputs, no speculation snapshot schedule is left pending on
	// the shared caches, and every captured snapshot is closed.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: eos}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
	r.spec = newSpeculation(r, &fakeMTPDraft{predict: predict})

	caches, tr := newMTPTestCaches(1)
	session := &cacheSession{caches: caches}
	ch := make(chan CompletionResponse) // unbuffered: every send must rendezvous

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-ch // the first streamed token
		cancel()
		// Stop reading: the next send blocks until the emit select sees
		// the cancelled context.
	}()

	position := 0
	req := Request{
		Responses:         ch,
		Tokens:            []int32{1},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	d := testDecoder(r, req, caches, []int32{1}, position)
	err := r.decode(ctx, req, session, d, 0)
	d.close()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("decode error = %v, want context.Canceled", err)
	}

	// Every committed token is recorded, and the caches never run ahead of
	// the record: the final recorded output is the round's next token,
	// emitted before it is ever forwarded, so the caches rest one short.
	rc := caches[0].(*fakeRewindableCache)
	if want := 1 + len(session.outputs) - 1; rc.Offset() != want {
		t.Fatalf("cache offset = %d, want %d (seed + recorded outputs %v minus the unforwarded final)", rc.Offset(), want, session.outputs)
	}
	if rc.pending.offsets != nil {
		t.Fatalf("speculation snapshot schedule left pending: %v", rc.pending.offsets)
	}
	for i, s := range tr.all {
		if s.closeCount == 0 {
			t.Fatalf("snapshot #%d [%d,%d) leaked: never closed", i, s.from, s.to)
		}
	}
}

// testDecoder builds the decoder TextGenerationPipeline would construct for
// this request; tests close it explicitly so close-time effects are visible
// to assertions.
func testDecoder(r *Runner, req Request, caches []cache.Cache, seed []int32, position int) decoder {
	if spec := r.spec.open(req, caches); spec != nil {
		return spec.decoder(seed, position)
	}
	return r.pipelinedDecoder(caches, seed, position)
}

// newMTPTestCaches returns n rewindable fake caches sharing one snapshot
// tracker, matching the cache.Cache the speculation helpers drive.
func newMTPTestCaches(n int) ([]cache.Cache, *snapshotTracker) {
	tr := &snapshotTracker{}
	caches := make([]cache.Cache, n)
	for i := range caches {
		caches[i] = &fakeRewindableCache{tracker: tr}
	}
	return caches, tr
}

// newMTPTestSession wraps caches in a cacheSession with a buffered response
// channel large enough to hold a short decode run without a reader.
func newMTPTestSession(caches []cache.Cache) (*cacheSession, chan CompletionResponse) {
	ch := make(chan CompletionResponse, 256)
	return &cacheSession{caches: caches}, ch
}

// testSpeculationSession wires a speculation engine around the runner's drafter and
// caches for tests that drive accept directly. A runner without a draft
// model gets a no-op drafter, since the engine requires one.
func testSpeculationSession(r *Runner, caches []cache.Cache) *speculationSession {
	s := r.spec
	if s == nil {
		s = &speculation{r: r}
	}
	var d drafter = nopDrafter{}
	if md := newMTPDrafter(s, caches); md != nil {
		d = md
	}
	return &speculationSession{spec: s, drafter: d, caches: caches}
}

// nopDrafter satisfies drafter for engine tests that supply candidates
// directly and never propose.
type nopDrafter struct{}

func (nopDrafter) propose(*mlx.Array, int) *draftCandidates { return nil }
func (nopDrafter) committed(_, _ *mlx.Array, _ int)         {}
func (nopDrafter) close()                                   {}

// scriptedCandidates builds draft candidates by running the real drafter
// against a fake whose prediction chain, starting from seed token 0, yields
// exactly the requested tokens. Using the real drafter means the proposal
// distributions match what the engine's acceptance expects.
func scriptedCandidates(r *Runner, tokens []int32) *draftCandidates {
	chain := map[int32]int32{}
	prev := int32(0)
	for _, tok := range tokens {
		chain[prev] = tok
		prev = tok
	}
	d := &mtpDrafter{spec: &speculation{r: r}, draft: &fakeMTPDraft{predict: chain}, target: r.Model.(base.MTPEmbeddingModel)}
	d.committed(mlx.FromValues([]int32{0}, 1, 1), mlx.Zeros(mlx.DTypeFloat32, 1, 1, mtpTestVocab), 0)
	return d.propose(mlx.FromValues([]int32{0}, 1), len(tokens))
}
