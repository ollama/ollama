package mlxrunner

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

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

	session, ch := newMTPTestSession(caches)
	position := caches[0].Offset()
	final := CompletionResponse{Done: true}
	generated := 0

	req := Request{Responses: ch, CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 100}}}
	next, accepted, done, err := r.acceptMTPDrafts(context.Background(), req, session, &decoder{tokenizer: r.Tokenizer}, caches, &position, baseLogits, candidates, &final, &generated)
	if err != nil {
		t.Fatalf("acceptMTPDrafts: %v", err)
	}
	if accepted != 3 {
		t.Fatalf("accepted = %d, want 3", accepted)
	}
	if done {
		t.Fatalf("done = true, want false")
	}
	if got := tokenID(next.Token); got != 5 {
		t.Fatalf("bonus token = %d, want 5", got)
	}
	if position != 3 {
		t.Fatalf("position = %d, want 3", position)
	}
	if got := caches[0].Offset(); got != 3 {
		t.Fatalf("cache offset = %d, want 3 (all drafts kept)", got)
	}
	if generated != 3 {
		t.Fatalf("generated = %d, want 3", generated)
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

	session, ch := newMTPTestSession(caches)
	position := caches[0].Offset()
	final := CompletionResponse{Done: true}
	generated := 0

	req := Request{Responses: ch, CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 100}}}
	next, accepted, done, err := r.acceptMTPDrafts(context.Background(), req, session, &decoder{tokenizer: r.Tokenizer}, caches, &position, baseLogits, candidates, &final, &generated)
	if err != nil {
		t.Fatalf("acceptMTPDrafts: %v", err)
	}
	if accepted != 1 {
		t.Fatalf("accepted = %d, want 1", accepted)
	}
	if done {
		t.Fatalf("done = true, want false")
	}
	if got := tokenID(next.Token); got != 3 {
		t.Fatalf("bonus token = %d, want 3 (target prediction at rejection)", got)
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
	// generation, no bonus token is produced, and the cache keeps both tokens.
	const eos int32 = 6
	predict := map[int32]int32{1: 2, 2: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})

	caches, _ := newMTPTestCaches(1)
	candidates := scriptedCandidates(r, []int32{2, eos})
	baseLogits := oneHotLogits([]int32{2}).Squeeze(1)

	session, ch := newMTPTestSession(caches)
	position := caches[0].Offset()
	final := CompletionResponse{Done: true, DoneReason: 1}
	generated := 0

	req := Request{Responses: ch, CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 100}}}
	next, accepted, done, err := r.acceptMTPDrafts(context.Background(), req, session, &decoder{tokenizer: r.Tokenizer}, caches, &position, baseLogits, candidates, &final, &generated)
	if err != nil {
		t.Fatalf("acceptMTPDrafts: %v", err)
	}
	if accepted != 2 {
		t.Fatalf("accepted = %d, want 2 (token + EOS)", accepted)
	}
	if !done {
		t.Fatalf("done = false, want true")
	}
	if next.Token != nil {
		t.Fatalf("bonus token = %d, want none after EOS", tokenID(next.Token))
	}
	if final.DoneReason != 0 {
		t.Fatalf("DoneReason = %d, want 0 (EOS)", final.DoneReason)
	}
	if position != 2 {
		t.Fatalf("position = %d, want 2", position)
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
	r.Draft = draft

	caches, _ := newMTPTestCaches(1)
	session, ch := newMTPTestSession(caches)
	position := 1 // one prefill token already processed

	req := Request{
		Responses:         ch,
		Tokens:            []int32{0},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	if err := r.runMTPDecode(context.Background(), req, session, caches, []int32{1}, &position, time.Now()); err != nil {
		t.Fatalf("runMTPDecode: %v", err)
	}

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
	// last target-seen position (the current token, offset 2), while the
	// extended token advances along the proposed chain.
	wantDraft := []draftCall{{2, 2}, {2, 3}, {2, 4}, {2, eos}}
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
	r.Draft = &fakeMTPDraft{predict: predict}

	if !r.useMTP(sampler.Options{Temperature: 1}) {
		t.Fatalf("useMTP rejected a sampled request")
	}

	caches, _ := newMTPTestCaches(1)
	session, ch := newMTPTestSession(caches)
	position := 1

	req := Request{
		Responses:         ch,
		Tokens:            []int32{0},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{Temperature: 1, Seed: 42, UseSeed: true},
	}
	if err := r.runMTPDecode(context.Background(), req, session, caches, []int32{1}, &position, time.Now()); err != nil {
		t.Fatalf("runMTPDecode: %v", err)
	}

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

// scriptedCandidates builds draft candidates by running the real generator
// against a draft whose prediction chain, starting from seed token 0, yields
// exactly the requested tokens. Using the real generator means the proposal
// distributions match what acceptMTPDrafts expects.
func scriptedCandidates(r *Runner, tokens []int32) *mtpDraftCandidates {
	chain := map[int32]int32{}
	prev := int32(0)
	for _, tok := range tokens {
		chain[prev] = tok
		prev = tok
	}
	draft := &fakeMTPDraft{predict: chain}
	target := r.Model.(base.MTPEmbeddingModel)
	seed := mlx.FromValues([]int32{0}, 1, 1)
	hidden := mlx.Zeros(mlx.DTypeFloat32, 1, 1, mtpTestVocab)
	return r.generateMTPDraftCandidates(draft, target, seed, hidden, nil, 0, len(tokens))
}
