package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
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
	for i, c := range caches {
		if i >= m.NumLayers() {
			break
		}
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

var _ base.Model = (*fakeMTPModel)(nil)

// fakeMTPDraft is a cacheless draft that extends b.InputIDs through predict;
// a map (not a step counter) keeps drafting consistent regardless of batching.
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

func (d *fakeMTPDraft) DraftCaches([]cache.Cache) []cache.Cache { return nil }

func (d *fakeMTPDraft) Draft(b *batch.Batch, caches []cache.Cache) (hidden, projected *mlx.Array) {
	mlx.Eval(b.InputIDs)
	prev := int32(b.InputIDs.Ints()[0])
	d.calls = append(d.calls, draftCall{position: b.SeqOffsets[0], from: prev})
	return oneHotLogits([]int32{d.predict[prev]}), mlx.Zeros(mlx.DTypeFloat32, 1, 1, mtpTestVocab)
}

// Unembed is the identity: the fake's hidden already is its one-hot logits.
func (d *fakeMTPDraft) Unembed(x *mlx.Array) *mlx.Array { return x }

var _ base.DraftModel = (*fakeMTPDraft)(nil)

// fakeKVDraft is a draft head with its own KV cache: it claims the trailing
// cache slot, writes its input ids there on every Draft call (advancing the
// offset like a real KV write), and records each call's offset, ids, and
// the identity of every fused hidden row. A target hidden row is one-hot
// (its hot index identifies which position it came from); the head's own
// projected hidden is all-zero and records as -1.
type fakeKVDraft struct {
	predict map[int32]int32
	extends []extendCall
}

// extendCall is one recorded Draft call: the absolute slot of the first
// entry written, the look-ahead token ids, and the hot index of each fused
// hidden row (-1 for the head's own projected hidden).
type extendCall struct {
	offset  int32
	ids     []int32
	hiddens []int32
}

func (d *fakeKVDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func (d *fakeKVDraft) DraftCaches(caches []cache.Cache) []cache.Cache {
	return caches[len(caches)-1:]
}

func (d *fakeKVDraft) Draft(b *batch.Batch, caches []cache.Cache) (hidden, projected *mlx.Array) {
	mlx.Eval(b.InputIDs, b.Hidden)
	rawIDs := b.InputIDs.Ints()
	ids := make([]int32, len(rawIDs))
	for i, id := range rawIDs {
		ids[i] = int32(id)
	}

	hot := make([]int32, b.Hidden.Dim(1))
	flat := b.Hidden.Floats()
	for r := range hot {
		hot[r] = -1
		for v := range mtpTestVocab {
			if flat[r*mtpTestVocab+v] != 0 {
				hot[r] = int32(v)
				break
			}
		}
	}
	d.extends = append(d.extends, extendCall{offset: b.SeqOffsets[0], ids: ids, hiddens: hot})

	if rc, ok := d.DraftCaches(caches)[0].(*fakeRewindableCache); ok {
		rc.feed(ids)
	}

	preds := make([]int32, len(ids))
	for i, id := range ids {
		preds[i] = d.predict[id]
	}
	return oneHotLogits(preds), mlx.Zeros(mlx.DTypeFloat32, 1, 1, mtpTestVocab)
}

// Unembed is the identity: the fake's hidden already is its one-hot logits.
func (d *fakeKVDraft) Unembed(x *mlx.Array) *mlx.Array { return x }

var _ base.DraftModel = (*fakeKVDraft)(nil)

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

	position := caches[0].Offset()

	spec := testSpeculationSession(r, caches)
	current := sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}
	results, accepted, err := spec.accept(&position, current, candidates)
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
	if position != 4 {
		t.Fatalf("position = %d, want 4 (current + accepted)", position)
	}
	if got := caches[0].Offset(); got != 4 {
		t.Fatalf("cache offset = %d, want 4 (current + all drafts kept)", got)
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

	position := caches[0].Offset()

	spec := testSpeculationSession(r, caches)
	current := sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}
	results, accepted, err := spec.accept(&position, current, candidates)
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
	if position != 2 {
		t.Fatalf("position = %d, want 2 (current + accepted)", position)
	}
	if got := caches[0].Offset(); got != 2 {
		t.Fatalf("cache offset = %d, want 2 (rolled back to accepted)", got)
	}
}

func TestAcceptMTPDraftsGreedyEOS(t *testing.T) {
	skipIfNoMLX(t)
	// The second accepted draft token is EOS: it is recorded but stops
	// generation and no bonus token is produced. The EOS's own KV is rolled
	// back so the caches rest one token behind the recorded outputs.
	const eos int32 = 6
	predict := map[int32]int32{1: 2, 2: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})

	caches, _ := newMTPTestCaches(1)
	candidates := scriptedCandidates(r, []int32{2, eos})

	position := caches[0].Offset()

	spec := testSpeculationSession(r, caches)
	current := sampler.Result{Token: mlx.FromValues([]int32{1}, 1)}
	results, accepted, err := spec.accept(&position, current, candidates)
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
		t.Fatalf("position = %d, want 2 (current + kept draft, EOS dropped)", position)
	}
	if got := caches[0].Offset(); got != 2 {
		t.Fatalf("cache offset = %d, want 2 (one behind the recorded outputs, EOS dropped)", got)
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
	// then one fused forward validating the current token and the 4 drafts
	// together at offset 2.
	wantForwards := []forwardCall{{offset: 1, n: 1}, {offset: 2, n: 5}}
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
	spec := r.spec.open(req, caches)
	if spec == nil || !spec.enabled {
		t.Fatalf("newSpeculationSession rejected a sampled request")
	}
	d := spec.decoder([]int32{1}, position)
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

func TestRunMTPDecodeWarmDrafter(t *testing.T) {
	skipIfNoMLX(t)
	// A drafter warmed by a prefill report proposes in the very first round:
	// the last prompt token seeds the decode loop and is forwarded fused
	// with the drafts, so generation runs no plain forward at all.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
	draft := &fakeMTPDraft{predict: predict}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(1)
	session, ch := newMTPTestSession(caches)
	position := 1 // one prefill token already processed

	req := Request{
		Responses:         ch,
		Tokens:            []int32{0},
		CompletionRequest: CompletionRequest{Options: api.Options{Runner: api.Runner{DraftNumPredict: 4}, NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	spec := r.spec.open(req, caches)
	// The prefill chunk's committed report: token 0 at slot 0 with its
	// hidden row, leaving the drafter ready to propose from slot 1.
	spec.committed(mlx.FromValues([]int32{0}, 1, 1), oneHotLogits([]int32{1}), 0)

	d := spec.decoder([]int32{1}, position)
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

	// One fused forward validates the seed token and all 4 proposals
	// together at the seed token's slot.
	wantForwards := []forwardCall{{offset: 1, n: 5}}
	model := r.Model.(*fakeMTPModel)
	if !slices.Equal(model.forwards, wantForwards) {
		t.Fatalf("target forwards = %v, want %v", model.forwards, wantForwards)
	}

	// Warm drafting anchors at the last reported slot (0) and extends the
	// seed token through the proposal chain.
	wantDraft := []draftCall{{0, 1}, {0, 2}, {0, 3}, {0, 4}}
	if !slices.Equal(draft.calls, wantDraft) {
		t.Fatalf("draft calls = %v, want %v", draft.calls, wantDraft)
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
	return r.pipelinedDecoder(nil, caches, seed, position)
}

func TestDecodeKVDraft(t *testing.T) {
	skipIfNoMLX(t)
	// A draft with its own KV cache mirroring the target chain
	// 1->2->3->4->5->6->EOS.
	// The decode seed catch-up writes the draft pair for the prompt token,
	// the first proposal comes from the catch-up's held logits (no draft
	// call), speculative entries are written at advancing slots, and the
	// post-accept rebuild rewrites the committed range from target hiddens.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: eos, eos: 0}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
	draft := &fakeKVDraft{predict: predict}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(2) // caches[0] target, caches[1] draft KV
	session, ch := newMTPTestSession(caches)
	position := 0

	req := Request{
		Responses:         ch,
		Tokens:            []int32{1},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	spec := r.spec.open(req, caches)
	if spec == nil || !spec.enabled || len(spec.spec.targets) != 1 {
		t.Fatalf("speculation engine not built around the draft caches")
	}
	defer spec.close()
	d := spec.decoder([]int32{1}, position)
	if err := r.decode(context.Background(), req, session, d, 0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	d.close()

	content, final := collectResponses(ch)
	if content != "23456" {
		t.Fatalf("content = %q, want %q", content, "23456")
	}
	if final.EvalCount != 5 || final.DoneReason != 0 {
		t.Fatalf("EvalCount = %d, DoneReason = %d, want 5 and 0", final.EvalCount, final.DoneReason)
	}
	if got := []int32{2, 3, 4, 5, 6, eos}; !slices.Equal(session.outputs, got) {
		t.Fatalf("session outputs = %v, want %v", session.outputs, got)
	}

	// All caches rest together at the trie frontier: the prompt token plus
	// five generated tokens; the EOS's KV is never committed.
	if got := caches[0].Offset(); got != 6 {
		t.Fatalf("target offset = %d, want 6", got)
	}
	if got := caches[1].Offset(); got != 6 {
		t.Fatalf("draft cache offset = %d, want 6 (lockstep with target)", got)
	}

	// Every committed draft slot S fuses look-ahead token x_{S+1} with the
	// target hidden at S (whose hot index equals the look-ahead id on this
	// mirrored chain); speculative steps fuse the head's own projections
	// (-1), seeded by the catch-up flush's held projection. The first
	// proposal of the round consumes the catch-up's held logits, so the
	// four-token draft makes only three head calls before the rebuild.
	wantExtends := []extendCall{
		{offset: 0, ids: []int32{2}, hiddens: []int32{2}},                             // frontier pair flushed at the first proposal
		{offset: 1, ids: []int32{3}, hiddens: []int32{-1}},                            // speculative step 2 (held projection)
		{offset: 2, ids: []int32{4}, hiddens: []int32{-1}},                            // speculative step 3 (projection)
		{offset: 3, ids: []int32{5}, hiddens: []int32{-1}},                            // speculative step 4 (projection)
		{offset: 1, ids: []int32{3, 4, 5, 6, eos}, hiddens: []int32{3, 4, 5, 6, eos}}, // committed pairs from the validated run + finish
	}
	if !reflect.DeepEqual(draft.extends, wantExtends) {
		t.Fatalf("draft extends = %+v, want %+v", draft.extends, wantExtends)
	}

	// The committed draft KV holds the look-ahead tokens, one per slot.
	if got, want := caches[1].(*fakeRewindableCache).tokens, []int32{2, 3, 4, 5, 6, eos}; !slices.Equal(got, want) {
		t.Fatalf("draft cache = %v, want %v", got, want)
	}
}

func TestDecodeKVDraftRejectionRebuildsFromTarget(t *testing.T) {
	skipIfNoMLX(t)
	// The draft mispredicts mid-chain: it proposes 6 where the target's own
	// next token is 4, so the round is accepted only up to the rejection and the
	// loop re-proposes from the target's correction. The speculative draft KV
	// written for the rejected proposals is rolled back and rewritten from the
	// validated run's target hiddens, so the committed draft cache holds the
	// target chain with no trace of the rejected tokens.
	const eos int32 = 7
	target := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: eos, eos: 0}
	r := mtpTestRunner(t, target, []int32{eos}, sampler.Options{})
	// The draft mirrors the target except at 3, where it proposes 6 (absent from
	// the target chain); once the target corrects 3->4 the next proposal
	// re-aligns on the shared chain.
	draft := &fakeKVDraft{predict: map[int32]int32{1: 2, 2: 3, 3: 6, 6: 0, 4: 5, 5: eos, eos: 0}}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(2) // caches[0] target, caches[1] draft KV
	session, ch := newMTPTestSession(caches)
	position := 0

	req := Request{
		Responses:         ch,
		Tokens:            []int32{1},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	spec := r.spec.open(req, caches)
	spec.limit = 4
	defer spec.close()
	d := spec.decoder([]int32{1}, position)
	if err := r.decode(context.Background(), req, session, d, 0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	d.close()

	content, final := collectResponses(ch)
	if content != "2345" {
		t.Fatalf("content = %q, want %q", content, "2345")
	}
	if final.DoneReason != 0 {
		t.Fatalf("DoneReason = %d, want 0 (EOS)", final.DoneReason)
	}
	if got := []int32{2, 3, 4, 5, eos}; !slices.Equal(session.outputs, got) {
		t.Fatalf("session outputs = %v, want %v", session.outputs, got)
	}

	// The divergent token was drafted into the speculative KV...
	drafted := false
	for _, e := range draft.extends {
		if slices.Contains(e.ids, 6) {
			drafted = true
		}
	}
	if !drafted {
		t.Fatalf("draft never proposed the divergent token 6; extends = %+v", draft.extends)
	}
	// ...but the rejection rolled it back: both caches rest in lockstep at the
	// target chain, and the committed draft KV holds only the validated
	// look-ahead tokens.
	if got := caches[0].Offset(); got != 5 {
		t.Fatalf("target offset = %d, want 5", got)
	}
	if got := caches[1].Offset(); got != 5 {
		t.Fatalf("draft cache offset = %d, want 5 (lockstep with target)", got)
	}
	if got, want := caches[1].(*fakeRewindableCache).tokens, []int32{2, 3, 4, 5, eos}; !slices.Equal(got, want) {
		t.Fatalf("draft cache = %v, want %v (rejected proposals rolled back)", got, want)
	}
}

func TestDecodeMaintainsDraftCacheWithoutDrafting(t *testing.T) {
	skipIfNoMLX(t)
	// A request that cannot speculate (logprobs) on a model whose draft has
	// its own KV cache still maintains it: the pipelined decoder
	// reports each forwarded token, the pairs wait in the pending list, and
	// one batched extend at close — its final pair completed by the decoder's
	// discarded in-flight sample — leaves the draft level with the target.
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: eos, eos: 0}
	opts := sampler.Options{Logprobs: true}
	r := mtpTestRunner(t, predict, []int32{eos}, opts)
	draft := &fakeKVDraft{predict: predict}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(2)
	session, ch := newMTPTestSession(caches)
	position := 0

	req := Request{
		Responses:         ch,
		Tokens:            []int32{1},
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       opts,
	}
	spec := r.spec.open(req, caches)
	if spec == nil || spec.enabled {
		t.Fatalf("want a maintain-only speculationSession, got %+v", spec)
	}
	d := spec.decoder([]int32{1}, position)
	if err := r.decode(context.Background(), req, session, d, 0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	d.close()
	spec.close() // the pipeline defers this before session close

	content, _ := collectResponses(ch)
	if content != "23456" {
		t.Fatalf("content = %q, want %q", content, "23456")
	}

	// The pipelined decoder forwards every emitted token, including the
	// ending EOS, so both caches rest at the full recorded path.
	if got := caches[0].Offset(); got != 7 {
		t.Fatalf("target offset = %d, want 7", got)
	}
	if got := caches[1].Offset(); got != 7 {
		t.Fatalf("draft cache offset = %d, want 7 (lockstep with target)", got)
	}

	// A session that never drafts defers every committed pair: the whole
	// generation arrives in one batched flush at finish, with the EOS slot's
	// pair completed by the in-flight sample (predict[eos] = 0) that decoding
	// discarded.
	wantExtends := []extendCall{
		{offset: 0, ids: []int32{2, 3, 4, 5, 6, eos, 0}, hiddens: []int32{2, 3, 4, 5, 6, eos, 0}},
	}
	if !reflect.DeepEqual(draft.extends, wantExtends) {
		t.Fatalf("draft extends = %+v, want %+v", draft.extends, wantExtends)
	}
	if got, want := caches[1].(*fakeRewindableCache).tokens, []int32{2, 3, 4, 5, 6, eos, 0}; !slices.Equal(got, want) {
		t.Fatalf("draft cache = %v, want %v", got, want)
	}
}

func TestFlushLevelsDraftCacheWithPrefill(t *testing.T) {
	skipIfNoMLX(t)
	// Prefill attaches its scheduled snapshots only at offsets every cache
	// has crossed, so the pipeline flushes the drafter's buffered pairs
	// first: after a prefill-sized committed report and a flush, the
	// draft cache covers every completed pair, one slot behind the target
	// (the frontier pair still awaits its look-ahead token).
	const eos int32 = 7
	predict := map[int32]int32{2: 3, 3: 4, 4: 5}
	r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
	draft := &fakeKVDraft{predict: predict}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(2)
	req := Request{
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	spec := r.spec.open(req, caches)
	defer spec.close()

	// The prompt's only chunk: tokens 1..4 at slots 0..3 with their hiddens.
	spec.committed(mlx.FromValues([]int32{1, 2, 3, 4}, 1, 4), oneHotLogits([]int32{1, 2, 3, 4}), 0)
	spec.flush()

	if got := caches[1].Offset(); got != 3 {
		t.Fatalf("draft cache offset after flush = %d, want 3 (all completed pairs)", got)
	}
	wantExtends := []extendCall{
		{offset: 0, ids: []int32{2, 3, 4}, hiddens: []int32{1, 2, 3}},
	}
	if !reflect.DeepEqual(draft.extends, wantExtends) {
		t.Fatalf("draft extends = %+v, want %+v", draft.extends, wantExtends)
	}
}

func TestCommittedRunBatchesPastFlushCap(t *testing.T) {
	skipIfNoMLX(t)
	// A committed run longer than the pending-flush cap still writes the draft
	// caches in a single head forward: the run's completed pairs coalesce into
	// one batched extend at the run's start, rather than splitting at the cap.
	const n = mtpPendingFlushTokens + 8
	predict := map[int32]int32{}
	tokens := make([]int32, n)
	for i := range tokens {
		tokens[i] = int32(i%7 + 1)
		predict[tokens[i]] = int32((i+1)%7 + 1)
	}

	r := mtpTestRunner(t, predict, []int32{0}, sampler.Options{})
	draft := &fakeKVDraft{predict: predict}
	r.spec = newSpeculation(r, draft)

	caches, _ := newMTPTestCaches(2)
	req := Request{
		CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}},
		SamplerOpts:       sampler.Options{},
	}
	spec := r.spec.open(req, caches)
	defer spec.close()

	// One prefill-sized chunk: n tokens at slots 0..n-1 with their hiddens.
	spec.committed(mlx.FromValues(tokens, 1, n), oneHotLogits(tokens), 0)
	spec.flush()

	if got := len(draft.extends); got != 1 {
		t.Fatalf("draft extends = %d calls, want 1 batched extend", got)
	}
	if got, want := len(draft.extends[0].ids), n-1; got != want {
		t.Fatalf("batched extend ids = %d, want %d (every completed pair)", got, want)
	}
	if got := draft.extends[0].offset; got != 0 {
		t.Fatalf("batched extend offset = %d, want 0", got)
	}
	if got := caches[1].Offset(); got != n-1 {
		t.Fatalf("draft cache offset = %d, want %d (all completed pairs)", got, n-1)
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

// testSpeculationSession wires a speculation engine around a drafter and caches for
// tests that drive accept directly. A runner without a draft model gets a
// no-op drafter, since the engine requires one.
func testSpeculationSession(r *Runner, caches []cache.Cache) *speculationSession {
	if r.spec != nil {
		r.spec.bind(caches)
		return &speculationSession{spec: r.spec, drafter: newMTPDrafter(r.spec)}
	}
	s := &speculation{r: r, caches: caches, targets: caches}
	return &speculationSession{spec: s, drafter: nopDrafter{}}
}

// nopDrafter satisfies drafter for engine tests that supply candidates
// directly and never propose.
type nopDrafter struct{}

func (nopDrafter) propose(*mlx.Array, int) *draftCandidates { return nil }
func (nopDrafter) committed(_, _ *mlx.Array, _ int)         {}
func (nopDrafter) finish(*mlx.Array)                        {}
func (nopDrafter) flush()                                   {}
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
	s := &speculation{r: r, draft: &fakeMTPDraft{predict: chain}}
	d := &mtpDrafter{spec: s}
	d.committed(mlx.FromValues([]int32{0}, 1, 1), mlx.Zeros(mlx.DTypeFloat32, 1, 1, mtpTestVocab), 0)
	return d.propose(mlx.FromValues([]int32{0}, 1), len(tokens))
}
