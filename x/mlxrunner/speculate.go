package mlxrunner

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

// drafter proposes speculative tokens for the engine to validate, learning
// the conversation through the committed-stream reports.
type drafter interface {
	// propose returns up to maxTokens draft tokens with their proposal
	// distributions, or nil to decode this round plainly.
	propose(current *mlx.Array, maxTokens int) *draftCandidates

	// committed reports a run of tokens committed to the target caches:
	// tokens[i] sits at slot position+i and hiddens row i is the target
	// hidden state at that slot. Runs arrive in slot order — the decode
	// seed, then each round's validated tokens.
	committed(tokens, hiddens *mlx.Array, position int)

	close()
}

type draftSchedule string

const (
	draftScheduleHeuristic draftSchedule = "heuristic"
	draftScheduleConstant  draftSchedule = "constant"
)

type specStats struct {
	iterations  int
	drafted     int
	accepted    int
	mismatches  int
	allAccepted int
	maxDraft    int
}

type specOptions struct {
	initialDraftTokens int
	maxDraftTokens     int
	draftSchedule      draftSchedule
}

func (r *Runner) loadSpecOptions(sample bool) specOptions {
	defaults := r.mtpDefaults(sample)

	opts := specOptions{
		initialDraftTokens: defaults.InitialDraftTokens,
		maxDraftTokens:     defaults.MaxDraftTokens,
		draftSchedule:      draftScheduleConstant,
	}
	if v := positiveEnvInt("OLLAMA_MLX_MTP_MAX_DRAFT_TOKENS"); v > 0 {
		opts.maxDraftTokens = v
	}
	if v := positiveEnvInt("OLLAMA_MLX_MTP_INITIAL_DRAFT_TOKENS"); v > 0 {
		opts.initialDraftTokens = v
	}
	if opts.initialDraftTokens > opts.maxDraftTokens {
		opts.initialDraftTokens = opts.maxDraftTokens
	}
	switch schedule := strings.ToLower(strings.TrimSpace(os.Getenv("OLLAMA_MLX_MTP_DRAFT_SCHEDULE"))); schedule {
	case "", string(draftScheduleConstant):
		opts.draftSchedule = draftScheduleConstant
	case string(draftScheduleHeuristic):
		opts.draftSchedule = draftScheduleHeuristic
	default:
		slog.Warn("invalid MTP env setting", "key", "OLLAMA_MLX_MTP_DRAFT_SCHEDULE", "value", schedule)
	}
	return opts
}

func positiveEnvInt(key string) int {
	raw := os.Getenv(key)
	if raw == "" {
		return 0
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		slog.Warn("invalid MTP env setting", "key", key, "value", raw)
		return 0
	}
	return v
}

// speculation is the persistent speculative-decoding subsystem of a Runner,
// one per loaded model. It holds the draft model; as the feature grows it
// also holds the cache partition and the depth state learned across requests.
// A nil *speculation means the checkpoint ships no draft head.
type speculation struct {
	r     *Runner
	draft base.DraftModel
}

// newSpeculation builds the speculative-decoding subsystem for a loaded model,
// or nil when the checkpoint ships no draft head.
func newSpeculation(r *Runner, draft base.DraftModel) *speculation {
	if draft == nil {
		return nil
	}
	return &speculation{r: r, draft: draft}
}

// speculationSession is the speculation cursor for one request over a speculation. A
// nil speculationSession is a plain decode.
type speculationSession struct {
	spec    *speculation
	drafter drafter
	caches  []cache.Cache
	opts    specOptions
	limit   int // current draft length
	stats   specStats
}

// open returns the speculation cursor for this request, or nil when the model
// carries no draft head or the request cannot speculate. Logprobs are not yet
// supported. A nil receiver (no draft head) decodes plainly.
func (s *speculation) open(request Request, caches []cache.Cache) *speculationSession {
	if s == nil {
		return nil
	}
	d := newMTPDrafter(s, caches)
	if d == nil {
		return nil
	}
	opts := request.SamplerOpts
	if !s.r.mtpDefaults(opts.Temperature != 0).Enabled {
		return nil
	}
	if opts.Logprobs || opts.TopLogprobs > 0 {
		return nil
	}

	specOpts := s.r.loadSpecOptions(opts.Temperature != 0)
	return &speculationSession{
		spec:    s,
		drafter: d,
		caches:  caches,
		opts:    specOpts,
		limit:   specOpts.initialDraftTokens,
		stats:   specStats{maxDraft: specOpts.initialDraftTokens},
	}
}

func (s *speculationSession) committed(tokens, hiddens *mlx.Array, position int) {
	if s == nil {
		return
	}
	s.drafter.committed(tokens, hiddens, position)
}

func (s *speculationSession) close() {
	if s == nil {
		return
	}
	s.drafter.close()
}

// speculativeDecoder decodes one speculative round per call: it forwards
// the current token — emitted by the previous call, so a token that ends
// generation is never forwarded — has the engine draft and validate ahead
// of it, and returns the round's accepted tokens followed by the engine's
// next token. The last returned token becomes the next call's current; the
// seed token primes current and is never returned.
type speculativeDecoder struct {
	s        *speculationSession
	position int
	current  sampler.Result // emitted (or the seed), not yet forwarded
}

func (s *speculationSession) decoder(seed []int32, position int) *speculativeDecoder {
	current := sampler.Result{Token: mlx.FromValues(seed, len(seed))}
	mlx.Pin(current.Arrays()...)
	return &speculativeDecoder{s: s, position: position, current: current}
}

func (st *speculativeDecoder) next(remaining int) ([]sampler.Result, error) {
	s := st.s
	r := s.spec.r

	hidden := r.Model.Forward(&batch.Batch{
		InputIDs:     tokenInput(st.current.Token),
		SeqOffsets:   []int32{int32(st.position)},
		SeqQueryLens: []int32{1},
	}, s.caches)
	st.position++

	results, next, err := s.round(&st.position, st.current, hidden, remaining)
	if err != nil {
		return nil, err
	}
	if next.Token != nil {
		results = append(results, next)
	}

	last := results[len(results)-1]
	mlx.Pin(last.Arrays()...)
	mlx.Unpin(st.current.Arrays()...)
	st.current = last
	mlx.AsyncEval(st.current.Arrays()...)
	return results, nil
}

func (st *speculativeDecoder) close() {
	mlx.Unpin(st.current.Arrays()...)
	st.s.logStats()
}

// round runs one speculative decode round for the just-forwarded current
// token and its hidden state: draft candidates after it, validate them
// against the target, and return the accepted run and the bonus or
// resampled next token.
func (s *speculationSession) round(position *int, current sampler.Result, hidden *mlx.Array, remaining int) (results []sampler.Result, next sampler.Result, err error) {
	r := s.spec.r
	s.stats.iterations++

	// A round emits the accepted drafts plus one more token (the bonus or
	// residual), so cap the draft one below the remaining budget to land that
	// extra token within it rather than overshooting. At remaining 1 the cap is
	// 0 and the last token decodes plainly.
	maxDraft := min(s.limit, remaining-1)
	candidates := s.drafter.propose(current.Token, maxDraft)
	baseLogits := lastLogits(r.Model.Unembed(hidden))
	if candidates == nil {
		s.drafter.committed(tokenInput(current.Token), lastHiddenRow(hidden), *position-1)
		return nil, r.Sampler.Sample([]int{pipelineSlot}, baseLogits), nil
	}

	draftCount := candidates.tokens.Dim(1)
	// hidden survives the sweep alongside the candidates: the post-accept
	// report fuses it into the committed stream.
	candidateArrays := append([]*mlx.Array{baseLogits, hidden}, candidates.Arrays()...)
	mlx.Pin(candidateArrays...)
	mlx.Sweep()
	defer mlx.Unpin(candidateArrays...)
	s.stats.drafted += draftCount

	results, accepted, err := s.accept(position, current, hidden, baseLogits, candidates)
	if err != nil {
		return nil, sampler.Result{}, err
	}
	// accept folds the bonus token into its results; surface it back out as
	// the next step's current.
	if len(results) > accepted {
		next = results[len(results)-1]
		results = results[:accepted]
	}

	s.stats.accepted += accepted
	if accepted == draftCount {
		s.stats.allAccepted++
	} else {
		s.stats.mismatches++
	}
	if s.opts.draftSchedule == draftScheduleHeuristic {
		if accepted == draftCount {
			s.limit = min(s.opts.maxDraftTokens, s.limit+2)
		} else {
			s.limit = max(1, s.limit-1)
		}
		s.stats.maxDraft = max(s.stats.maxDraft, s.limit)
	}
	return results, next, nil
}

// logStats reports the per-request speculation summary.
func (s *speculationSession) logStats() {
	acceptance := 0.0
	if s.stats.drafted > 0 {
		acceptance = float64(s.stats.accepted) / float64(s.stats.drafted)
	}
	avgDraft := 0.0
	avgAccepted := 0.0
	if s.stats.iterations > 0 {
		avgDraft = float64(s.stats.drafted) / float64(s.stats.iterations)
		avgAccepted = float64(s.stats.accepted) / float64(s.stats.iterations)
	}
	slog.Info("speculative decode stats", "drafted", s.stats.drafted, "accepted", s.stats.accepted, "acceptance", acceptance, "iterations", s.stats.iterations, "avg_draft", avgDraft, "avg_accepted", avgAccepted, "mismatches", s.stats.mismatches, "all_accepted", s.stats.allAccepted, "max_draft", s.stats.maxDraft, "draft_schedule", s.opts.draftSchedule)
}

// draftCandidates is one round's draft tokens and the proposal distribution
// each was sampled from, weighed against the target during acceptance.
type draftCandidates struct {
	tokens *mlx.Array
	dist   sampler.Distribution
}

func (c *draftCandidates) Arrays() []*mlx.Array {
	if c == nil {
		return nil
	}
	return append([]*mlx.Array{c.tokens}, c.dist.Arrays()...)
}

// scheduleSpeculation schedules per-token snapshots at offsets
// [before, before+draftCount) on every cache, so the speculative forward
// captures a rollback point before each draft token's write.
func scheduleSpeculation(caches []cache.Cache, before, draftCount int) {
	offsets := make([]int, draftCount)
	for i := range offsets {
		offsets[i] = before + i
	}
	for _, c := range caches {
		if c != nil {
			c.PrepareSnapshots(offsets)
		}
	}
}

// commitSpeculation rolls every cache back to before+accepted, keeping only
// the accepted prefix; full acceptance needs no restore. Rollback tries a
// live rewind first (Restore(nil)) and falls back to the captured snapshot.
func commitSpeculation(caches []cache.Cache, accepted, draftCount, before int) {
	target := before + accepted
	for _, c := range caches {
		if c == nil {
			continue
		}
		snaps := c.TakeSnapshots()
		if accepted < draftCount {
			// Close the snapshots we won't restore from before restoring: a
			// snapshot restore on a wrapped RotatingKVCache copies out every
			// outstanding lazy snapshot before it rebuilds the buffer, so
			// dropping the unused ones first stops that copy-out from
			// materializing snapshots we are about to discard anyway.
			for i, s := range snaps {
				if s != nil && i != accepted {
					s.Close()
					snaps[i] = nil
				}
			}
			if !c.Restore(nil, target) && !c.Restore(snaps[accepted], target) {
				panic(fmt.Sprintf("speculation: cache restore to %d failed", target))
			}
		}
		for _, s := range snaps {
			if s != nil {
				s.Close()
			}
		}
	}
}

// accept accepts the longest draft prefix that survives rejection sampling
// against the target model. At temperature 0 the distributions are point
// masses, so acceptance reduces to argmax-match. It returns the accepted
// drafts followed by the target's own next token — the residual at the
// rejection point, or the bonus past a fully accepted run — so a round
// yields accepted+1 tokens, except when an accepted EOS ends generation,
// where the run stops at the EOS with no continuation. The accepted run is
// reported back to the drafter with its hidden states.
//
// The NumPredict budget is the decode loop's to enforce; the drafter is
// already capped to the remaining budget, so an accepted run never
// overshoots, and a legitimate token past the budget is left for decode to
// drop rather than cut here.
func (s *speculationSession) accept(position *int, current sampler.Result, hidden, baseLogits *mlx.Array, candidates *draftCandidates) (results []sampler.Result, accepted int, err error) {
	r := s.spec.r
	before := *position
	draftCount := candidates.tokens.Dim(1)
	scheduleSpeculation(s.caches, before, draftCount)

	// Every exit between schedule and commit must drain the snapshot
	// schedule and roll the speculative writes back out of the live caches:
	// an undrained schedule panics the next PrepareSnapshots, and
	// uncommitted speculative tokens reach the trie through session.close.
	committed := false
	commit := func(keep int) {
		if committed {
			return
		}
		committed = true
		commitSpeculation(s.caches, keep, draftCount, before)
	}
	defer commit(0)

	hiddenSeq := r.Model.Forward(&batch.Batch{
		InputIDs:     candidates.tokens,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(draftCount)},
	}, s.caches)

	targetDist := r.Sampler.Distribution(pipelineSlot, validationLogits(r, baseLogits, hiddenSeq), candidates.tokens)
	draftDist := candidates.dist
	acceptedMask := r.sampleAcceptedMask(targetDist.SliceRows(0, draftCount), draftDist, candidates.tokens)
	mlx.Eval(candidates.tokens, acceptedMask)

	draftIDs := candidates.tokens.Ints()
	acceptedFlags := acceptedMask.Ints()
	for _, ok := range acceptedFlags {
		if ok == 0 {
			break
		}
		accepted++
	}
	if accepted > draftCount {
		return nil, 0, fmt.Errorf("speculation validation accepted %d tokens for %d draft tokens", accepted, draftCount)
	}

	// Find where an accepted EOS ends the run, before committing, so the cut
	// is known while the per-token rollback snapshots still exist.
	commitIDs := make([]int32, 0, accepted+1)
	done := false
	for i, id := range draftIDs[:accepted] {
		commitIDs = append(commitIDs, int32(id))
		if r.Tokenizer.IsEOS(int32(id)) {
			done = true
			accepted = i + 1
			break
		}
	}

	commit(accepted)
	*position = before + accepted

	// Report the validated run — current plus the accepted drafts, with the
	// hidden state at each token's own slot — to the drafter before
	// returning, so even a cancelled emission leaves the drafter's state
	// describing exactly what the target caches hold.
	runIDs := append([]int32{int32(current.Token.Int())}, commitIDs...)
	runHiddens := lastHiddenRow(hidden)
	if accepted > 0 {
		runHiddens = runHiddens.Concatenate(1, hiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, accepted), mlx.Slice()))
	}
	s.drafter.committed(mlx.FromValues(runIDs, 1, len(runIDs)), runHiddens, before-1)

	results = draftResults(draftIDs[:accepted])
	if done {
		r.Sampler.Commit(pipelineSlot, commitIDs)
		return results, accepted, nil
	}

	var nextToken *mlx.Array
	if accepted == draftCount {
		nextToken = r.sampleTokenAt(targetDist, draftCount)
	} else {
		nextToken = r.sampleResidualToken(targetDist, draftDist, accepted)
	}
	mlx.Eval(nextToken)
	nextID := int32(nextToken.Int())
	commitIDs = append(commitIDs, nextID)
	r.Sampler.Commit(pipelineSlot, commitIDs)

	results = append(results, sampler.Result{Token: nextToken})
	return results, accepted, nil
}

// validationLogits stacks the current token's logits ahead of the draft
// positions' logits so row i scores draft i.
func validationLogits(r *Runner, baseLogits, hiddenSeq *mlx.Array) *mlx.Array {
	seqLogits := r.Model.Unembed(hiddenSeq)
	return baseLogits.ExpandDims(1).Concatenate(1, seqLogits)
}

func (r *Runner) sampleAcceptedMask(targetDist, draftDist sampler.Distribution, draftTokens *mlx.Array) *mlx.Array {
	p := targetDist.Prob(draftTokens)
	q := draftDist.Prob(draftTokens)
	acceptP := mlx.Minimum(p.Divide(q), mlx.FromValue(float32(1)))
	return r.Sampler.Bernoulli(pipelineSlot, acceptP).AsType(mlx.DTypeInt32)
}

func (r *Runner) sampleTokenAt(dist sampler.Distribution, index int) *mlx.Array {
	return r.Sampler.SampleDistribution(pipelineSlot, dist.SliceRows(index, index+1))
}

func (r *Runner) sampleResidualToken(targetDist, draftDist sampler.Distribution, index int) *mlx.Array {
	residual := targetDist.SliceRows(index, index+1).ResidualAgainst(draftDist.SliceRows(index, index+1))
	return tokenVector(r.Sampler.SampleDistribution(pipelineSlot, residual))
}

// draftResults wraps accepted draft ids as sampler results; drafts carry no
// logprobs, so only the token id is set.
func draftResults(ids []int) []sampler.Result {
	results := make([]sampler.Result, len(ids))
	for i, id := range ids {
		results[i] = sampler.Result{Token: mlx.FromValues([]int32{int32(id)}, 1)}
	}
	return results
}

func tokenInput(token *mlx.Array) *mlx.Array {
	switch token.NumDims() {
	case 0:
		return token.Reshape(1, 1)
	case 1:
		return token.ExpandDims(-1)
	case 2:
		return token
	default:
		panic(fmt.Sprintf("token must be rank 0, 1, or 2, got rank %d", token.NumDims()))
	}
}

func tokenVector(token *mlx.Array) *mlx.Array {
	switch token.NumDims() {
	case 0:
		return token.Reshape(1)
	case 1:
		return token
	default:
		panic(fmt.Sprintf("sampled token must be rank 0 or 1, got rank %d", token.NumDims()))
	}
}
