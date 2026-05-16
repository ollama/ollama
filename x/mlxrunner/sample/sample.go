package sample

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Options struct {
	Temperature      float32
	TopP             float32
	MinP             float32
	TopK             int
	RepeatLastN      int
	RepeatPenalty    float32
	PresencePenalty  float32
	FrequencyPenalty float32
	Seed             int
	UseSeed          bool

	// Logprobs causes Sample to populate Result.Logprob with the selected
	// token's log-probability. TopLogprobs (when > 0) adds top-K pairs.
	Logprobs    bool
	TopLogprobs int
}

// Result bundles the outputs of one decode step. Logprob/TopTokens/
// TopLogprobs are populated whenever any registered slot has Logprobs
// (respectively TopLogprobs>0). Consumers need to filter by their
// per-slot Options.
type Result struct {
	Token       *mlx.Array // sampled token ids,       shape [B]
	Logprob     *mlx.Array // sampled-token logprobs,  shape [B,1];    nil unless any registered slot has Logprobs
	TopTokens   *mlx.Array // top-K token ids,         shape [B,maxK]; nil unless any registered slot has TopLogprobs>0
	TopLogprobs *mlx.Array // top-K logprobs,          shape [B,maxK]; same
}

// Arrays returns the tensor fields as a slice so callers can drive the mlx
// lifecycle verbs (Pin, Unpin, Eval, AsyncEval) over the whole group. Unset
// fields stay nil; the mlx helpers skip them.
func (r Result) Arrays() []*mlx.Array {
	return []*mlx.Array{r.Token, r.Logprob, r.TopTokens, r.TopLogprobs}
}

// Distribution is the filtered probability distribution used by the sampler.
// When IDs is nil, Probs is dense over the vocabulary. When IDs is set, Probs
// is sparse over the token ids in IDs, preserving GPU residency for the
// top-k-first path used by normal and speculative sampling.
type Distribution struct {
	IDs   *mlx.Array // sparse token ids, shape [B,K]; nil for dense distributions
	Probs *mlx.Array // probabilities, shape [B,K] or [B,V]
}

// Arrays returns the tensor fields for mlx lifecycle management.
func (d Distribution) Arrays() []*mlx.Array {
	return []*mlx.Array{d.IDs, d.Probs}
}

// Rows returns the number of rows in the distribution.
func (d Distribution) Rows() int {
	if d.Probs == nil {
		return 0
	}
	return d.Probs.Dim(0)
}

// SliceRows returns a row slice while preserving sparse/dense layout.
func (d Distribution) SliceRows(start, stop int) Distribution {
	out := Distribution{Probs: d.Probs.Slice(mlx.Slice(start, stop), mlx.Slice())}
	if d.IDs != nil {
		out.IDs = d.IDs.Slice(mlx.Slice(start, stop), mlx.Slice())
	}
	return out
}

// SampleWithKey draws one token per row using key when supplied.
func (d Distribution) SampleWithKey(key *mlx.Array) *mlx.Array {
	choice := logitsFromProbs(d.Probs).CategoricalWithKey(-1, key).AsType(mlx.DTypeInt32)
	if d.IDs == nil {
		return choice
	}
	return d.IDs.TakeAlongAxis(choice.ExpandDims(-1), -1).Squeeze(-1).AsType(mlx.DTypeInt32)
}

// Prob returns the probability assigned to one token per row.
func (d Distribution) Prob(tokens *mlx.Array) *mlx.Array {
	switch tokens.NumDims() {
	case 2:
		if tokens.Dim(0) == 1 {
			tokens = tokens.Squeeze(0)
		} else if tokens.Dim(1) == 1 {
			tokens = tokens.Squeeze(1)
		}
	case 0:
		tokens = tokens.Reshape(1)
	}
	return d.ProbsForIDs(tokens.ExpandDims(-1)).Squeeze(-1)
}

// ProbsForIDs returns probabilities for each requested token id. ids must be
// rank-2 [B,N], matching the distribution rows.
func (d Distribution) ProbsForIDs(ids *mlx.Array) *mlx.Array {
	if d.IDs == nil {
		return d.Probs.TakeAlongAxis(ids, -1)
	}
	eq := d.IDs.ExpandDims(-1).Equal(ids.ExpandDims(1))
	values := mlx.Where(eq, d.Probs.ExpandDims(-1), mlx.FromValue(float32(0)))
	return values.SumAxis(1, false)
}

// ResidualAgainst returns the Leviathan/Chen rejection distribution
// proportional to max(target - draft, 0). Sparse target distributions stay
// sparse over the target support; tokens outside target support have zero mass.
func (d Distribution) ResidualAgainst(draft Distribution) Distribution {
	if d.IDs != nil {
		diff := d.Probs.Subtract(draft.ProbsForIDs(d.IDs))
		return Distribution{IDs: d.IDs, Probs: normalizeProbs(mlx.Maximum(diff, mlx.FromValue(float32(0))))}
	}
	if draft.IDs != nil {
		panic("sample.Distribution.ResidualAgainst: dense target with sparse draft is unsupported")
	}
	diff := d.Probs.Subtract(draft.Probs)
	return Distribution{Probs: normalizeProbs(mlx.Maximum(diff, mlx.FromValue(float32(0))))}
}

// LogProbs returns dense log-probabilities, scattering sparse distributions
// into a full-vocabulary tensor when needed.
func (d Distribution) LogProbs(vocab int) *mlx.Array {
	logProbs := logitsFromProbs(d.Probs)
	if d.IDs == nil {
		return logProbs
	}
	out := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, d.Probs.Dim(0), vocab), float32(math.Inf(-1)))
	return out.PutAlongAxis(d.IDs, logProbs, -1)
}

// ConcatenateDistributions concatenates distribution rows. All inputs must use
// the same sparse/dense layout.
func ConcatenateDistributions(dists []Distribution) Distribution {
	if len(dists) == 0 {
		return Distribution{}
	}
	probs := make([]*mlx.Array, 0, len(dists))
	ids := make([]*mlx.Array, 0, len(dists))
	sparse := dists[0].IDs != nil
	for _, d := range dists {
		if (d.IDs != nil) != sparse {
			panic("sample.ConcatenateDistributions: mixed sparse and dense distributions")
		}
		probs = append(probs, d.Probs)
		if sparse {
			ids = append(ids, d.IDs)
		}
	}
	out := Distribution{Probs: mlx.Concatenate(probs, 0)}
	if sparse {
		out.IDs = mlx.Concatenate(ids, 0)
	}
	return out
}

// Sampler is a batched, slot-based sampler. Sequences are registered with
// Add and released with Remove. Each Sample call takes a subset of
// registered slots (in any order) with their [B,V] logits, samples one
// token per row, and appends it to that slot's ring-buffer history. Slots
// not named in a given call are untouched.
type Sampler struct {
	slots []*slotState
	byID  map[int]*slotState

	// history is the pooled ring-buffer storage, [B, W] int32. Row i
	// belongs to slots[i]; W is max(RepeatLastN) across penalty slots.
	// Allocated on the first penalty slot, rebuilt only in Add/Remove.
	history *mlx.Array

	// allSameOpts: every registered slot shares Options. When true the
	// canonical shared value is s.slots[0].opts.
	allSameOpts bool

	// anyLogprobs / maxTopLogprobs: compute-for-all output config.
	// Sample populates Logprob (and Top* when maxTopLogprobs>0) whenever
	// any registered slot requests them, even if that slot isn't in the
	// current call.
	anyLogprobs    bool
	maxTopLogprobs int

	// numCtx is the runner's context window; normalize uses it to
	// resolve the repeat_last_n == -1 sentinel.
	numCtx int
}

type slotState struct {
	opts          Options
	historyLen    int
	randomCounter uint64
}

type slotCtx struct {
	opts    Options
	history *mlx.Array // 2D [B, W] when penalties are configured; nil otherwise
}

// New constructs an empty sampler with no registered slots. numCtx is
// the runner's context window and must be positive.
func New(numCtx int) *Sampler {
	return &Sampler{
		byID:        make(map[int]*slotState),
		allSameOpts: true,
		numCtx:      numCtx,
	}
}

// historyWidth returns the column count of the pooled history tensor,
// or 0 when no penalty slot has forced it to be allocated.
func (s *Sampler) historyWidth() int {
	if s.history == nil {
		return 0
	}
	return s.history.Dim(1)
}

func (o Options) usesHistory() bool {
	// RepeatLastN == 0 disables the penalty ring per the repeat_last_n API
	// contract (0 = disabled), overriding any penalty coefficients.
	if o.RepeatLastN == 0 {
		return false
	}
	return o.RepeatPenalty != 1 || o.PresencePenalty != 0 || o.FrequencyPenalty != 0
}

func (o Options) normalize(numCtx int) Options {
	if o.RepeatPenalty <= 0 {
		o.RepeatPenalty = 1
	}
	// Resolve the repeat_last_n == -1 sentinel ("-1 = num_ctx") against
	// the caller's context window.
	if o.RepeatLastN < 0 {
		o.RepeatLastN = numCtx
	}
	if !o.usesHistory() {
		// Zero the ring capacity so slots that differ only in a spurious
		// RepeatLastN still batch together and don't inflate pool width.
		o.RepeatLastN = 0
	}
	if o.Seed < 0 {
		o.UseSeed = false
	}
	if !o.UseSeed {
		// Keep unseeded callers on the same batching path even when a
		// meaningless Seed value is present in an Options literal.
		o.Seed = 0
	}
	return o
}

// Add registers a sequence under seqID. The last RepeatLastN entries of
// priorTokens seed the ring buffer.
func (s *Sampler) Add(seqID int, opts Options, priorTokens []int32) {
	if _, dup := s.byID[seqID]; dup {
		panic(fmt.Sprintf("sample.Sampler.Add: seqID %d already registered", seqID))
	}

	opts = opts.normalize(s.numCtx)
	slot := &slotState{
		opts: opts,
	}

	// Grow the pool to hold this slot's row. The pool is lazy — the first
	// penalty slot allocates it — and thereafter every registered slot
	// gets a row (rows for non-penalty slots are zero and never read).
	// Invariant: s.history is pinned whenever non-nil.
	if s.history != nil || opts.usesHistory() {
		targetWidth := max(opts.RepeatLastN, s.historyWidth())
		newRow := makeHistoryRow(priorTokens, opts.RepeatLastN, targetWidth)

		var pool *mlx.Array
		switch {
		case s.history == nil && len(s.slots) == 0:
			pool = newRow
		case s.history == nil:
			// First penalty slot with non-penalty slots already registered;
			// seed zero rows so s.slots and pool row indices stay aligned.
			zeros := mlx.Zeros(mlx.DTypeInt32, len(s.slots), targetWidth)
			pool = zeros.Concatenate(0, newRow)
		case targetWidth > s.historyWidth():
			pad := mlx.Zeros(mlx.DTypeInt32, s.history.Dim(0), targetWidth-s.historyWidth())
			pool = s.history.Concatenate(1, pad).Concatenate(0, newRow)
		default:
			pool = s.history.Concatenate(0, newRow)
		}

		mlx.Pin(pool)
		mlx.Unpin(s.history)
		s.history = pool

		if opts.usesHistory() {
			// Cap on seed so the next write's ring position
			// (historyLen % RepeatLastN) lands at 0, overwriting the
			// oldest entry when the ring was filled from priors.
			slot.historyLen = min(len(priorTokens), opts.RepeatLastN)
		}
	}

	s.slots = append(s.slots, slot)
	s.byID[seqID] = slot
	s.recomputeInvariants()
}

// makeHistoryRow builds a [1, width] int32 row with the last repeatLastN
// entries of priorTokens packed into [0, min(len, repeatLastN)), zeros
// elsewhere.
func makeHistoryRow(priorTokens []int32, repeatLastN, width int) *mlx.Array {
	take := min(len(priorTokens), repeatLastN)
	if take <= 0 {
		return mlx.Zeros(mlx.DTypeInt32, 1, width)
	}
	row := make([]int32, width)
	copy(row, priorTokens[len(priorTokens)-take:])
	return mlx.NewArrayInt32(row, []int32{1, int32(width)})
}

// recomputeInvariants refreshes allSameOpts and anyLogprobs/maxTopLogprobs
// from s.slots. Called at the end of Add and Remove.
func (s *Sampler) recomputeInvariants() {
	if len(s.slots) == 0 {
		s.allSameOpts = true
		s.anyLogprobs = false
		s.maxTopLogprobs = 0
		return
	}
	first := s.slots[0].opts
	s.allSameOpts = true
	s.anyLogprobs = false
	s.maxTopLogprobs = 0
	for _, slot := range s.slots {
		if slot.opts != first {
			s.allSameOpts = false
		}
		if slot.opts.Logprobs {
			s.anyLogprobs = true
			if slot.opts.TopLogprobs > s.maxTopLogprobs {
				s.maxTopLogprobs = slot.opts.TopLogprobs
			}
		}
	}
}

// Remove releases the slot. The pool tensor is rebuilt to drop the row.
func (s *Sampler) Remove(seqID int) {
	slot, ok := s.byID[seqID]
	if !ok {
		return
	}
	delete(s.byID, seqID)

	row := slices.Index(s.slots, slot)
	s.slots = slices.Delete(s.slots, row, row+1)
	s.recomputeInvariants()

	if s.history == nil {
		return
	}

	n := s.history.Dim(0)
	var newHistory *mlx.Array
	switch {
	case n == 1:
		newHistory = nil
	case row == 0:
		newHistory = s.history.Slice(mlx.Slice(1, n), mlx.Slice())
	case row == n-1:
		newHistory = s.history.Slice(mlx.Slice(0, row), mlx.Slice())
	default:
		before := s.history.Slice(mlx.Slice(0, row), mlx.Slice())
		after := s.history.Slice(mlx.Slice(row+1, n), mlx.Slice())
		newHistory = before.Concatenate(0, after)
	}

	mlx.Pin(newHistory)
	mlx.Unpin(s.history)
	s.history = newHistory
}

// Free releases the pooled history tensor and resets the sampler to the
// New-equivalent state so it may be reused.
func (s *Sampler) Free() {
	mlx.Unpin(s.history)
	*s = Sampler{
		byID:        make(map[int]*slotState),
		allSameOpts: true,
		numCtx:      s.numCtx,
	}
}

// Sample draws one token per row of logits ([B,V]); seqIDs[i] names the
// slot whose logits live at row i. Each sampled token is appended to its
// slot's ring. Slots not named in seqIDs are untouched.
func (s *Sampler) Sample(seqIDs []int, logits *mlx.Array) Result {
	if len(seqIDs) == 0 {
		return Result{}
	}

	slots := make([]*slotState, len(seqIDs))
	for i, id := range seqIDs {
		slot, ok := s.byID[id]
		if !ok {
			panic(fmt.Sprintf("sample.Sampler.Sample: seqID %d not registered", id))
		}
		slots[i] = slot
	}

	var token *mlx.Array
	if opts0, ok := s.canBatch(slots); ok {
		token = s.sampleTokensUniform(slots, opts0, logits)
	} else {
		token = s.sampleTokensSerial(slots, logits)
	}

	res := Result{Token: token}
	if s.anyLogprobs {
		// Log-softmax over original logits so every row holds a truthful
		// value (compute-for-all; consumers filter per-slot). Subtract
		// max first for numerical stability in the logsumexp.
		lp := logits.AsType(mlx.DTypeFloat32)
		lp = lp.Subtract(lp.MaxAxis(-1, true))
		lp = lp.Subtract(lp.LogsumexpAxis(-1, true))
		res.Logprob = lp.TakeAlongAxis(token.ExpandDims(-1), -1)
		if s.maxTopLogprobs > 0 {
			k := s.maxTopLogprobs
			if vocab := lp.Dim(lp.NumDims() - 1); k > vocab {
				k = vocab
			}
			// Argpartition on the negated values places the K largest
			// (unsorted) in positions [0:K].
			idx := lp.Negative().ArgpartitionAxis(k-1, -1).Slice(mlx.Slice(), mlx.Slice(0, k))
			res.TopTokens = idx.AsType(mlx.DTypeInt32)
			res.TopLogprobs = lp.TakeAlongAxis(idx, -1)
		}
	}
	return res
}

// Distribution applies this slot's sampling transforms to logits without
// mutating sampler state. Row i is built as if draftTokens[:i] had already
// been appended to the slot history. logits must be [R,V] or [1,R,V].
func (s *Sampler) Distribution(seqID int, logits *mlx.Array, draftTokens *mlx.Array) Distribution {
	slot, logits, draftTokens := s.speculativeInputs("Distribution", seqID, logits, draftTokens)
	rows := logits.Dim(0)

	var hist *mlx.Array
	if slot.opts.usesHistory() {
		if s.history == nil {
			panic(fmt.Sprintf("sample.Sampler.Distribution: seqID %d has no history", seqID))
		}
		if slot.historyLen < slot.opts.RepeatLastN {
			return s.speculativeDistributionSerial(slot, logits, draftTokens)
		}
		hist = s.speculativeHistory(slot, draftTokens, rows)
	}

	return slot.distribution(&slotCtx{opts: slot.opts, history: hist}, logits)
}

// SpeculativeScores applies this slot's sampling transforms to logits without
// mutating sampler state and returns dense log-probability scores for sampled
// decoding. Greedy decoding returns the penalty-adjusted logits.
func (s *Sampler) SpeculativeScores(seqID int, logits *mlx.Array, draftTokens *mlx.Array) *mlx.Array {
	slot, logits, draftTokens := s.speculativeInputs("SpeculativeScores", seqID, logits, draftTokens)
	rows := logits.Dim(0)

	var hist *mlx.Array
	if slot.opts.usesHistory() {
		if s.history == nil {
			panic(fmt.Sprintf("sample.Sampler.SpeculativeScores: seqID %d has no history", seqID))
		}
		if slot.historyLen < slot.opts.RepeatLastN {
			return s.speculativeScoresSerial(slot, logits, draftTokens)
		}
		hist = s.speculativeHistory(slot, draftTokens, rows)
	}

	return slot.speculativeScores(&slotCtx{opts: slot.opts, history: hist}, logits)
}

// SampleDistribution draws from a precomputed distribution while advancing
// seqID's deterministic RNG stream when a seed is configured.
func (s *Sampler) SampleDistribution(seqID int, dist Distribution) *mlx.Array {
	slot := s.mustSlot("SampleDistribution", seqID)
	return dist.SampleWithKey(slot.nextRandomKey())
}

// Bernoulli samples boolean outcomes while advancing seqID's deterministic RNG
// stream when a seed is configured.
func (s *Sampler) Bernoulli(seqID int, p *mlx.Array) *mlx.Array {
	slot := s.mustSlot("Bernoulli", seqID)
	return mlx.BernoulliWithKey(p, slot.nextRandomKey())
}

func (s *Sampler) mustSlot(caller string, seqID int) *slotState {
	slot, ok := s.byID[seqID]
	if !ok {
		panic(fmt.Sprintf("sample.Sampler.%s: seqID %d not registered", caller, seqID))
	}
	return slot
}

func (s *Sampler) speculativeInputs(caller string, seqID int, logits *mlx.Array, draftTokens *mlx.Array) (*slotState, *mlx.Array, *mlx.Array) {
	slot := s.mustSlot(caller, seqID)

	if logits.NumDims() == 3 {
		if logits.Dim(0) != 1 {
			panic(fmt.Sprintf("sample.Sampler.%s: only batch size 1 is supported", caller))
		}
		logits = logits.Squeeze(0)
	}
	if logits.NumDims() != 2 {
		panic(fmt.Sprintf("sample.Sampler.%s: logits must be rank 2 or 3, got rank %d", caller, logits.NumDims()))
	}

	if draftTokens != nil && draftTokens.NumDims() == 1 {
		draftTokens = draftTokens.ExpandDims(0)
	}
	return slot, logits, draftTokens
}

// Commit appends already-selected tokens to seqID's repeat-penalty history.
// It is used after speculative sampling once the accepted continuation is
// known. Normal Sample calls continue to mutate history themselves.
func (s *Sampler) Commit(seqID int, tokens []int32) {
	if len(tokens) == 0 {
		return
	}
	slot, ok := s.byID[seqID]
	if !ok {
		panic(fmt.Sprintf("sample.Sampler.Commit: seqID %d not registered", seqID))
	}
	if !slot.opts.usesHistory() {
		return
	}
	if s.history == nil {
		panic(fmt.Sprintf("sample.Sampler.Commit: seqID %d has no history", seqID))
	}

	row := slices.Index(s.slots, slot)
	width := s.historyWidth()
	take := min(len(tokens), slot.opts.RepeatLastN)
	startLen := slot.historyLen + len(tokens) - take
	writeTokens := tokens[len(tokens)-take:]
	flatOffsets := make([]int32, take)
	for i := range take {
		ringPos := (startLen + i) % slot.opts.RepeatLastN
		flatOffsets[i] = int32(row*width + ringPos)
	}

	flatIdx := mlx.NewArrayInt32(flatOffsets, []int32{int32(take), 1})
	values := mlx.NewArrayInt32(writeTokens, []int32{int32(take), 1})
	flatHist := s.history.Reshape(s.history.Dim(0)*width, 1)
	s.history.Set(flatHist.PutAlongAxis(flatIdx, values, 0).Reshape(s.history.Dim(0), width))
	slot.historyLen += len(tokens)
}

func (s *Sampler) speculativeDistributionSerial(slot *slotState, logits *mlx.Array, draftTokens *mlx.Array) Distribution {
	rows := logits.Dim(0)
	draftCount := 0
	if draftTokens != nil {
		draftCount = draftTokens.Dim(1)
	}
	row := slices.Index(s.slots, slot)
	baseFill := min(slot.historyLen, slot.opts.RepeatLastN)
	var base *mlx.Array
	if baseFill > 0 {
		base = s.history.Slice(mlx.Slice(row, row+1), mlx.Slice(0, baseFill))
	}

	dists := make([]Distribution, 0, rows)
	for i := range rows {
		rowLogits := logits.Slice(mlx.Slice(i, i+1), mlx.Slice())
		hist := base
		prefixLen := min(i, draftCount)
		if prefixLen > 0 {
			prefix := draftTokens.Slice(mlx.Slice(), mlx.Slice(0, prefixLen))
			if hist == nil {
				hist = prefix
			} else {
				hist = hist.Concatenate(1, prefix)
			}
			if hist.Dim(1) > slot.opts.RepeatLastN {
				hist = hist.Slice(mlx.Slice(), mlx.Slice(hist.Dim(1)-slot.opts.RepeatLastN, mlx.End))
			}
		}
		dists = append(dists, slot.distribution(&slotCtx{opts: slot.opts, history: hist}, rowLogits))
	}
	return ConcatenateDistributions(dists)
}

func (s *Sampler) speculativeScoresSerial(slot *slotState, logits *mlx.Array, draftTokens *mlx.Array) *mlx.Array {
	return s.speculativeDistributionSerial(slot, logits, draftTokens).LogProbs(logits.Dim(logits.NumDims() - 1))
}

func (s *Sampler) speculativeHistory(slot *slotState, draftTokens *mlx.Array, rows int) *mlx.Array {
	row := slices.Index(s.slots, slot)
	width := slot.opts.RepeatLastN
	base := s.history.Slice(mlx.Slice(row, row+1), mlx.Slice(0, width))
	base = mlx.Tile(base, []int32{int32(rows), 1})
	next := slot.historyLen % width
	draftCount := 0
	if draftTokens != nil {
		draftCount = draftTokens.Dim(1)
	}
	if draftCount == 0 {
		return base
	}

	sourceIdx := make([]int32, rows*width)
	writeMask := make([]bool, rows*width)
	for i := range rows {
		prefixLen := min(i, draftCount)
		for j := range prefixLen {
			pos := (next + j) % width
			sourceIdx[i*width+pos] = int32(j)
			writeMask[i*width+pos] = true
		}
	}

	draftRows := mlx.Tile(draftTokens, []int32{int32(rows), 1})
	idx := mlx.NewArrayInt32(sourceIdx, []int32{int32(rows), int32(width)})
	mask := mlx.FromValues(writeMask, rows, width)
	values := draftRows.TakeAlongAxis(idx, 1)
	return mlx.Where(mask, values, base)
}

func (slot *slotState) speculativeScores(ctx *slotCtx, logits *mlx.Array) *mlx.Array {
	if slot.opts.Temperature == 0 {
		return slot.baseScores(ctx, logits)
	}
	return slot.distribution(ctx, logits).LogProbs(logits.Dim(logits.NumDims() - 1))
}

// canBatch reports whether the call can take the uniform batched path.
// All slots must share Options; when penalties are active the call must
// additionally cover every registered slot in registration order with a
// full ring, because the uniform path indexes the pool positionally.
func (s *Sampler) canBatch(slots []*slotState) (Options, bool) {
	if !s.allSameOpts {
		return Options{}, false
	}
	// slots is non-empty (Sample guards) and every slot is registered,
	// so s.slots[0].opts is the canonical shared value.
	shared := s.slots[0].opts
	// TODO(pdevine): Before using multi-slot batching with seeded stochastic sampling,
	// make sure each row gets its own per-slot random key instead of sharing
	// slots[0]'s key through one batched categorical op.
	if !shared.usesHistory() {
		return shared, true
	}
	if len(slots) != len(s.slots) {
		return Options{}, false
	}
	for i, slot := range slots {
		if s.slots[i] != slot || slot.historyLen < shared.RepeatLastN {
			return Options{}, false
		}
	}
	return shared, true
}

// sampleTokensUniform runs one fused sampling pass over the whole batch.
// Reached only when canBatch is true, which lets the pool be used in place
// with a single PutAlongAxis write-back and no gather.
func (s *Sampler) sampleTokensUniform(slots []*slotState, opts Options, logits *mlx.Array) *mlx.Array {
	B := len(slots)

	var hist *mlx.Array
	if opts.usesHistory() {
		hist = s.history
		if s.historyWidth() > opts.RepeatLastN {
			hist = hist.Slice(mlx.Slice(), mlx.Slice(0, opts.RepeatLastN))
		}
	}

	ctx := &slotCtx{opts: opts, history: hist}
	token := slots[0].sample(ctx, logits)
	if opts.UseSeed && opts.Temperature != 0 {
		// TODO: This only keeps counters aligned; it does not give each slot
		// an independent key for the batched draw.
		for _, slot := range slots[1:] {
			slot.randomCounter++
		}
	}

	if !opts.usesHistory() {
		return token
	}

	writeIdxData := make([]int32, B)
	for i, slot := range slots {
		writeIdxData[i] = int32(slot.historyLen % opts.RepeatLastN)
		slot.historyLen++
	}
	writeIdx := mlx.NewArrayInt32(writeIdxData, []int32{int32(B), 1})

	s.history.Set(s.history.PutAlongAxis(writeIdx, token.ExpandDims(-1), 1))
	return token
}

// sampleTokensSerial samples each slot against its own row of logits.
func (s *Sampler) sampleTokensSerial(slots []*slotState, logits *mlx.Array) *mlx.Array {
	perSlotTokens := make([]*mlx.Array, len(slots))

	rowOf := make(map[*slotState]int, len(s.slots))
	for i, slot := range s.slots {
		rowOf[slot] = i
	}

	for i, slot := range slots {
		row := logits.Slice(mlx.Slice(i, i+1), mlx.Slice())

		var hist *mlx.Array
		if slot.opts.usesHistory() && slot.historyLen > 0 && s.history != nil {
			poolRow := rowOf[slot]
			fill := min(slot.historyLen, slot.opts.RepeatLastN)
			hist = s.history.Slice(
				mlx.Slice(poolRow, poolRow+1),
				mlx.Slice(0, fill),
			)
		}

		ctx := &slotCtx{opts: slot.opts, history: hist}
		perSlotTokens[i] = slot.sample(ctx, row)
	}

	token := mlx.Concatenate(perSlotTokens, 0)

	if s.history != nil {
		// For each writing slot collect its flat (row-major) pool offset
		// and the call-order position of its token. One PutAlongAxis on a
		// flat view of the pool scatters all writes in a single op.
		flatOffsets := make([]int32, 0, len(slots))
		tokenPos := make([]int32, 0, len(slots))
		for i, slot := range slots {
			if !slot.opts.usesHistory() {
				continue
			}
			ringPos := slot.historyLen % slot.opts.RepeatLastN
			flatOffsets = append(flatOffsets, int32(rowOf[slot]*s.historyWidth()+ringPos))
			tokenPos = append(tokenPos, int32(i))
			slot.historyLen++
		}

		if len(flatOffsets) > 0 {
			m := len(flatOffsets)
			flatIdx := mlx.NewArrayInt32(flatOffsets, []int32{int32(m), 1})
			writingTokens := token
			if m != len(slots) {
				tokenPosIdx := mlx.NewArrayInt32(tokenPos, []int32{int32(m)})
				writingTokens = token.TakeAxis(tokenPosIdx, 0)
			}
			flatHist := s.history.Reshape(s.history.Dim(0)*s.historyWidth(), 1)
			s.history.Set(flatHist.PutAlongAxis(flatIdx, writingTokens.ExpandDims(-1), 0).Reshape(s.history.Dim(0), s.historyWidth()))
		}
	}
	return token
}

func (slot *slotState) sample(ctx *slotCtx, logits *mlx.Array) *mlx.Array {
	if slot.opts.Temperature == 0 {
		return slot.baseScores(ctx, logits).Argmax(-1, false).AsType(mlx.DTypeInt32)
	}
	return slot.distribution(ctx, logits).SampleWithKey(slot.nextRandomKey())
}

func (slot *slotState) nextRandomKey() *mlx.Array {
	if !slot.opts.UseSeed {
		return nil
	}
	seed := mixSeed(uint64(slot.opts.Seed), slot.randomCounter)
	slot.randomCounter++
	return mlx.RandomKey(seed)
}

const (
	// SplitMix64 constants used to decorrelate nearby (seed, counter) pairs.
	splitMix64Weyl       = 0x9e3779b97f4a7c15
	splitMix64Mul1       = 0xbf58476d1ce4e5b9
	splitMix64Mul2       = 0x94d049bb133111eb
	splitMix64Shift1     = 30
	splitMix64Shift2     = 27
	splitMix64FinalShift = 31
)

func mixSeed(seed, counter uint64) uint64 {
	z := seed + splitMix64Weyl*(counter+1)
	z = (z ^ (z >> splitMix64Shift1)) * splitMix64Mul1
	z = (z ^ (z >> splitMix64Shift2)) * splitMix64Mul2
	return z ^ (z >> splitMix64FinalShift)
}

func (slot *slotState) baseScores(ctx *slotCtx, logits *mlx.Array) *mlx.Array {
	scores := logits
	if slot.opts.usesHistory() {
		scores = penalty(ctx, scores)
	}
	return scores
}

func (slot *slotState) distribution(ctx *slotCtx, logits *mlx.Array) Distribution {
	scores := slot.baseScores(ctx, logits)
	if slot.opts.Temperature <= 0 {
		ids := scores.Argmax(-1, false).AsType(mlx.DTypeInt32).ExpandDims(-1)
		probs := mlx.AddScalar(ids.AsType(mlx.DTypeFloat32).Multiply(mlx.FromValue(float32(0))), 1)
		return Distribution{IDs: ids, Probs: probs}
	}

	vocab := scores.Dim(scores.NumDims() - 1)
	if slot.opts.TopK > 0 && slot.opts.TopK < vocab {
		return sparseDistribution(ctx.opts, scores)
	}
	return denseDistribution(ctx.opts, scores)
}

func sparseDistribution(opts Options, scores *mlx.Array) Distribution {
	ids := scores.Negative().ArgpartitionAxis(opts.TopK-1, -1).Slice(mlx.Slice(), mlx.Slice(0, opts.TopK)).AsType(mlx.DTypeInt32)
	topScores := scores.TakeAlongAxis(ids, -1).AsType(mlx.DTypeFloat32)
	probs := mlx.SoftmaxAxis(mlx.DivScalar(topScores, opts.Temperature), -1, true)
	probs = applyTopPProbs(probs, opts.TopP)
	probs = applyMinPProbs(probs, opts.MinP)
	return Distribution{IDs: ids, Probs: normalizeProbs(probs)}
}

func denseDistribution(opts Options, scores *mlx.Array) Distribution {
	probs := mlx.SoftmaxAxis(mlx.DivScalar(scores.AsType(mlx.DTypeFloat32), opts.Temperature), -1, true)
	probs = applyTopPProbs(probs, opts.TopP)
	probs = applyMinPProbs(probs, opts.MinP)
	return Distribution{Probs: normalizeProbs(probs)}
}

func applyTopPProbs(probs *mlx.Array, topP float32) *mlx.Array {
	if topP <= 0 || topP >= 1 {
		return probs
	}
	order := probs.Negative().ArgsortAxis(-1)
	sorted := probs.TakeAlongAxis(order, -1)
	prevCumProbs := sorted.Cumsum(-1, false, true).Subtract(sorted)
	keep := prevCumProbs.Less(mlx.FromValue(topP))
	filtered := mlx.Where(keep, sorted, mlx.FromValue(float32(0)))
	return mlx.Zeros(probs.DType(), probs.Dims()...).PutAlongAxis(order, filtered, -1)
}

func applyMinPProbs(probs *mlx.Array, minP float32) *mlx.Array {
	if minP <= 0 || minP > 1 {
		return probs
	}
	threshold := mlx.MulScalar(probs.MaxAxis(-1, true), minP)
	return mlx.Where(probs.Less(threshold), mlx.FromValue(float32(0)), probs)
}

func normalizeProbs(probs *mlx.Array) *mlx.Array {
	sum := mlx.Maximum(probs.SumAxis(-1, true), mlx.FromValue(float32(1e-20)))
	return probs.Divide(sum)
}

func logitsFromProbs(probs *mlx.Array) *mlx.Array {
	positive := mlx.Maximum(probs, mlx.FromValue(float32(1e-20)))
	logits := mlx.Log(positive)
	return mlx.Where(probs.LessEqual(mlx.FromValue(float32(0))), mlx.FromValue(float32(math.Inf(-1))), logits)
}

func penalty(ctx *slotCtx, scores *mlx.Array) *mlx.Array {
	tokenIndices := ctx.history
	if tokenIndices == nil {
		return scores
	}

	if ctx.opts.RepeatPenalty != 1 || ctx.opts.PresencePenalty != 0 {
		adjusted := scores.TakeAlongAxis(tokenIndices, -1)
		if ctx.opts.RepeatPenalty != 1 {
			factor := mlx.Where(
				adjusted.Less(mlx.FromValue(float32(0))),
				mlx.FromValue(ctx.opts.RepeatPenalty),
				mlx.FromValue(1/ctx.opts.RepeatPenalty),
			)
			adjusted = adjusted.Multiply(factor)
		}
		if ctx.opts.PresencePenalty != 0 {
			adjusted = mlx.AddScalar(adjusted, -ctx.opts.PresencePenalty)
		}
		scores = scores.PutAlongAxis(tokenIndices, adjusted, -1)
	}

	if ctx.opts.FrequencyPenalty != 0 {
		scores = scores.ScatterAddAxis(tokenIndices, mlx.FromValue(-ctx.opts.FrequencyPenalty), -1)
	}

	return scores
}
