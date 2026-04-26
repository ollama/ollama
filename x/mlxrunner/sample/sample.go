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
	opts       Options
	transforms []transform
	historyLen int
}

type slotCtx struct {
	opts    Options
	history *mlx.Array // 2D [B, W] when penalties are configured; nil otherwise
}

type transform func(*slotCtx, *mlx.Array) *mlx.Array

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
	return o
}

func (o Options) buildTransforms() []transform {
	var ts []transform
	if o.usesHistory() {
		ts = append(ts, penalty)
	}

	hasTopP := o.TopP > 0 && o.TopP < 1
	hasTopK := o.TopK > 0
	switch {
	case hasTopP:
		// topKTopP always does a full descending sort for the top-P
		// cumulative mask and opportunistically masks top-K during the
		// same pass when it is also configured.
		ts = append(ts, topKTopP)
	case hasTopK:
		// Argpartition (partial sort) is cheaper than a full sort.
		ts = append(ts, topK)
	}

	if o.MinP != 0 {
		ts = append(ts, minP)
	}

	if o.Temperature == 0 {
		ts = append(ts, greedy)
	} else {
		ts = append(ts, temperature)
	}
	return ts
}

// Add registers a sequence under seqID. The last RepeatLastN entries of
// priorTokens seed the ring buffer.
func (s *Sampler) Add(seqID int, opts Options, priorTokens []int32) {
	if _, dup := s.byID[seqID]; dup {
		panic(fmt.Sprintf("sample.Sampler.Add: seqID %d already registered", seqID))
	}

	opts = opts.normalize(s.numCtx)
	slot := &slotState{
		opts:       opts,
		transforms: opts.buildTransforms(),
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

// sampleTokensUniform runs one fused transform pass over the whole batch.
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
	scores := logits
	for _, t := range slots[0].transforms {
		scores = t(ctx, scores)
	}
	token := scores

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

// sampleTokensSerial runs each slot's transforms against its own row of
// logits.
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
		scores := row
		for _, t := range slot.transforms {
			scores = t(ctx, scores)
		}
		perSlotTokens[i] = scores
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

func greedy(_ *slotCtx, scores *mlx.Array) *mlx.Array {
	return scores.Argmax(-1, false).AsType(mlx.DTypeInt32)
}

func temperature(ctx *slotCtx, scores *mlx.Array) *mlx.Array {
	return mlx.DivScalar(scores, ctx.opts.Temperature).Categorical(-1).AsType(mlx.DTypeInt32)
}

// topKTopP applies top-P in a descending sort pass and, when top-K is also
// configured, masks any surviving value below the K-th largest in the same
// pass. Callers dispatch here whenever top-P is enabled — the top-K-only case
// uses a cheaper partial sort via the topK transform.
func topKTopP(ctx *slotCtx, scores *mlx.Array) *mlx.Array {
	vocab := scores.Dim(scores.NumDims() - 1)
	applyTopK := ctx.opts.TopK > 0 && ctx.opts.TopK < vocab

	order := scores.Negative().ArgsortAxis(-1)
	sorted := scores.TakeAlongAxis(order, -1)
	negInf := mlx.FromValue(float32(math.Inf(-1)))

	// Top-P: in descending order, keep tokens whose exclusive cumulative
	// probability is still below TopP.
	probs := mlx.SoftmaxAxis(sorted, -1, true)
	prevCumProbs := probs.Cumsum(-1, false, true).Subtract(probs)
	keep := prevCumProbs.Less(mlx.FromValue(ctx.opts.TopP))
	sorted = mlx.Where(keep, sorted, negInf)

	out := scores.PutAlongAxis(order, sorted, -1)

	// Top-K: sorted is already in descending order, so positions [K, V) are
	// the ones to drop. Scatter -inf through their original-layout indices
	// (order[K:]). Positional (not value-based) so exactly K tokens survive —
	// ties at the K-th logit get broken by the sort order rather than
	// promoted through the filter.
	if applyTopK {
		dropOrder := order.Slice(mlx.Slice(), mlx.Slice(ctx.opts.TopK, mlx.End))
		out = out.PutAlongAxis(dropOrder, negInf, -1)
	}

	return out
}

func minP(ctx *slotCtx, scores *mlx.Array) *mlx.Array {
	if ctx.opts.MinP <= 0 || ctx.opts.MinP > 1 {
		return scores
	}

	maxScore := scores.MaxAxis(-1, true)
	threshold := mlx.AddScalar(maxScore, float32(math.Log(float64(ctx.opts.MinP))))

	return mlx.Where(
		scores.Less(threshold),
		mlx.FromValue(float32(math.Inf(-1))),
		scores,
	)
}

func topK(ctx *slotCtx, scores *mlx.Array) *mlx.Array {
	if ctx.opts.TopK <= 0 {
		return scores
	}
	vocab := scores.Dim(scores.NumDims() - 1)
	if ctx.opts.TopK >= vocab {
		return scores
	}

	mask := scores.Negative().ArgpartitionAxis(ctx.opts.TopK-1, -1).Slice(mlx.Slice(), mlx.Slice(ctx.opts.TopK, mlx.End))
	return scores.PutAlongAxis(mask, mlx.FromValue(float32(math.Inf(-1))), -1)
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
