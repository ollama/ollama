package sample

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Transform func(*Sampler, *mlx.Array) *mlx.Array

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

type Sampler struct {
	Options

	history    *mlx.Array
	historyLen int
	transforms []Transform
}

// Result bundles the outputs of one decode step. The logprob tensors are
// populated only when the sampler is configured to report them.
type Result struct {
	Token       *mlx.Array // sampled token id, shape [B]
	Logprob     *mlx.Array // sampled-token logprob, shape [B,1]; nil unless Logprobs
	TopTokens   *mlx.Array // top-K token ids, shape [B,K]; nil unless TopLogprobs > 0
	TopLogprobs *mlx.Array // top-K logprobs, shape [B,K]; nil unless TopLogprobs > 0
}

// Arrays returns the tensor fields as a slice so callers can drive the mlx
// lifecycle verbs (Pin, Unpin, Eval, AsyncEval) over the whole group. Unset
// fields stay nil; the mlx helpers skip them.
func (r Result) Arrays() []*mlx.Array {
	return []*mlx.Array{r.Token, r.Logprob, r.TopTokens, r.TopLogprobs}
}

func New(opts Options) *Sampler {
	if opts.RepeatPenalty <= 0 {
		opts.RepeatPenalty = 1
	}

	s := &Sampler{Options: opts}

	var transforms []Transform
	if s.usesHistory() {
		transforms = append(transforms, penalty)
	}

	hasTopP := opts.TopP > 0 && opts.TopP < 1
	hasTopK := opts.TopK > 0
	switch {
	case hasTopP:
		// topKTopP always does a full descending sort for the top-P
		// cumulative mask and opportunistically masks top-K during the
		// same pass when it is also configured.
		transforms = append(transforms, topKTopP)
	case hasTopK:
		// Argpartition (partial sort) is cheaper than a full sort.
		transforms = append(transforms, topK)
	}

	if opts.MinP != 0 {
		transforms = append(transforms, minP)
	}

	if opts.Temperature == 0 {
		transforms = append(transforms, greedy)
	} else {
		transforms = append(transforms, temperature)
	}

	s.transforms = transforms
	return s
}

func (s *Sampler) usesHistory() bool {
	return s.RepeatPenalty != 1 || s.PresencePenalty != 0 || s.FrequencyPenalty != 0
}

func (s *Sampler) setHistory(history *mlx.Array, historyLen int) {
	if history != nil {
		mlx.Pin(history)
	}
	if s.history != nil {
		mlx.Unpin(s.history)
	}
	s.history = history
	s.historyLen = historyLen
}

func (s *Sampler) ResetHistory(history []int32) {
	if !s.usesHistory() {
		return
	}
	if s.RepeatLastN > 0 && len(history) > s.RepeatLastN {
		history = history[len(history)-s.RepeatLastN:]
	}
	if len(history) == 0 {
		s.setHistory(nil, 0)
		return
	}

	tokens := append([]int32(nil), history...)
	s.setHistory(mlx.NewArrayInt32(tokens, []int32{int32(len(tokens))}), len(tokens))
}

func (s *Sampler) AppendToken(token *mlx.Array) {
	if !s.usesHistory() || token == nil {
		return
	}

	next := token.AsType(mlx.DTypeInt32)
	nextLen := next.Size()

	if s.history != nil && s.historyLen > 0 {
		next = s.history.Concatenate(0, next)
		nextLen += s.historyLen
	}

	if s.RepeatLastN > 0 && nextLen > s.RepeatLastN {
		trim := nextLen - s.RepeatLastN
		next = next.Slice(mlx.Slice(trim, nextLen))
		nextLen = s.RepeatLastN
	}

	s.setHistory(next, nextLen)
}

func (s *Sampler) Free() {
	s.setHistory(nil, 0)
}

// Sample runs the configured transform chain on the raw per-token logits
// and returns the sampled token id plus, when configured, the reported
// log-probability tensors for the selected token and the top-K tokens.
func (s *Sampler) Sample(logits *mlx.Array) Result {
	scores := logits
	for _, transform := range s.transforms {
		scores = transform(s, scores)
	}
	res := Result{Token: scores}

	if s.Logprobs {
		// Compute log_softmax in fp32 and subtract the max before
		// logsumexp so the final subtraction stays on small values.
		// Otherwise it cancels two large numbers and loses precision.
		lp := logits.AsType(mlx.DTypeFloat32)
		lp = lp.Subtract(lp.MaxAxis(-1, true))
		lp = lp.Subtract(lp.Logsumexp(true))
		res.Logprob = lp.TakeAlongAxis(res.Token.ExpandDims(-1), -1)
		if k := s.TopLogprobs; k > 0 {
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

func greedy(_ *Sampler, scores *mlx.Array) *mlx.Array {
	return scores.Argmax(-1, false)
}

func temperature(s *Sampler, scores *mlx.Array) *mlx.Array {
	return mlx.DivScalar(scores, s.Temperature).Categorical(-1)
}

// topKTopP applies top-P in a descending sort pass and, when top-K is also
// configured, masks any surviving value below the K-th largest in the same
// pass. Callers dispatch here whenever top-P is enabled — the top-K-only
// case uses a cheaper partial sort via the topK transform.
func topKTopP(s *Sampler, scores *mlx.Array) *mlx.Array {
	vocab := scores.Dim(scores.NumDims() - 1)
	applyTopK := s.TopK > 0 && s.TopK < vocab

	order := scores.Negative().ArgsortAxis(-1)
	sorted := scores.TakeAlongAxis(order, -1)
	negInf := mlx.FromValue(float32(math.Inf(-1)))

	// Top-P: in descending order, keep tokens whose exclusive cumulative
	// probability is still below s.TopP.
	probs := mlx.SoftmaxAxis(sorted, -1, true)
	prevCumProbs := probs.Cumsum(-1, false, true).Subtract(probs)
	keep := prevCumProbs.Less(mlx.FromValue(s.TopP))
	sorted = mlx.Where(keep, sorted, negInf)

	out := scores.PutAlongAxis(order, sorted, -1)

	// Top-K: sorted is already in descending order, so positions [K, V)
	// are the ones to drop. Scatter -inf through their original-layout
	// indices (order[K:]). Positional (not value-based) so exactly K
	// tokens survive — ties at the K-th logit get broken by the sort
	// order rather than promoted through the filter.
	if applyTopK {
		dropOrder := order.Slice(mlx.Slice(), mlx.Slice(s.TopK, mlx.End))
		out = out.PutAlongAxis(dropOrder, negInf, -1)
	}

	return out
}

func minP(s *Sampler, scores *mlx.Array) *mlx.Array {
	if s.MinP <= 0 || s.MinP > 1 {
		return scores
	}

	maxScore := scores.MaxAxis(-1, true)
	threshold := mlx.AddScalar(maxScore, float32(math.Log(float64(s.MinP))))

	return mlx.Where(
		scores.Less(threshold),
		mlx.FromValue(float32(math.Inf(-1))),
		scores,
	)
}

func topK(s *Sampler, scores *mlx.Array) *mlx.Array {
	if s.TopK <= 0 {
		return scores
	}

	vocab := scores.Dim(scores.NumDims() - 1)
	if s.TopK >= vocab {
		return scores
	}

	mask := scores.Negative().ArgpartitionAxis(s.TopK-1, -1).Slice(mlx.Slice(), mlx.Slice(s.TopK, mlx.End))
	return scores.PutAlongAxis(mask, mlx.FromValue(float32(math.Inf(-1))), -1)
}

func penalty(s *Sampler, scores *mlx.Array) *mlx.Array {
	if s.historyLen == 0 {
		return scores
	}

	tokenIndices := s.history
	if scores.NumDims() > 1 {
		tokenIndices = tokenIndices.ExpandDims(0)
	}

	if s.RepeatPenalty != 1 || s.PresencePenalty != 0 {
		adjusted := scores.TakeAlongAxis(tokenIndices, -1)
		if s.RepeatPenalty != 1 {
			factor := mlx.Where(
				adjusted.Less(mlx.FromValue(float32(0))),
				mlx.FromValue(s.RepeatPenalty),
				mlx.FromValue(1/s.RepeatPenalty),
			)
			adjusted = adjusted.Multiply(factor)
		}
		if s.PresencePenalty != 0 {
			adjusted = mlx.AddScalar(adjusted, -s.PresencePenalty)
		}
		scores = scores.PutAlongAxis(tokenIndices, adjusted, -1)
	}

	if s.FrequencyPenalty != 0 {
		scores = scores.ScatterAddAxis(tokenIndices, mlx.FromValue(-s.FrequencyPenalty), -1)
	}

	return scores
}
