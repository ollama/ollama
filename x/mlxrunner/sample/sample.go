//go:build mlx

package sample

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Sampler interface {
	Sample(*mlx.Array, []int32) *mlx.Array
}

func New(temp, top_p, min_p float32, top_k, repeatLastN int, repeatPenalty, presencePenalty, frequencyPenalty float32) Sampler {
	var samplers []Sampler
	if repeatLastN > 0 && (repeatPenalty != 1 || presencePenalty != 0 || frequencyPenalty != 0) {
		samplers = append(samplers, Penalty{
			RepeatLastN:      repeatLastN,
			RepeatPenalty:    repeatPenalty,
			PresencePenalty:  presencePenalty,
			FrequencyPenalty: frequencyPenalty,
		})
	}

	if temp == 0 {
		samplers = append(samplers, greedy{})
	} else {
		samplers = append(samplers, Distribution{
			Temperature: temp,
			TopK:        top_k,
			TopP:        top_p,
			MinP:        min_p,
		})
	}
	return chain(samplers)
}

type greedy struct{}

func (greedy) Sample(logits *mlx.Array, _ []int32) *mlx.Array {
	return logits.Argmax(-1, false)
}

type chain []Sampler

func (c chain) Sample(logits *mlx.Array, history []int32) *mlx.Array {
	for _, sampler := range c {
		logits = sampler.Sample(logits, history)
	}
	return logits
}

type Distribution struct {
	Temperature float32
	TopK        int
	TopP        float32
	MinP        float32
}

func (d Distribution) Sample(logits *mlx.Array, _ []int32) *mlx.Array {
	filtered, indices := d.filter(logits)
	sample := filtered.Categorical(-1)
	if indices == nil {
		return sample
	}

	positions := sample.ExpandDims(1)
	return indices.TakeAlongAxis(positions, -1).Squeeze(1)
}

func (d Distribution) filter(logits *mlx.Array) (*mlx.Array, *mlx.Array) {
	candidates := logits
	var candidateIndices *mlx.Array

	if d.TopK > 0 && d.TopK < logits.Dim(logits.NumDims()-1) {
		partitions := logits.Negative().ArgpartitionAxis(d.TopK-1, -1)
		switch logits.NumDims() {
		case 1:
			candidateIndices = partitions.Slice(mlx.Slice(0, d.TopK))
		default:
			candidateIndices = partitions.Slice(mlx.Slice(), mlx.Slice(0, d.TopK))
		}
		candidates = logits.TakeAlongAxis(candidateIndices, -1)
	}

	if d.Temperature != 1 {
		candidates = mlx.DivScalar(candidates, d.Temperature)
	}

	if !d.needsProbabilityFilters() {
		return candidates, candidateIndices
	}

	order := candidates.Negative().ArgsortAxis(-1)
	sortedLogits := candidates.TakeAlongAxis(order, -1)
	sortedProbs := mlx.SoftmaxAxis(candidates, -1, true).TakeAlongAxis(order, -1)

	remove := d.topPRemovalMask(sortedProbs)
	if d.MinP > 0 {
		minPRemove := d.minPRemovalMask(sortedProbs)
		if remove == nil {
			remove = minPRemove
		} else {
			remove = remove.LogicalOr(minPRemove)
		}
	}

	if remove == nil {
		return candidates, candidateIndices
	}

	negInf := mlx.FromValue(float32(math.Inf(-1)))
	filtered := mlx.Where(remove, negInf, sortedLogits)
	return candidates.PutAlongAxis(order, filtered, -1), candidateIndices
}

func (d Distribution) needsProbabilityFilters() bool {
	return (d.TopP > 0 && d.TopP < 1) || d.MinP > 0
}

func (d Distribution) topPRemovalMask(sortedProbs *mlx.Array) *mlx.Array {
	if d.TopP <= 0 || d.TopP >= 1 {
		return nil
	}

	threshold := mlx.NewScalarArray(d.TopP)
	prevCum := sortedProbs.Cumsum(-1, false, true).Subtract(sortedProbs)
	return prevCum.GreaterEqual(threshold)
}

func (d Distribution) minPRemovalMask(sortedProbs *mlx.Array) *mlx.Array {
	if d.MinP <= 0 {
		return nil
	}

	var maxProb *mlx.Array
	switch sortedProbs.NumDims() {
	case 1:
		maxProb = sortedProbs.Slice(mlx.Slice(0, 1))
	default:
		maxProb = sortedProbs.Slice(mlx.Slice(), mlx.Slice(0, 1))
	}

	threshold := mlx.MulScalar(maxProb, d.MinP)
	return sortedProbs.Less(threshold)
}

type Penalty struct {
	RepeatLastN      int
	RepeatPenalty    float32
	PresencePenalty  float32
	FrequencyPenalty float32
}

func (p Penalty) Sample(logprobs *mlx.Array, history []int32) *mlx.Array {
	if len(history) == 0 {
		return logprobs
	}

	window := p.RepeatLastN
	if window <= 0 || window > len(history) {
		window = len(history)
	}

	counts := make(map[int32]int, window)
	order := make([]int32, 0, window)
	for _, token := range history[len(history)-window:] {
		if token < 0 {
			continue
		}
		if counts[token] == 0 {
			order = append(order, token)
		}
		counts[token]++
	}
	if len(order) == 0 {
		return logprobs
	}

	indexShape := []int32{int32(len(order))}
	valueShape := []int{len(order)}
	if logprobs.NumDims() > 1 {
		indexShape = []int32{1, int32(len(order))}
		valueShape = []int{1, len(order)}
	}

	indices := mlx.NewArrayInt32(order, indexShape)
	selected := logprobs.TakeAlongAxis(indices, -1)
	mlx.Eval(selected)

	values := selected.Floats()
	for i, token := range order {
		v := values[i]
		if p.RepeatPenalty != 1 {
			if v < 0 {
				v *= p.RepeatPenalty
			} else {
				v /= p.RepeatPenalty
			}
		}
		if p.PresencePenalty != 0 {
			v -= p.PresencePenalty
		}
		if p.FrequencyPenalty != 0 {
			v -= p.FrequencyPenalty * float32(counts[token])
		}
		values[i] = v
	}

	return logprobs.PutAlongAxis(indices, mlx.FromValues(values, valueShape...), -1)
}
