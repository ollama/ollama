package sample

import (
	"cmp"
	"errors"
	"math"
	"slices"

	pq "github.com/emirpasic/gods/v2/queues/priorityqueue"
	"gonum.org/v1/gonum/floats"
)

type Transform interface {
	Apply([]float64) ([]float64, error)
}

// TODO(parthsareen): potentially cache softmax values
func softmax(logits []float64) []float64 {
	var sum float64
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(v)
		sum += probs[i]
	}
	floats.Scale(1/sum, probs)
	return probs
}

type Temperature float64

func (t Temperature) Apply(logits []float64) ([]float64, error) {
	if t == 0 {
		return nil, errors.New("use Greedy sampler instead of Temperature(0)")
	}
	if t < 0 || t > 2 {
		return nil, errors.New("temperature must be between 0 and 2")
	}
	temp := math.Max(float64(t), 1e-7)

	// subtracting max logit to avoid under/overflow
	maxLogit := slices.Max(logits)
	for i := range logits {
		logits[i] = (logits[i] - maxLogit) / temp
	}

	return logits, nil
}

type logitMap struct {
	index int
	logit float64
}

func logitMapComparator(a, b logitMap) int {
	return -cmp.Compare(a.logit, b.logit)
}

type TopK int

// TODO(parthsareen): avoid having to check all logits after this transform
func (k TopK) Apply(logits []float64) ([]float64, error) {
	if k <= 0 {
		return nil, errors.New("k must be greater than 0")
	}
	if int(k) >= len(logits) {
		return logits, nil
	}

	q := pq.NewWith(logitMapComparator)
	for i, logit := range logits {
		q.Enqueue(logitMap{index: i, logit: logit})
	}

	validLogits := make(map[int]float64)
	for range k {
		logitMap, _ := q.Dequeue()
		validLogits[logitMap.index] = logitMap.logit
	}

	for i := range logits {
		if _, ok := validLogits[i]; !ok {
			logits[i] = math.Inf(-1)
		}
	}

	return logits, nil
}

type TopP float64

func (p TopP) Apply(logits []float64) ([]float64, error) {
	if p <= 0 || p >= 1 {
		return nil, errors.New("p must be between 0 and 1")
	}

	probs := softmax(logits)
	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}

	// sort in descending order
	slices.SortFunc(indices, func(i, j int) int {
		return cmp.Compare(probs[j], probs[i])
	})

	var cumSum float64
	for i, idx := range indices {
		cumSum += probs[idx]
		if cumSum > float64(p) {
			for _, idx := range indices[i+1:] {
				logits[idx] = math.Inf(-1)
			}
			break
		}
	}
	return logits, nil
}

type MinP float64

func (p MinP) Apply(logits []float64) ([]float64, error) {
	if p <= 0 || p >= 1 {
		return nil, errors.New("p must be between 0 and 1")
	}

	probs := softmax(logits)
	threshold := slices.Max(probs) * float64(p)

	for i, prob := range probs {
		if prob < threshold {
			logits[i] = math.Inf(-1)
		}
	}

	return logits, nil
}
