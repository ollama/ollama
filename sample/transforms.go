package sample

import (
	"cmp"
	"math"
	"slices"

	pq "github.com/emirpasic/gods/v2/queues/priorityqueue"
)

type Transform interface {
	Apply([]float64) []float64
}

// TODO(parthsareen): potentially cache softmax values
func softmax(logits []float64) []float64 {
	var sum float64
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(v)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

type Temperature float64

func (t Temperature) Apply(logits []float64) []float64 {
	temp := math.Max(float64(t), 1e-7)

	// subtracting max logit to avoid under/overflow
	maxLogit := slices.Max(logits)
	for i := range logits {
		logits[i] = (logits[i] - maxLogit) / temp
	}

	return logits
}

type logitMap struct {
	index int
	logit float64
}

type TopK int

// TODO(parthsareen): avoid having to check all logits after this transform
func (k TopK) Apply(logits []float64) []float64 {
	if int(k) >= len(logits) {
		return logits
	}
	q := pq.NewWith(func(a, b logitMap) int {
		return -cmp.Compare(a.logit, b.logit)
	})

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

	return logits
}

type TopP float64

func (p TopP) Apply(logits []float64) []float64 {
	probs := softmax(logits)
	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}

	// sort in descending order
	slices.SortFunc(indices, func(i, j int) int {
		return cmp.Compare(probs[j], probs[i])
	})

	var sum float64
	for i, idx := range indices {
		sum += probs[idx]
		if sum > float64(p) {
			for _, idx := range indices[i+1:] {
				logits[idx] = math.Inf(-1)
			}
			break
		}
	}
	return logits
}

type MinP float64

func (p MinP) Apply(logits []float64) []float64 {
	probs := softmax(logits)
	threshold := slices.Max(probs) * float64(p)

	for i, prob := range probs {
		if prob < threshold {
			logits[i] = math.Inf(-1)
		}
	}

	return logits
}
