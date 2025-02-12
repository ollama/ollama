package sample

import (
	"cmp"
	"errors"
	"math"
	"slices"

	pq "github.com/emirpasic/gods/v2/queues/priorityqueue"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Transform interface {
	Apply([]float64) ([]float64, error)
}

type Sampler interface {
	Sample([]float32, ...Transform) (int, error)
}

// TODO(parthsareen): potentially cache softmax values
func softmax(logits []float64) []float64 {
	var sum float64
	tt := make([]float64, len(logits))
	for i, v := range logits {
		tt[i] = math.Exp(v)
		sum += tt[i]
	}
	floats.Scale(1/sum, tt)
	return tt
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

type weighted struct {
	src rand.Source
}

func Weighted(seed *int64) Sampler {
	var src rand.Source
	if seed != nil {
		src = rand.NewSource(uint64(*seed))
	}
	return weighted{src: src}
}

func (s weighted) Sample(logits []float32, transforms ...Transform) (int, error) {
	logits64 := make([]float64, len(logits))
	for i, v := range logits {
		logits64[i] = float64(v)
	}

	var err error
	for _, t := range transforms {
		logits64, err = t.Apply(logits64)
		if err != nil {
			return -1, err
		}
	}

	logitsCopy := make([]float64, 0, len(logits))
	indices := make([]int, 0, len(logits))
	for i, logit := range logits64 {
		if !math.IsInf(logit, -1) {
			logitsCopy = append(logitsCopy, logit)
			indices = append(indices, i)
		}
	}

	if len(logitsCopy) == 0 {
		return -1, errors.New("no valid logits found for weighed sampling")
	}

	probs := softmax(logitsCopy)
	w := sampleuv.NewWeighted(probs, s.src)
	if idx, ok := w.Take(); ok {
		return indices[idx], nil
	}
	return -1, errors.New("weighed sampler failed, no valid token found")
}
