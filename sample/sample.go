package sample

import (
	"errors"
	"math"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Sampler interface {
	Sample([]float64) ([]float64, error)
}

type Temperature float64

func (s Temperature) Sample(logits []float64) ([]float64, error) {
	if s < 0 || s > 1 {
		return nil, errors.New("temperature must be between 0 and 1")
	}

	copiedLogits := append([]float64(nil), logits...)
	// Greedy sampling
	if s == 0 {
		return []float64{floats.Max(copiedLogits)}, nil
	}
	floats.Scale(1.0/float64(s), copiedLogits)
	return copiedLogits, nil
}

type softmax struct{}

func Softmax() Sampler {
	return softmax{}
}

func (softmax) Sample(logits []float64) ([]float64, error) {
	return computeSoftmax(logits)
}

func computeSoftmax(logits []float64) ([]float64, error) {
	copiedLogits := make([]float64, len(logits))
	copy(copiedLogits, logits)
	for i := range copiedLogits {
		copiedLogits[i] = math.Exp(copiedLogits[i])
	}

	floatSum := floats.Sum(copiedLogits)
	if floatSum == 0 {
		return nil, errors.New("no valid tokens found")
	}
	floats.Scale(1.0/floatSum, copiedLogits)
	return copiedLogits, nil
}

type TopK int

func (k TopK) Sample(logits []float64) ([]float64, error) {
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}
	if int(k) >= len(logits) {
		return logits, nil
	}

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		return logits[indices[i]] > logits[indices[j]]
	})

	for _, idx := range indices[k:] {
		logits[idx] = math.NaN()
	}

	return logits, nil
}

type TopP float32

func (p TopP) Sample(logits []float64) ([]float64, error) {
	if p <= 0 || p >= 1 {
		return nil, errors.New("p must be between 0 and 1")
	}

	probs, err := computeSoftmax(logits)
	if err != nil {
		return nil, err
	}

	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return probs[indices[i]] > probs[indices[j]]
	})

	cumSum := 0.0
	for i, idx := range indices {
		cumSum += probs[idx]
		if cumSum > float64(p) {
			for _, idx := range indices[i+1:] {
				logits[idx] = math.NaN()
			}
			break
		}
	}
	return logits, nil
}

type MinP float32

func (p MinP) Sample(logits []float64) ([]float64, error) {
	if p <= 0 || p >= 1 {
		return nil, errors.New("p must be between 0 and 1")
	}

	probs, err := computeSoftmax(logits)
	if err != nil {
		return nil, err
	}
	copiedProbs := make([]float64, len(probs))
	copy(copiedProbs, probs)

	sort.Slice(copiedProbs, func(i, j int) bool { return copiedProbs[i] > copiedProbs[j] })

	maxProb := floats.Max(probs)
	probThreshold := float64(p) * maxProb

	for i := range probs {
		if probs[i] < probThreshold {
			logits[i] = math.NaN()
		}
	}

	return logits, nil
}

type weighed struct{}

func Weighed() Sampler {
	return weighed{}
}

func (s weighed) Sample(logits []float64) ([]float64, error) {
	logitsCopy := make([]float64, 0, len(logits))
	indices := make([]int, 0, len(logits))
	// the uv sampler does not support NaN values
	for i, logit := range logits {
		if !math.IsNaN(logit) {
			logitsCopy = append(logitsCopy, logit)
			indices = append(indices, i)
		}
	}

	if len(logitsCopy) == 0 {
		return nil, errors.New("no valid tokens found")
	}

	w := sampleuv.NewWeighted(logitsCopy, nil)
	if v, ok := w.Take(); ok {
		return []float64{float64(indices[v])}, nil
	}
	return nil, errors.New("weighed sampler failed")
}

func Sample(tokenID []float64, samplers ...Sampler) ([]float64, error) {
	var err error
	for _, sampler := range samplers {
		tokenID, err = sampler.Sample(tokenID)
		if err != nil {
			return nil, err
		}
	}
	return tokenID, nil
}
