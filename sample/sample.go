package sample

import (
	"cmp"
	"errors"
	"math"
	"slices"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Sampler interface {
	Sample([]float64) ([]float64, error)
}

type Temperature float64

func (t Temperature) Sample(logits []float64) ([]float64, error) {
	if t < 0 || t > 2 {
		return nil, errors.New("temperature must be between 0 and 2")
	}

	// subtracting max logit to avoid under/overflow
	maxLogit := floats.Max(logits)

	temp := math.Max(float64(t), 1e-7)
	for i := range logits {
		logits[i] = (logits[i] - maxLogit) / temp
	}

	return logits, nil
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

	// sort in descending order
	slices.SortFunc(indices, func(i, j int) int {
		return cmp.Compare(logits[j], logits[i])
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

	// sort in descending order
	slices.SortFunc(indices, func(i, j int) int {
		return cmp.Compare(probs[j], probs[i])
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

	slices.Sort(copiedProbs)

	maxProb := copiedProbs[len(copiedProbs)-1]
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

	logitsCopy, err := computeSoftmax(logitsCopy)
	if err != nil {
		return nil, err
	}

	w := sampleuv.NewWeighted(logitsCopy, nil)
	if v, ok := w.Take(); ok {
		// returns the token ID
		return []float64{float64(indices[v])}, nil
	}
	return nil, errors.New("weighed sampler failed")
}

// TODO: remove after next PR merge
type greedy struct{}

func Greedy() Sampler {
	return greedy{}
}

func (greedy) Sample(logits []float64) ([]float64, error) {
	return []float64{float64(floats.MaxIdx(logits))}, nil
}

func Sample(logits []float64, samplers ...Sampler) ([]float64, error) {
	var err error
	for _, sampler := range samplers {
		if sampler == Temperature(0) {
			// early return with greedy if temperature is 0
			logits, err = Greedy().Sample(logits)
			if err != nil {
				return nil, err
			}
			return logits, nil
		}
		logits, err = sampler.Sample(logits)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}
