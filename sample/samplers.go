package sample

import (
	"errors"
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Sampler interface {
	Sample([]float32) (int32, error)
}

type weighted struct {
	src        rand.Source
	transforms []Transform
}

func Weighted(seed *int64, transforms ...Transform) Sampler {
	var src rand.Source
	if seed != nil {
		src = rand.NewSource(uint64(*seed))
	}
	return weighted{src: src, transforms: transforms}
}

func (s weighted) Sample(logits []float32) (int32, error) {
	logits64 := make([]float64, len(logits))
	for i, v := range logits {
		logits64[i] = float64(v)
	}

	var err error
	for _, t := range s.transforms {
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
		return int32(indices[idx]), nil
	}
	return -1, errors.New("weighed sampler failed, no valid token found")
}

type greedy struct {
	transforms []Transform
}

func Greedy(transforms ...Transform) Sampler {
	return greedy{transforms: transforms}
}

func (s greedy) Sample(logits []float32) (int32, error) {
	logits64 := make([]float64, len(logits))
	for i, v := range logits {
		logits64[i] = float64(v)
	}

	var err error
	for _, t := range s.transforms {
		logits64, err = t.Apply(logits64)
		if err != nil {
			return -1, err
		}
	}

	return int32(floats.MaxIdx(logits64)), nil
}
