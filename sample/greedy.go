package sample

import "gonum.org/v1/gonum/floats"

type greedy struct{}

func Greedy() Sampler {
	return greedy{}
}

func (s greedy) Sample(logits []float32, transforms ...Transform) (int, error) {
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

	return floats.MaxIdx(logits64), nil
}
