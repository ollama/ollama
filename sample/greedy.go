package sample

import "gonum.org/v1/gonum/floats"

type greedy struct{}

func Greedy() Sampler {
	return greedy{}
}

func (s greedy) Sample(t []float64) ([]float64, error) {
	return []float64{float64(floats.MaxIdx(t))}, nil
}
