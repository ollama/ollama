package sample

import "gonum.org/v1/gonum/floats"

type greedy struct{}

func Greedy() Sampler {
	return greedy{}
}

func (s greedy) Sample(t []float64) (int, error) {
	return floats.MaxIdx(t), nil
}
