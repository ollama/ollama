package sample

import (
	"slices"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Sampler interface {
	Sample([]float64) ([]float64, error)
}

type Temperature float64

func (s Temperature) Sample(t []float64) ([]float64, error) {
	floats.Div(t, slices.Repeat([]float64{float64(s)}, len(t)))
	return t, nil
}

type softmax struct{}

func Softmax() Sampler {
	return softmax{}
}

func (softmax) Sample(t []float64) ([]float64, error) {
	return t, nil
}

type TopK int

func (s TopK) Sample(t []float64) ([]float64, error) {
	return t, nil
}

type TopP float32

func (s TopP) Sample(t []float64) ([]float64, error) {
	return t, nil
}

type MinP float32

func (s MinP) Sample(t []float64) ([]float64, error) {
	return t, nil
}

type weighed struct{}

func Weighed() Sampler {
	return weighed{}
}

func (s weighed) Sample(t []float64) ([]float64, error) {
	w := sampleuv.NewWeighted(t, nil)
	if v, ok := w.Take(); ok {
		return []float64{float64(v)}, nil
	}

	return t, nil
}

func Sample(floats []float64, samplers ...Sampler) ([]float64, error) {
	var err error
	for _, sampler := range samplers {
		floats, err = sampler.Sample(floats)
		if err != nil {
			return nil, err
		}
	}

	return floats, nil
}
