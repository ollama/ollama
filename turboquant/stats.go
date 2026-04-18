package turboquant

import "math"

type Stats struct {
	MSE         float32
	RMSE        float32
	MeanAbsErr  float32
	MaxAbsErr   float32
}

func Compare(reference, approx []float32) Stats {
	var s Stats
	if len(reference) == 0 || len(reference) != len(approx) {
		return s
	}
	for i := range reference {
		err := reference[i] - approx[i]
		s.MSE += err * err
		s.MeanAbsErr += abs32(err)
		if abs32(err) > s.MaxAbsErr {
			s.MaxAbsErr = abs32(err)
		}
	}
	s.MSE /= float32(len(reference))
	s.RMSE = float32(math.Sqrt(float64(s.MSE)))
	s.MeanAbsErr /= float32(len(reference))
	return s
}

func maxAbs(values []float32) float32 {
	var m float32
	for _, v := range values {
		if abs32(v) > m {
			m = abs32(v)
		}
	}
	return m
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
