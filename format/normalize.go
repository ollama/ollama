package format

import "math"

func Normalize(vec []float64) []float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}

	sum = math.Sqrt(sum)

	var norm float64

	if sum > 0 {
		norm = 1.0 / sum
	} else {
		norm = 0.0
	}

	for i := range vec {
		vec[i] *= norm
	}
	return vec
}
