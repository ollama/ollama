package format

import "math"

func Normalize(vec []float32) []float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}

	sum = math.Sqrt(sum)

	var norm float32

	if sum > 0 {
		norm = float32(1.0 / sum)
	} else {
		norm = 0.0
	}

	for i := range vec {
		vec[i] *= norm
	}
	return vec
}
