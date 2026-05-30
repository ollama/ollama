package turboquant

import "math"

type namedVector struct {
	name   string
	values []float32
}

func deterministicCorpus() []namedVector {
	return []namedVector{
		{name: "ramp-16", values: []float32{-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
		{name: "alternating-16", values: []float32{1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8}},
		{name: "sparse-17", values: []float32{0, 0, 0, 7, 0, 0, -5, 0, 0, 0, 9, 0, 0, 0, 0, -3, 0}},
		{name: "constant-31", values: filledVector(31, 1.5)},
		{name: "zero-64", values: filledVector(64, 0)},
		{name: "random-65", values: pseudoRandomVector(65, 0x1234abcd)},
	}
}

func pseudoRandomVector(n int, seed uint64) []float32 {
	rng := splitmix64(seed)
	out := make([]float32, n)
	for i := range out {
		u := float32(rng.next()&0xffff) / 65535
		out[i] = (u * 10) - 5
	}
	return out
}

func filledVector(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func vectorNorm(values []float32) float32 {
	var sum float64
	for _, v := range values {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum))
}

func meanMSEForPreset(preset Preset) (float32, error) {
	var total float32
	corpus := deterministicCorpus()
	for _, tc := range corpus {
		encoded, err := EncodeVector(tc.values, preset)
		if err != nil {
			return 0, err
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			return 0, err
		}
		decoded, _, err := DecodeVector(data)
		if err != nil {
			return 0, err
		}
		total += Compare(tc.values, decoded).MSE
	}
	return total / float32(len(corpus)), nil
}
