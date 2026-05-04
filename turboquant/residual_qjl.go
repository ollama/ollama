package turboquant

import "math"

const qjlUnbiasScale = 1.2533141373155001 // sqrt(pi / 2)

type ResidualSketch struct {
	Seed      uint64
	Scale     float32 // residual L2 norm; retained name keeps the old struct shape stable
	SketchDim uint16
	Signs     []byte
}

func encodeResidual(rotated, approx []float32, sketchSpec any, seed uint64) ResidualSketch {
	var sketchRows int
	switch v := sketchSpec.(type) {
	case int:
		sketchRows = v
	case Preset:
		sketchRows = v.KeyQJLRows(len(rotated))
	default:
		return ResidualSketch{}
	}
	if sketchRows <= 0 || len(rotated) == 0 {
		return ResidualSketch{}
	}

	residual := make([]float32, len(rotated))
	var l2 float64
	for i := range rotated {
		delta := rotated[i] - approx[i]
		residual[i] = delta
		l2 += float64(delta * delta)
	}

	if l2 == 0 {
		return ResidualSketch{
			Seed:      seed,
			SketchDim: uint16(sketchRows),
			Signs:     make([]byte, expectedPackedBytes(sketchRows, 1)),
		}
	}

	signBits := make([]uint8, sketchRows)
	for row := range sketchRows {
		if gaussianProjectionDot(residual, seed, row) >= 0 {
			signBits[row] = 1
		}
	}

	return ResidualSketch{
		Seed:      seed,
		Scale:     float32(math.Sqrt(l2)),
		SketchDim: uint16(sketchRows),
		Signs:     packBits(signBits, 1),
	}
}

func reconstructResidual(dim int, sketch ResidualSketch) []float32 {
	out := make([]float32, dim)
	if dim == 0 || sketch.SketchDim == 0 || sketch.Scale == 0 {
		return out
	}

	signBits := unpackBits(sketch.Signs, 1, int(sketch.SketchDim))
	scale := float32(qjlUnbiasScale) * sketch.Scale / float32(sketch.SketchDim)
	for row, bit := range signBits {
		sign := float32(-1)
		if bit == 1 {
			sign = 1
		}
		// Residual reconstruction stays on float32 accumulation today; if a backend-specific half path is introduced, it needs an explicit FP32-accumulate audit before rollout.
		for col := range dim {
			out[col] += sign * gaussianProjectionEntry(sketch.Seed, row, col) * scale
		}
	}
	return out
}

func residualDotCorrection(queryRot []float32, sketch ResidualSketch) float32 {
	if len(queryRot) == 0 || sketch.SketchDim == 0 || sketch.Scale == 0 {
		return 0
	}

	signBits := unpackBits(sketch.Signs, 1, int(sketch.SketchDim))
	var total float32
	for row, bit := range signBits {
		sign := float32(-1)
		if bit == 1 {
			sign = 1
		}
		total += sign * gaussianProjectionDot(queryRot, sketch.Seed, row)
	}

	correction := float32(qjlUnbiasScale) * sketch.Scale * (total / float32(sketch.SketchDim))
	queryNorm := float32(math.Sqrt(float64(dotSelf(queryRot))))
	if queryNorm == 0 {
		return 0
	}
	maxCorrection := sketch.Scale * queryNorm
	if correction > maxCorrection {
		return maxCorrection
	}
	if correction < -maxCorrection {
		return -maxCorrection
	}
	if sketch.Scale < 1e-6 {
		return 0
	}
	return correction
}

// PrecomputeCorrectionVec builds the vector w = (√π/2 · residualNorm / sketchDim) · Σ_j sign_j · G_j,
// where G_j is row j of the random Gaussian projection matrix and sign_j is the stored QJL sign bit.
// Scoring then reduces to dot(queryRotated, w), replacing the per-query O(dim²) Gaussian projection
// loop with a single O(dim) dot product.
//
// The returned slice has length dim and is nil when the sketch carries no correction (Scale==0).
func PrecomputeCorrectionVec(sketch ResidualSketch, dim int) []float32 {
	if sketch.SketchDim == 0 || sketch.Scale == 0 {
		return nil
	}
	out := make([]float32, dim)
	signBits := unpackBits(sketch.Signs, 1, int(sketch.SketchDim))
	scale := float32(qjlUnbiasScale) * sketch.Scale / float32(sketch.SketchDim)
	for row, bit := range signBits {
		sign := float32(-1)
		if bit == 1 {
			sign = 1
		}
		sv := sign * scale
		for col := range dim {
			out[col] += sv * gaussianProjectionEntry(sketch.Seed, row, col)
		}
	}
	return out
}

func gaussianProjectionDot(values []float32, seed uint64, row int) float32 {
	var out float32
	for col, value := range values {
		out += value * gaussianProjectionEntry(seed, row, col)
	}
	return out
}

func gaussianProjectionEntry(seed uint64, row, col int) float32 {
	local := splitmix64(seed ^ uint64(row+1)*0x9e3779b97f4a7c15 ^ uint64(col+1)*0xbf58476d1ce4e5b9)
	return float32(gaussianFloat64(&local))
}

// BuildQJLProjection generates the deterministic random Gaussian projection
// matrix used by the QJL residual sketch.  The returned slice is row-major
// with shape [qjlRows, headDim] (each row is one projection vector).
func BuildQJLProjection(headDim, qjlRows int, seed uint64) []float32 {
	data := make([]float32, headDim*qjlRows)
	for row := range qjlRows {
		for col := range headDim {
			data[row*headDim+col] = gaussianProjectionEntry(seed, row, col)
		}
	}
	return data
}

func dotSelf(values []float32) float32 {
	var out float32
	for _, value := range values {
		out += value * value
	}
	return out
}
