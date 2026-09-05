package nn

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// RopeParameters carries common RoPE metadata embedded in model configs.
type RopeParameters struct {
	RopeTheta                     float32 `json:"rope_theta"`
	RopeType                      string  `json:"rope_type"`
	Type                          string  `json:"type"`
	PartialRotaryFactor           float32 `json:"partial_rotary_factor"`
	Factor                        float32 `json:"factor"`
	OriginalMaxPositionEmbeddings int32   `json:"original_max_position_embeddings"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
	AttentionFactor               float32 `json:"attention_factor"`
	MScale                        float32 `json:"mscale"`
	MScaleAllDim                  float32 `json:"mscale_all_dim"`
	Truncate                      *bool   `json:"truncate"`
	MropeSection                  []int32 `json:"mrope_section"`
}

// TypeName returns rope_type when present, falling back to type.
func (rp *RopeParameters) TypeName() string {
	if rp == nil {
		return ""
	}
	if rp.RopeType != "" {
		return rp.RopeType
	}
	return rp.Type
}

// BuildYarnRopeFreqs returns YaRN rotary frequencies and the mscale value.
func BuildYarnRopeFreqs(dim int, base float32, rp *RopeParameters) (*mlx.Array, float32) {
	freqs, attentionFactor := YarnRopeFreqValues(dim, base, rp)
	if freqs == nil {
		return nil, attentionFactor
	}
	arr := mlx.FromValues(freqs, len(freqs))
	mlx.Eval(arr)
	return arr, attentionFactor
}

// YarnRopeFreqValues returns YaRN rotary frequencies and the mscale value in
// CPU memory. The frequencies are wavelengths, matching MLX's custom RoPE API.
func YarnRopeFreqValues(dim int, base float32, rp *RopeParameters) ([]float32, float32) {
	if rp == nil || dim <= 0 {
		return nil, 1
	}
	factor := rp.Factor
	if factor <= 0 {
		factor = 1
	}
	attentionFactor := rp.AttentionFactor
	if attentionFactor == 0 {
		if rp.MScale != 0 && rp.MScaleAllDim != 0 {
			attentionFactor = yarnMScale(factor, rp.MScale) / yarnMScale(factor, rp.MScaleAllDim)
		} else {
			attentionFactor = yarnMScale(factor, 1)
		}
	}
	if factor <= 1 {
		return nil, attentionFactor
	}

	originalMax := rp.OriginalMaxPositionEmbeddings
	if originalMax <= 0 {
		originalMax = 4096
	}
	betaFast := rp.BetaFast
	if betaFast == 0 {
		betaFast = 32
	}
	betaSlow := rp.BetaSlow
	if betaSlow == 0 {
		betaSlow = 1
	}
	half := dim / 2
	truncate := rp.Truncate == nil || *rp.Truncate
	low, high := yarnCorrectionRange(betaFast, betaSlow, dim, base, originalMax, truncate)
	freqs := make([]float32, half)
	for i := range half {
		posFreq := math.Pow(float64(base), float64(2*i)/float64(dim))
		invExtrapolation := 1.0 / posFreq
		invInterpolation := 1.0 / (float64(factor) * posFreq)
		ramp := yarnRamp(float64(i), low, high)
		mask := 1 - ramp
		inv := invInterpolation*(1-mask) + invExtrapolation*mask
		freqs[i] = float32(1.0 / inv)
	}
	return freqs, attentionFactor
}

func yarnMScale(factor, multiplier float32) float32 {
	if factor <= 1 {
		return 1
	}
	return float32(0.1*float64(multiplier)*math.Log(float64(factor)) + 1)
}

func yarnCorrectionRange(betaFast, betaSlow float32, dim int, base float32, maxPosition int32, truncate bool) (float64, float64) {
	findDim := func(rot float32) float64 {
		return float64(dim) * math.Log(float64(maxPosition)/(float64(rot)*2*math.Pi)) / (2 * math.Log(float64(base)))
	}
	low := findDim(betaFast)
	high := findDim(betaSlow)
	if truncate {
		low = math.Floor(low)
		high = math.Ceil(high)
	}
	low = math.Max(low, 0)
	high = math.Min(high, float64(dim-1))
	if low == high {
		high += 0.001
	}
	return low, high
}

func yarnRamp(i, low, high float64) float64 {
	v := (i - low) / (high - low)
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// ScaleRotaryPart applies YaRN's mscale to only the rotated dimensions.
func ScaleRotaryPart(x *mlx.Array, ropeDim int, scale float32) *mlx.Array {
	if scale == 1 {
		return x
	}
	dims := x.Dims()
	last := dims[len(dims)-1]
	if ropeDim >= last {
		return mlx.MulScalar(x, scale)
	}
	start := make([]int32, len(dims))
	stopRot := make([]int32, len(dims))
	stopPass := make([]int32, len(dims))
	startPass := make([]int32, len(dims))
	for i, dim := range dims {
		stopRot[i] = int32(dim)
		stopPass[i] = int32(dim)
	}
	stopRot[len(dims)-1] = int32(ropeDim)
	startPass[len(dims)-1] = int32(ropeDim)
	rot := mlx.MulScalar(mlx.SliceStartStop(x, start, stopRot), scale)
	pass := mlx.SliceStartStop(x, startPass, stopPass)
	return mlx.Concatenate([]*mlx.Array{rot, pass}, -1)
}
