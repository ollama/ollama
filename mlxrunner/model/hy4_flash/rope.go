package hy4_flash

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
)

type rotaryPositions struct {
	batch      *batch.Batch
	groupStart int
	stride     int
}

type rotaryMemoKey struct {
	shape      attentionShape
	length     int
	groupStart int
	stride     int
	dtype      mlx.DType
}

type rotaryPhases struct{ cos, sin *mlx.Array }

func (r rotaryPositions) apply(x *mlx.Array, shape attentionShape) *mlx.Array {
	key := rotaryMemoKey{shape: shape, length: x.Dim(2), groupStart: r.groupStart, stride: r.stride, dtype: x.DType()}
	cached, ok := r.batch.Memo.Get(key)
	if !ok {
		cached = r.phases(key, x.Dim(0))
		r.batch.Memo.Put(key, cached)
	}
	phase := cached.(rotaryPhases)
	n := shape.rotary / 2
	paired := x.AsType(mlx.DTypeFloat32).Reshape(x.Dim(0), x.Dim(1), x.Dim(2), n, 2)
	even := paired.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, 1))
	odd := paired.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2))
	real := mlx.Sub(mlx.Mul(even, phase.cos), mlx.Mul(odd, phase.sin))
	imag := mlx.Add(mlx.Mul(odd, phase.cos), mlx.Mul(even, phase.sin))
	return mlx.Concatenate([]*mlx.Array{real, imag}, -1).Reshape(x.Dims()...).AsType(x.DType())
}

func (r rotaryPositions) phases(key rotaryMemoKey, rows int) rotaryPhases {
	half := key.shape.rotary / 2
	invFreq := make([]float32, half)
	axis := make([]int, half)
	for i := range half {
		invFreq[i] = float32(math.Pow(float64(key.shape.theta), -float64(i)/float64(half)))
		channel := i % 3
		if channel > 0 && i < 3*key.shape.sections[channel] {
			axis[i] = channel
		}
	}
	angles := make([]float32, rows*key.length*half)
	for row := range rows {
		var layout *positionLayout
		if row < len(r.batch.Layout) {
			layout, _ = r.batch.Layout[row].(*positionLayout)
		}
		start, stride := r.batch.SeqOffsets[row], int32(1)
		if r.stride > 0 {
			start, stride = int32(r.groupStart), int32(r.stride)
		}
		for i := range key.length {
			pos := start + int32(i)*stride
			coords := [3]int32{pos, pos, pos}
			if layout != nil {
				if int(pos) < len(layout.positions) {
					coords = layout.positions[pos]
				} else {
					pos += layout.delta
					coords = [3]int32{pos, pos, pos}
				}
			}
			for j, freq := range invFreq {
				angles[(row*key.length+i)*half+j] = float32(coords[axis[j]]) * freq
			}
		}
	}
	x := mlx.FromValues(angles, rows, 1, key.length, half, 1)
	// The checkpoint rounds the phase tables to the activation dtype before
	// the FP32 rotation. Use the same tables for text and image coordinates.
	return rotaryPhases{
		cos: mlx.Cos(x).AsType(key.dtype).AsType(mlx.DTypeFloat32),
		sin: mlx.Sin(x).AsType(key.dtype).AsType(mlx.DTypeFloat32),
	}
}
