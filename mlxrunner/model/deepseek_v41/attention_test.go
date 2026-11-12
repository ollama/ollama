package deepseek_v41

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestAttentionBatchesDifferentCacheLengths(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const width = 32
		identity := make([]float32, width*width)
		for i := range width {
			identity[i*width+i] = 1
		}
		weight := mlx.FromValues(identity, width, width)
		packed, scales, _ := mlx.Quantize(weight, 32, 8, "mxfp8")
		output, err := newGroupedOutput(packed, scales, 1, width, width)
		if err != nil {
			t.Fatal(err)
		}
		norm := nn.NewRMSNorm(mlx.FromValues([]float32{
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		}, width), 1e-6)
		layer := &transformerLayer{Attention: &latentAttention{
			QueryLow: nn.NewLinear(weight, nil), QueryHigh: nn.NewLinear(weight, nil),
			KeyValue: nn.NewLinear(weight, nil), OutputLow: output, OutputHigh: nn.NewLinear(weight, nil),
			QueryNorm: norm, KeyNorm: norm, Sink: mlx.Zeros(mlx.DTypeFloat32, 1),
			Frequencies: []float32{1}, Heads: 1, HeadWidth: width, Window: 3,
		}}
		layer.Attention.Compressor = &compressor{
			KeyValue: nn.NewLinear(weight, nil), Gate: nn.NewLinear(weight, nil), Norm: norm, Ratio: 2,
		}
		indexWeight := make([]float32, width)
		indexWeight[0] = 1
		layer.Attention.Indexer = &sparseIndexer{
			Query: nn.NewLinear(weight, nil), Weights: nn.NewLinear(mlx.FromValues(indexWeight, 1, width), nil),
			Key: nn.NewLinear(weight, nil), KeyNorm: norm, Heads: 1, Width: width, TopK: 1,
		}
		vectors := func(values ...float32) *mlx.Array {
			data := make([]float32, len(values)*width)
			for i, value := range values {
				data[i*width] = value
				data[i*width+1] = value / 2
			}
			return mlx.FromValues(data, 1, len(values), width)
		}
		seed := func(values ...float32) *cache.KVCache {
			state := cache.NewKVCache()
			ids := make([]int32, len(values))
			positions := make([]int32, len(values))
			for i := range positions {
				positions[i] = int32(i)
			}
			b := &batch.Batch{InputIDs: mlx.FromValues(ids, 1, len(ids)), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(ids))}}
			layer.attend(vectors(values...), mlx.FromValues(positions, 1, len(positions)), b, state, []*sparseState{{}})
			return state
		}
		batchedOne, batchedTwo := seed(1, 2, 3, 4), seed(5, 6)
		layers, err := cache.Batch([][]cache.Cache{{batchedOne}, {batchedTwo}})
		if err != nil {
			t.Fatal(err)
		}
		b := &batch.Batch{InputIDs: mlx.FromValues([]int32{0, 0}, 2, 1), SeqOffsets: []int32{4, 2}, SeqQueryLens: []int32{1, 1}}
		input := mlx.Concatenate([]*mlx.Array{vectors(7), vectors(8)}, 0)
		got := layer.attend(input, mlx.FromValues([]int32{4, 2}, 2, 1), b, layers[0], []*sparseState{{}, {}})
		one := layer.attend(vectors(7), mlx.FromValues([]int32{4}, 1, 1), b.Row(0), seed(1, 2, 3, 4), []*sparseState{{}})
		two := layer.attend(vectors(8), mlx.FromValues([]int32{2}, 1, 1), b.Row(1), seed(5, 6), []*sparseState{{}})
		want := mlx.Concatenate([]*mlx.Array{one, two}, 0)
		mlx.Eval(got, want)
		compareFloats(t, "mixed-offset batched attention", got.Floats(), want.Floats(), 1e-4)
	})
}

func TestCompressorCarriesPartialGroupAcrossChunks(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		identity := make([]float32, 16)
		for i := range 4 {
			identity[i*4+i] = 1
		}
		c := &compressor{
			KeyValue: nn.NewLinear(mlx.FromValues(identity, 4, 4), nil),
			Gate:     nn.NewLinear(mlx.FromValues(make([]float32, 16), 4, 4), nil),
			Norm:     nn.NewRMSNorm(mlx.FromValues([]float32{1, 1, 1, 1}, 4), 1e-6),
			Ratio:    2,
		}
		x := mlx.FromValues([]float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}, 1, 4, 4)
		full, _, _, _ := c.encode(x, nil, 0, []float32{0})
		_, _, firstValue, firstScore := c.encode(x.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()), nil, 0, []float32{0})
		zero := mlx.Zeros(mlx.DTypeFloat32, 1, 1, 1, 4)
		previous := mlx.Transpose(mlx.Concatenate([]*mlx.Array{
			zero, zero, mlx.ExpandDims(firstValue, 2), mlx.ExpandDims(firstScore, 2),
		}, 2), 0, 2, 1, 3)
		continued, _, _, _ := c.encode(x.Slice(mlx.Slice(), mlx.Slice(1, 4), mlx.Slice()), previous, 1, []float32{0})
		want := full.Slice(mlx.Slice(), mlx.Slice(1, 4), mlx.Slice())
		mlx.Eval(continued, want)
		compareFloats(t, "continued compression", continued.Floats(), want.Floats(), 1e-5)
	})
}

func TestGroupedOutputKeepsHeadGroupsSeparate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		values := make([]float32, 2*32)
		for i := range values {
			if i < 32 {
				values[i] = .25
			} else {
				values[i] = .5
			}
		}
		packed, scales, _ := mlx.Quantize(mlx.FromValues(values, 2, 32), 32, 8, "mxfp8")
		projection, err := newGroupedOutput(packed, scales, 2, 32, 1)
		if err != nil {
			t.Fatal(err)
		}
		input := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 1, 2, 32), 1)
		result := projection.forward(input)
		mlx.Eval(result)
		compareFloats(t, "grouped output", result.Floats(), []float32{8, 16}, .03)
	})
}

func TestRotateTailPreservesLeadingChannelsAndInverts(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		x := mlx.FromValues([]float32{10, 20, 1, 0, 10, 20, 1, 0}, 1, 2, 1, 4)
		positions := mlx.FromValues([]int32{0, 1}, 1, 2)
		rotated := rotateTail(x, positions, []float32{1}, false)
		recovered := rotateTail(rotated, positions, []float32{1}, true)
		mlx.Eval(rotated, recovered)
		compareFloats(t, "rotated", rotated.Floats(), []float32{10, 20, 1, 0, 10, 20, float32(math.Cos(1)), float32(math.Sin(1))}, 1e-5)
		compareFloats(t, "inverse", recovered.Floats(), x.Floats(), 1e-5)
	})
}

func TestWindowAttentionIncludesSinkAndCausalLimit(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		q := mlx.Zeros(mlx.DTypeFloat32, 1, 3, 1, 2)
		kv := mlx.FromValues([]float32{2, 0, 4, 0, 8, 0}, 1, 3, 2)
		sink := mlx.FromValues([]float32{0}, 1)
		result := attendWindow(q, kv, nil, nil, sink, 0, 0, 2, 0, 1)
		mlx.Eval(result)
		compareFloats(t, "prefill", result.Floats(), []float32{1, 0, 2, 0, 4, 0}, 1e-5)

		last := q.Slice(mlx.Slice(), mlx.Slice(2, 3), mlx.Slice(), mlx.Slice())
		tail := kv.Slice(mlx.Slice(), mlx.Slice(1, 3), mlx.Slice())
		decode := attendWindow(last, tail, nil, nil, sink, 2, 1, 2, 0, 1)
		mlx.Eval(decode)
		compareFloats(t, "decode", decode.Floats(), []float32{4, 0}, 1e-5)
	})
}

func TestAttendWindowIncludesCompletedCompressedGroups(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		q := mlx.FromValues(make([]float32, 3*2), 1, 3, 1, 2)
		kv := mlx.FromValues([]float32{1, 0, 2, 0, 3, 0}, 1, 3, 2)
		compressed := mlx.FromValues([]float32{10, 0}, 1, 1, 2)
		sink := mlx.FromValues([]float32{0}, 1)
		result := attendWindow(q, kv, compressed, nil, sink, 0, 0, 2, 2, 1)
		mlx.Eval(result)
		compareFloats(t, "compressed attention", result.Floats(), []float32{.5, 0, 3.25, 0, 3.75, 0}, 1e-5)
	})
}

func TestAttendWindowUsesSelectedCompressedPositions(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		q := mlx.Zeros(mlx.DTypeFloat32, 1, 3, 1, 2)
		kv := mlx.FromValues([]float32{1, 0, 2, 0, 3, 0}, 1, 3, 2)
		compressed := mlx.FromValues([]float32{10, 0, 20, 0}, 1, 2, 2)
		selection := mlx.FromValues([]int32{-1, 0, 1}, 1, 3, 1)
		sink := mlx.FromValues([]float32{0}, 1)
		result := attendWindow(q, kv, compressed, selection, sink, 0, 0, 2, 1, 1)
		mlx.Eval(result)
		compareFloats(t, "selected compressed attention", result.Floats(), []float32{.5, 0, 3.25, 0, 6.25, 0}, 1e-5)
	})
}
