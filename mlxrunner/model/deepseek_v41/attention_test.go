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
		rotated := mlx.RotateTail(x, positions, []float32{1}, false)
		recovered := mlx.RotateTail(rotated, positions, []float32{1}, true)
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

func TestSparseWindowAttentionMatchesGraph(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("sparse window kernel requires a GPU")
		}
		const length, context, compressedCount, heads, width, window, topK = 33, 70, 64, 64, 512, 7, 32
		queryValues := make([]float32, length*heads*width)
		keyValues := make([]float32, context*width)
		compressedValues := make([]float32, compressedCount*width)
		selected := make([]int32, length*topK)
		sinks := make([]float32, heads)
		for i := range queryValues {
			queryValues[i] = float32((i*17)%29-14) / 19
		}
		for i := range keyValues {
			keyValues[i] = float32((i*13)%31-15) / 17
		}
		for i := range compressedValues {
			compressedValues[i] = float32((i*7)%23-11) / 13
		}
		for i := range selected {
			selected[i] = int32((i * 7) % compressedCount)
			if i%11 == 0 {
				selected[i] = -1
			}
		}
		for i := range sinks {
			sinks[i] = float32(i%7-3) / 10
		}
		compressed := mlx.FromValues(compressedValues, 1, compressedCount, width)
		selection := mlx.FromValues(selected, 1, length, topK)
		sink := mlx.FromValues(sinks, heads)
		for _, dtype := range []mlx.DType{mlx.DTypeBFloat16, mlx.DTypeFloat32} {
			q := mlx.FromValues(queryValues, 1, length, heads, width).AsType(dtype)
			for _, rawType := range []mlx.DType{mlx.DTypeBFloat16, mlx.DTypeFloat32} {
				kv := mlx.FromValues(keyValues, 1, context, width).AsType(rawType)
				kernel, ok := mlx.SparseWindowAttention(q, kv, compressed, selection, sink, 25, 5, window, .125)
				if !ok {
					t.Fatal("GPU sparse window kernel was unavailable")
				}
				got := attendWindow(q, kv, compressed, selection, sink, 25, 5, window, 1, .125).AsType(mlx.DTypeFloat32)
				want := attendWindowGraph(q, kv, compressed, selection, sink, 25, 5, window, 1, .125).AsType(mlx.DTypeFloat32)
				kernel = kernel.AsType(mlx.DTypeFloat32)
				mlx.Eval(got, want, kernel)
				tolerance := 1e-4
				if dtype == mlx.DTypeBFloat16 {
					tolerance = 1e-3
				}
				compareFloats(t, "sparse window wrapper", got.Floats(), want.Floats(), tolerance)
				compareFloats(t, "sparse window kernel", kernel.Floats(), want.Floats(), tolerance)
			}
		}
	})
}

func TestSparseWindowAttentionKeepsBatchRowsSeparate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("sparse window kernel requires a GPU")
		}
		const batchSize, length, heads, width, rawCount, compressedCount, topK = 2, 1, 3, 64, 12, 13, 5
		queryValues := make([]float32, batchSize*length*heads*width)
		rawValues := make([]float32, batchSize*rawCount*width)
		compressedValues := make([]float32, batchSize*compressedCount*width)
		selected := make([]int32, batchSize*length*topK)
		for i := range queryValues {
			queryValues[i] = float32(i%23-11) / 17
		}
		for i := range rawValues {
			rawValues[i] = float32(i%31-15) / 19
		}
		for i := range compressedValues {
			compressedValues[i] = float32(i%37-18) / 21
		}
		for i := range topK {
			selected[i] = int32((i * 3) % compressedCount)
			selected[topK+i] = -1
		}
		q := mlx.FromValues(queryValues, batchSize, length, heads, width).AsType(mlx.DTypeBFloat16)
		raw := mlx.FromValues(rawValues, batchSize, rawCount, width).AsType(mlx.DTypeBFloat16)
		compressed := mlx.FromValues(compressedValues, batchSize, compressedCount, width)
		selection := mlx.FromValues(selected, batchSize, length, topK)
		sink := mlx.FromValues([]float32{.2, -.1, .4}, heads)
		got, ok := mlx.SparseWindowAttention(q, raw, compressed, selection, sink, 14, 5, 4, .125)
		if !ok {
			t.Fatal("GPU sparse window kernel was unavailable")
		}
		wantRows := make([]*mlx.Array, batchSize)
		for row := range batchSize {
			part := mlx.Slice(row, row+1)
			wantRows[row] = attendWindowGraph(q.Slice(part, mlx.Slice(), mlx.Slice(), mlx.Slice()),
				raw.Slice(part, mlx.Slice(), mlx.Slice()),
				compressed.Slice(part, mlx.Slice(), mlx.Slice()),
				selection.Slice(part, mlx.Slice(), mlx.Slice()), sink, 14, 5, 4, 1, .125)
		}
		want := mlx.Concatenate(wantRows, 0).AsType(mlx.DTypeFloat32)
		got = got.AsType(mlx.DTypeFloat32)
		mlx.Eval(got, want)
		compareFloats(t, "batch rows with masked selection", got.Floats(), want.Floats(), 1e-3)
	})
}

func TestSparseWindowAttentionModelDecodeShape(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("sparse window kernel requires a GPU")
		}
		const width, heads, window, topK, compressedCount = 512, 64, 128, 512, 1024
		qValues := make([]float32, heads*width)
		rawValues := make([]float32, window*width)
		compressedValues := make([]float32, compressedCount*width)
		selected := make([]int32, topK)
		for i := range qValues {
			qValues[i] = float32(i%37-18) / 29
		}
		for i := range rawValues {
			rawValues[i] = float32(i%31-15) / 23
		}
		for i := range compressedValues {
			compressedValues[i] = float32(i%41-20) / 31
		}
		for i := range selected {
			selected[i] = int32((i * 17) % compressedCount)
			if i%13 == 0 {
				selected[i] = -1
			}
		}
		q := mlx.FromValues(qValues, 1, 1, heads, width).AsType(mlx.DTypeBFloat16)
		raw := mlx.FromValues(rawValues, 1, window, width).AsType(mlx.DTypeBFloat16)
		compressed := mlx.FromValues(compressedValues, 1, compressedCount, width)
		selection := mlx.FromValues(selected, 1, 1, topK)
		sink := mlx.Zeros(mlx.DTypeFloat32, heads)
		got, ok := mlx.SparseWindowAttention(q, raw, compressed, selection, sink, 1023, 896, window, 1/float32(math.Sqrt(width)))
		if !ok {
			t.Fatal("GPU sparse window kernel was unavailable at model decode shape")
		}
		want := attendWindowGraph(q, raw, compressed, selection, sink, 1023, 896, window, 1, 1/float32(math.Sqrt(width)))
		got, want = got.AsType(mlx.DTypeFloat32), want.AsType(mlx.DTypeFloat32)
		mlx.Eval(got, want)
		compareFloats(t, "model decode attention", got.Floats(), want.Floats(), 1e-3)
	})
}
