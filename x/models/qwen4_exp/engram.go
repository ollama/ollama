package qwen4_exp

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func (n *streamRMSNorm) Forward(x *mlx.Array, eps float32) *mlx.Array {
	originalShape := x.Dims()
	if originalShape[len(originalShape)-1] == n.Weight.Size() {
		shape := append([]int(nil), originalShape[:len(originalShape)-1]...)
		shape = append(shape, n.Weight.Dim(0), n.Weight.Dim(1))
		x = x.Reshape(shape...)
	}
	dtype := x.DType()
	values := x.AsType(mlx.DTypeFloat32)
	meanSquare := mlx.Mean(mlx.Mul(values, values), -1, true)
	invRMS := mlx.Div(mlx.NewScalarArray(1), mlx.AddScalar(meanSquare, eps).Sqrt())
	normalized := mlx.Mul(mlx.Mul(values, invRMS), n.Weight.AsType(mlx.DTypeFloat32)).AsType(dtype)
	return normalized.Reshape(originalShape...)
}

func (p *PLE) hashes(b *batch.Batch, history *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	tokens := b.InputIDs.AsType(mlx.DTypeInt64)
	joined := mlx.Concatenate([]*mlx.Array{history, tokens}, 1)
	length := b.InputIDs.Dim(1)
	shifts := make([]*mlx.Array, cfg.NGramSize)
	eos := mlx.FromValues([]int64{cfg.EOSTokenID}, 1)
	for k := range cfg.NGramSize {
		start := int(cfg.NGramSize - 1 - k)
		shifts[k] = mlx.SliceStartStop(joined,
			[]int32{0, int32(start)},
			[]int32{int32(joined.Dim(0)), int32(start + length)},
		)
		for previous := int32(1); previous <= k; previous++ {
			boundary := mlx.SliceStartStop(joined,
				[]int32{0, cfg.NGramSize - 1 - previous},
				[]int32{int32(joined.Dim(0)), cfg.NGramSize - 1 - previous + int32(length)},
			)
			shifts[k] = mlx.Where(boundary.Equal(eos), eos, shifts[k])
		}
	}

	hashes := make([]*mlx.Array, 0, (cfg.NGramSize-1)*cfg.HeadsPerNGram)
	head := 0
	for ngram := int32(2); ngram <= cfg.NGramSize; ngram++ {
		multiplier := mlx.Take(p.LayerMultipliers, mlx.FromValue(0), 0)
		mix := mlx.Mul(shifts[0], multiplier)
		for k := int32(1); k < ngram; k++ {
			multiplier = mlx.Take(p.LayerMultipliers, mlx.FromValue(int(k)), 0)
			mix = mix.BitwiseXor(mlx.Mul(shifts[k], multiplier))
		}
		for range cfg.HeadsPerNGram {
			modulus := mlx.Take(p.HeadVocabSizes, mlx.FromValue(head), 0)
			hashes = append(hashes, mix.Remainder(modulus))
			head++
		}
	}
	return mlx.Add(mlx.Stack(hashes, 2), p.HeadOffsets), tokens
}

func (p *PLE) lookup(globalIDs *mlx.Array) *mlx.Array {
	rows := mlx.FromValues([]int64{int64(p.ShardRows)}, 1)
	shardIDs := globalIDs.FloorDivide(rows)
	localIDs := globalIDs.Remainder(rows)
	zeroID := mlx.FromValues([]int64{0}, 1)

	// The checkpoint partitions one logical table into equal contiguous row
	// shards. This portable fallback touches every shard, then masks and sums
	// the selected rows. It prioritizes reference bring-up over performance;
	// production support should replace it with a fused split-table lookup.
	var result *mlx.Array
	for i, embedding := range p.EmbeddingShards {
		mask := shardIDs.Equal(mlx.FromValues([]int64{int64(i)}, 1))
		indices := mlx.Where(mask, localIDs, zeroID)
		values := embedding.Forward(indices)
		values = mlx.Mul(values, mlx.ExpandDims(mask, -1).AsType(values.DType()))
		if result == nil {
			result = values
		} else {
			result = mlx.Add(result, values)
		}
	}
	return result
}

// Forward implements the published Engram dataflow against the checkpoint's
// fused four-stream projections and split embedding table.
func (p *PLE) Forward(hidden *mlx.Array, b *batch.Batch, tokens *engramCache, cfg *Config) *mlx.Array {
	dims := hidden.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	streams := mlx.Reshape(hidden, B, L, cfg.HCCount, cfg.HiddenSize)

	hashes, inputIDs := p.hashes(b, tokens.get(b), cfg)
	embeddings := p.lookup(hashes)
	embeddings = mlx.Reshape(embeddings, B, L, cfg.HiddenSize)

	key := mlx.Reshape(p.KeyProj.Forward(embeddings), B, L, cfg.HCCount, cfg.HiddenSize)
	key = p.NormKey.Forward(key, cfg.RMSNormEps)
	query := p.NormQuery.Forward(streams, cfg.RMSNormEps)
	gate := mlx.DivScalar(mlx.Sum(mlx.Mul(key, query), -1, false), float32(math.Sqrt(float64(cfg.HiddenSize))))
	gate = mlx.Mul(mlx.Maximum(gate.Abs(), mlx.NewScalarArray(1e-6)).Sqrt(), gate.Sign())
	gate = mlx.ExpandDims(mlx.Sigmoid(gate), -1)

	value := mlx.ExpandDims(p.ValueProj.Forward(embeddings), 2)
	value = mlx.Mul(gate, value)
	convInput := p.NormConv.Forward(value, cfg.RMSNormEps)
	convInput = mlx.Reshape(convInput, B, L, cfg.HCCount*cfg.HiddenSize)
	convHistory := tokens.getConvHistory(b, convInput.DType())
	conv := p.Conv.Forward(mlx.Concatenate([]*mlx.Array{convHistory, convInput}, 1))
	conv = mlx.Reshape(mlx.SiLU(conv), B, L, cfg.HCCount, cfg.HiddenSize)

	tokens.put(b, inputIDs, convInput)
	return mlx.Reshape(mlx.Add(value, conv), B, L, cfg.HCCount*cfg.HiddenSize).AsType(hidden.DType())
}
