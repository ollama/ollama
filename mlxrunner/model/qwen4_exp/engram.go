package qwen4_exp

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// PLE is the model's n-gram embedding block. The checkpoint stores one logical
// table in split vocabulary shards.
type PLE struct {
	Conv      *nn.Conv1d
	KeyProj   nn.LinearLayer
	ValueProj nn.LinearLayer
	NormConv  *streamRMSNorm
	NormKey   *streamRMSNorm
	NormQuery *streamRMSNorm

	LayerMultipliers *mlx.Array
	HeadOffsets      *mlx.Array
	HeadVocabSizes   *mlx.Array
	EmbeddingShards  []nn.EmbeddingLayer
	ShardRows        int
	hostEmbedding    *hostPLETable
}

func (m *Model) loadPLE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*PLE, error) {
	p := &PLE{}
	var err error
	convWeight, err := requiredArray(tensors, prefix+".conv1d.weight", prefix+".conv1d")
	if err != nil {
		return nil, err
	}
	if convWeight.NumDims() != 3 || convWeight.Dim(1) != 1 || convWeight.Dim(2) != int(m.PLEConvKernelSize) {
		return nil, fmt.Errorf("%s.conv1d has shape %v, want [%d, 1, %d]", prefix, convWeight.Dims(), m.HCCount*m.HiddenSize, m.PLEConvKernelSize)
	}
	convWeight = mlx.Transpose(convWeight, 0, 2, 1)
	p.Conv = nn.NewConv1d(convWeight, nil, 1, 0, m.NGramSize, m.HCCount*m.HiddenSize)
	if p.KeyProj, err = requiredLinear(linears, prefix+".key_proj"); err != nil {
		return nil, err
	}
	if p.ValueProj, err = requiredLinear(linears, prefix+".value_proj"); err != nil {
		return nil, err
	}
	if p.NormConv, err = loadStreamNorm(tensors, prefix+".norm_conv", m.Config); err != nil {
		return nil, err
	}
	if p.NormKey, err = loadStreamNorm(tensors, prefix+".norm_key", m.Config); err != nil {
		return nil, err
	}
	if p.NormQuery, err = loadStreamNorm(tensors, prefix+".norm_query", m.Config); err != nil {
		return nil, err
	}
	if p.LayerMultipliers, err = requiredArray(tensors, prefix+".ple_embedding.layer_multipliers"); err != nil {
		return nil, err
	}
	if p.HeadOffsets, err = requiredArray(tensors, prefix+".ple_embedding.ngram_heads_offsets"); err != nil {
		return nil, err
	}
	if p.HeadVocabSizes, err = requiredArray(tensors, prefix+".ple_embedding.ngram_heads_vocab_sizes"); err != nil {
		return nil, err
	}

	if table := m.hostPLE[prefix]; table != nil {
		p.hostEmbedding = table
		p.ShardRows = table.rows
		return p, nil
	}

	p.EmbeddingShards = make([]nn.EmbeddingLayer, m.SplitNGramParts)
	for i := range m.SplitNGramParts {
		path := fmt.Sprintf("%s.ple_embedding.ngram_embedding.shard_%d", prefix, i)
		weight, err := requiredArray(tensors, path+".weight")
		if err != nil {
			return nil, err
		}
		if i == 0 {
			p.ShardRows = weight.Dim(0)
		} else if weight.Dim(0) != p.ShardRows {
			return nil, fmt.Errorf("%s has %d rows, want %d", path, weight.Dim(0), p.ShardRows)
		}
		p.EmbeddingShards[i] = model.MakeEmbeddingLayer(tensors, path, m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
		if p.EmbeddingShards[i] == nil {
			return nil, fmt.Errorf("missing embedding %q", path)
		}
	}
	return p, nil
}

func (p *PLE) NewCache(cfg *Config) *engramCache {
	return newEngramCache(int(cfg.NGramSize-1), int((cfg.PLEConvKernelSize-1)*cfg.NGramSize), int(cfg.HCCount*cfg.HiddenSize), cfg.EOSTokenID)
}

func (p *PLE) hashes(b *batch.Batch, history *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	tokens := b.InputIDs.AsType(mlx.DTypeInt64)
	joined := mlx.Concatenate([]*mlx.Array{history, tokens}, 1)
	length := b.InputIDs.Dim(1)
	shifts := make([]*mlx.Array, cfg.NGramSize)
	eos := mlx.FromValues([]int64{cfg.EOSTokenID}, 1)
	for k := range cfg.NGramSize {
		start := int(cfg.NGramSize - 1 - k)
		shifts[k] = joined.Slice(mlx.Slice(), mlx.Slice(start, start+length))
		for previous := int32(1); previous <= k; previous++ {
			start := int(cfg.NGramSize - 1 - previous)
			boundary := joined.Slice(mlx.Slice(), mlx.Slice(start, start+length))
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
	if p.hostEmbedding != nil {
		return p.hostEmbedding.lookup(globalIDs)
	}

	rows := mlx.FromValues([]int64{int64(p.ShardRows)}, 1)
	shardIDs := globalIDs.FloorDivide(rows)
	localIDs := globalIDs.Remainder(rows)
	zeroID := mlx.FromValues([]int64{0}, 1)

	// The checkpoint partitions one logical table into equal contiguous row
	// shards. Non-NVFP4 tables remain resident: gather each shard and mask out
	// rows belonging to the others. NVFP4 tables use the host-backed path above.
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
