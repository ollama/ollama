package deepseek_v41

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type transformerLayer struct {
	AttentionMix  connectionMix
	FFNMix        connectionMix
	Attention     *latentAttention
	FFN           *mixtureOfExperts
	AttentionNorm *nn.RMSNorm
	FFNNorm       *nn.RMSNorm
	Engram        *engramLayer
}

func requireTensor(tensors map[string]*mlx.Array, name string) (*mlx.Array, error) {
	array := tensors[name]
	if array == nil {
		return nil, fmt.Errorf("missing DeepSeek tensor %s", name)
	}
	return array, nil
}

func loadConnectionMix(tensors map[string]*mlx.Array, prefix string, cfg *Config) (connectionMix, error) {
	projection, err := requireTensor(tensors, prefix+"_fn")
	if err != nil {
		return connectionMix{}, err
	}
	base, err := requireTensor(tensors, prefix+"_base")
	if err != nil {
		return connectionMix{}, err
	}
	scale, err := requireTensor(tensors, prefix+"_scale")
	if err != nil {
		return connectionMix{}, err
	}
	count := (cfg.HCMult + 2) * cfg.HCMult
	if projection.DType() != mlx.DTypeFloat32 || base.DType() != mlx.DTypeFloat32 || scale.DType() != mlx.DTypeFloat32 ||
		projection.NumDims() != 2 || projection.Dim(0) != count || projection.Dim(1) != cfg.HCMult*cfg.HiddenSize ||
		base.Size() != count || scale.Size() != 3 {
		return connectionMix{}, fmt.Errorf("invalid hyper-connection tensors at %s", prefix)
	}
	return connectionMix{
		Projection: projection, Base: base, Scale: scale, Copies: int32(cfg.HCMult),
		Iterations: cfg.HCSinkhornIters, NormEpsilon: float32(cfg.RMSNormEps), MixEpsilon: float32(cfg.HCEps),
	}, nil
}

func loadTransformerLayer(tensors map[string]*mlx.Array, quant map[string]*model.TensorQuantInfo, cfg *Config, index int, table *hostEngramTable) (*transformerLayer, error) {
	prefix := fmt.Sprintf("layers.%d", index)
	var layer transformerLayer
	var err error
	if layer.AttentionMix, err = loadConnectionMix(tensors, prefix+".hc_attn", cfg); err != nil {
		return nil, err
	}
	if layer.FFNMix, err = loadConnectionMix(tensors, prefix+".hc_ffn", cfg); err != nil {
		return nil, err
	}
	attnNorm, err := requireTensor(tensors, prefix+".attn_norm.weight")
	if err != nil {
		return nil, err
	}
	ffnNorm, err := requireTensor(tensors, prefix+".ffn_norm.weight")
	if err != nil {
		return nil, err
	}
	layer.AttentionNorm = nn.NewRMSNorm(attnNorm, float32(cfg.RMSNormEps))
	layer.FFNNorm = nn.NewRMSNorm(ffnNorm, float32(cfg.RMSNormEps))
	if layer.Attention, err = loadLatentAttention(tensors, quant, cfg, prefix+".attn", index); err != nil {
		return nil, err
	}
	if layer.FFN, err = loadMixtureOfExperts(tensors, quant, cfg, prefix+".ffn"); err != nil {
		return nil, err
	}
	if table != nil {
		layer.Engram, err = loadEngramLayer(tensors, quant, cfg, prefix+".engram", table)
		if err != nil {
			return nil, err
		}
	}
	return &layer, nil
}

func loadLatentAttention(tensors map[string]*mlx.Array, quant map[string]*model.TensorQuantInfo, cfg *Config, prefix string, index int) (*latentAttention, error) {
	var err error
	a := &latentAttention{
		Heads: int32(cfg.NumAttentionHeads), HeadWidth: int32(cfg.HeadDim), Window: cfg.SlidingWindow,
		Frequencies: rotaryFrequencies(cfg, cfg.CompressRatios[index] > 0),
	}
	if a.QueryLow, err = makeProjection(tensors, quant, prefix+".wq_a"); err != nil {
		return nil, err
	}
	if a.QueryHigh, err = makeProjection(tensors, quant, prefix+".wq_b"); err != nil {
		return nil, err
	}
	if a.KeyValue, err = makeProjection(tensors, quant, prefix+".wkv"); err != nil {
		return nil, err
	}
	if a.OutputHigh, err = makeProjection(tensors, quant, prefix+".wo_b"); err != nil {
		return nil, err
	}
	packed, err := requireTensor(tensors, prefix+".wo_a.weight")
	if err != nil {
		return nil, err
	}
	scales, err := requireTensor(tensors, prefix+".wo_a.weight_scale")
	if err != nil {
		return nil, err
	}
	a.OutputLow, err = newGroupedOutput(packed, scales, cfg.OGroups, cfg.NumAttentionHeads*cfg.HeadDim/cfg.OGroups, cfg.OLoRARank)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", prefix, err)
	}
	qNorm, err := requireTensor(tensors, prefix+".q_norm.weight")
	if err != nil {
		return nil, err
	}
	kNorm, err := requireTensor(tensors, prefix+".kv_norm.weight")
	if err != nil {
		return nil, err
	}
	a.QueryNorm = nn.NewRMSNorm(qNorm, float32(cfg.RMSNormEps))
	a.KeyNorm = nn.NewRMSNorm(kNorm, float32(cfg.RMSNormEps))
	a.Sink, err = requireTensor(tensors, prefix+".attn_sink")
	if err != nil {
		return nil, err
	}
	for _, source := range cfg.KVSourceLayerIDs {
		if source != index {
			continue
		}
		c := &compressor{Ratio: cfg.CompressRatios[index]}
		if c.KeyValue, err = makeProjection(tensors, quant, prefix+".compressor.wkv"); err != nil {
			return nil, err
		}
		if c.Ratio > 1 {
			if c.Gate, err = makeProjection(tensors, quant, prefix+".compressor.wgate"); err != nil {
				return nil, err
			}
		}
		weight, err := requireTensor(tensors, prefix+".compressor.norm.weight")
		if err != nil {
			return nil, err
		}
		c.Norm = nn.NewRMSNorm(weight, float32(cfg.RMSNormEps))
		a.Compressor = c
		break
	}
	for _, source := range cfg.IndexSourceLayerIDs {
		if source != index {
			continue
		}
		idx := &sparseIndexer{
			Heads: cfg.IndexNumHeads, Width: cfg.IndexHeadDim, TopK: cfg.IndexTopK,
			CandidateSource: index == cfg.CandidateSourceLayerID,
			UsesCandidates:  index > cfg.CandidateSourceLayerID,
			CandidateBlocks: cfg.CandidateTopKBlocks, CandidateBlockSize: cfg.CandidateBlockSize,
		}
		if idx.Query, err = makeProjection(tensors, quant, prefix+".indexer.wq_b"); err != nil {
			return nil, err
		}
		if idx.Weights, err = makeProjection(tensors, quant, prefix+".indexer.weights_proj"); err != nil {
			return nil, err
		}
		if a.Compressor != nil {
			if idx.Key, err = makeProjection(tensors, quant, prefix+".indexer.wk"); err != nil {
				return nil, err
			}
			weight, err := requireTensor(tensors, prefix+".indexer.k_norm.weight")
			if err != nil {
				return nil, err
			}
			idx.KeyNorm = nn.NewRMSNorm(weight, float32(cfg.RMSNormEps))
		}
		a.Indexer = idx
		break
	}
	return a, nil
}

func loadMixtureOfExperts(tensors map[string]*mlx.Array, quant map[string]*model.TensorQuantInfo, cfg *Config, prefix string) (*mixtureOfExperts, error) {
	gateWeight, err := requireTensor(tensors, prefix+".gate.weight")
	if err != nil {
		return nil, err
	}
	bias, err := requireTensor(tensors, prefix+".gate.bias")
	if err != nil {
		return nil, err
	}
	m := &mixtureOfExperts{
		Gate: &expertGate{
			Weight: gateWeight, Bias: bias, ImageBias: tensors[prefix+".gate.bias_vl"],
			TopK: int32(cfg.NumExpertsPerToken), Scale: float32(cfg.RoutedScalingFactor),
		}, Limit: float32(cfg.SwiGLULimit),
		Shared: make([]nn.LinearLayer, 3),
	}
	for _, projection := range []struct {
		name          string
		output, input int
		bank          *expertBank
	}{
		{"w1", cfg.MoeIntermediateSize, cfg.HiddenSize, &m.GateUp},
		{"w2", cfg.HiddenSize, cfg.MoeIntermediateSize, &m.Down},
		{"w3", cfg.MoeIntermediateSize, cfg.HiddenSize, &m.Up},
	} {
		*projection.bank, err = loadExpertBank(tensors, prefix+".experts."+projection.name,
			cfg.NumRoutedExperts, projection.output, projection.input)
		if err != nil {
			return nil, err
		}
	}
	for i, name := range []string{"w1", "w2", "w3"} {
		m.Shared[i], err = makeProjection(tensors, quant, prefix+".shared_experts."+name)
		if err != nil {
			return nil, err
		}
	}
	return m, nil
}

func loadEngramLayer(tensors map[string]*mlx.Array, quant map[string]*model.TensorQuantInfo, cfg *Config, prefix string, table *hostEngramTable) (*engramLayer, error) {
	projection, err := makeProjection(tensors, quant, prefix+".wkv")
	if err != nil {
		return nil, err
	}
	query, err := requireTensor(tensors, prefix+".q_weight")
	if err != nil {
		return nil, err
	}
	key, err := requireTensor(tensors, prefix+".k_weight")
	if err != nil {
		return nil, err
	}
	if query.NumDims() != 2 || key.NumDims() != 2 ||
		query.Dim(0) != cfg.HCMult || query.Dim(1) != cfg.HiddenSize || key.Dim(0) != cfg.HCMult || key.Dim(1) != cfg.HiddenSize {
		return nil, fmt.Errorf("invalid engram gate dimensions at %s", prefix)
	}
	return &engramLayer{
		Table: table, Project: projection, Query: query, Key: key,
		Copies: int32(cfg.HCMult), Width: int32(cfg.HiddenSize), Epsilon: float32(cfg.RMSNormEps),
	}, nil
}
