package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/models/qwen3_5"
)

type hyperConnection struct {
	Norm         *streamRMSNorm
	InputMixDown nn.LinearLayer
	InputMixUp   nn.LinearLayer
	BlockInject  nn.LinearLayer
}

type linearAttention struct {
	QKV nn.LinearLayer
	Z   nn.LinearLayer
	B   nn.LinearLayer
	A   nn.LinearLayer
	Out nn.LinearLayer

	Conv       *nn.Conv1d
	NormWeight *mlx.Array
	DtBias     *mlx.Array
	ALog       *mlx.Array
	AExp       *mlx.Array
}

type attentionIndexer struct {
	QKProj nn.LinearLayer
	QNorm  *nn.RMSNorm
	KNorm  *nn.RMSNorm
}

type fullAttention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	Indexer *attentionIndexer

	SideCacheIndex int
}

type sparseMoE struct {
	Gate           nn.LinearLayer
	GateUpExperts  *mlx.Array
	GateUpScales   *mlx.Array
	GateUpBiases   *mlx.Array
	DownExperts    *mlx.Array
	DownScales     *mlx.Array
	DownBiases     *mlx.Array
	GateUpGroup    int
	GateUpBits     int
	GateUpMode     string
	DownGroup      int
	DownBits       int
	DownMode       string
	SharedGate     nn.LinearLayer
	SharedGateProj nn.LinearLayer
	SharedUpProj   nn.LinearLayer
	SharedDownProj nn.LinearLayer
}

func supportsGatherQMM(mode string, bits int) bool {
	switch mode {
	case "affine":
		return bits == 4 || bits == 8
	case "mxfp8":
		return bits == 8
	case "nvfp4", "mxfp4":
		return bits == 4
	default:
		return false
	}
}

func loadPackedExperts(tensors map[string]*mlx.Array, base string, m *Model) (weight, scales, biases *mlx.Array, group, bits int, mode string, err error) {
	var key string
	for _, candidate := range []string{base, base + ".weight"} {
		if weight = tensors[candidate]; weight != nil {
			key = candidate
			break
		}
	}
	if weight == nil {
		return nil, nil, nil, 0, 0, "", fmt.Errorf("missing packed expert tensor %q", base)
	}
	scales = tensors[key+"_scale"]
	if scales == nil {
		return mlx.Transpose(weight, 0, 2, 1), nil, nil, 0, 0, "", nil
	}
	biases = tensors[key+"_qbias"]
	group, bits, mode = model.ResolveLinearQuantParams(m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant, key, weight, scales)
	if !supportsGatherQMM(mode, bits) {
		return nil, nil, nil, 0, 0, "", fmt.Errorf("packed expert tensor %q uses unsupported quantization mode=%q bits=%d", key, mode, bits)
	}
	return weight, scales, biases, group, bits, mode, nil
}

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
}

type streamRMSNorm struct {
	Weight *mlx.Array
}

type Layer struct {
	AttentionConnection *hyperConnection
	MLPConnection       *hyperConnection
	Linear              *linearAttention
	Attention           *fullAttention
	MoE                 *sparseMoE
	PLE                 *PLE
}

type MTPHead struct {
	EmbeddingNorm *nn.RMSNorm
	HiddenNorm    *nn.RMSNorm
	FCEmbedding   nn.LinearLayer
	FCHidden      nn.LinearLayer
	Mixer         *hyperConnection
	Layer         *Layer
}

func requiredArray(tensors map[string]*mlx.Array, names ...string) (*mlx.Array, error) {
	for _, name := range names {
		if value := tensors[name]; value != nil {
			return value, nil
		}
	}
	return nil, fmt.Errorf("missing tensor %q", names[0])
}

func requiredLinear(linears model.LinearFactory, name string) (nn.LinearLayer, error) {
	value := linears.Make(name)
	if value == nil {
		return nil, fmt.Errorf("missing linear %q", name)
	}
	return value, nil
}

func loadNorm(tensors map[string]*mlx.Array, name string, eps float32) (*nn.RMSNorm, error) {
	weight, err := requiredArray(tensors, name+".weight", name)
	if err != nil {
		return nil, err
	}
	// Qwen 4 RMSNorm weights are stored zero-centered.
	return nn.NewRMSNorm(mlx.AddScalar(weight, 1), eps), nil
}

func loadStreamNorm(tensors map[string]*mlx.Array, name string, cfg *Config) (*streamRMSNorm, error) {
	weight, err := requiredArray(tensors, name+".weight", name)
	if err != nil {
		return nil, err
	}
	if weight.Size() != int(cfg.HCCount*cfg.HiddenSize) {
		return nil, fmt.Errorf("%s has %d elements, want %d", name, weight.Size(), cfg.HCCount*cfg.HiddenSize)
	}
	weight = mlx.Reshape(mlx.AddScalar(weight, 1), cfg.HCCount, cfg.HiddenSize)
	return &streamRMSNorm{Weight: weight}, nil
}

func loadHyperConnection(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, withInjection bool, cfg *Config) (*hyperConnection, error) {
	h := &hyperConnection{}
	var err error
	if h.Norm, err = loadStreamNorm(tensors, prefix+".hc_norm", cfg); err != nil {
		return nil, err
	}
	if h.InputMixDown, err = requiredLinear(linears, prefix+".input_mix_weight_down"); err != nil {
		return nil, err
	}
	if h.InputMixUp, err = requiredLinear(linears, prefix+".input_mix_weight_up"); err != nil {
		return nil, err
	}
	if withInjection {
		if h.BlockInject, err = requiredLinear(linears, prefix+".block_inject_weight"); err != nil {
			return nil, err
		}
	}
	return h, nil
}

func loadLinearAttention(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*linearAttention, error) {
	a := &linearAttention{}
	var err error
	for name, dst := range map[string]*nn.LinearLayer{
		"in_proj_qkv": &a.QKV,
		"in_proj_z":   &a.Z,
		"in_proj_b":   &a.B,
		"in_proj_a":   &a.A,
		"out_proj":    &a.Out,
	} {
		*dst, err = requiredLinear(linears, prefix+"."+name)
		if err != nil {
			return nil, err
		}
	}
	convWeight, err := requiredArray(tensors, prefix+".conv1d.weight", prefix+".conv1d")
	if err != nil {
		return nil, err
	}
	if convWeight.NumDims() == 3 && convWeight.Dim(1) == 1 {
		convWeight = mlx.Reshape(convWeight, int32(convWeight.Dim(0)), int32(convWeight.Dim(2)))
	}
	if convWeight.NumDims() != 2 {
		return nil, fmt.Errorf("%s.conv1d weight must reduce to 2D, got %dD", prefix, convWeight.NumDims())
	}
	a.Conv = nn.NewConv1d(mlx.ExpandDims(convWeight, 2), nil, 1, 0, 1, int32(convWeight.Dim(0)))
	if a.NormWeight, err = requiredArray(tensors, prefix+".norm.weight", prefix+".norm"); err != nil {
		return nil, err
	}
	if a.DtBias, err = requiredArray(tensors, prefix+".dt_bias", prefix+".dt_proj"); err != nil {
		return nil, err
	}
	if a.ALog, err = requiredArray(tensors, prefix+".A_log", prefix+".a_log"); err != nil {
		return nil, err
	}
	a.AExp = mlx.Exp(a.ALog.AsType(mlx.DTypeFloat32))
	return a, nil
}

func loadFullAttention(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, eps float32) (*fullAttention, error) {
	a := &fullAttention{}
	var err error
	for name, dst := range map[string]*nn.LinearLayer{
		"q_proj": &a.QProj,
		"k_proj": &a.KProj,
		"v_proj": &a.VProj,
		"o_proj": &a.OProj,
	} {
		*dst, err = requiredLinear(linears, prefix+"."+name)
		if err != nil {
			return nil, err
		}
	}
	if a.QNorm, err = loadNorm(tensors, prefix+".q_norm", eps); err != nil {
		return nil, err
	}
	if a.KNorm, err = loadNorm(tensors, prefix+".k_norm", eps); err != nil {
		return nil, err
	}

	idx := &attentionIndexer{}
	if idx.QKProj, err = requiredLinear(linears, prefix+".indexer.index_qk_proj"); err != nil {
		return nil, err
	}
	if idx.QNorm, err = loadNorm(tensors, prefix+".indexer.q_layernorm", eps); err != nil {
		return nil, err
	}
	if idx.KNorm, err = loadNorm(tensors, prefix+".indexer.k_layernorm", eps); err != nil {
		return nil, err
	}
	a.Indexer = idx
	return a, nil
}

func (m *Model) loadSparseMoE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*sparseMoE, error) {
	moe := &sparseMoE{}
	var err error
	if moe.Gate, err = requiredLinear(linears, prefix+".gate"); err != nil {
		return nil, err
	}
	if moe.GateUpExperts, moe.GateUpScales, moe.GateUpBiases, moe.GateUpGroup, moe.GateUpBits, moe.GateUpMode, err = loadPackedExperts(tensors, prefix+".experts.gate_up_proj", m); err != nil {
		return nil, err
	}
	if moe.DownExperts, moe.DownScales, moe.DownBiases, moe.DownGroup, moe.DownBits, moe.DownMode, err = loadPackedExperts(tensors, prefix+".experts.down_proj", m); err != nil {
		return nil, err
	}
	for name, dst := range map[string]*nn.LinearLayer{
		"shared_expert_gate":      &moe.SharedGate,
		"shared_expert.gate_proj": &moe.SharedGateProj,
		"shared_expert.up_proj":   &moe.SharedUpProj,
		"shared_expert.down_proj": &moe.SharedDownProj,
	} {
		*dst, err = requiredLinear(linears, prefix+"."+name)
		if err != nil {
			return nil, err
		}
	}
	return moe, nil
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

func (m *Model) loadLayer(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix, kind string, layer int) (*Layer, error) {
	l := &Layer{}
	var err error
	if l.AttentionConnection, err = loadHyperConnection(linears, tensors, prefix+".attn_hyper_connection", true, m.Config); err != nil {
		return nil, err
	}
	if l.MLPConnection, err = loadHyperConnection(linears, tensors, prefix+".mlp_hyper_connection", true, m.Config); err != nil {
		return nil, err
	}
	if kind == "linear_attention" {
		if l.Linear, err = loadLinearAttention(linears, tensors, prefix+".linear_attn"); err != nil {
			return nil, err
		}
	} else if l.Attention, err = loadFullAttention(linears, tensors, prefix+".self_attn", m.RMSNormEps); err != nil {
		return nil, err
	}
	if l.MoE, err = m.loadSparseMoE(linears, tensors, prefix+".mlp"); err != nil {
		return nil, err
	}

	for _, id := range m.PLELayerIDs {
		// PLE layer IDs are one-based and identify the layer receiving the
		// additive n-gram features before its attention branch.
		if int(id-1) == layer {
			l.PLE, err = m.loadPLE(linears, tensors, prefix+".ple")
			if err != nil {
				return nil, err
			}
		}
	}
	return l, nil
}

func (m *Model) loadMTP(linears model.LinearFactory, tensors map[string]*mlx.Array) (*MTPHead, error) {
	head := &MTPHead{}
	var err error
	if head.EmbeddingNorm, err = loadNorm(tensors, "mtp.pre_fc_norm_embedding", m.RMSNormEps); err != nil {
		return nil, err
	}
	if head.HiddenNorm, err = loadNorm(tensors, "mtp.pre_fc_norm_hidden", m.RMSNormEps); err != nil {
		return nil, err
	}
	if head.FCEmbedding, err = requiredLinear(linears, "mtp.fc_embedding"); err != nil {
		return nil, err
	}
	if head.FCHidden, err = requiredLinear(linears, "mtp.fc_hidden"); err != nil {
		return nil, err
	}
	if head.Mixer, err = loadHyperConnection(linears, tensors, "mtp.hyper_connection_mixer", false, m.Config); err != nil {
		return nil, err
	}
	if head.Layer, err = m.loadLayer(linears, tensors, "mtp.layers.0", m.Config.MTP.LayerTypes[0], -1); err != nil {
		return nil, err
	}
	return head, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
	var err error
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.language_model.embed_tokens", m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight")
	}
	if m.LMHead, err = requiredLinear(linears, "lm_head"); err != nil {
		return err
	}
	if m.OutputMixer, err = loadHyperConnection(linears, tensors, "model.language_model.hyper_connection_mixer", false, m.Config); err != nil {
		return err
	}

	sideCacheIndex := len(m.Layers)
	for i := range m.Layers {
		prefix := fmt.Sprintf("model.language_model.layers.%d", i)
		if m.Layers[i], err = m.loadLayer(linears, tensors, prefix, m.LayerTypes[i], i); err != nil {
			return fmt.Errorf("load layer %d: %w", i, err)
		}
		if m.Layers[i].Attention != nil {
			m.Layers[i].Attention.SideCacheIndex = sideCacheIndex
			sideCacheIndex++
		}
	}
	if m.MTP, err = m.loadMTP(linears, tensors); err != nil {
		return fmt.Errorf("load MTP: %w", err)
	}
	m.Vision, err = qwen3_5.NewVisionAdapter(m.configData, tensors, qwen3_5.VisionAdapterConfig{
		HiddenSize:   m.HiddenSize,
		RopeDim:      m.RopeDim,
		RopeTheta:    m.RopeParameters.RopeTheta,
		MropeSection: m.RopeParameters.MRoPESection,
		QuantGroup:   m.quantGroup,
		QuantBits:    m.quantBits,
		QuantMode:    m.quantMode,
		TensorQuant:  m.tensorQuant,
	})
	if err != nil {
		return fmt.Errorf("load vision tower: %w", err)
	}

	return nil
}
