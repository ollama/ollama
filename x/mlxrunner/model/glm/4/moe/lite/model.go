package glm

import (
	"encoding/json"
	"math"

	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type Options struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`

	QLoraRank     int `json:"q_lora_rank"`
	KVLoraRank    int `json:"kv_lora_rank"`
	QKRopeHeadDim int `json:"qk_rope_head_dim"`
	QKNopeHeadDim int `json:"qk_nope_head_dim"`

	NumRoutedExperts    int     `json:"n_routed_experts"`
	NumSharedExperts    int     `json:"n_shared_experts"`
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`
	RoutedScalingFactor float32 `json:"routed_scaling_factor"`
	NormTopKProb        bool    `json:"norm_topk_prob"`
	FirstKDenseReplace  int     `json:"first_k_dense_replace"`

	mlx.RoPE
}

type Model struct {
	EmbedTokens mlx.Embedding `weight:"model.embed_tokens"`
	Layers      []Layer       `weight:"model.layers"`
	Norm        mlx.RMSNorm   `weight:"model.norm"`
	LMHead      mlx.Linear    `weight:"lm_head"`

	Options
}

func (m Model) NumLayers() int {
	return len(m.Layers)
}

func (m Model) Forward(inputs *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := inputs.Dim(0), inputs.Dim(1)
	h := m.EmbedTokens.Forward(inputs)
	for i, layer := range m.Layers {
		h = layer.Forward(h, caches[i], B, L, m.Options)
	}

	h = m.Norm.Forward(h, m.RMSNormEps)
	return h
}

func (m Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

type Layer struct {
	InputLayernorm         mlx.RMSNorm `weight:"input_layernorm"`
	Attention              Attention   `weight:"self_attn"`
	PostAttentionLayernorm mlx.RMSNorm `weight:"post_attention_layernorm"`
	MLP                    MLP         `weight:"mlp"`
}

func (m Layer) Forward(h *mlx.Array, cache cache.Cache, B, L int, opts Options) *mlx.Array {
	r := h
	h = m.InputLayernorm.Forward(h, opts.RMSNormEps)
	h = m.Attention.Forward(h, cache, B, L, opts)
	h = h.Add(r)

	r = h
	h = m.PostAttentionLayernorm.Forward(h, opts.RMSNormEps)
	h = m.MLP.Forward(h, B, L, opts)
	h = h.Add(r)
	return h
}

type MultiLinear struct {
	Weight mlx.Array `weight:"weight"`
}

func (m MultiLinear) Forward(x *mlx.Array) *mlx.Array {
	return x.Matmul(m.Weight.Transpose(0, 2, 1))
}

type Attention struct {
	QAProj      mlx.Linear  `weight:"q_a_proj"`
	QALayernorm mlx.RMSNorm `weight:"q_a_layernorm"`
	QBProj      mlx.Linear  `weight:"q_b_proj"`

	KVAProjWithMQA mlx.Linear  `weight:"kv_a_proj_with_mqa"`
	KVALayernorm   mlx.RMSNorm `weight:"kv_a_layernorm"`
	KVBProj        mlx.Linear  `weight:"kv_b_proj"`

	embedQ     MultiLinear
	unembedOut MultiLinear

	OProj mlx.Linear `weight:"o_proj"`
}

func (m *Attention) AfterLoad(root *model.Root) ([]*mlx.Array, error) {
	bts, err := root.ReadFile("config.json")
	if err != nil {
		return nil, err
	}

	var opts struct {
		NumAttentionHeads int `json:"num_attention_heads"`
		QKNopeHeadDim     int `json:"qk_nope_head_dim"`
		KVLoraRank        int `json:"kv_lora_rank"`
	}
	if err := json.Unmarshal(bts, &opts); err != nil {
		return nil, err
	}

	w := m.KVBProj.Weight.Reshape(opts.NumAttentionHeads, -1, opts.KVLoraRank)
	m.embedQ.Weight.Set(w.Slice(mlx.Slice(), mlx.Slice(0, opts.QKNopeHeadDim), mlx.Slice()).Transpose(0, 2, 1))
	m.unembedOut.Weight.Set(w.Slice(mlx.Slice(), mlx.Slice(opts.QKNopeHeadDim, 0), mlx.Slice()))

	return []*mlx.Array{
		&m.embedQ.Weight,
		&m.unembedOut.Weight,
	}, nil
}

func (m Attention) Forward(hiddenStates *mlx.Array, cache cache.Cache, B, L int, opts Options) *mlx.Array {
	query := m.QAProj.Forward(hiddenStates)
	query = m.QALayernorm.Forward(query, opts.RMSNormEps)
	query = m.QBProj.Forward(query)

	query = query.Reshape(B, L, opts.NumAttentionHeads, -1)
	query = query.Transpose(0, 2, 1, 3)

	queryNope := query.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, opts.QKNopeHeadDim))
	queryRope := query.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(opts.QKNopeHeadDim, 0))

	compressedKV := m.KVAProjWithMQA.Forward(hiddenStates)

	keyRope := compressedKV.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(opts.KVLoraRank, 0))
	keyRope = keyRope.Reshape(B, L, 1, opts.QKRopeHeadDim)
	keyRope = keyRope.Transpose(0, 2, 1, 3)

	kvCompressed := compressedKV.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, opts.KVLoraRank))

	var offset int
	if cache != nil {
		offset = cache.Offset()
	}

	queryRope = opts.RoPE.Forward(queryRope, offset)
	keyRope = opts.RoPE.Forward(keyRope, offset)

	key := m.KVALayernorm.Forward(kvCompressed, opts.RMSNormEps).
		ExpandDims(1).
		Concatenate(3, keyRope)

	if cache != nil {
		key, _ = cache.Update(key, mlx.Zeros(mlx.DTypeBFloat16, B, 1, L, 0))
	}

	value := key.Clone().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, opts.KVLoraRank))
	query = m.embedQ.Forward(queryNope).Concatenate(3, queryRope)

	attention := mlx.ScaledDotProductAttention(query, key, value, nil, float32(1.0/math.Sqrt(float64(opts.QKNopeHeadDim+opts.QKRopeHeadDim))))
	attention = m.unembedOut.Forward(attention)
	attention = attention.Transpose(0, 2, 1, 3).Reshape(B, L, -1)
	return m.OProj.Forward(attention)
}

type MLP interface {
	Forward(*mlx.Array, int, int, Options) *mlx.Array
}

type dense struct {
	GateProj mlx.Linear `weight:"gate_proj"`
	UpProj   mlx.Linear `weight:"up_proj"`
	DownProj mlx.Linear `weight:"down_proj"`
}

func (m dense) Forward(h *mlx.Array, _, _ int, opts Options) *mlx.Array {
	h = mlx.SILU(m.GateProj.Forward(h)).Multiply(m.UpProj.Forward(h))
	return m.DownProj.Forward(h)
}

type Gate struct {
	Gate           mlx.Linear `weight:"gate"`
	CorrectionBias mlx.Array  `weight:"gate.e_score_correction_bias"`
}

var expertSelect *mlx.Closure

func ExpertSelect(opts Options) *mlx.Closure {
	if expertSelect == nil {
		expertSelect = mlx.Compile(func(inputs []*mlx.Array) []*mlx.Array {
			scores, correctionBias := inputs[0], inputs[1]

			scores = scores.Sigmoid()
			original := scores
			scores = scores.Add(correctionBias)

			indices := scores.Negative().ArgpartitionAxis(opts.NumExpertsPerTok-1, -1)
			indices = indices.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, opts.NumExpertsPerTok))

			scores = original.TakeAlongAxis(indices, -1)
			if opts.NumExpertsPerTok > 1 && opts.NormTopKProb {
				scores = scores.Divide(scores.SumAxis(-1, true).Add(mlx.FromValue[float32](1e-20)))
			}

			scores = scores.Multiply(mlx.FromValue(opts.RoutedScalingFactor))
			return []*mlx.Array{indices, scores}
		}, false)
	}

	return expertSelect
}

func (m Gate) Forward(h *mlx.Array, opts Options) (indices, scores *mlx.Array) {
	outputs := ExpertSelect(opts).Call([]*mlx.Array{
		m.Gate.Forward(h).AsType(mlx.DTypeFloat32),
		&m.CorrectionBias,
	})
	return outputs[0], outputs[1]
}

type sparse struct {
	Gate

	Experts []dense `weight:"experts"`
	fused   struct {
		GateProj mlx.Linear
		UpProj   mlx.Linear
		DownProj mlx.Linear
	}

	SharedExperts dense `weight:"shared_experts"`
}

func (m *sparse) AfterLoad(*model.Root) ([]*mlx.Array, error) {
	w1 := make([]*mlx.Array, len(m.Experts))
	w2 := make([]*mlx.Array, len(m.Experts))
	w3 := make([]*mlx.Array, len(m.Experts))

	for i := range m.Experts {
		w1[i] = &m.Experts[i].GateProj.Weight
		w2[i] = &m.Experts[i].UpProj.Weight
		w3[i] = &m.Experts[i].DownProj.Weight
	}

	m.fused.GateProj.Weight.Set(w1[0].StackAxis(0, w1[1:]...))
	m.fused.UpProj.Weight.Set(w2[0].StackAxis(0, w2[1:]...))
	m.fused.DownProj.Weight.Set(w3[0].StackAxis(0, w3[1:]...))

	return []*mlx.Array{
		&m.fused.GateProj.Weight,
		&m.fused.UpProj.Weight,
		&m.fused.DownProj.Weight,
	}, nil
}

func (m sparse) Forward(h *mlx.Array, B, L int, opts Options) *mlx.Array {
	indices, scores := m.Gate.Forward(h, opts)
	scores = scores.ExpandDims(-1)

	flat := h.ExpandDims(-2).ExpandDims(-2).Reshape(-1, 1, 1, opts.HiddenSize)
	indices = indices.Reshape(-1, opts.NumExpertsPerTok)

	sort := B*L >= 64
	var inverseOrder *mlx.Array
	if sort {
		indicesAll := indices.Flatten(0, len(indices.Dims())-1)
		order := indicesAll.ArgsortAxis(0)
		inverseOrder = order.ArgsortAxis(0)
		flat = flat.Squeeze(1).TakeAxis(order.FloorDivide(mlx.FromValue(opts.NumExpertsPerTok)), 0).ExpandDims(1)
		indices = indicesAll.TakeAxis(order, 0).Reshape(B*L*opts.NumExpertsPerTok, 1)
	}

	experts := mlx.SILU(m.fused.GateProj.Gather(flat, nil, indices, sort)).
		Multiply(m.fused.UpProj.Gather(flat, nil, indices, sort))
	experts = m.fused.DownProj.Gather(experts, nil, indices, sort)

	if sort {
		experts = experts.Squeeze(2).Squeeze(1).TakeAxis(inverseOrder, 0)
		experts = experts.Reshape(-1, opts.NumExpertsPerTok, opts.HiddenSize)
	} else {
		experts = experts.Squeeze(2)
	}

	experts = experts.Reshape(B, L, opts.NumExpertsPerTok, opts.HiddenSize)
	experts = experts.Multiply(scores).SumAxis(-2, false).AsType(experts.DType())
	experts = experts.Add(m.SharedExperts.Forward(h, B, L, opts))
	return experts.Reshape(B, L, -1)
}

func init() {
	base.Register("Glm4MoeLiteForCausalLM", func(root *model.Root) (base.Model, error) {
		bts, err := root.ReadFile("config.json")
		if err != nil {
			return nil, err
		}

		var opts Options
		if err := json.Unmarshal(bts, &opts); err != nil {
			return nil, err
		}

		opts.RoPE = mlx.RoPE{
			Dims:        opts.QKRopeHeadDim,
			Traditional: true,
			Base:        opts.RopeTheta,
			Scale:       1,
		}

		layers := make([]Layer, opts.NumHiddenLayers)
		for i := range layers {
			if i < opts.FirstKDenseReplace {
				layers[i].MLP = &dense{}
			} else {
				layers[i].MLP = &sparse{Experts: make([]dense, opts.NumRoutedExperts)}
			}
		}

		return &Model{
			Layers:  layers,
			Options: opts,
		}, nil
	})
}
