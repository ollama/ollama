package deepseek2

// uses deepseek 2 architecture but written based on deepseek 3 model

import (
	"cmp"
	"fmt"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Options struct {
	isMLA               bool
	numExpertsUsed      int
	numExperts          int
	normTopKProb        bool
	routedScalingFactor float32

	kvLoraRank,
	qkNopeHeadDim,
	qkRopeHeadDim,
	kqNopeHeadDim,
	qkHeadDim int
	qLoraRank int
	vHeadDim  int

	hiddenSize,
	numHeads,
	numKVHeads,
	originalContextLength int

	eps,
	ropeBase,
	ropeScale float32
	kqScale float64

	attentionTemperatureScale      float32
	attentionTemperatureLength     int
	attentionTemperatureFloorScale int
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, t, p ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, t, p, o.qkRopeHeadDim, o.ropeBase, 1./o.ropeScale,
		rope.WithOriginalContextLength(o.originalContextLength),
		rope.WithExtrapolationFactor(1.),
		rope.WithAttentionFactor(float32(1.0/(1.0+0.1*math.Log(float64(o.ropeScale))))),
	)
}

type Attention struct {
	Q *nn.Linear `gguf:"attn_q"`

	QA     *nn.Linear  `gguf:"attn_q_a"`
	QANorm *nn.RMSNorm `gguf:"attn_q_a_norm"`
	QB     *nn.Linear  `gguf:"attn_q_b"`

	KVA     *nn.Linear  `gguf:"attn_kv_a_mqa"`
	KVANorm *nn.RMSNorm `gguf:"attn_kv_a_norm"`
	KVB     *nn.Linear  `gguf:"attn_kv_b"`

	KB *nn.Linear `gguf:"attn_k_b"`
	VB *nn.Linear `gguf:"attn_v_b"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"`
}

func (attn *Attention) Forward(ctx ml.Context, hiddenStates, positions, attentionScales ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	seqLength := hiddenStates.Dim(1)

	var query ml.Tensor
	if opts.qLoraRank == 0 {
		query = attn.Q.Forward(ctx, hiddenStates)
	} else {
		query = attn.QA.Forward(ctx, hiddenStates)
		query = attn.QANorm.Forward(ctx, query, opts.eps)
		query = attn.QB.Forward(ctx, query)
	}

	query = query.Reshape(ctx, query.Dim(0)/opts.numHeads, opts.numHeads, seqLength)
	queryChunks := query.ChunkSections(ctx, 0, opts.qkNopeHeadDim, opts.qkRopeHeadDim)

	compressedKV := attn.KVA.Forward(ctx, hiddenStates)
	kPass := compressedKV.Slice(ctx, 0, 0, opts.kvLoraRank, 1)
	kRot := compressedKV.View(ctx,
		opts.kvLoraRank*compressedKV.Stride(0), opts.qkRopeHeadDim,
		compressedKV.Stride(1), 1,
		compressedKV.Stride(1), compressedKV.Dim(1),
	)

	qRot := opts.applyRotaryPositionEmbeddings(ctx, queryChunks[1], positions)
	kRot = opts.applyRotaryPositionEmbeddings(ctx, kRot, positions)
	kPass = attn.KVANorm.Forward(ctx, kPass, opts.eps)

	var attention ml.Tensor

	if !opts.isMLA { // v3
		kPass = attn.KVB.Forward(ctx, kPass)

		kv := kPass.Reshape(ctx, kPass.Dim(0)/opts.numKVHeads, opts.numKVHeads, seqLength)
		kvChunks := kv.ChunkSections(ctx, 0, opts.kqNopeHeadDim, opts.vHeadDim)

		kRot = kRot.Repeat(ctx, 1, queryChunks[0].Dim(1))
		query = qRot.Concat(ctx, queryChunks[0], 0)
		key := kRot.Concat(ctx, kvChunks[0], 0)

		if attentionScales != nil {
			query = query.Mul(ctx, attentionScales)
		}

		attention = nn.Attention(ctx, query, key, kvChunks[1], opts.kqScale, cache)
	} else { // v3.1
		qPass := queryChunks[0].Permute(ctx, 0, 2, 1, 3)
		qPassAbsorb := attn.KB.Forward(ctx, qPass)
		qPassAbsorb = qPassAbsorb.Permute(ctx, 0, 2, 1, 3)

		query = qRot.Concat(ctx, qPassAbsorb, 0)
		kPass = kPass.Reshape(ctx, opts.kvLoraRank, 1, seqLength)
		key := kRot.Concat(ctx, kPass, 0)
		value := kPass

		if attentionScales != nil {
			query = query.Mul(ctx, attentionScales)
		}

		attention = nn.AttentionWithVMLA(ctx, query, key, value, nil, attn.VB.Weight, opts.kqScale, cache)
	}

	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention)
}

type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type sparse struct {
	Router       *nn.Linear `gguf:"ffn_gate_inp"`
	Gate         *nn.Linear `gguf:"ffn_gate_exps"`
	Up           *nn.Linear `gguf:"ffn_up_exps"`
	Down         *nn.Linear `gguf:"ffn_down_exps"`
	SharedExpert *dense     `gguf:",suf:_shexp"`
	ExpProbsBias ml.Tensor  `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *sparse) Moe(ctx ml.Context, hiddenStates, topKIndices, topKWeights ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	upStates := moe.Up.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = moe.Gate.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = hiddenStates.SILU(ctx, upStates)

	experts := moe.Down.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	experts = experts.Mul(ctx, topKWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}
	return nextStates
}

func (moe *sparse) topKIndices(ctx ml.Context, scores ml.Tensor, opts *Options) ml.Tensor {
	if moe.ExpProbsBias != nil {
		scores = scores.Add(ctx, moe.ExpProbsBias)
	}
	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
	return topKIndices
}

func (moe *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	residuals := hiddenStates

	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	scores := routerLogits.Sigmoid(ctx)
	topKIndices := moe.topKIndices(ctx, scores, opts)
	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)

	if opts.normTopKProb {
		topKWeights = topKWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		topKWeights = topKWeights.Div(ctx, topKWeights.SumRows(ctx))
		topKWeights = topKWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}

	topKWeights = topKWeights.Scale(ctx, float64(opts.routedScalingFactor))
	hiddenStates = moe.Moe(ctx, hiddenStates, topKIndices, topKWeights, opts)
	sharedExpertResult := moe.SharedExpert.Forward(ctx, residuals, opts)

	hiddenStates = hiddenStates.Add(ctx, sharedExpertResult)
	return hiddenStates
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Attention     *Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     MLP
}

func (t *Layer) Forward(ctx ml.Context, hiddenStates, positions, attentionScales, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	fmt.Println("[LAYER] In the new engine")
	residual := hiddenStates
	hiddenStates = t.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.Attention.Forward(ctx, hiddenStates, positions, attentionScales, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = t.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = t.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)
	return hiddenStates
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`

	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`

	*Options
}

func New(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))
	fmt.Printf("[MODEL DEBUG] Creating model with %d layers\n", c.Uint("block_count"))

	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count"))
	for i := range layers {
		if i < firstDenseLayerIndex {
			layers[i].MLP = &dense{}
		} else {
			layers[i].MLP = &sparse{}
		}
	}

	mScale := float32(1.0 + float64(c.Float("rope.scaling.yarn_log_multiplier"))*math.Log(float64(c.Float("rope.scaling.factor"))))
	kqScale := float64(mScale) * float64(mScale) / math.Sqrt(float64(c.Uint("attention.key_length")))

	isMLA := c.Uint("attention.key_length_mla") != 0 && c.Uint("attention.value_length_mla") != 0
	keyLength := int(cmp.Or(c.Uint("attention.key_length_mla"), c.Uint("attention.key_length")))
	valueLength := int(cmp.Or(c.Uint("attention.value_length_mla"), c.Uint("attention.value_length")))

	var pre []string
	fmt.Println("[TOKENIZER] Using tokenizer", c.String("tokenizer.ggml.pre"))
	switch c.String("tokenizer.ggml.pre") {
	case "deepseek-v3":
		pre = []string{
			// Split regex into multiple parts (according to DeepSeek3's regex)
			"\\p{N}{1,3}",
			`[一-龥぀-ゟ゠-ヿ]+`,
			"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
		}
	case "tekken":
		fmt.Println("[TOKENIZER] Using Tekken tokenizer")
		pre = []string{
			"[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
		}
	case "deepseek-llm":
		// TODO: these models haven't been vetted so skip for now
		// pre = []string{
		// 	"[\r\n]",
		// 	"\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
		// 	"\\s?[!-/:-~！-／：-～‘-‟　-。]+",
		// 	"\\s+$",
		// 	"[一-龥ࠀ-一가-퟿]+",
		// 	"[0-9]",
		// }
		fallthrough
	default:
		return nil, model.ErrUnsupportedTokenizer
	}

	// DEBUG: Check tokenizer vocabulary loading
	tokens := c.Strings("tokenizer.ggml.tokens")
	tokenTypes := c.Ints("tokenizer.ggml.token_type")
	merges := c.Strings("tokenizer.ggml.merges")

	fmt.Printf("[TOKENIZER DEBUG] Loading vocabulary:\n")
	fmt.Printf("[TOKENIZER DEBUG] - Tokens count: %d\n", len(tokens))
	fmt.Printf("[TOKENIZER DEBUG] - Token types count: %d\n", len(tokenTypes))
	fmt.Printf("[TOKENIZER DEBUG] - Merges count: %d\n", len(merges))
	fmt.Printf("[TOKENIZER DEBUG] - BOS token ID: %d\n", c.Uint("tokenizer.ggml.bos_token_id"))
	fmt.Printf("[TOKENIZER DEBUG] - EOS token ID: %d\n", c.Uint("tokenizer.ggml.eos_token_id"))
	fmt.Printf("[TOKENIZER DEBUG] - Add BOS: %v\n", c.Bool("tokenizer.ggml.add_bos_token", true))
	fmt.Printf("[TOKENIZER DEBUG] - Add EOS: %v\n", c.Bool("tokenizer.ggml.add_eos_token", false))

	if len(tokens) > 0 {
		maxShow := 10
		if len(tokens) < maxShow {
			maxShow = len(tokens)
		}
		fmt.Printf("[TOKENIZER DEBUG] First %d tokens: %v\n", maxShow, tokens[:maxShow])
	} else {
		fmt.Printf("[TOKENIZER DEBUG] ERROR: No tokens loaded from GGUF!\n")
	}

	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: tokens,
				Types:  tokenTypes,
				Merges: merges,
				AddBOS: false, // c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			pre...,
		),
		Layers: layers,
		Options: &Options{
			isMLA:          isMLA,
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			eps:            c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:       c.Float("rope.freq_base"),
			ropeScale:      c.Float("rope.scaling.factor", 1),
			numExperts:     int(c.Uint("expert_count")),
			numExpertsUsed: int(c.Uint("expert_used_count")),
			normTopKProb:   c.Bool("expert_weights_norm", true),

			qLoraRank:     int(c.Uint("attention.q_lora_rank")),
			kvLoraRank:    int(c.Uint("attention.kv_lora_rank")),
			qkHeadDim:     keyLength,
			vHeadDim:      valueLength,
			qkRopeHeadDim: int(c.Uint("rope.dimension_count")),
			qkNopeHeadDim: keyLength - int(c.Uint("rope.dimension_count")),
			kqNopeHeadDim: keyLength - int(c.Uint("rope.dimension_count")),

			routedScalingFactor:   c.Float("expert_weights_scale"),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),

			// TODO: double check these values
			attentionTemperatureScale:      c.Float("attention.temperature_scale", 1.0),
			attentionTemperatureLength:     int(c.Uint("attention.temperature_length")),
			attentionTemperatureFloorScale: int(c.Uint("attention.temperature_floor_scale", 8192)),

			kqScale: kqScale,
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func (m Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	// DEBUG: Check TokenEmbedding initialization
	if m.TokenEmbedding == nil {
		panic("DEBUG: m.TokenEmbedding is nil - 'token_embd' tensor not found in GGUF")
	}

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	// Temperature tuning - used by mistral-large
	var attentionScales ml.Tensor
	if m.attentionTemperatureScale != 0.0 {
		nTokens := len(batch.Positions)
		scales := make([]float32, nTokens)

		for i, pos := range batch.Positions {
			posFloat := float64(pos)
			scaleValue := math.Log(math.Floor((posFloat+1.0)/float64(m.attentionTemperatureFloorScale))+1.0)*float64(m.attentionTemperatureScale) + 1.0
			scales[i] = float32(scaleValue)
		}

		attentionScales = ctx.Input().FromFloats(scales, 1, 1, nTokens)
	}

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, attentionScales, outputs, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("deepseek2", New)
}
