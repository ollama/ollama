package convert

import (
	"cmp"
	"fmt"
	"log/slog"
	"math"
	"os"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type graniteModel struct {
	GraniteMoEParameters
	ModelParameters
	ModelType             string  `json:"model_type"` // TBD: Why is this not in ModelParameters?
	NLayers               uint32  `json:"n_layers"`
	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	NLayer                uint32  `json:"n_layer"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NCtx                  uint32  `json:"n_ctx"`
	HiddenSize            uint32  `json:"hidden_size"`
	NEmbd                 uint32  `json:"n_embd"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NInner                uint32  `json:"n_inner"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NHead                 uint32  `json:"n_head"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScaling           struct {
		Type                            string  `json:"type"`
		RopeType                        string  `json:"rope_type"`
		Factor                          float32 `json:"factor"`
		LowFrequencyFactor              float32 `json:"low_freq_factor"`
		HighFrequencyFactor             float32 `json:"high_freq_factor"`
		OriginalMaxPositionalEmbeddings uint32  `json:"original_max_positional_embeddings"`

		factors ropeFactor
	} `json:"rope_scaling"`
	RMSNormEPS          float32 `json:"rms_norm_eps"`
	LayerNormEPS        float32 `json:"layer_norm_eps"`
	LayerNormEpsilon    float32 `json:"layer_norm_epsilon"`
	NormEpsilon         float32 `json:"norm_epsilon"`
	HeadDim             uint32  `json:"head_dim"`
	AttentionMultiplier float32 `json:"attention_multiplier"`
	EmbeddingMultiplier float32 `json:"embedding_multiplier"`
	ResidualMultiplier  float32 `json:"residual_multiplier"`
	LogitsScaling       float32 `json:"logits_scaling"`
}

type GraniteMoEParameters struct {
	NumExperts     uint32 `json:"num_local_experts"`
	NumExpertsUsed uint32 `json:"num_experts_per_tok"`
}

func (p *graniteModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		"block_sparse_moe.router.layer", "ffn_gate_inp", // granitemoe
		"block_sparse_moe.output_linear", "ffn_down_exps", // granitemoe
		"block_sparse_moe.input_linear", "ffn_gate_exps", // granitemoe
	}
}

var _ ModelConverter = (*graniteModel)(nil)

func (p *graniteModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)

	// General metadata
	kv["general.type"] = "model"

	// SHOULD be set from config.json, "torch_dtype", but "finetunes" can alter; often overridden looking at first_tensor.dtype
	kv["general.file_type"] = uint32(1) // GGUF constant (LlamaFileType(IntEnum)), MOSTLY_F16 = 1

	// architecture-specific values (i.e. from HF config.json, tokenizer.json, etc.)
	modelType := p.ModelType
	kv["general.architecture"] = modelType

	kv[modelType+".vocab_size"] = p.VocabSize
	kv[modelType+".block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers, p.NLayer)

	if contextLength := cmp.Or(p.MaxPositionEmbeddings, p.NCtx); contextLength > 0 {
		kv[modelType+".context_length"] = contextLength
	}

	if embeddingLength := cmp.Or(p.HiddenSize, p.NEmbd); embeddingLength > 0 {
		kv[modelType+".embedding_length"] = cmp.Or(p.HiddenSize, p.NEmbd)
	}

	if feedForwardLength := cmp.Or(p.IntermediateSize, p.NInner); feedForwardLength > 0 {
		kv[modelType+".feed_forward_length"] = cmp.Or(p.IntermediateSize, p.NInner)
	}

	if headCount := cmp.Or(p.NumAttentionHeads, p.NHead); headCount > 0 {
		kv[modelType+".attention.head_count"] = cmp.Or(p.NumAttentionHeads, p.NHead)
		kv[modelType+".rope.dimension_count"] = p.HiddenSize / headCount
	}

	if p.RopeTheta > 0 {
		kv[modelType+".rope.freq_base"] = p.RopeTheta
	}

	if p.RopeScaling.Type == "linear" {
		kv[modelType+".rope.scaling.type"] = p.RopeScaling.Type
		kv[modelType+".rope.scaling.factor"] = p.RopeScaling.Factor
	} else if p.RopeScaling.RopeType == "llama3" {
		dim := p.HiddenSize / p.NumAttentionHeads
		for i := uint32(0); i < dim; i += 2 {
			factor := cmp.Or(p.RopeScaling.Factor, 8.0)
			factorLow := cmp.Or(p.RopeScaling.LowFrequencyFactor, 1.0)
			factorHigh := cmp.Or(p.RopeScaling.HighFrequencyFactor, 4.0)

			original := cmp.Or(p.RopeScaling.OriginalMaxPositionalEmbeddings, 8192)
			lambdaLow := float32(original) / factorLow
			lambdaHigh := float32(original) / factorHigh

			lambda := 2 * math.Pi * math.Pow(float64(p.RopeTheta), float64(i)/float64(dim))
			if lambda < float64(lambdaHigh) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0)
			} else if lambda > float64(lambdaLow) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, factor)
			} else {
				smooth := (float32(original)/float32(lambda) - factorLow) / (factorHigh - factorLow)
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0/((1-smooth)/factor+smooth))
			}
		}
	}

	if p.NumKeyValueHeads > 0 {
		kv[modelType+".attention.head_count_kv"] = p.NumKeyValueHeads
	}

	if p.RMSNormEPS > 0 {
		kv[modelType+".attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon, p.NormEpsilon); layerNormEpsilon > 0 {
		kv[modelType+".attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	if p.HeadDim > 0 {
		kv[modelType+".attention.key_length"] = p.HeadDim
		kv[modelType+".attention.value_length"] = p.HeadDim
	}

	// Granite multipliers
	if p.AttentionMultiplier != 0 {
		kv[modelType+".attention.scale"] = p.AttentionMultiplier
	}
	if p.EmbeddingMultiplier != 0 {
		kv[modelType+".embedding_scale"] = p.EmbeddingMultiplier
	}
	if p.ResidualMultiplier != 0 {
		kv[modelType+".residual_scale"] = p.ResidualMultiplier
	}
	if p.LogitsScaling != 0 {
		kv[modelType+".logit_scale"] = p.LogitsScaling
	}

	// Granite Moe params
	kv[modelType+".expert_count"] = p.NumExperts
	kv[modelType+".expert_used_count"] = p.NumExpertsUsed

	return kv
}

// In general, the GraniteMoE FFN tensor has dimensions (40, 512, 1536)
// - i.e. (batch size, sequence length, embedding dimension)
// as represented in the model's config.json:
//   - "num_local_experts": 40, "intermediate_size": 512, "hidden_size": 1536,
func (p *graniteModel) getFeedForwardLengthFromFFNShape(ffnTensor []uint64) (uint64, error) {
	if len(ffnTensor) != 3 {
		err := fmt.Errorf("FFN Tensor must be size 3, actual: %v", len(ffnTensor))
		return 0, err
	}
	return ffnTensor[1], nil
}

func (p *graniteModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	if p.RopeScaling.factors != nil {
		out = append(out, ggml.Tensor{
			Name:     "rope_freqs.weight",
			Kind:     0,
			Shape:    []uint64{uint64(len(p.RopeScaling.factors))},
			WriterTo: p.RopeScaling.factors,
		})
	}

	ffn_dim_64 := uint64(p.IntermediateSize)

	for _, t := range ts {
		if strings.HasSuffix(t.Name(), "attn_q.weight") ||
			strings.HasSuffix(t.Name(), "attn_k.weight") {
			t.SetRepacker(p.repack)
		} else if index := strings.Index(t.Name(), "ffn_gate_exps.weight"); index > -1 {
			// In modeling_granitemoe, the JetMoe implementation of parallel experts
			// is used. This essentially merges w1 (i.e., "ffn_gate_exp") and w3 (i.e., "ffn_gate_up")
			// into a single tensor with 2x the hidden size that is then split during forward.
			merged_ffn_length, err := p.getFeedForwardLengthFromFFNShape(t.Shape())
			if err != nil {
				slog.Error(err.Error())
				os.Exit(1) // TODO: what is convention to exit gracefully if error during conversion?
			}
			// Assure tensor is actually 2x size expected
			if merged_ffn_length != 2*ffn_dim_64 {
				slog.Error(fmt.Sprintf("Merged FFN tensor size (%v) must be 2 * intermediate_size (%v)", merged_ffn_length, ffn_dim_64))
				os.Exit(1)
			}

			// The tensor package uses (int) for dimensions, so we should verify before casting from (uint64)s
			if merged_ffn_length > uint64(math.MaxInt64) { // Check for overflow
				slog.Error(fmt.Sprintf("Cannot convert (uint64) value: %v to (int):", merged_ffn_length))
				os.Exit(1)
			}

			// Duplicate (copy) "ffn_gate_exp" Tensor to "ffn_gate_up" Tensor
			block_prefix := t.Name()[:index]
			ffn_gate_up_exp_name := block_prefix + "ffn_up_exps.weight"
			int_merged_ffn_length := int(merged_ffn_length)
			var newFfnShape []uint64 = []uint64{t.Shape()[0], ffn_dim_64, t.Shape()[2]}

			gate := t.Clone()
			gate.SetRepacker(p.getMoESparseInputRepacker(nil, tensor.S(0, int_merged_ffn_length/2), nil))

			up := t.Clone()
			up.SetRepacker(p.getMoESparseInputRepacker(nil, tensor.S(int_merged_ffn_length/2, int_merged_ffn_length), nil))

			gtGate := &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    newFfnShape,
				WriterTo: gate, // Note: same safetensor (cloned) data on both "gate" and "up" gguf.Tensors
			}
			out = append(out, *gtGate)

			gtUp := &ggml.Tensor{
				Name:     ffn_gate_up_exp_name,
				Kind:     t.Kind(),
				Shape:    newFfnShape,
				WriterTo: up, // Note: same safetensor (cloned) data on both "gate" and "up" gguf.Tensors
			}
			out = append(out, *gtUp)

			continue
		}

		// append gguf tensors to the output list
		out = append(out, ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *graniteModel) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		dims = append(dims, int(dim))
	}

	var heads uint32
	if strings.HasSuffix(name, "attn_q.weight") {
		heads = p.NumAttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight") {
		heads = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor for repack: %s", name)
	}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
	if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
		return nil, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
		return nil, err
	}

	if err := n.Reshape(dims...); err != nil {
		return nil, err
	}

	if err := n.Transpose(); err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}

// GraniteMoe models, in the HF safetensor format, are encoded such that the
// "ffn_gate_exps" and "ffn_gate_up" tensors are merged into the "ffn_gate_exps" tensor
// and need to be "sliced" in half from 1024->512 in this "repack" function.
func (p *graniteModel) getMoESparseInputRepacker(tSlice ...tensor.Slice) Repacker {
	return func(name string, data []float32, shape []uint64) (f32s []float32, err error) {
		dims := make([]int, len(shape))
		for i, dim := range shape {
			dims[i] = int(dim)
		}

		var t tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

		// Note: we should only be slicing the dim[1] value
		if tSlice != nil {
			slog.Debug(fmt.Sprintf("slicing: tensor: %s using tSlice[1]: start: %v, end: %v, step: %v\n", name, tSlice[1].Start(), tSlice[1].End(), tSlice[1].Step()))
			t, err = t.Slice(tSlice...)
			if err != nil {
				return nil, err
			}
		} else {
			slog.Error(fmt.Sprintf("Cannot repack tensor %s. Slice is nil", name))
			os.Exit(1)
		}

		// "realize" the data for the "Slice" (which only creates a view)
		t = tensor.Materialize(t)

		// flatten tensor so it can be return as a vector
		if err := t.Reshape(t.Shape().TotalSize()); err != nil {
			return nil, err
		}

		// convert tensor to float32 slice
		f32s, err = native.VectorF32(t.(*tensor.Dense))
		return f32s, err
	}
}
