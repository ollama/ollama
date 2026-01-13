package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type nomicbertModel struct {
	ModelParameters
	NLayers               uint32  `json:"n_layers"`
	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	LayerNormEPS          float32 `json:"layer_norm_eps"`
	LayerNormEpsilon      float32 `json:"layer_norm_epsilon"`
	RopeFreqBase          float32 `json:"rope_theta"`
	normalizeEmbeddings   bool
	PoolingType           uint32

	// MoE parameters (only present in v2 models)
	NumExperts      uint32 `json:"num_local_experts"`
	NumExpertsUsed  uint32 `json:"num_experts_per_tok"`
	MoEEveryNLayers uint32 `json:"moe_every_n_layers"`
}

var (
	_ ModelConverter = (*nomicbertModel)(nil)
	_ moreParser     = (*nomicbertModel)(nil)
)

func (p *nomicbertModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err != nil {
		return err
	}

	var modules []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}

	if err := json.Unmarshal(bts, &modules); err != nil {
		return err
	}

	var pooling string
	for _, m := range modules {
		switch m.Type {
		case "sentence_transformers.models.Pooling":
			pooling = m.Path
		case "sentence_transformers.models.Normalize":
			p.normalizeEmbeddings = true
		}
	}

	if pooling != "" {
		bts, err := fs.ReadFile(fsys, filepath.Join(pooling, "config.json"))
		if err != nil {
			return err
		}

		var pc struct {
			PoolingModeCLSToken   bool `json:"pooling_mode_cls_token"`
			PoolingModeMeanTokens bool `json:"pooling_mode_mean_tokens"`
		}

		if err := json.Unmarshal(bts, &pc); err != nil {
			return err
		}

		if pc.PoolingModeMeanTokens {
			p.PoolingType = 1
		} else if pc.PoolingModeCLSToken {
			p.PoolingType = 2
		}
	}

	return nil
}

func (p *nomicbertModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)

	// Determine architecture based on MoE parameters (following qwen3 pattern)
	arch := "nomic-bert"
	if p.MoEEveryNLayers > 0 {
		arch += "-moe"
	}

	kv["general.architecture"] = arch
	kv["attention.causal"] = false
	kv["pooling_type"] = p.PoolingType
	kv["normalize_embeddings"] = p.normalizeEmbeddings

	kv["block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers)

	if contextLength := p.MaxPositionEmbeddings; contextLength > 0 {
		kv["context_length"] = contextLength
	}

	if embeddingLength := p.HiddenSize; embeddingLength > 0 {
		kv["embedding_length"] = p.HiddenSize
	}

	if feedForwardLength := p.IntermediateSize; feedForwardLength > 0 {
		kv["feed_forward_length"] = p.IntermediateSize
	}

	if headCount := p.NumAttentionHeads; headCount > 0 {
		kv["attention.head_count"] = p.NumAttentionHeads
	}

	if kvHeadCount := p.NumKeyValueHeads; kvHeadCount > 0 {
		kv["attention.head_count_kv"] = p.NumKeyValueHeads
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon); layerNormEpsilon > 0 {
		kv["attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	if p.RopeFreqBase > 0 {
		kv["rope.freq_base"] = p.RopeFreqBase
	}

	// MoE specific parameters (only if MoE is enabled)
	if p.NumExperts > 0 {
		kv["expert_count"] = p.NumExperts
	}

	if p.NumExpertsUsed > 0 {
		kv["expert_used_count"] = p.NumExpertsUsed
	}

	if p.MoEEveryNLayers > 0 {
		kv["moe_every_n_layers"] = p.MoEEveryNLayers
	}

	kv["tokenizer.ggml.model"] = "bert"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)

	// convert to phantom space tokens
	for i, e := range t.Tokens {
		switch {
		case strings.HasPrefix(e, "[") && strings.HasSuffix(e, "]"):
			// noop - keep special tokens as-is
		case strings.HasPrefix(e, "##"):
			t.Tokens[i] = e[2:]
		default:
			t.Tokens[i] = "\u2581" + e
		}
	}

	kv["tokenizer.ggml.tokens"] = t.Tokens

	return kv
}

func (p *nomicbertModel) Tensors(ts []Tensor) []*ggml.Tensor {
	out := make([]*ggml.Tensor, 0, len(ts))
	for _, t := range ts {
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) {
			continue
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (nomicbertModel) Replacements() []string {
	return []string{
		"encoder.layer", "blk",
		"encoder.layers", "blk",
		"embeddings.word_embeddings", "token_embd",
		"embeddings.token_type_embeddings", "token_types",
		"embeddings.LayerNorm", "token_embd_norm",

		"attention.self.qkv", "attn_qkv",

		"attention.output.dense", "attn_output",
		"attention.output.LayerNorm", "attn_output_norm",

		"mlp.up", "ffn_up",
		"mlp.down", "ffn_down",

		"mlp.router", "ffn_gate_inp",
		"mlp.experts.up", "ffn_up_exps",
		"mlp.experts.down", "ffn_down_exps",

		"intermediate.dense", "ffn_up",
		"output.dense", "ffn_down",
		"output.LayerNorm", "layer_output_norm",
	}
}
