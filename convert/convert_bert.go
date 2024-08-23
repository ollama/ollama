package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/llm"
)

type bertModel struct {
	ModelParameters
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
	LayerNormEPS          float32 `json:"layer_norm_eps"`
	LayerNormEpsilon      float32 `json:"layer_norm_epsilon"`
	NormEpsilon           float32 `json:"norm_epsilon"`

	PoolingType uint32
}

var (
	_ ModelConverter = (*bertModel)(nil)
	_ moreParser     = (*bertModel)(nil)
)

func (p *bertModel) parseMore(fsys fs.FS) error {
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
		if m.Type == "sentence_transformers.models.Pooling" {
			pooling = m.Path
			break
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

func (p *bertModel) KV(t *Tokenizer) llm.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "bert"
	kv["bert.attention.causal"] = false
	kv["bert.pooling_type"] = p.PoolingType

	kv["bert.block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers, p.NLayer)

	if contextLength := cmp.Or(p.MaxPositionEmbeddings, p.NCtx); contextLength > 0 {
		kv["bert.context_length"] = contextLength
	}

	if embeddingLength := cmp.Or(p.HiddenSize, p.NEmbd); embeddingLength > 0 {
		kv["bert.embedding_length"] = cmp.Or(p.HiddenSize, p.NEmbd)
	}

	if feedForwardLength := cmp.Or(p.IntermediateSize, p.NInner); feedForwardLength > 0 {
		kv["bert.feed_forward_length"] = cmp.Or(p.IntermediateSize, p.NInner)
	}

	if headCount := cmp.Or(p.NumAttentionHeads, p.NHead); headCount > 0 {
		kv["bert.attention.head_count"] = cmp.Or(p.NumAttentionHeads, p.NHead)
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon, p.NormEpsilon); layerNormEpsilon > 0 {
		kv["bert.attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	kv["tokenizer.ggml.model"] = "bert"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)

	// convert to phantom space tokens
	for i, e := range t.Tokens {
		if strings.HasPrefix(e, "[") && strings.HasSuffix(e, "]") {
			// noop
		} else if strings.HasPrefix(e, "##") {
			t.Tokens[i] = e[2:]
		} else {
			t.Tokens[i] = "\u2581" + e
		}
	}

	kv["tokenizer.ggml.tokens"] = t.Tokens

	return kv
}

func (p *bertModel) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor
	for _, t := range ts {
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) {
			continue
		}

		out = append(out, llm.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (bertModel) Replacements() []string {
	return []string{
		"encoder.layer", "blk",
		"encoder.layers", "blk",
		"embeddings.word_embeddings", "token_embd",
		"embeddings.token_type_embeddings", "token_types",
		"embeddings.LayerNorm", "token_embd_norm",
		"embeddings.position_embeddings", "position_embd",
		"attention.self.query", "attn_q",
		"attention.self.key", "attn_k",
		"attention.self.value", "attn_v",
		"attention.output.dense", "attn_output",
		"attention.output.LayerNorm", "attn_output_norm",
		"intermediate.dense", "ffn_up",
		"output.dense", "ffn_down",
		"output.LayerNorm", "layer_output_norm",
	}
}
