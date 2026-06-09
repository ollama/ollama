package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type embeddingGemmaModel struct {
	gemmaModel
	RopeLocalTheta float32 `json:"rope_local_base_freq"`
	RopeTheta      float32 `json:"rope_theta"`
	SlidingWindow  uint32  `json:"sliding_window"`

	poolingType  uint32
	denseModules []embeddingGemmaDenseModule
}

type embeddingGemmaDenseModule struct {
	path       string
	tensorName string
	in, out    uint32
}

var (
	_ ModelConverter    = (*embeddingGemmaModel)(nil)
	_ moreParser        = (*embeddingGemmaModel)(nil)
	_ extraTensorParser = (*embeddingGemmaModel)(nil)
	_ tokenizerAdjuster = (*embeddingGemmaModel)(nil)
)

func (m *embeddingGemmaModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma-embedding"
	kv["gemma-embedding.context_length"] = cmp.Or(m.MaxPositionEmbeddings, uint32(2048))
	kv["gemma-embedding.embedding_length"] = m.HiddenSize
	kv["gemma-embedding.block_count"] = m.HiddenLayers
	kv["gemma-embedding.feed_forward_length"] = m.IntermediateSize
	kv["gemma-embedding.attention.head_count"] = m.NumAttentionHeads
	kv["gemma-embedding.attention.head_count_kv"] = m.NumKeyValueHeads
	kv["gemma-embedding.attention.layer_norm_rms_epsilon"] = cmp.Or(m.RMSNormEPS, float32(1e-6))
	kv["gemma-embedding.attention.key_length"] = m.HeadDim
	kv["gemma-embedding.attention.value_length"] = m.HeadDim
	kv["gemma-embedding.attention.sliding_window"] = m.SlidingWindow
	kv["gemma-embedding.rope.freq_base"] = cmp.Or(m.RopeTheta, float32(1000000.0))
	kv["gemma-embedding.rope.freq_base_swa"] = cmp.Or(m.RopeLocalTheta, float32(10000.0))
	kv["gemma-embedding.pooling_type"] = cmp.Or(m.poolingType, uint32(1))

	for _, dense := range m.denseModules {
		kv["gemma-embedding."+dense.tensorName+"_feat_in"] = dense.in
		kv["gemma-embedding."+dense.tensorName+"_feat_out"] = dense.out
	}

	return kv
}

func (m *embeddingGemmaModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return errors.New("embeddinggemma requires sentence-transformers modules.json")
		}
		return err
	}

	var modules []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}

	if err := json.Unmarshal(bts, &modules); err != nil {
		return err
	}

	m.poolingType = 1
	m.denseModules = nil
	for _, module := range modules {
		switch module.Type {
		case "sentence_transformers.models.Pooling":
			poolingType, err := embeddingGemmaPoolingType(fsys, module.Path)
			if err != nil {
				return err
			}
			if poolingType != 0 {
				m.poolingType = poolingType
			}
		case "sentence_transformers.models.Dense":
			dense, ok, err := embeddingGemmaDenseModuleConfig(fsys, module.Path)
			if err != nil {
				return err
			}
			if ok {
				m.denseModules = append(m.denseModules, dense)
			}
		}
	}

	slices.SortFunc(m.denseModules, func(a, b embeddingGemmaDenseModule) int {
		return strings.Compare(a.tensorName, b.tensorName)
	})

	if len(m.denseModules) != 2 ||
		m.denseModules[0].tensorName != "dense_2" ||
		m.denseModules[1].tensorName != "dense_3" {
		return errors.New("embeddinggemma requires sentence-transformers 2_Dense and 3_Dense modules")
	}

	return nil
}

func (m *embeddingGemmaModel) adjustTokenizer(t *Tokenizer) {
	n := int(m.VocabSize)
	if n == 0 || len(t.Vocabulary.Tokens) <= n {
		return
	}

	t.Vocabulary.Tokens = t.Vocabulary.Tokens[:n]
	if len(t.Vocabulary.Scores) > n {
		t.Vocabulary.Scores = t.Vocabulary.Scores[:n]
	}
	if len(t.Vocabulary.Types) > n {
		t.Vocabulary.Types = t.Vocabulary.Types[:n]
	}
}

func embeddingGemmaPoolingType(fsys fs.FS, modulePath string) (uint32, error) {
	if modulePath == "" {
		return 0, nil
	}

	bts, err := fs.ReadFile(fsys, path.Join(modulePath, "config.json"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}

	var cfg struct {
		PoolingModeMeanTokens bool `json:"pooling_mode_mean_tokens"`
		PoolingModeCLSToken   bool `json:"pooling_mode_cls_token"`
	}
	if err := json.Unmarshal(bts, &cfg); err != nil {
		return 0, err
	}

	switch {
	case cfg.PoolingModeMeanTokens:
		return 1, nil
	case cfg.PoolingModeCLSToken:
		return 2, nil
	default:
		return 0, nil
	}
}

func embeddingGemmaDenseModuleConfig(fsys fs.FS, modulePath string) (embeddingGemmaDenseModule, bool, error) {
	tensorName, ok := embeddingGemmaDenseTensorName(modulePath)
	if !ok {
		return embeddingGemmaDenseModule{}, false, nil
	}

	weightsPath := path.Join(modulePath, "model.safetensors")
	if _, err := fs.Stat(fsys, weightsPath); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return embeddingGemmaDenseModule{}, false, nil
		}
		return embeddingGemmaDenseModule{}, false, err
	}

	bts, err := fs.ReadFile(fsys, path.Join(modulePath, "config.json"))
	if err != nil {
		return embeddingGemmaDenseModule{}, false, err
	}

	var cfg struct {
		InFeatures  uint32 `json:"in_features"`
		OutFeatures uint32 `json:"out_features"`
		Bias        bool   `json:"bias"`
	}
	if err := json.Unmarshal(bts, &cfg); err != nil {
		return embeddingGemmaDenseModule{}, false, err
	}
	if cfg.InFeatures == 0 || cfg.OutFeatures == 0 {
		return embeddingGemmaDenseModule{}, false, errors.New("embeddinggemma dense layer config missing in/out features")
	}
	if cfg.Bias {
		return embeddingGemmaDenseModule{}, false, fmt.Errorf("embeddinggemma dense layer %s has unsupported bias", modulePath)
	}

	return embeddingGemmaDenseModule{
		path:       weightsPath,
		tensorName: tensorName,
		in:         cfg.InFeatures,
		out:        cfg.OutFeatures,
	}, true, nil
}

func embeddingGemmaDenseTensorName(modulePath string) (string, bool) {
	switch modulePath {
	case "2_Dense":
		return "dense_2", true
	case "3_Dense":
		return "dense_3", true
	default:
		return "", false
	}
}

func (m *embeddingGemmaModel) extraTensors(fsys fs.FS) ([]Tensor, error) {
	var extra []Tensor
	for _, dense := range m.denseModules {
		ts, err := parseSafetensors(fsys, strings.NewReplacer("linear.", dense.tensorName+"."), dense.path)
		if err != nil {
			return nil, err
		}

		foundWeight := false
		for _, t := range ts {
			if t.Name() == dense.tensorName+".weight" {
				extra = append(extra, t)
				foundWeight = true
			}
		}
		if !foundWeight {
			return nil, fmt.Errorf("embeddinggemma dense module %s missing linear.weight", dense.path)
		}
	}

	return extra, nil
}

func (m *embeddingGemmaModel) Tensors(ts []Tensor) []*ggml.Tensor {
	out := make([]*ggml.Tensor, 0, len(ts))
	for _, t := range ts {
		name := t.Name()
		if name == "norm.weight" {
			name = "output_norm.weight"
		}
		if strings.HasSuffix(name, "_norm.weight") {
			t.SetRepacker(m.addOne)
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (m *embeddingGemmaModel) Replacements() []string {
	return []string{
		"embed_tokens.", "token_embd.",
		"layers.", "blk.",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
	}
}
