package qwen3

import (
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type embedModel struct {
	model.Base
	model.BytePairEncoding

	*Model
	poolingType pooling.Type
}

func (m *embedModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates, err := m.forward(ctx, batch)
	if err != nil {
		return nil, err
	}

	hiddenStates = m.poolingType.Forward(ctx, hiddenStates)
	hiddenStates = hiddenStates.L2Norm(ctx, 1e-12)
	return hiddenStates, nil
}

func newEmbed(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))
	for i := range layers {
		layers[i].MLP = &dense{}
	}
	m := embedModel{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Model: &Model{
			Layers: layers,
			Options: &Options{
				hiddenSize:     int(c.Uint("embedding_length")),
				numHeads:       int(c.Uint("attention.head_count")),
				numKVHeads:     int(c.Uint("attention.head_count_kv")),
				keyLength:      int(c.Uint("attention.key_length")),
				valueLength:    int(c.Uint("attention.value_length")),
				eps:            c.Float("attention.layer_norm_rms_epsilon"),
				ropeBase:       c.Float("rope.freq_base"),
				ropeScale:      c.Float("rope.freq_scale", 1),
				numExperts:     int(c.Uint("expert_count")),
				numExpertsUsed: int(c.Uint("expert_used_count")),
				normTopKProb:   c.Bool("norm_top_k_prob", true),
			},
		},
		poolingType: pooling.Type(c.Uint("pooling_type")),
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}
