package gemma3

import (
	"errors"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type embedModel struct {
	model.Base
	model.SentencePieceModel

	*TextModel
	PoolingType uint32

	Dense [2]*nn.Linear `gguf:"dense"`
}

func (m *embedModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	batch.Outputs = batch.Positions // return all positions
	hiddenStates := m.TextModel.Forward(ctx, batch, m.Cache)

	switch m.PoolingType {
	case 0: // None
	case 1: // Mean
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
		hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	default:
		return nil, errors.New("unsupported pooling type")
	}

	for _, dense := range m.Dense {
		hiddenStates = dense.Forward(ctx, hiddenStates)
	}

	return hiddenStates, nil
}

func newEmbedModel(c fs.Config) (model.Model, error) {
	m := &embedModel{
		SentencePieceModel: model.NewSentencePieceModel(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Scores: c.Floats("tokenizer.ggml.scores"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{
						int32(c.Uint("tokenizer.ggml.eos_token_id")),
						int32(c.Uint("tokenizer.ggml.eot_token_id", 106)),
					},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
		),
		TextModel:   newTextModel(c),
		PoolingType: c.Uint("pooling_type", 0),
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewSWACache(int32(c.Uint("attention.sliding_window")), m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)

	return m, nil
}
