package convert

import (
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/uppercaveman/ollama-server/llm"
)

type GemmaModel struct {
	ModelData
}

func addOnes(data []float32, vectorSize int) ([]float32, error) {
	n := tensor.New(tensor.WithShape(vectorSize), tensor.WithBacking(data))
	ones := tensor.Ones(tensor.Float32, vectorSize)

	n, err := n.Add(ones)
	if err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 0)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}


	return f32s, nil
}

func (m *GemmaModel) GetTensors() error {
	t, err := m.Format.GetTensors(m.Path, m.Params)
	if err != nil {
		return err
	}

	slog.Debug(fmt.Sprintf("Total tensors: %d", len(t)))
	for _, l := range t {
		if strings.HasSuffix(l.Name, "norm.weight") {
			wt := l.WriterTo.(safetensorWriterTo)
			wt.repacker = m.Repack
			l.WriterTo = wt
		}
		m.Tensors = append(m.Tensors, l)
	}

	return nil
}

func (m *GemmaModel) LoadVocab() error {
	v, err := LoadSentencePieceTokens(m.Path, m.Params)
	if err != nil {
		return err
	}
	m.Vocab = v
	return nil
}

func (m *GemmaModel) Repack(_ string, data []float32, shape []uint64) ([]float32, error) {
	return addOnes(data, int(shape[0]))
}

func (m *GemmaModel) WriteGGUF(ws io.WriteSeeker) error {
	kv := llm.KV{
		"general.architecture":                   "gemma",
		"general.name":                           m.Name,
		"gemma.context_length":                   uint32(m.Params.ContextSize),
		"gemma.embedding_length":                 uint32(m.Params.HiddenSize),
		"gemma.block_count":                      uint32(m.Params.HiddenLayers),
		"gemma.feed_forward_length":              uint32(m.Params.IntermediateSize),
		"gemma.attention.head_count":             uint32(m.Params.AttentionHeads),
		"gemma.attention.head_count_kv":          uint32(m.Params.KeyValHeads),
		"gemma.attention.layer_norm_rms_epsilon": float32(m.Params.NormEPS),
		"gemma.attention.key_length":             uint32(m.Params.HeadDimension),
		"gemma.attention.value_length":           uint32(m.Params.HeadDimension),
		"general.file_type":                      uint32(1),
		"tokenizer.ggml.model":                   "llama",

		"tokenizer.ggml.tokens":     m.Vocab.Tokens,
		"tokenizer.ggml.scores":     m.Vocab.Scores,
		"tokenizer.ggml.token_type": m.Vocab.Types,

		"tokenizer.ggml.bos_token_id":     uint32(m.Params.BoSTokenID),
		"tokenizer.ggml.eos_token_id":     uint32(m.Params.EoSTokenID),
		"tokenizer.ggml.padding_token_id": uint32(m.Params.PaddingTokenID),
		"tokenizer.ggml.unknown_token_id": uint32(3),
		"tokenizer.ggml.add_bos_token":    true,
		"tokenizer.ggml.add_eos_token":    false,
	}

	return llm.NewGGUFV3(m.Params.ByteOrder).Encode(ws, kv, m.Tensors)
}
