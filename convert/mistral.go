package convert

import (
	"io"
	"regexp"

	"github.com/uppercaveman/ollama-server/llm"
)

type MistralModel struct {
	ModelData
}

func (m *MistralModel) GetTensors() error {
	t, err := m.Format.GetTensors(m.Path, m.Params)
	if err != nil {
		return err
	}

	pattern := `^blk\.[0-9]+\.attn_(?P<layer>q|k)\.weight$`
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}

	for _, l := range t {
		matches := re.FindAllStringSubmatch(l.Name, -1)
		if len(matches) > 0 {
			wt := l.WriterTo.(safetensorWriterTo)
			wt.repacker = m.Repack
			l.WriterTo = wt
		}
		m.Tensors = append(m.Tensors, l)
	}

	return nil
}

func (m *MistralModel) LoadVocab() error {
	v, err := LoadSentencePieceTokens(m.Path, m.Params)
	if err != nil {
		return err
	}
	m.Vocab = v
	return nil
}

func (m *MistralModel) WriteGGUF(ws io.WriteSeeker) error {
	kv := llm.KV{
		"general.architecture":                   "llama",
		"general.name":                           m.Name,
		"llama.context_length":                   uint32(m.Params.ContextSize),
		"llama.embedding_length":                 uint32(m.Params.HiddenSize),
		"llama.block_count":                      uint32(m.Params.HiddenLayers),
		"llama.feed_forward_length":              uint32(m.Params.IntermediateSize),
		"llama.rope.dimension_count":             uint32(m.Params.HiddenSize / m.Params.AttentionHeads),
		"llama.attention.head_count":             uint32(m.Params.AttentionHeads),
		"llama.attention.head_count_kv":          uint32(m.Params.KeyValHeads),
		"llama.attention.layer_norm_rms_epsilon": float32(m.Params.NormEPS),
		"general.file_type":                      uint32(1),
		"tokenizer.ggml.model":                   "llama",

		"tokenizer.ggml.tokens":     m.Vocab.Tokens,
		"tokenizer.ggml.scores":     m.Vocab.Scores,
		"tokenizer.ggml.token_type": m.Vocab.Types,

		"tokenizer.ggml.bos_token_id":     uint32(m.Params.BoSTokenID),
		"tokenizer.ggml.eos_token_id":     uint32(m.Params.EoSTokenID),
		"tokenizer.ggml.add_bos_token":    true,
		"tokenizer.ggml.add_eos_token":    false,
		"tokenizer.ggml.unknown_token_id": uint32(0),
	}

	return llm.NewGGUFV3(m.Params.ByteOrder).Encode(ws, kv, m.Tensors)
}

func (m *MistralModel) Repack(name string, data []float32, shape []uint64) ([]float32, error) {
	return llamaRepack(name, m.Params, data, shape)
}
