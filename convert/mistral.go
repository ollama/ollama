package convert

import (
	"github.com/ollama/ollama/llm"
	"os"
)

type MistralModel struct {
	Path    string
	Name    string
	Params  *Params
	Vocab   *Vocab
	Tensors []llm.Tensor
}

func (m *MistralModel) GetTensors() error {
	t, err := GetSafeTensors(m.Path, m.Params)
	if err != nil {
		return err
	}
	m.Tensors = t
	return nil
}

func (m *MistralModel) LoadVocab() error {
	v, err := LoadSentencePieceTokens(m.Path, m.Params.VocabSize)
	if err != nil {
		return err
	}
	m.Vocab = v
	return nil
}

func (m *MistralModel) WriteGGUF() (string, error) {
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
		"llama.rope.freq_base":                   float32(m.Params.RopeFreqBase),
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

	f, err := os.CreateTemp("", "ollama-gguf")
	if err != nil {
		return "", err
	}
	defer f.Close()

	mod := llm.NewGGUFV3(m.Params.ByteOrder)
	if err := mod.Encode(f, kv, m.Tensors); err != nil {
		return "", err
	}

	return f.Name(), nil
}
