package convert

import (
	"os"

	"github.com/ollama/ollama/llm"
)

type GemmaModel struct {
	Path    string
	Name    string
	Params  *Params
	Vocab   *Vocab
	Tensors []llm.Tensor
}

func (m *GemmaModel) GetTensors() error {
	t, err := GetSafeTensors(m.Path, m.Params)
	if err != nil {
		return err
	}
	m.Tensors = t
	return nil
}

func (m *GemmaModel) LoadVocab() error {
	v, err := LoadSentencePieceTokens(m.Path, m.Params.VocabSize)
	if err != nil {
		return err
	}
	m.Vocab = v
	return nil
}

func (m *GemmaModel) WriteGGUF() (string, error) {
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
