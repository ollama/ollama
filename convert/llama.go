package convert

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/uppercaveman/ollama-server/llm"
)

type LlamaModel struct {
	ModelData
}

func (m *LlamaModel) GetTensors() error {
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
			switch m.Format.(type) {
			case *TorchFormat:
				wt := l.WriterTo.(torchWriterTo)
				wt.repacker = m.Repack
				l.WriterTo = wt
			case *SafetensorFormat:
				wt := l.WriterTo.(safetensorWriterTo)
				wt.repacker = m.Repack
				l.WriterTo = wt
			}
		}
		m.Tensors = append(m.Tensors, l)
	}

	return nil
}

func (m *LlamaModel) LoadVocab() (err error) {
	pre, ts, merges, err := parseTokens(filepath.Join(m.Path, "tokenizer.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}

	m.Vocab = &Vocab{}
	for _, t := range ts {
		m.Vocab.Tokens = append(m.Vocab.Tokens, t.Content)
		m.Vocab.Types = append(m.Vocab.Types, t.Type())
	}

	m.Vocab.Merges = merges
	m.Params.PreTokenizer = pre
	return nil
}

func (m *LlamaModel) WriteGGUF(ws io.WriteSeeker) error {
	kv := llm.KV{
		"general.architecture":                   "llama",
		"general.name":                           m.Name,
		"llama.vocab_size":                       uint32(len(m.Vocab.Tokens)),
		"llama.context_length":                   uint32(m.Params.ContextSize),
		"llama.embedding_length":                 uint32(m.Params.HiddenSize),
		"llama.block_count":                      uint32(m.Params.HiddenLayers),
		"llama.feed_forward_length":              uint32(m.Params.IntermediateSize),
		"llama.rope.freq_base":                   float32(m.Params.RopeFrequencyBase),
		"llama.rope.dimension_count":             uint32(m.Params.HiddenSize / m.Params.AttentionHeads),
		"llama.attention.head_count":             uint32(m.Params.AttentionHeads),
		"llama.attention.head_count_kv":          uint32(m.Params.KeyValHeads),
		"llama.attention.layer_norm_rms_epsilon": float32(m.Params.NormEPS),
		"general.file_type":                      uint32(1),
		"tokenizer.ggml.model":                   "gpt2",

		"tokenizer.ggml.pre":        m.Params.PreTokenizer,
		"tokenizer.ggml.tokens":     m.Vocab.Tokens,
		"tokenizer.ggml.token_type": m.Vocab.Types,

		"tokenizer.ggml.bos_token_id":     uint32(m.Params.BoSTokenID),
		"tokenizer.ggml.eos_token_id":     uint32(m.Params.EoSTokenID),
		"tokenizer.ggml.unknown_token_id": uint32(0),
	}

	if len(m.Vocab.Merges) > 0 {
		kv["tokenizer.ggml.merges"] = m.Vocab.Merges
	} else {
		kv["tokenizer.ggml.scores"] = m.Vocab.Scores
	}

	return llm.NewGGUFV3(m.Params.ByteOrder).Encode(ws, kv, m.Tensors)
}

func (m *LlamaModel) Repack(name string, data []float32, shape []uint64) ([]float32, error) {
	return llamaRepack(name, m.Params, data, shape)
}

func llamaRepack(name string, params *Params, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		if dim != 0 {
			dims = append(dims, int(dim))
		}
	}

	var heads int
	if strings.HasSuffix(name, "attn_q.weight") {
		heads = params.AttentionHeads
	} else if strings.HasSuffix(name, "attn_k.weight") {
		heads = cmp.Or(params.KeyValHeads, params.AttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor name: %s", name)
	}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
	if err := n.Reshape(append([]int{heads, 2, dims[0] / heads / 2}, dims[1:]...)...); err != nil {
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
