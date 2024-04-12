package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"os"
	"regexp"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"github.com/x448/float16"

	"github.com/ollama/ollama/llm"
)

type LlamaModel struct {
	ModelData
}

func llamaLayerHandler(w io.Writer, r torchWriterTo) error {
	slog.Debug(fmt.Sprintf("repacking layer '%s'", r.t.Name))

	data := r.storage.(*pytorch.HalfStorage).Data
	tData := make([]uint16, len(data))
	for cnt, v := range data {
		tData[cnt] = uint16(float16.Fromfloat32(v))
	}

	var err error
	var heads uint32
	if strings.Contains(r.t.Name, "attn_q") {
		heads = uint32(r.params.AttentionHeads)
	} else if strings.Contains(r.t.Name, "attn_k") {
		heads = uint32(r.params.KeyValHeads)
		if heads == 0 {
			heads = uint32(r.params.AttentionHeads)
		}
	} else {
		return fmt.Errorf("unknown layer type")
	}

	slog.Debug(fmt.Sprintf("heads = %d", heads))

	tData, err = llamaRepack(tData, int(heads), r.t.Shape)
	if err != nil {
		return err
	}

	if err = binary.Write(w, r.bo, tData); err != nil {
		return err
	}
	return nil
}

func llamaRepack(data []uint16, heads int, shape []uint64) ([]uint16, error) {
	n := tensor.New(tensor.WithShape(int(shape[0]), int(shape[1])), tensor.WithBacking(data))
	origShape := n.Shape().Clone()

	// reshape the tensor and swap axes 1 and 2 to unpack the layer for gguf
	if err := n.Reshape(heads, 2, origShape[0]/heads/2, origShape[1]); err != nil {
		return nil, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
		return nil, err
	}

	if err := n.Reshape(origShape...); err != nil {
		return nil, err
	}

	if err := n.Transpose(); err != nil {
		return nil, err
	}
	newN, err := native.SelectU16(n, 1)
	if err != nil {
		return nil, err
	}

	var fullTensor []uint16
	for _, v := range newN {
		fullTensor = append(fullTensor, v...)
	}
	return fullTensor, nil
}

func (m *LlamaModel) GetTensors() error {
	t, err := m.Format.GetTensors(m.Path, m.Params)
	if err != nil {
		return err
	}

	m.Tensors = []llm.Tensor{}

	pattern := `^blk\.[0-9]+\.attn_(?P<layer>q|k)\.weight$`
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}

	for _, l := range t {
		matches := re.FindAllStringSubmatch(l.Name, -1)
		if len(matches) > 0 {
			slog.Debug(fmt.Sprintf("setting handler for: %s", l.Name))
			wt := l.WriterTo.(torchWriterTo)
			wt.handler = llamaLayerHandler
			l.WriterTo = wt
		}
		m.Tensors = append(m.Tensors, l)
	}

	return nil
}

func (m *LlamaModel) LoadVocab() error {
	var v *Vocab
	var err error

	slog.Debug("loading vocab")
	v, err = LoadSentencePieceTokens(m.Path, m.Params)
	if err != nil {
		return err
	}

	slog.Debug("vocab loaded")

	m.Vocab = v
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
		"tokenizer.ggml.unknown_token_id": uint32(0),
		"tokenizer.ggml.add_bos_token":    true,
		"tokenizer.ggml.add_eos_token":    false,
	}

	f, err := os.CreateTemp("", "ollama-gguf")
	if err != nil {
		return err
	}
	defer f.Close()

	return llm.NewGGUFV3(m.Params.ByteOrder).Encode(f, kv, m.Tensors)
}
