package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/d4l3k/go-bfloat16"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"github.com/x448/float16"

	"github.com/ollama/ollama/llm"
)

type MistralModel struct {
	ModelData
}

func mistralLayerHandler(w io.Writer, r safetensorWriterTo, f *os.File) error {
	layerSize := r.end - r.start

	var err error
	tData := make([]uint16, layerSize/2)
	if err = binary.Read(f, r.bo, tData); err != nil {
		return err
	}

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

	tData, err = repack(tData, int(heads), r.t.Shape)
	if err != nil {
		return err
	}

	var buf []byte
	for _, n := range tData {
		buf = r.bo.AppendUint16(buf, n)
	}

	tempBuf := make([]uint16, len(tData))
	tDataF32 := bfloat16.DecodeFloat32(buf)
	for cnt, v := range tDataF32 {
		tDataF16 := float16.Fromfloat32(v)
		tempBuf[cnt] = uint16(tDataF16)
	}

	if err = binary.Write(w, r.bo, tempBuf); err != nil {
		return err
	}
	return nil
}

func repack(data []uint16, heads int, shape []uint64) ([]uint16, error) {
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

func (m *MistralModel) GetTensors() error {
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
			wt := l.WriterTo.(safetensorWriterTo)
			wt.handler = mistralLayerHandler
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
