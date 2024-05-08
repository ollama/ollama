package convert

import (
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
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

func llamaTorchLayerHandler(w io.Writer, r torchWriterTo) error {

	var tData []uint16
	switch r.storage.(type) {
	case *pytorch.HalfStorage:
		data := r.storage.(*pytorch.HalfStorage).Data
		tData = make([]uint16, len(data))
		for cnt, v := range data {
			tData[cnt] = uint16(float16.Fromfloat32(v))
		}
	case *pytorch.BFloat16Storage:
		data := r.storage.(*pytorch.BFloat16Storage).Data
		tData = make([]uint16, len(data))

		for cnt, v := range data {
			tData[cnt] = uint16(float16.Fromfloat32(v))
		}
	default:
		return fmt.Errorf("unknown storage type for torch")
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
			switch m.Format.(type) {
			case *TorchFormat:
				wt := l.WriterTo.(torchWriterTo)
				wt.handler = llamaTorchLayerHandler
				l.WriterTo = wt
			case *SafetensorFormat:
				wt := l.WriterTo.(safetensorWriterTo)
				wt.handler = mistralLayerHandler
				l.WriterTo = wt
			}
		}
		m.Tensors = append(m.Tensors, l)
	}

	return nil
}

func (m *LlamaModel) LoadVocab() error {
	v := &Vocab{
		Tokens: []string{},
		Types:  []int32{},
		Merges: []string{},
	}

	tokpath := filepath.Join(m.Path, "tokenizer.json")
	slog.Debug(fmt.Sprintf("looking for %s", tokpath))
	if _, err := os.Stat(tokpath); !os.IsNotExist(err) {
		t, err := newTokenizer(tokpath)
		if err != nil {
			return err
		}

		for _, tok := range t.Model.Tokens {
			v.Tokens = append(v.Tokens, tok.Content)
			var tokType int32
			switch {
			case tok.Special:
				tokType = 3
			case tok.UserDefined:
				tokType = 4
			default:
				tokType = 1
			}
			v.Types = append(v.Types, tokType)
		}
		v.Merges = t.Model.Merges
	} else {
		slog.Debug("loading sentence piece vocab")
		v, err = LoadSentencePieceTokens(m.Path, m.Params)
		if err != nil {
			return err
		}

		slog.Debug("vocab loaded")

	}
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
		"llama.rope.freq_base":                   float32(m.Params.RopeFrequencyBase),
		"llama.rope.dimension_count":             uint32(m.Params.HiddenSize / m.Params.AttentionHeads),
		"llama.attention.head_count":             uint32(m.Params.AttentionHeads),
		"llama.attention.head_count_kv":          uint32(m.Params.KeyValHeads),
		"llama.attention.layer_norm_rms_epsilon": float32(m.Params.NormEPS),
		"general.file_type":                      uint32(2),
		"tokenizer.ggml.model":                   "gpt2",

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
