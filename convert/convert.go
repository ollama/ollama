package convert

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"

	"github.com/d4l3k/go-bfloat16"
	"github.com/mitchellh/mapstructure"
	"github.com/x448/float16"
	"google.golang.org/protobuf/proto"

	"github.com/ollama/ollama/convert/sentencepiece"
	"github.com/ollama/ollama/llm"
)

type Params struct {
	Architectures    []string `json:"architectures"`
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`       // n_embd
	HiddenLayers     int      `json:"num_hidden_layers"` // n_layer
	ContextSize      int      `json:"max_position_embeddings"`
	IntermediateSize int      `json:"intermediate_size"`
	AttentionHeads   int      `json:"num_attention_heads"` // n_head
	KeyValHeads      int      `json:"num_key_value_heads"`
	NormEPS          float64  `json:"rms_norm_eps"`
	RopeFreqBase     float64  `json:"rope_theta"`
	BoSTokenID       int      `json:"bos_token_id"`
	EoSTokenID       int      `json:"eos_token_id"`
	HeadDimension    int      `json:"head_dim"`
	PaddingTokenID   int      `json:"pad_token_id"`

	ByteOrder
}

type ByteOrder interface {
	binary.ByteOrder
	binary.AppendByteOrder
}

type MetaData struct {
	Type    string `mapstructure:"dtype"`
	Shape   []int  `mapstructure:"shape"`
	Offsets []int  `mapstructure:"data_offsets"`
}

type ModelArch interface {
	GetTensors() error
	LoadVocab() error
	WriteGGUF() (string, error)
}

type ModelData struct {
	Path    string
	Name    string
	Params  *Params
	Vocab   *Vocab
	Tensors []llm.Tensor
}

func ReadSafeTensors(fn string, offset uint64, params *Params) ([]llm.Tensor, uint64, error) {
	f, err := os.Open(fn)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	var jsonSize uint64
	if err := binary.Read(f, binary.LittleEndian, &jsonSize); err != nil {
		return nil, 0, err
	}

	buf := make([]byte, jsonSize)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		return nil, 0, err
	}

	d := json.NewDecoder(bytes.NewBuffer(buf))
	d.UseNumber()
	var parsed map[string]interface{}
	if err = d.Decode(&parsed); err != nil {
		return nil, 0, err
	}

	var keys []string
	for k := range parsed {
		keys = append(keys, k)
	}

	slices.Sort(keys)

	slog.Info("converting layers")

	var tensors []llm.Tensor
	for _, k := range keys {
		vals := parsed[k].(map[string]interface{})
		var data MetaData
		if err = mapstructure.Decode(vals, &data); err != nil {
			return nil, 0, err
		}

		var size uint64
		var kind uint32
		switch len(data.Shape) {
		case 0:
			// metadata
			continue
		case 1:
			// convert to float32
			kind = 0
			size = uint64(data.Shape[0] * 4)
		case 2:
			// convert to float16
			kind = 1
			size = uint64(data.Shape[0] * data.Shape[1] * 2)
		}

		ggufName, err := GetTensorName(k)
		if err != nil {
			slog.Error("%v", err)
			return nil, 0, err
		}

		shape := []uint64{0, 0, 0, 0}
		for i := range data.Shape {
			shape[i] = uint64(data.Shape[i])
		}

		t := llm.Tensor{
			Name:   ggufName,
			Kind:   kind,
			Offset: offset,
			Shape:  shape[:],
		}

		t.WriterTo = safetensorWriterTo{
			t:        &t,
			params:   params,
			bo:       params.ByteOrder,
			filename: fn,
			start:    uint64(data.Offsets[0]),
			end:      uint64(data.Offsets[1]),
			padding:  8 + jsonSize,
		}

		slog.Debug(fmt.Sprintf("%v", t))
		tensors = append(tensors, t)
		offset += size
	}
	return tensors, offset, nil
}

func GetSafeTensors(dirpath string, params *Params) ([]llm.Tensor, error) {
	var tensors []llm.Tensor
	files, err := filepath.Glob(filepath.Join(dirpath, "/model-*.safetensors"))
	if err != nil {
		return nil, err
	}

	var offset uint64
	for _, f := range files {
		var t []llm.Tensor
		var err error
		t, offset, err = ReadSafeTensors(f, offset, params)
		if err != nil {
			slog.Error("%v", err)
			return nil, err
		}
		tensors = append(tensors, t...)
	}
	return tensors, nil
}

func GetParams(dirpath string) (*Params, error) {
	f, err := os.Open(filepath.Join(dirpath, "config.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var params Params

	d := json.NewDecoder(f)
	err = d.Decode(&params)
	if err != nil {
		return nil, err
	}

	params.ByteOrder = binary.LittleEndian
	return &params, nil
}

// Details on gguf's tokenizer can be found at:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#tokenizer
type Vocab struct {
	Tokens []string
	Scores []float32
	Types  []int32
}

func LoadSentencePieceTokens(dirpath string, vocabSize int) (*Vocab, error) {
	slog.Info(fmt.Sprintf("reading vocab from %s", filepath.Join(dirpath, "tokenizer.model")))
	in, err := os.ReadFile(filepath.Join(dirpath, "tokenizer.model"))
	if err != nil {
		return nil, err
	}

	// To regenerate sentencepiece from the protobufs use:
	// protoc -I=./ --go_out=./ sentencepiece_model.proto
	modelProto := &sentencepiece.ModelProto{}
	if err := proto.Unmarshal(in, modelProto); err != nil {
		return nil, err
	}

	v := &Vocab{
		Tokens: make([]string, 0),
		Scores: make([]float32, 0),
		Types:  make([]int32, 0),
	}

	pieces := modelProto.GetPieces()
	for _, p := range pieces {
		v.Tokens = append(v.Tokens, p.GetPiece())
		v.Scores = append(v.Scores, p.GetScore())
		t := p.GetType()
		switch t {
		case sentencepiece.ModelProto_SentencePiece_UNKNOWN:
		case sentencepiece.ModelProto_SentencePiece_CONTROL:
		case sentencepiece.ModelProto_SentencePiece_UNUSED:
		case sentencepiece.ModelProto_SentencePiece_BYTE:
		default:
			t = sentencepiece.ModelProto_SentencePiece_NORMAL
		}
		v.Types = append(v.Types, int32(t))
	}

	slog.Info(fmt.Sprintf("vocab size: %d", len(v.Tokens)))

	// add any additional tokens
	addIn, err := os.ReadFile(filepath.Join(dirpath, "added_tokens.json"))
	if os.IsNotExist(err) {
		return v, nil
	} else if err != nil {
		return nil, err
	}

	slog.Info("reading user defined tokens")

	var extraTokenData map[string]int
	if err := json.Unmarshal(addIn, &extraTokenData); err != nil {
		return nil, err
	}

	type token struct {
		key string
		pos int
	}

	extraTokens := make([]token, 0)
	for k, id := range extraTokenData {
		extraTokens = append(extraTokens, token{k, id})
	}

	slices.SortFunc(extraTokens, func(a, b token) int {
		return cmp.Compare(a.pos, b.pos)
	})

	numToks := len(v.Tokens)

	for cnt, t := range extraTokens {
		// the token id should match the specific index for the total number of tokens
		if t.pos != cnt+numToks {
			return nil, fmt.Errorf("token ID '%d' for '%s' doesn't match total token size", t.pos, t.key)
		}
		v.Tokens = append(v.Tokens, t.key)
		v.Scores = append(v.Scores, -1000.0)
		v.Types = append(v.Types, int32(llm.GGUFTokenUserDefined))
	}
	slog.Info(fmt.Sprintf("vocab size w/ extra tokens: %d", len(v.Tokens)))

	if vocabSize > len(v.Tokens) {
		missingTokens := vocabSize - len(v.Tokens)
		slog.Warn(fmt.Sprintf("vocab is missing %d tokens", missingTokens))
		for cnt := 0; cnt < missingTokens; cnt++ {
			v.Tokens = append(v.Tokens, fmt.Sprintf("<dummy%05d>", cnt+1))
			v.Scores = append(v.Scores, -1)
			v.Types = append(v.Types, int32(llm.GGUFTokenUserDefined))
		}
	}

	return v, nil
}

func GetTensorName(n string) (string, error) {
	tMap := map[string]string{
		"model.embed_tokens.weight":                           "token_embd.weight",
		"model.layers.(\\d+).input_layernorm.weight":          "blk.$1.attn_norm.weight",
		"model.layers.(\\d+).mlp.down_proj.weight":            "blk.$1.ffn_down.weight",
		"model.layers.(\\d+).mlp.gate_proj.weight":            "blk.$1.ffn_gate.weight",
		"model.layers.(\\d+).mlp.up_proj.weight":              "blk.$1.ffn_up.weight",
		"model.layers.(\\d+).post_attention_layernorm.weight": "blk.$1.ffn_norm.weight",
		"model.layers.(\\d+).self_attn.k_proj.weight":         "blk.$1.attn_k.weight",
		"model.layers.(\\d+).self_attn.o_proj.weight":         "blk.$1.attn_output.weight",
		"model.layers.(\\d+).self_attn.q_proj.weight":         "blk.$1.attn_q.weight",
		"model.layers.(\\d+).self_attn.v_proj.weight":         "blk.$1.attn_v.weight",
		"lm_head.weight":    "output.weight",
		"model.norm.weight": "output_norm.weight",
	}

	v, ok := tMap[n]
	if ok {
		return v, nil
	}

	// quick hack to rename the layers to gguf format
	for k, v := range tMap {
		re := regexp.MustCompile(k)
		newName := re.ReplaceAllString(n, v)
		if newName != n {
			return newName, nil
		}
	}

	return "", fmt.Errorf("couldn't find a layer name for '%s'", n)
}

type safetensorWriterTo struct {
	t *llm.Tensor

	params *Params
	bo     ByteOrder

	filename string

	start, end, padding uint64
	handler             func(w io.Writer, r safetensorWriterTo, f *os.File) error
}

func (r safetensorWriterTo) WriteTo(w io.Writer) (n int64, err error) {
	f, err := os.Open(r.filename)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if _, err = f.Seek(int64(r.padding+r.start), 0); err != nil {
		return 0, err
	}

	// use the handler if one is present
	if r.handler != nil {
		return 0, r.handler(w, r, f)
	}

	remaining := r.end - r.start

	bufSize := uint64(10240)
	var finished bool
	for {
		data := make([]byte, min(bufSize, remaining))

		b, err := io.ReadFull(f, data)
		remaining -= uint64(b)

		if err == io.EOF || remaining <= 0 {
			finished = true
		} else if err != nil {
			return 0, err
		}

		// convert bfloat16 -> ieee float32
		tDataF32 := bfloat16.DecodeFloat32(data)

		switch r.t.Kind {
		case 0:
			if err := binary.Write(w, r.bo, tDataF32); err != nil {
				return 0, err
			}
		case 1:
			// convert float32 -> float16
			tempBuf := make([]uint16, len(data)/2)
			for cnt, v := range tDataF32 {
				tDataF16 := float16.Fromfloat32(v)
				tempBuf[cnt] = uint16(tDataF16)
			}
			if err := binary.Write(w, binary.LittleEndian, tempBuf); err != nil {
				return 0, err
			}
		}
		if finished {
			break
		}
	}
	return 0, nil
}

func GetModelArchFromParams(name, dirPath string, params *Params) (ModelArch, error) {
	switch len(params.Architectures) {
	case 0:
		return nil, fmt.Errorf("No architecture specified to convert")
	case 1:
		switch params.Architectures[0] {
		case "MistralForCausalLM":
			return &MistralModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
				},
			}, nil
		case "GemmaForCausalLM":
			return &GemmaModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
				},
			}, nil
		default:
			return nil, fmt.Errorf("Models based on '%s' are not yet supported", params.Architectures[0])
		}
	}

	return nil, fmt.Errorf("Unknown error")
}
