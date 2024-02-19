package convert

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"slices"

	"github.com/golang/protobuf/proto"
	"github.com/mitchellh/mapstructure"
	//"gorgonia.org/tensor"
	//"gorgonia.org/tensor/native"
	//"github.com/d4l3k/go-bfloat16"
	//"github.com/x448/float16"

	"github.com/jmorganca/ollama/convert/sentencepiece"
	"github.com/jmorganca/ollama/llm"
)

type Params struct {
	VocabSize        int     `json:"vocab_size"`
	HiddenSize       int     `json:"hidden_size"`       // n_embd
	HiddenLayers     int     `json:"num_hidden_layers"` // n_layer
	ContextSize      int     `json:"max_position_embeddings"`
	IntermediateSize int     `json:"intermediate_size"`
	AttentionHeads   int     `json:"num_attention_heads"` // n_head
	KeyValHeads      int     `json:"num_key_value_heads"`
	NormEPS          float64 `json:"rms_norm_eps"`
	RopeFreqBase     float64 `json:"rope_theta"`
}

type MetaData struct {
	Type    string `mapstructure:"dtype"`
	Shape   []int  `mapstructure:"shape"`
	Offsets []int  `mapstructure:"data_offsets"`
}

func ReadSafeTensors(fn string, offset uint64) ([]llm.Tensor, uint64, error) {
	f, err := os.Open(fn)
	if err != nil {
		return []llm.Tensor{}, 0, err
	}
	defer f.Close()

	// read the first 8 bytes of the file
	buf := make([]byte, 8)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		return []llm.Tensor{}, 0, err
	}

	// convert to little endian uint64
	num := binary.LittleEndian.Uint64(buf)

	buf = make([]byte, num)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		return []llm.Tensor{}, 0, err
	}

	// todo remove this
	if err := ioutil.WriteFile(fmt.Sprintf("%s-config.json", fn), buf, 0644); err != nil {
		return []llm.Tensor{}, 0, err
	}

	d := json.NewDecoder(bytes.NewBuffer(buf))
	d.UseNumber()
	var parsed map[string]interface{}
	if err = d.Decode(&parsed); err != nil {
		return []llm.Tensor{}, 0, err
	}

	var keys []string
	for k := range parsed {
		keys = append(keys, k)
	}

	slices.SortFunc(keys, func(a, b string) int {
		return cmp.Compare(a, b)
	})

	var tensors []llm.Tensor
	for _, k := range keys {
		vals := parsed[k].(map[string]interface{})
		var data MetaData
		if err = mapstructure.Decode(vals, &data); err != nil {
			return []llm.Tensor{}, 0, err
		}

		var size uint64
		var kind uint32
		switch len(data.Shape) {
		case 0:
			// metadata
			continue
		case 1:
			fmt.Printf("%s is a vector\n", k)
			// convert to float32
			kind = 0
			size = uint64(data.Shape[0] * 4)
		case 2:
			// convert to float16
			kind = 1
			size = uint64(data.Shape[0] * data.Shape[1] * 2)
			//size = uint64((data.Offsets[1] - data.Offsets[0]) * 2)
		}

		ggufName, err := GetTensorName(k)
		if err != nil {
			fmt.Println(err)
			return []llm.Tensor{}, 0, err
		}

		shape := [4]uint64{0, 0, 0, 0}
		for cnt, s := range data.Shape {
			shape[cnt] = uint64(s)
		}

		t := llm.Tensor{
			Name:           ggufName,
			Kind:           kind,
			Offset:         offset,
			Shape:          shape,
			FileName:       fn,
			OffsetPadding:  8 + num,
			FileOffsets:    []uint64{uint64(data.Offsets[0]), uint64(data.Offsets[1])},
		}
		tensors = append(tensors, t)
		offset += size
	}
	return tensors, offset, nil
}

func GetSafeTensors(dirpath string) ([]llm.Tensor, error) {
	var tensors []llm.Tensor
	files, err := filepath.Glob(dirpath + "/model-*.safetensors")
	if err != nil {
		return []llm.Tensor{}, nil
	}

	var offset uint64
	for _, f := range files {
		var t []llm.Tensor
		var err error
		t, offset, err = ReadSafeTensors(f, offset)
		if err != nil {
			fmt.Println(err)
			return []llm.Tensor{}, err
		}
		tensors = append(tensors, t...)
	}
	return tensors, nil
}

func GetParams(dirpath string) (*Params, error) {
	f, err := os.Open(dirpath + "/config.json")
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

	return &params, nil
}

type Vocab struct {
	Tokens []string
	Scores []float32
	Types  []int32
}

func LoadTokens(dirpath string) (*Vocab, error) {
	in, err := ioutil.ReadFile(dirpath + "/tokenizer.model")
	if err != nil {
		return nil, err
	}

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
		v.Types = append(v.Types, int32(t))
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
		"lm_head.weight":                                      "output.weight",
		"model.norm.weight":                                   "output_norm.weight",
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

func WriteGGUF(tensors []llm.Tensor, params *Params, vocab *Vocab) error {
	c := llm.ContainerGGUF{
		ByteOrder: binary.LittleEndian,
	}

	m := llm.NewGGUFModel(&c)
	m.Tensors = tensors
	m.KV["general.architecture"] = "llama"
	m.KV["general.name"] = ".." // XXX changeme
	m.KV["llama.context_length"] = uint32(params.ContextSize)
	m.KV["llama.embedding_length"] = uint32(params.HiddenSize)
	m.KV["llama.block_count"] = uint32(params.HiddenLayers)
	m.KV["llama.feed_forward_length"] = uint32(params.IntermediateSize)
	m.KV["llama.rope.dimension_count"] = uint32(128)
	m.KV["llama.attention.head_count"] = uint32(params.AttentionHeads)
	m.KV["llama.attention.head_count_kv"] = uint32(params.KeyValHeads)
	m.KV["llama.attention.layer_norm_rms_epsilon"] = float32(params.NormEPS)
	m.KV["llama.rope.freq_base"] = float32(params.RopeFreqBase)
	m.KV["general.file_type"] = uint32(1)
	m.KV["tokenizer.ggml.model"] = "llama"

	m.KV["tokenizer.ggml.tokens"] = vocab.Tokens
	m.KV["tokenizer.ggml.scores"] = vocab.Scores
	m.KV["tokenizer.ggml.token_type"] = vocab.Types

	m.KV["tokenizer.ggml.bos_token_id"] = uint32(1)
	m.KV["tokenizer.ggml.eos_token_id"] = uint32(2)
	m.KV["tokenizer.ggml.unknown_token_id"] = uint32(0)
	m.KV["tokenizer.ggml.add_bos_token"] = true
	m.KV["tokenizer.ggml.add_eos_token"] = false
	m.KV["tokenizer.chat_template"] = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}" // XXX removeme

	c.V3.NumTensor = uint64(len(tensors))
	c.V3.NumKV = uint64(len(m.KV))

	fmt.Printf("c - %#v\n", c)

	f, err := os.Create("output.bin")
	if err != nil {
		return err
	}
	defer f.Close()

	err = m.Encode(f)
	if err != nil {
		return err
	}

	return nil
}
