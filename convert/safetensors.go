package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"

	"github.com/uppercaveman/ollama-server/llm"
)

type safetensorWriterTo struct {
	t *llm.Tensor

	params *Params
	bo     ByteOrder

	filename string
	dtype    string

	offset, size int64
	repacker     func(string, []float32, []uint64) ([]float32, error)
}

type safetensorMetadata struct {
	Type    string   `json:"dtype"`
	Shape   []uint64 `json:"shape"`
	Offsets []int64  `json:"data_offsets"`
}

type SafetensorFormat struct{}

func (m *SafetensorFormat) GetTensors(dirpath string, params *Params) ([]llm.Tensor, error) {
	var tensors []llm.Tensor
	matches, err := filepath.Glob(filepath.Join(dirpath, "*.safetensors"))
	if err != nil {
		return nil, err
	}

	var offset uint64
	for _, f := range matches {
		var t []llm.Tensor
		var err error
		t, offset, err = m.readTensors(f, offset, params)
		if err != nil {
			return nil, err
		}

		tensors = append(tensors, t...)
	}
	return tensors, nil
}

func (m *SafetensorFormat) readTensors(fn string, offset uint64, params *Params) ([]llm.Tensor, uint64, error) {
	f, err := os.Open(fn)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	var n int64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, 0, err
	}

	b := bytes.NewBuffer(make([]byte, 0, n))
	if _, err = io.CopyN(b, f, n); err != nil {
		return nil, 0, err
	}

	var headers map[string]safetensorMetadata
	if err := json.NewDecoder(b).Decode(&headers); err != nil {
		return nil, 0, err
	}

	var keys []string
	for key := range headers {
		if !strings.HasSuffix(key, "self_attn.rotary_embd.inv_freq") {
			keys = append(keys, key)
		}
	}

	slices.Sort(keys)

	var tensors []llm.Tensor
	for _, key := range keys {
		value := headers[key]

		var kind uint32
		switch len(value.Shape) {
		case 0:
			// valuedata
			continue
		case 2:
			kind = 1
		}

		name, err := m.GetLayerName(key)
		if err != nil {
			return nil, 0, err
		}

		shape := make([]uint64, len(value.Shape))
		copy(shape, value.Shape)

		pad := func(s int64) int64 {
			return 8 + n + s
		}

		t := llm.Tensor{
			Name:   name,
			Kind:   kind,
			Offset: offset,
			Shape:  shape[:],
		}

		t.WriterTo = safetensorWriterTo{
			t:        &t,
			params:   params,
			bo:       params.ByteOrder,
			filename: fn,
			dtype:    value.Type,
			offset:   pad(value.Offsets[0]),
			size:     pad(value.Offsets[1]) - pad(value.Offsets[0]),
		}

		offset += t.Size()
		tensors = append(tensors, t)
	}

	return tensors, offset, nil
}

func (m *SafetensorFormat) GetParams(dirpath string) (*Params, error) {
	f, err := os.Open(filepath.Join(dirpath, "config.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var params Params

	if err := json.NewDecoder(f).Decode(&params); err != nil {
		return nil, err
	}

	params.ByteOrder = binary.LittleEndian
	return &params, nil
}

func (m *SafetensorFormat) GetLayerName(n string) (string, error) {
	directMap := map[string]string{
		"model.embed_tokens.weight": "token_embd.weight",
		"lm_head.weight":            "output.weight",
		"model.norm.weight":         "output_norm.weight",
	}

	tMap := map[string]string{
		"model.layers.(\\d+).input_layernorm.weight":                    "blk.$1.attn_norm.weight",
		"model.layers.(\\d+).mlp.down_proj.weight":                      "blk.$1.ffn_down.weight",
		"model.layers.(\\d+).mlp.gate_proj.weight":                      "blk.$1.ffn_gate.weight",
		"model.layers.(\\d+).mlp.up_proj.weight":                        "blk.$1.ffn_up.weight",
		"model.layers.(\\d+).post_attention_layernorm.weight":           "blk.$1.ffn_norm.weight",
		"model.layers.(\\d+).self_attn.k_proj.weight":                   "blk.$1.attn_k.weight",
		"model.layers.(\\d+).self_attn.o_proj.weight":                   "blk.$1.attn_output.weight",
		"model.layers.(\\d+).self_attn.q_proj.weight":                   "blk.$1.attn_q.weight",
		"model.layers.(\\d+).self_attn.v_proj.weight":                   "blk.$1.attn_v.weight",
		"model.layers.(\\d+).block_sparse_moe.gate.weight":              "blk.$1.ffn_gate_inp.weight",
		"model.layers.(\\d+).block_sparse_moe.experts.(\\d+).w1.weight": "blk.$1.ffn_gate.$2.weight",
		"model.layers.(\\d+).block_sparse_moe.experts.(\\d+).w2.weight": "blk.$1.ffn_down.$2.weight",
		"model.layers.(\\d+).block_sparse_moe.experts.(\\d+).w3.weight": "blk.$1.ffn_up.$2.weight",
	}

	v, ok := directMap[n]
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

func (r safetensorWriterTo) WriteTo(w io.Writer) (n int64, err error) {
	f, err := os.Open(r.filename)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if _, err = f.Seek(r.offset, io.SeekStart); err != nil {
		return 0, err
	}

	var f32s []float32
	switch r.dtype {
	case "F32":
		f32s = make([]float32, r.size/4)
		if err = binary.Read(f, r.bo, f32s); err != nil {
			return 0, err
		}
	case "F16":
		u16s := make([]uint16, r.size/2)
		if err = binary.Read(f, r.bo, u16s); err != nil {
			return 0, err
		}

		for _, b := range u16s {
			f32s = append(f32s, float16.Frombits(b).Float32())
		}

	case "BF16":
		u8s := make([]uint8, r.size)
		if err = binary.Read(f, r.bo, u8s); err != nil {
			return 0, err
		}

		f32s = bfloat16.DecodeFloat32(u8s)
	default:
		return 0, fmt.Errorf("unknown data type: %s", r.dtype)
	}

	if r.repacker != nil {
		f32s, err = r.repacker(r.t.Name, f32s, r.t.Shape)
		if err != nil {
			return 0, err
		}
	}

	switch r.t.Kind {
	case 0:
		return 0, binary.Write(w, r.bo, f32s)
	case 1:
		f16s := make([]uint16, len(f32s))
		for i := range f32s {
			f16s[i] = float16.Fromfloat32(f32s[i]).Bits()
		}

		return 0, binary.Write(w, r.bo, f16s)
	default:
		return 0, fmt.Errorf("unknown storage type: %d", r.t.Kind)
	}
}

func (m *SafetensorFormat) GetModelArch(name, dirPath string, params *Params) (ModelArch, error) {
	switch len(params.Architectures) {
	case 0:
		return nil, fmt.Errorf("No architecture specified to convert")
	case 1:
		switch params.Architectures[0] {
		case "LlamaForCausalLM":
			return &LlamaModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
					Format: m,
				},
			}, nil
		case "MistralForCausalLM":
			return &MistralModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
					Format: m,
				},
			}, nil
		case "MixtralForCausalLM":
			return &MixtralModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
					Format: m,
				},
			}, nil
		case "GemmaForCausalLM":
			return &GemmaModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
					Format: m,
				},
			}, nil
		default:
			return nil, fmt.Errorf("Models based on '%s' are not yet supported", params.Architectures[0])
		}
	}

	return nil, fmt.Errorf("Unknown error")
}
