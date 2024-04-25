package convert

import (
	"bytes"
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

	"github.com/ollama/ollama/llm"
)

type safetensorWriterTo struct {
	t *llm.Tensor

	params *Params
	bo     ByteOrder

	filename string

	start, end, padding uint64
	handler             func(w io.Writer, r safetensorWriterTo, f *os.File) error
}

type tensorMetaData struct {
	Type    string `mapstructure:"dtype"`
	Shape   []int  `mapstructure:"shape"`
	Offsets []int  `mapstructure:"data_offsets"`
}

type SafetensorFormat struct{}

func (m *SafetensorFormat) GetTensors(dirpath string, params *Params) ([]llm.Tensor, error) {
	slog.Debug("getting tensor data")
	var tensors []llm.Tensor
	files, err := filepath.Glob(filepath.Join(dirpath, "/model-*.safetensors"))
	if err != nil {
		return nil, err
	}

	var offset uint64
	for _, f := range files {
		var t []llm.Tensor
		var err error
		t, offset, err = m.readTensors(f, offset, params)
		if err != nil {
			slog.Error(err.Error())
			return nil, err
		}
		tensors = append(tensors, t...)
	}
	slog.Debug(fmt.Sprintf("all tensors = %d", len(tensors)))
	return tensors, nil
}

func (m *SafetensorFormat) readTensors(fn string, offset uint64, params *Params) ([]llm.Tensor, uint64, error) {
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
		var data tensorMetaData
		if err = mapstructure.Decode(vals, &data); err != nil {
			slog.Error("couldn't decode properly")
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

		ggufName, err := m.GetLayerName(k)
		if err != nil {
			slog.Error(err.Error())
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

		offset += size
		tensors = append(tensors, t)
	}

	slog.Debug(fmt.Sprintf("total tensors for file = %d", len(tensors)))
	slog.Debug(fmt.Sprintf("offset = %d", offset))

	return tensors, offset, nil
}

func (m *SafetensorFormat) GetParams(dirpath string) (*Params, error) {
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
			if err := binary.Write(w, r.bo, tempBuf); err != nil {
				return 0, err
			}
		}
		if finished {
			break
		}
	}
	return 0, nil
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
