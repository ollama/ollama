package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"regexp"
	"slices"

	"github.com/d4l3k/go-bfloat16"
	"github.com/jmorganca/ollama/llm"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
	"github.com/x448/float16"
)

type SafetensorMetadata struct {
	Type    string   `json:"type"`
	Shape   []uint64 `json:"shape"`
	Offsets []int    `json:"data_offsets"`
}

func ggmlTensorName(n string) (string, error) {
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

func ParseSafetensor(filename string, params *Params) ([]*llm.Tensor, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var n uint64
	if err := binary.Read(file, binary.LittleEndian, &n); err != nil {
		return nil, err
	}

	slog.Info("ParseSafetensor", "size", n)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, file, int64(n)); err != nil {
		return nil, err
	}

	var headers map[string]SafetensorMetadata
	if err := json.NewDecoder(&b).Decode(&headers); err != nil {
		return nil, err
	}

	var keys []string
	for k := range headers {
		keys = append(keys, k)
	}

	slices.Sort(keys)

	var tensors []*llm.Tensor
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

		name, err := ggmlTensorName(key)
		if err != nil {
			return nil, err
		}

		shape := []uint64{0, 0, 0, 0}
		copy(shape, value.Shape)

		tensors = append(tensors, &llm.Tensor{
			Name:  name,
			Kind:  kind,
			Shape: shape,
			WriterTo: safetensor{
				name:        name,
				kind:        kind,
				shape:       value.Shape, // use the original shape
				bo:          params.ByteOrder,
				headCount:   uint32(params.AttentionHeads),
				headCountKV: uint32(params.KeyValHeads),
				filename:    filename,
				start:       uint64(value.Offsets[0]),
				end:         uint64(value.Offsets[1]),
				padding:     8 + n,
			},
		})
	}

	return tensors, nil
}

type safetensor struct {
	name  string
	kind  uint32
	shape []uint64

	bo          ByteOrder
	headCount   uint32
	headCountKV uint32

	filename string

	start, end, padding uint64
}

func (r safetensor) repack(data []uint16, heads int) ([]uint16, error) {
	n := tensor.New(tensor.WithShape(int(r.shape[0]), int(r.shape[1])), tensor.WithBacking(data))
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

func (r safetensor) WriteTo(w io.Writer) (n int64, err error) {
	f, err := os.Open(r.filename)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	if _, err = f.Seek(int64(r.padding+r.start), 0); err != nil {
		return 0, err
	}

	pattern := `^blk\.[0-9]+\.attn_(?P<layer>q|k)\.weight$`
	re, err := regexp.Compile(pattern)
	if err != nil {
		return 0, err
	}

	matches := re.FindAllStringSubmatch(r.name, -1)
	if len(matches) > 0 {
		layerSize := r.end - r.start

		var err error
		tData := make([]uint16, layerSize/2)
		if err = binary.Read(f, r.bo, tData); err != nil {
			return 0, err
		}

		layerType := matches[0][re.SubexpIndex("layer")]
		var heads uint32
		switch layerType {
		case "q":
			heads = r.headCount
		case "k":
			heads = r.headCountKV
			if heads == 0 {
				heads = r.headCount
			}
		}

		tData, err = r.repack(tData, int(heads))
		if err != nil {
			return 0, err
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
			return 0, err
		}
	} else {
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

			switch r.kind {
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
	}

	return 0, nil
}
