package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/ollama/ollama/util/bufioutil"
)

type GGML struct {
	container
	model
}

type model interface {
	KV() KV
	Tensors() Tensors
}

type KV map[string]any

func (kv KV) u64(key string) uint64 {
	switch v := kv[key].(type) {
	case uint64:
		return v
	case uint32:
		return uint64(v)
	case float64:
		return uint64(v)
	default:
		return 0
	}
}

func (kv KV) Architecture() string {
	if s, ok := kv["general.architecture"].(string); ok {
		return s
	}

	return "unknown"
}

func (kv KV) Kind() string {
	if s, ok := kv["general.type"].(string); ok {
		return s
	}

	return "unknown"
}

func (kv KV) ParameterCount() uint64 {
	return kv.u64("general.parameter_count")
}

func (kv KV) FileType() fileType {
	if u64 := kv.u64("general.file_type"); u64 > 0 {
		return fileType(uint32(u64))
	}

	return fileTypeUnknown
}

func (kv KV) BlockCount() uint64 {
	return kv.u64(fmt.Sprintf("%s.block_count", kv.Architecture()))
}

func (kv KV) HeadCount() uint64 {
	return kv.u64(fmt.Sprintf("%s.attention.head_count", kv.Architecture()))
}

func (kv KV) HeadCountKV() uint64 {
	if headCountKV := kv.u64(fmt.Sprintf("%s.attention.head_count_kv", kv.Architecture())); headCountKV > 0 {
		return headCountKV
	}

	return 1
}

func (kv KV) EmbeddingHeadCount() uint64 {
	if heads := kv.HeadCount(); heads > 0 {
		return kv.EmbeddingLength() / kv.HeadCount()
	}

	return 0
}

func (kv KV) EmbeddingHeadCountK() uint64 {
	if k := kv.u64(fmt.Sprintf("%s.attention.key_length", kv.Architecture())); k > 0 {
		return k
	}

	return kv.EmbeddingHeadCount()
}

func (kv KV) EmbeddingHeadCountV() uint64 {
	if v := kv.u64(fmt.Sprintf("%s.attention.value_length", kv.Architecture())); v > 0 {
		return v
	}

	return kv.EmbeddingHeadCount()
}

func (kv KV) GQA() uint64 {
	return kv.HeadCount() / kv.HeadCountKV()
}

func (kv KV) EmbeddingLength() uint64 {
	return kv.u64(fmt.Sprintf("%s.embedding_length", kv.Architecture()))
}

func (kv KV) ContextLength() uint64 {
	return kv.u64(fmt.Sprintf("%s.context_length", kv.Architecture()))
}

func (kv KV) ChatTemplate() string {
	s, _ := kv["tokenizer.chat_template"].(string)
	return s
}

type Tensors struct {
	Items  []*Tensor
	Offset uint64
}

func (ts Tensors) Layers() map[string]Layer {
	layers := make(map[string]Layer)
	for _, t := range ts.Items {
		parts := strings.Split(t.Name, ".")
		if parts[0] == "blk" {
			// join first and second part, e.g. blk.%d
			parts = append([]string{fmt.Sprintf("%s.%s", parts[0], parts[1])}, parts[2:]...)
		}

		if _, ok := layers[parts[0]]; !ok {
			layers[parts[0]] = make(Layer)
		}

		layers[parts[0]][strings.Join(parts[1:], ".")] = t
	}

	return layers
}

type Layer map[string]*Tensor

func (l Layer) size() (size uint64) {
	for _, t := range l {
		size += t.Size()
	}

	return size
}

type Tensor struct {
	Name   string `json:"name"`
	Kind   uint32 `json:"kind"`
	Offset uint64 `json:"-"`

	// Shape is the number of elements in each dimension
	Shape []uint64 `json:"shape"`

	io.WriterTo `json:"-"`
}

func (t Tensor) block() (n int) {
	if _, err := fmt.Sscanf(t.Name, "blk.%d.", &n); err != nil {
		return -1
	}

	return
}

func (t Tensor) blockSize() uint64 {
	switch t.Kind {
	case 0, 1, 24, 25, 26, 27, 28, 30: // F32, F16, I8, I16, I32, I64, F64, BF16
		return 1
	case 2, 3, 4, 5, 6, 7, 8, 9, 20: // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, IQ4_NL
		return 32
	default: // All others
		return 256
	}
}

func (t Tensor) typeSize() uint64 {
	blockSize := t.blockSize()

	switch t.Kind {
	case 0: // FP32
		return 4
	case 1: // FP16
		return 2
	case 2: // Q4_0
		return 2 + blockSize/2
	case 3: // Q4_1
		return 2 + 2 + blockSize/2
	case 6: // Q5_0
		return 2 + 4 + blockSize/2
	case 7: // Q5_1
		return 2 + 2 + 4 + blockSize/2
	case 8: // Q8_0
		return 2 + blockSize
	case 9: // Q8_1
		return 4 + 4 + blockSize
	case 10: // Q2_K
		return blockSize/16 + blockSize/4 + 2 + 2
	case 11: // Q3_K
		return blockSize/8 + blockSize/4 + 12 + 2
	case 12: // Q4_K
		return 2 + 2 + 12 + blockSize/2
	case 13: // Q5_K
		return 2 + 2 + 12 + blockSize/8 + blockSize/2
	case 14: // Q6_K
		return blockSize/2 + blockSize/4 + blockSize/16 + 2
	case 15: // Q8_K
		return 2 + blockSize + 2*blockSize/16
	case 16: // IQ2_XXS
		return 2 + 2*blockSize/8
	case 17: // IQ2_XS
		return 2 + 2*blockSize/8 + blockSize/32
	case 18: // IQ3_XXS
		return 2 + blockSize/4 + blockSize/8
	case 19: // IQ1_S
		return 2 + blockSize/8 + blockSize/16
	case 20: // IQ4_NL
		return 2 + blockSize/2
	case 21: // IQ3_S
		return 2 + blockSize/4 + blockSize/8 + blockSize/32 + 4
	case 22: // IQ2_S
		return 2 + blockSize/4 + blockSize/16
	case 23: // IQ4_XS
		return 2 + 2 + blockSize/2 + blockSize/64
	case 24: // I8
		return 1
	case 25: // I16
		return 2
	case 26: // I32
		return 4
	case 27: // I64
		return 8
	case 28: // F64
		return 8
	case 29: // IQ1_M
		return blockSize/8 + blockSize/16 + blockSize/32
	case 30: // BF16
		return 2
	default:
		return 0
	}
}

func (t Tensor) parameters() uint64 {
	var count uint64 = 1
	for _, n := range t.Shape {
		count *= n
	}
	return count
}

func (t Tensor) Size() uint64 {
	return t.parameters() * t.typeSize() / t.blockSize()
}

type container interface {
	Name() string
	Decode(io.ReadSeeker) (model, error)
}

const (
	// Magic constant for `ggml` files (unversioned).
	FILE_MAGIC_GGML = 0x67676d6c
	// Magic constant for `ggml` files (versioned, ggmf).
	FILE_MAGIC_GGMF = 0x67676d66
	// Magic constant for `ggml` files (versioned, ggjt).
	FILE_MAGIC_GGJT = 0x67676a74
	// Magic constant for `ggla` files (LoRA adapter).
	FILE_MAGIC_GGLA = 0x67676C61
	// Magic constant for `gguf` files (versioned, gguf)
	FILE_MAGIC_GGUF_LE = 0x46554747
	FILE_MAGIC_GGUF_BE = 0x47475546
)

var ErrUnsupportedFormat = errors.New("unsupported model format")

func DetectGGMLType(b []byte) string {
	switch binary.LittleEndian.Uint32(b[:4]) {
	case FILE_MAGIC_GGML:
		return "ggml"
	case FILE_MAGIC_GGMF:
		return "ggmf"
	case FILE_MAGIC_GGJT:
		return "ggjt"
	case FILE_MAGIC_GGLA:
		return "ggla"
	case FILE_MAGIC_GGUF_LE, FILE_MAGIC_GGUF_BE:
		return "gguf"
	default:
		return ""
	}
}

// DecodeGGML decodes a GGML model from the given reader.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func DecodeGGML(rs io.ReadSeeker, maxArraySize int) (*GGML, int64, error) {
	if maxArraySize == 0 {
		maxArraySize = 1024
	}

	rs = bufioutil.NewBufferedSeeker(rs, 32<<10)

	var magic uint32
	if err := binary.Read(rs, binary.LittleEndian, &magic); err != nil {
		return nil, 0, err
	}

	var c container
	switch magic {
	case FILE_MAGIC_GGML, FILE_MAGIC_GGMF, FILE_MAGIC_GGJT:
		return nil, 0, ErrUnsupportedFormat
	case FILE_MAGIC_GGLA:
		c = &containerGGLA{}
	case FILE_MAGIC_GGUF_LE:
		c = &containerGGUF{ByteOrder: binary.LittleEndian, maxArraySize: maxArraySize}
	case FILE_MAGIC_GGUF_BE:
		c = &containerGGUF{ByteOrder: binary.BigEndian, maxArraySize: maxArraySize}
	default:
		return nil, 0, errors.New("invalid file magic")
	}

	model, err := c.Decode(rs)
	if err != nil {
		return nil, 0, err
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, 0, err
	}

	// final model type
	return &GGML{
		container: c,
		model:     model,
	}, offset, nil
}

func (llm GGML) GraphSize(context, batch uint64) (partialOffload, fullOffload uint64) {
	embedding := llm.KV().EmbeddingLength()
	heads := llm.KV().HeadCount()
	headsKV := llm.KV().HeadCountKV()
	vocab := uint64(llm.KV()["tokenizer.ggml.tokens"].(*array).size)

	embeddingHeads := llm.KV().EmbeddingHeadCount()
	embeddingHeadsK := llm.KV().EmbeddingHeadCountK()

	layers := llm.Tensors().Layers()

	switch llm.KV().Architecture() {
	case "llama":
		fullOffload = max(
			4*batch*(1+4*embedding+context*(1+heads)),
			4*batch*(embedding+vocab),
		)

		partialOffload = 4 * batch * embedding
		partialOffload += max(
			4*batch*(1+embedding+max(context, embedding))+embedding*embedding*9/16+4*context*(batch*heads+embeddingHeads*headsKV),
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
		)

		if ffnGateExpsWeight, ok := layers["blk.0"]["ffn_gate_exps.weight"]; ok {
			// mixtral 8x22b
			ff := uint64(llm.KV()["llama.feed_forward_length"].(uint32))
			partialOffload = max(
				3*ffnGateExpsWeight.Size()+4*batch*(2*ff+headsKV+embedding+context+embeddingHeads*headsKV),
				4*(context*batch*heads+context*embeddingHeads*headsKV+batch*1024+embeddingHeads*headsKV*batch),
			)
		} else if ffnGateWeight, ok := layers["blk.0"]["ffn_gate.0.weight"]; ok {
			// mixtral 8x7b
			ffnGateWeight1 := ffnGateWeight.Shape[1]
			fullOffload = 4 * batch * (2 + 3*embedding + context*(1+heads) + 2*headsKV + ffnGateWeight1)
			partialOffload = max(
				4*batch*(3+embeddingHeads*headsKV+embedding+context*(1+heads)+ffnGateWeight1)+(embedding*embedding+3*embedding*headsKV*ffnGateWeight1)*9/16,
				4*batch*(1+2*embedding+context*(1+heads))+embedding*(6*context*headsKV/heads+embedding*9/16),
			)
		}
	case "gemma", "gemma2":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(2+context+context*heads+2*embedding+2*embeddingHeadsK*heads),
		)

		partialOffload = max(
			4*embedding*batch+embedding*vocab*105/128+4*vocab*batch,
			4*batch*(2*embedding+1+2*embeddingHeadsK*heads+context+context*heads)+
				4*embeddingHeadsK*context*8+
				embedding*embeddingHeadsK*heads*9/16,
		)
	case "command-r":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(2+4*embedding+context*(1+heads)),
		)

		partialOffload = max(
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
			4*batch*(1+2*embedding+context*(1+heads))+4*embedding*context+embedding*embedding*9/16,
		)
	case "qwen2":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(1+2*embedding+context+context*heads),
		)

		partialOffload = max(
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
			4*(batch*(1+2*embedding+context*(1+heads))+embedding*(1+context)),
		)
	case "phi2":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(1+4*embedding+context+context*heads),
		)

		partialOffload = max(
			4*batch*(2*embedding+vocab)+embedding*vocab*105/128,
			4*batch*(2+3*embedding+context+context*heads),
		)
	case "stablelm":
		fullOffload = 4 * batch * (context*(1+heads) + 3*embedding + 2)
		partialOffload = max(
			4*batch*(vocab+2*embedding),
			fullOffload,
		)
	case "deepseek2":
		fullOffload = max(
			4*batch*(3*embedding+vocab),
			4*batch*(3*embedding+2+context*(1+headsKV)+2*embeddingHeadsK*headsKV),
		)

		partialOffload = max(
			4*batch*(3*embedding+vocab)+embedding*vocab*105/128,
			4*batch*(2*embedding+1+2*embeddingHeadsK*headsKV+context+context*headsKV)+4*embeddingHeadsK*context*headsKV+embedding*embeddingHeadsK*headsKV*9/16,
		)
	case "chatglm":
		fullOffload = 4 * batch * (embedding + vocab)
		partialOffload = 4*batch*(embedding+vocab) + embedding*vocab*105/128
		if qkvBias, ok := layers["blk.0"]["attn_qkv.bias"]; ok {
			fullOffload = max(
				fullOffload,
				4*batch*(2+
					2*embedding+
					context+
					context*heads+
					embeddingHeadsK*heads+
					qkvBias.Shape[0]),
			)

			partialOffload = max(
				partialOffload,
				4*batch*(1+
					2*embedding+
					embeddingHeadsK*heads+
					context+
					context*heads)+
					4*embeddingHeadsK*context+
					4*context*embeddingHeadsK+
					4*qkvBias.Shape[0],
			)
		}
	}

	return
}
