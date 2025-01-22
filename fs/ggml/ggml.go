package ggml

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/util/bufioutil"
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

func (kv KV) Architecture() string {
	return kv.String("general.architecture", "unknown")
}

func (kv KV) Kind() string {
	return kv.String("general.type", "unknown")
}

func (kv KV) ParameterCount() uint64 {
	return keyValue[uint64](kv, "general.parameter_count")
}

func (kv KV) FileType() fileType {
	if t := kv.Uint("general.file_type"); t > 0 {
		return fileType(t)
	}

	return fileTypeUnknown
}

func (kv KV) BlockCount() uint64 {
	return uint64(kv.Uint("block_count"))
}

func (kv KV) EmbeddingLength() uint64 {
	return uint64(kv.Uint("embedding_length"))
}

func (kv KV) HeadCount() uint64 {
	return uint64(kv.Uint("attention.head_count"))
}

func (kv KV) HeadCountKV() uint64 {
	return uint64(kv.Uint("attention.head_count_kv", 1))
}

func (kv KV) EmbeddingHeadCount() uint64 {
	if heads := kv.HeadCount(); heads > 0 {
		return kv.EmbeddingLength() / heads
	}

	return 0
}

func (kv KV) EmbeddingHeadCountK() uint64 {
	return uint64(kv.Uint("attention.key_length", uint32(kv.EmbeddingHeadCount())))
}

func (kv KV) EmbeddingHeadCountV() uint64 {
	return uint64(kv.Uint("attention.value_length", uint32(kv.EmbeddingHeadCount())))
}

func (kv KV) GQA() uint64 {
	return kv.HeadCount() / kv.HeadCountKV()
}

func (kv KV) ContextLength() uint64 {
	return uint64(kv.Uint("context_length"))
}

func (kv KV) ChatTemplate() string {
	return kv.String("tokenizer.chat_template")
}

func (kv KV) String(key string, defaultValue ...string) string {
	return keyValue(kv, key, append(defaultValue, "")...)
}

func (kv KV) Uint(key string, defaultValue ...uint32) uint32 {
	return keyValue(kv, key, append(defaultValue, 0)...)
}

func (kv KV) Float(key string, defaultValue ...float32) float32 {
	return keyValue(kv, key, append(defaultValue, 0)...)
}

func (kv KV) Strings(key string, defaultValue ...[]string) []string {
	r := keyValue(kv, key, &array{})
	s := make([]string, r.size)
	for i := range r.size {
		s[i] = r.values[i].(string)
	}

	return s
}

func (kv KV) Uints(key string, defaultValue ...[]uint32) []uint32 {
	r := keyValue(kv, key, &array{})
	s := make([]uint32, r.size)
	for i := range r.size {
		s[i] = uint32(r.values[i].(int32))
	}

	return s
}

func keyValue[T string | uint32 | uint64 | float32 | *array](kv KV, key string, defaultValue ...T) T {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key]; ok {
		return val.(T)
	}

	slog.Warn("key not found", "key", key, "default", defaultValue[0])
	return defaultValue[0]
}

type Tensors struct {
	items  []*Tensor
	Offset uint64
}

func (s Tensors) Items(prefix ...string) []*Tensor {
	if len(prefix) == 0 {
		return s.items
	}

	var items []*Tensor
	for _, t := range s.items {
		if strings.HasPrefix(t.Name, prefix[0]) {
			items = append(items, t)
		}
	}

	return items
}

func (ts Tensors) Layers() map[string]Layer {
	layers := make(map[string]Layer)
	for _, t := range ts.items {
		parts := strings.Split(t.Name, ".")
		if i := slices.Index(parts, "blk"); i > 0 {
			parts = append([]string{
				strings.Join(parts[:i], "."),
				strings.Join(parts[i:i+2], "."),
			}, parts[i+2:]...)
		} else if i == 0 {
			parts = append([]string{
				strings.Join(parts[i:i+2], "."),
			}, parts[i+2:]...)
		}

		if _, ok := layers[parts[0]]; !ok {
			layers[parts[0]] = make(Layer)
		}

		layers[parts[0]][strings.Join(parts[1:], ".")] = t
	}

	return layers
}

type Layer map[string]*Tensor

func (l Layer) Size() (size uint64) {
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

func DetectContentType(b []byte) string {
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

// Decode decodes a GGML model from the given reader.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func Decode(rs io.ReadSeeker, maxArraySize int) (*GGML, int64, error) {
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

func (llm GGML) GraphSize(context, batch uint64, kvCacheType string) (kv, partialOffload, fullOffload uint64) {
	embedding := llm.KV().EmbeddingLength()
	heads := llm.KV().HeadCount()
	headsKV := llm.KV().HeadCountKV()
	vocab := uint64(llm.KV()["tokenizer.ggml.tokens"].(*array).size)

	embeddingHeads := llm.KV().EmbeddingHeadCount()
	embeddingHeadsK := llm.KV().EmbeddingHeadCountK()
	embeddingHeadsV := llm.KV().EmbeddingHeadCountV()

	layers := llm.Tensors().Layers()

	bytesPerElement := kvCacheBytesPerElement(kvCacheType)
	kv = uint64(float64(context*llm.KV().BlockCount()*(embeddingHeadsK+embeddingHeadsV)*headsKV) * bytesPerElement)

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
	case "mllama":
		var visionTokens, tiles uint64 = 1601, 4

		if crossAttentionLayers, ok := llm.KV()["mllama.attention.cross_attention_layers"].(*array); ok {
			kv = headsKV *
				(embeddingHeadsK + embeddingHeadsV) * // one for K, one for V
				(2* // sizeof(float16)
					(llm.KV().BlockCount()-uint64(crossAttentionLayers.size))* // num non-cross attention layers
					context +
					4* // sizeof(float32)
						uint64(crossAttentionLayers.size)* // num cross attention layers
						visionTokens*
						tiles)
		}

		fullOffload = max(
			4*batch*(2+3*embedding+embeddingHeadsK*heads+context*(1+heads)),
			// vocab graph
			4*batch*(embedding+vocab),
		)

		var ropeFreqsCount uint64
		if ropeFreqs, ok := llm.Tensors().Layers()["rope_freqs"]; ok {
			if ropeFreqsWeights, ok := ropeFreqs["weights"]; ok {
				ropeFreqsCount = ropeFreqsWeights.parameters()
			}
		}

		partialOffload = max(
			4*(batch*
				(2*embedding+1+context*(1+heads)+embeddingHeadsK*heads)+
				ropeFreqsCount+
				embeddingHeadsK*context*headsKV),
			// vocab graph
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
		)
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

// SupportsKVCacheType checks if the requested cache type is supported
func (llm GGML) SupportsKVCacheType(cacheType string) bool {
	return slices.Contains([]string{"f16", "q8_0", "q4_0"}, cacheType)
}

// SupportsFlashAttention checks if the model supports flash attention
func (llm GGML) SupportsFlashAttention() bool {
	_, isEmbedding := llm.KV()[fmt.Sprintf("%s.pooling_type", llm.KV().Architecture())]
	if isEmbedding {
		return false
	}

	// Check head counts match and are non-zero
	headCountK := llm.KV().EmbeddingHeadCountK()
	headCountV := llm.KV().EmbeddingHeadCountV()
	return headCountK != 0 && headCountV != 0 && headCountK == headCountV
}

// kvCacheBytesPerElement returns the number of bytes per element for a given KV cache type
func kvCacheBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q8_0":
		return 1 // 1/2 of fp16
	case "q4_0":
		return 0.5 // 1/4 of fp16
	default:
		return 2 // f16 (default)
	}
}
