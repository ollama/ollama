package ggml

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/util/bufioutil"
	"github.com/ollama/ollama/ml"
)

type GGML struct {
	container
	model
	Length int64
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
	val, _ := keyValue(kv, "general.parameter_count", uint64(0))
	return val
}

func (kv KV) FileType() FileType {
	if t := kv.Uint("general.file_type"); t > 0 {
		return FileType(t)
	}

	return FileTypeUnknown
}

func (kv KV) BlockCount() uint64 {
	return uint64(kv.Uint("block_count"))
}

func (kv KV) EmbeddingLength() uint64 {
	return uint64(kv.Uint("embedding_length"))
}

func (kv KV) HeadCount() []uint64 {
	headCountDefault := uint32(1)
	headCount := kv.UintOrArrayValueAsArray("attention.head_count", headCountDefault)
	if len(headCount) == 1 {
		headCountDefault = headCount[0]
	}
	nLayers := int(kv.BlockCount())
	if len(headCount) > nLayers {
		slog.Warn("got more elements of attention.head_count than layers", "len(headCount)", len(headCount), "layers", nLayers)
	}
	out := make([]uint64, nLayers)
	for i := range nLayers {
		if i >= len(headCount) {
			out[i] = uint64(headCountDefault)
		} else {
			out[i] = uint64(headCount[i])
		}
	}
	return out
}

func (kv KV) HeadCountMax() uint64 {
	return uint64(kv.UintOrMaxArrayValue("attention.head_count", 1))
}

func (kv KV) HeadCountMin() uint64 {
	return uint64(kv.UintOrMinArrayValue("attention.head_count", 1))
}

func (kv KV) HeadCountKV() []uint64 {
	headCountKVDefault := uint32(1)
	headCountKV := kv.UintOrArrayValueAsArray("attention.head_count_kv", headCountKVDefault)
	if len(headCountKV) == 1 {
		headCountKVDefault = headCountKV[0]
	}
	nLayers := int(kv.BlockCount())
	if len(headCountKV) > nLayers {
		slog.Warn("got more elements of attention.head_count than layers", "len(headCountKV)", len(headCountKV), "layers", nLayers)
	}
	out := make([]uint64, nLayers)
	for i := range nLayers {
		if i >= len(headCountKV) {
			out[i] = uint64(headCountKVDefault)
		} else {
			out[i] = uint64(headCountKV[i])
		}
	}
	return out
}

func (kv KV) HeadCountKVMax() uint64 {
	return uint64(kv.UintOrMaxArrayValue("attention.head_count_kv", 1))
}

func (kv KV) HeadCountKVMin() uint64 {
	return uint64(kv.UintOrMinArrayValue("attention.head_count_kv", 1))
}

func (kv KV) EmbeddingHeadCountMax() uint64 {
	if heads := kv.HeadCountMin(); heads > 0 {
		return kv.EmbeddingLength() / heads
	}

	return 0
}

func (kv KV) EmbeddingHeadCountK() uint64 {
	return uint64(kv.Uint("attention.key_length", uint32(kv.EmbeddingHeadCountMax())))
}

func (kv KV) EmbeddingHeadCountV() uint64 {
	return uint64(kv.Uint("attention.value_length", uint32(kv.EmbeddingHeadCountMax())))
}

func (kv KV) ContextLength() uint64 {
	return uint64(kv.Uint("context_length"))
}

func (kv KV) ChatTemplate() string {
	return kv.String("tokenizer.chat_template")
}

// ssm architecture parameters

func (kv KV) SSMConvKernel() uint64 {
	return uint64(kv.Uint("ssm.conv_kernel"))
}

func (kv KV) SSMInnerSize() uint64 {
	return uint64(kv.Uint("ssm.inner_size"))
}

func (kv KV) SSMStateSize() uint64 {
	return uint64(kv.Uint("ssm.state_size"))
}

func (kv KV) SSMGroupCount() uint64 {
	return uint64(kv.Uint("ssm.group_count"))
}

// general types

func (kv KV) String(key string, defaultValue ...string) string {
	val, _ := keyValue(kv, key, append(defaultValue, "")...)
	return val
}

func (kv KV) Uint(key string, defaultValue ...uint32) uint32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

func (kv KV) Float(key string, defaultValue ...float32) float32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

func (kv KV) Bool(key string, defaultValue ...bool) bool {
	val, _ := keyValue(kv, key, append(defaultValue, false)...)
	return val
}

func (kv KV) UintOrMaxArrayValue(key string, defaultValue uint32) uint32 {
	_, max := kv.UintOrArrayValue(key, defaultValue)
	return max
}

func (kv KV) UintOrMinArrayValue(key string, defaultValue uint32) uint32 {
	min, _ := kv.UintOrArrayValue(key, defaultValue)
	return min
}

func (kv KV) UintOrArrayValue(key string, defaultValue uint32) (uint32, uint32) {
	arrVal := kv.UintOrArrayValueAsArray(key, defaultValue)
	return slices.Min(arrVal), slices.Max(arrVal)
}

func (kv KV) UintOrArrayValueAsArray(key string, defaultValue uint32) []uint32 {
	if u32, ok := keyValue(kv, key, uint32(0)); ok {
		return []uint32{u32}
	} else if u32s, ok := keyValue(kv, key, &array[uint32]{}); ok {
		return u32s.values
	} else if i32s, ok := keyValue(kv, key, &array[int32]{}); ok {
		dst := make([]uint32, len(i32s.values))
		for i, v := range i32s.values {
			if v < 0 {
				slog.Warn("array values are unexpectedly negative", "key", key, "i", i, "v", v)
			}
			dst[i] = uint32(v)
		}
		return dst
	}

	return []uint32{defaultValue}
}

func (kv KV) Strings(key string, defaultValue ...[]string) []string {
	val, _ := keyValue(kv, key, &array[string]{values: append(defaultValue, []string(nil))[0]})
	return val.values
}

func (kv KV) Ints(key string, defaultValue ...[]int32) []int32 {
	val, _ := keyValue(kv, key, &array[int32]{values: append(defaultValue, []int32(nil))[0]})
	return val.values
}

func (kv KV) Uints(key string, defaultValue ...[]uint32) []uint32 {
	val, _ := keyValue(kv, key, &array[uint32]{values: append(defaultValue, []uint32(nil))[0]})
	return val.values
}

func (kv KV) Floats(key string, defaultValue ...[]float32) []float32 {
	val, _ := keyValue(kv, key, &array[float32]{values: append(defaultValue, []float32(nil))[0]})
	return val.values
}

func (kv KV) Bools(key string, defaultValue ...[]bool) []bool {
	val, _ := keyValue(kv, key, &array[bool]{values: append(defaultValue, []bool(nil))[0]})
	return val.values
}

func (kv KV) Len() int {
	return len(kv)
}

func (kv KV) Keys() iter.Seq[string] {
	return maps.Keys(kv)
}

func (kv KV) Value(key string) any {
	return kv[key]
}

func (kv KV) OllamaEngineRequired() bool {
	return slices.Contains([]string{
		"bert",
		"deepseek2",
		"deepseekocr",
		"gemma3",
		"gemma3n",
		"gptoss", "gpt-oss",
		"llama4",
		"mistral3",
		"mllama",
		"nomic-bert",
		"olmo3",
		"qwen25vl",
		"qwen3", "qwen3moe",
		"qwen3vl", "qwen3vlmoe",
	}, kv.Architecture())
}

type valueTypes interface {
	uint8 | int8 | uint16 | int16 |
		uint32 | int32 | uint64 | int64 |
		string | float32 | float64 | bool
}

type arrayValueTypes interface {
	*array[uint8] | *array[int8] | *array[uint16] | *array[int16] |
		*array[uint32] | *array[int32] | *array[uint64] | *array[int64] |
		*array[string] | *array[float32] | *array[float64] | *array[bool]
}

func keyValue[T valueTypes | arrayValueTypes](kv KV, key string, defaultValue ...T) (T, bool) {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key].(T); ok {
		return val, true
	}

	slog.Debug("key with type not found", "key", key, "default", defaultValue[0])
	return defaultValue[0], false
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

func (ts Tensors) GroupLayers() map[string]Layer {
	layers := make(map[string]Layer)
	for _, t := range ts.items {
		parts := strings.Split(t.Name, ".")
		if index := slices.IndexFunc(parts, func(s string) bool { return s == "blk" || s == "mm" }); index != -1 {
			if len(parts) > index+2 {
				// blk and mm should have a number after them, join it
				parts = append(
					[]string{strings.Join(parts[:index+2], ".")},
					parts[index+2:]...)
			}
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
		return math.MaxInt
	}

	return
}

func (t Tensor) blockSize() uint64 {
	return TensorType(t.Kind).BlockSize()
}

func (t TensorType) BlockSize() uint64 {
	switch t {
	case
		TensorTypeF32,
		TensorTypeF16,
		TensorTypeI8,
		TensorTypeI16,
		TensorTypeI32,
		TensorTypeI64,
		TensorTypeF64,
		TensorTypeBF16:
		return 1
	case
		TensorTypeQ4_0,
		TensorTypeQ4_1,
		TensorTypeQ5_0,
		TensorTypeQ5_1,
		TensorTypeQ8_0,
		TensorTypeQ8_1,
		tensorTypeIQ4_NL,
		4, TensorTypeMXFP4:
		return 32
	default:
		return 256
	}
}

func (t Tensor) typeSize() uint64 {
	return TensorType(t.Kind).TypeSize()
}

func (t TensorType) TypeSize() uint64 {
	blockSize := t.BlockSize()

	switch t {
	case TensorTypeF32:
		return 4
	case TensorTypeF16:
		return 2
	case TensorTypeQ4_0:
		return 2 + blockSize/2
	case TensorTypeQ4_1:
		return 2 + 2 + blockSize/2
	case TensorTypeQ5_0:
		return 2 + 4 + blockSize/2
	case TensorTypeQ5_1:
		return 2 + 2 + 4 + blockSize/2
	case TensorTypeQ8_0:
		return 2 + blockSize
	case TensorTypeQ8_1:
		return 2 + 2 + blockSize
	case TensorTypeQ2_K:
		return blockSize/16 + blockSize/4 + 2 + 2
	case TensorTypeQ3_K:
		return blockSize/8 + blockSize/4 + 12 + 2
	case TensorTypeQ4_K:
		return 2 + 2 + 12 + blockSize/2
	case TensorTypeQ5_K:
		return 2 + 2 + 12 + blockSize/8 + blockSize/2
	case TensorTypeQ6_K:
		return blockSize/2 + blockSize/4 + blockSize/16 + 2
	case TensorTypeQ8_K:
		return 4 + blockSize + 2*blockSize/16
	case tensorTypeIQ2_XXS:
		return 2 + 2*blockSize/8
	case tensorTypeIQ2_XS:
		return 2 + 2*blockSize/8 + blockSize/32
	case tensorTypeIQ3_XXS:
		return 2 + blockSize/4 + blockSize/8
	case tensorTypeIQ1_S:
		return 2 + blockSize/8 + blockSize/16
	case tensorTypeIQ4_NL:
		return 2 + blockSize/2
	case tensorTypeIQ3_S:
		return 2 + blockSize/4 + blockSize/8 + blockSize/32 + 4
	case tensorTypeIQ2_S:
		return 2 + blockSize/4 + blockSize/16
	case tensorTypeIQ4_XS:
		return 2 + 2 + blockSize/2 + blockSize/64
	case TensorTypeI8:
		return 1
	case TensorTypeI16:
		return 2
	case TensorTypeI32:
		return 4
	case TensorTypeI64:
		return 8
	case TensorTypeF64:
		return 8
	case tensorTypeIQ1_M:
		return blockSize/8 + blockSize/16 + blockSize/32
	case TensorTypeBF16:
		return 2
	case 4, TensorTypeMXFP4:
		return 1 + blockSize/2
	default:
		return 0
	}
}

func (t Tensor) Elements() uint64 {
	var count uint64 = 1
	for _, n := range t.Shape {
		count *= n
	}
	return count
}

func (t Tensor) Size() uint64 {
	return t.Elements() * t.typeSize() / t.blockSize()
}

func (t Tensor) Type() string {
	return TensorType(t.Kind).String()
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
// maxArraySize. If the maxArraySize is negative, all arrays are collected.
func Decode(rs io.ReadSeeker, maxArraySize int) (*GGML, error) {
	rs = bufioutil.NewBufferedSeeker(rs, 32<<10)

	var magic uint32
	if err := binary.Read(rs, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}

	var c container
	switch magic {
	case FILE_MAGIC_GGUF_LE:
		c = &containerGGUF{ByteOrder: binary.LittleEndian, maxArraySize: maxArraySize}
	case FILE_MAGIC_GGUF_BE:
		c = &containerGGUF{ByteOrder: binary.BigEndian, maxArraySize: maxArraySize}
	default:
		return nil, errors.New("invalid file magic")
	}

	model, err := c.Decode(rs)
	if err != nil {
		return nil, err
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	// final model type
	return &GGML{
		container: c,
		model:     model,
		Length:    offset,
	}, nil
}

func (f GGML) GraphSize(context, batch uint64, numParallel int, kvCacheType string, useFlashAttention ml.FlashAttentionType) (kv []uint64, partialOffload, fullOffload uint64) {
	context *= uint64(numParallel)

	embedding := f.KV().EmbeddingLength()
	heads := f.KV().HeadCountMax()
	headsArr := f.KV().HeadCount()
	headsKV := f.KV().HeadCountKVMax()
	headsKVArr := f.KV().HeadCountKV()
	vocab := uint64(f.KV()["tokenizer.ggml.tokens"].(*array[string]).size)

	embeddingHeads := f.KV().EmbeddingHeadCountMax()
	embeddingHeadsK := f.KV().EmbeddingHeadCountK()
	embeddingHeadsV := f.KV().EmbeddingHeadCountV()

	layers := f.Tensors().GroupLayers()

	bytesPerElement := kvCacheBytesPerElement(kvCacheType)

	// Default for models unless special-cased below. These defaults mirror the
	// cache usage in llama.cpp under the assumption that models without special
	// cases below will use the llamarunner and caching will be handled by the
	// llama.cpp layer.
	//
	// This also assumes that a layer without heads or headsKV set is recurrent
	// which is usually the case. Some models (eg nemotronh) use "blocks" in
	// place of layers where some are MLP blocks that don't have any cache.
	// Models like this will need a special case below to be accurately
	// estimated.
	var kvTotal uint64
	kv = make([]uint64, f.KV().BlockCount())
	kvSizeAttn := uint64(0)
	kvSizeRecurrent := uint64(0)
	for i := range kv {
		headsL := headsArr[i]
		headsKVL := headsKVArr[i]
		if headsL > 0 && headsKVL > 0 {
			// full attention layer
			// NOTE: Assumes uniform values for all attn layers
			kv[i] = uint64(float64(context*(embeddingHeadsK+embeddingHeadsV)*headsKVL) * bytesPerElement)
			kvSizeAttn += kv[i]
		} else {
			// recurrent layer
			ssmDConv := f.KV().SSMConvKernel()
			ssmDState := f.KV().SSMStateSize()
			ssmDInner := f.KV().SSMInnerSize()
			ssmNGroups := f.KV().SSMGroupCount()
			nEmbdR := uint64(0)
			if ssmDConv > 0 {
				nEmbdR = (ssmDConv - 1) * (ssmDInner + 2*ssmNGroups*ssmDState)
			}
			nEmbdS := ssmDState * ssmDInner

			// recurrent always uses F32 in llama.cpp backend
			// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp#L18644
			bytesPerElementRecurrent := kvCacheBytesPerElement("f32")

			kv[i] = (nEmbdR + nEmbdS) * uint64(bytesPerElementRecurrent)
			kvSizeRecurrent += kv[i]
		}
		kvTotal += kv[i]
	}
	slog.Debug("default cache size estimate", "attention MiB", float32(kvSizeAttn)/(1024.*1024.), "attention bytes", kvSizeAttn, "recurrent MiB", float32(kvSizeRecurrent)/(1024.*1024.), "recurrent bytes", kvSizeRecurrent)

	switch f.KV().Architecture() {
	case "llama", "llama4":
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
			ff := uint64(f.KV().Uint("feed_forward_length"))
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

		crossAttentionLayers := f.KV().Ints("attention.cross_attention_layers")
		for i := range kv {
			if slices.Contains(crossAttentionLayers, int32(i)) {
				kv[i] = headsKV * (embeddingHeadsK + embeddingHeadsV) *
					4 * // sizeof(float32)
					visionTokens *
					tiles
			}
		}

		fullOffload = max(
			4*batch*(2+3*embedding+embeddingHeadsK*heads+context*(1+heads)),
			// vocab graph
			4*batch*(embedding+vocab),
		)

		var ropeFreqsCount uint64
		if ropeFreqs, ok := f.Tensors().GroupLayers()["rope_freqs"]; ok {
			if ropeFreqsWeights, ok := ropeFreqs["weights"]; ok {
				ropeFreqsCount = ropeFreqsWeights.Elements()
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
	case "gemma", "gemma2", "gemma3", "gemma3n":
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

		if f.KV().Architecture() == "gemma3n" {
			fullOffload *= 4
			partialOffload *= 4
		}

		// Gemma2 also has sliding window attention but we only have an optimized implementation in the Ollama
		// engine. Gemma3 always uses the Ollama engine.
		if f.KV().Architecture() == "gemma3" {
			const gemma3GlobalCacheCount = 6
			slidingWindow := (uint64(numParallel) * uint64(f.KV().Uint("attention.sliding_window"))) + batch
			for i := range kv {
				// Every 6th layer is a global layer, which is the full context size that has already been set. The other
				// layers are the smaller local (sliding) layers.
				if (i+1)%gemma3GlobalCacheCount != 0 {
					kv[i] = uint64(float64(slidingWindow*(embeddingHeadsK+embeddingHeadsV)*headsKV) * bytesPerElement)
				}
			}
		}
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
	case "gptoss", "gpt-oss":
		kv = make([]uint64, f.KV().BlockCount())
		for i := range kv {
			kv[i] = uint64(float64((embeddingHeadsK+embeddingHeadsV)*headsKV) * bytesPerElement)
			if i%2 == 0 {
				kv[i] *= (uint64(numParallel)*4096 + batch)
			} else {
				kv[i] *= context
			}
		}

		partialOffload = 2 * f.KV().HeadCountMax() / cmp.Or(f.KV().HeadCountKVMin(), 1) * kvTotal / 6
		if useFlashAttention == ml.FlashAttentionEnabled {
			// rough estimate of graph size with flash attention on
			partialOffload = (4*uint64(numParallel) + context>>10 + 110) * format.MebiByte
		}
	}

	return
}

// SupportsKVCacheType checks if the requested cache type is supported
func (f GGML) SupportsKVCacheType(cacheType string) bool {
	if cacheType == "" || cacheType == "f16" {
		return true
	}

	return slices.Contains([]string{"q8_0", "q4_0"}, cacheType)
}

// KVCacheTypeIsQuantized checks if the requested cache type is a quantized type
func (f GGML) KVCacheTypeIsQuantized(cacheType string) bool {
	if cacheType == "" || cacheType == "f16" || cacheType == "f32" || cacheType == "bf16" {
		return false
	}
	return true
}

// SupportsFlashAttention checks if the model supports flash attention
func (f GGML) SupportsFlashAttention() bool {
	_, isEmbedding := f.KV()[fmt.Sprintf("%s.pooling_type", f.KV().Architecture())]
	if isEmbedding {
		return false
	}

	if arch := f.KV().Architecture(); slices.Contains([]string{"gemma2"}, arch) {
		return false
	}

	// Check head counts match and are non-zero
	headCountK := f.KV().EmbeddingHeadCountK()
	headCountV := f.KV().EmbeddingHeadCountV()
	return headCountK != 0 && headCountV != 0 && headCountK == headCountV
}

// FlashAttention checks if the model should enable flash attention
func (f GGML) FlashAttention() bool {
	return slices.Contains([]string{
		"bert",
		"gemma3",
		"gptoss", "gpt-oss",
		"mistral3",
		"olmo3",
		"qwen3", "qwen3moe",
		"qwen3vl", "qwen3vlmoe",
	}, f.KV().String("general.architecture"))
}

// kvCacheBytesPerElement returns the number of bytes per element for a given KV cache type
func kvCacheBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q8_0":
		return 1 // 1/2 of fp16
	case "q4_0":
		return 0.5 // 1/4 of fp16
	case "f32":
		return 4 // f32 (default for recurrent)
	default:
		return 2 // f16 (default)
	}
}
