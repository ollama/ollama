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

func (kv KV) HeadCountMax() uint64 {
	// TODO(drifkin): using the max value can cause an overestimation. In the
	// future if array values become more popular, we can adapt the more invasive
	// <https://github.com/ollama/ollama/pull/10225>
	return uint64(kv.UintOrMaxArrayValue("attention.head_count", 1))
}

func (kv KV) HeadCountMin() uint64 {
	return uint64(kv.UintOrMinArrayValue("attention.head_count", 1))
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
	if u32, ok := keyValue(kv, key, uint32(0)); ok {
		return u32, u32
	} else if u32s, ok := keyValue(kv, key, &array[uint32]{}); ok {
		min := slices.Min(u32s.values)
		max := slices.Max(u32s.values)
		return min, max
	} else if i32s, ok := keyValue(kv, key, &array[int32]{}); ok {
		min := slices.Min(i32s.values)
		max := slices.Max(i32s.values)
		if min < 0 || max < 0 {
			slog.Warn("array values are unexpectedly negative", "key", key, "min", min, "max", max)
		}
		return uint32(min), uint32(max)
	}

	return defaultValue, defaultValue
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

func (kv KV) OllamaEngineRequired() bool {
	return slices.Contains([]string{
		"gemma3",
		"gemma3n",
		"mistral3",
		"llama4",
		"mllama",
		"qwen25vl",
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
		return -1
	}

	return
}

func (t Tensor) blockSize() uint64 {
	return (TensorType)(t.Kind).BlockSize()
}

func (t TensorType) BlockSize() uint64 {
	switch t {
	case
		0,  // F32
		1,  // F16
		24, // I8
		25, // I16
		26, // I32
		27, // I64
		28, // F64
		30: // BF16
		return 1
	case
		2,  // Q4_0
		3,  // Q4_1
		6,  // Q5_0
		7,  // Q5_1
		8,  // Q8_0
		9,  // Q8_1
		20: // IQ4_NL
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

func (f GGML) GraphSize(context, batch uint64, numParallel int, kvCacheType string) (kv []uint64, partialOffload, fullOffload uint64) {
	embedding := f.KV().EmbeddingLength()
	heads := f.KV().HeadCountMax()
	headsKV := f.KV().HeadCountKVMax()
	vocab := uint64(f.KV()["tokenizer.ggml.tokens"].(*array[string]).size)

	embeddingHeads := f.KV().EmbeddingHeadCountMax()
	embeddingHeadsK := f.KV().EmbeddingHeadCountK()
	embeddingHeadsV := f.KV().EmbeddingHeadCountV()

	layers := f.Tensors().GroupLayers()

	bytesPerElement := kvCacheBytesPerElement(kvCacheType)
	kv = make([]uint64, f.KV().BlockCount())
	for i := range kv {
		kv[i] = uint64(float64(context*(embeddingHeadsK+embeddingHeadsV)*headsKV) * bytesPerElement)
	}

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
	}

	return
}

func (llm GGML) VisionGraphSize() (weights, graphSize uint64) {
	if llm.KV().Uint("vision.block_count") == 0 {
		return
	}

	for name, layer := range llm.Tensors().GroupLayers() {
		if name == "v" || strings.HasPrefix(name, "v.") {
			for _, tensor := range layer {
				weights += tensor.Size()
			}
		}
	}

	imageSize := uint64(llm.KV().Uint("vision.image_size"))
	patchSize := uint64(llm.KV().Uint("vision.patch_size"))
	if patchSize == 0 {
		slog.Warn("unknown patch size for vision model")
		return
	}

	numChannels := uint64(llm.KV().Uint("vision.num_channels"))

	numPatches := (imageSize / patchSize) * (imageSize / patchSize)
	if _, ok := llm.Tensors().GroupLayers()["v"]["class_embd"]; ok {
		numPatches++
	}

	headCount := uint64(llm.KV().Uint("vision.attention.head_count"))
	embeddingLength := uint64(llm.KV().Uint("vision.embedding_length"))

	switch llm.KV().Architecture() {
	case "mllama":
		numPaddedPatches := numPatches + 8 - (numPatches%8)%8

		maxNumTiles := uint64(llm.KV().Uint("vision.max_num_tiles"))

		graphSize = 4 * (8 +
			imageSize*imageSize*numChannels*maxNumTiles +
			embeddingLength*numPatches*maxNumTiles +
			9*embeddingLength*numPaddedPatches*maxNumTiles +
			numPaddedPatches*maxNumTiles*numPaddedPatches*maxNumTiles*headCount)
	case "gemma3", "mistral3":
		graphSize = 4 * (imageSize*imageSize*numChannels +
			embeddingLength*patchSize +
			numPatches*numPatches*headCount)
	case "qwen25vl":
		maxPixels := uint64(llm.KV().Uint("vision.max_pixels", 28*28*1280))

		numPatches := maxPixels / (patchSize * patchSize)

		graphSize = 4 * (maxPixels*numChannels + // Original image storage
			// Normalized pixels
			maxPixels*numChannels +
			// Patches storage (numPatches * channels * patchSize^2)
			numPatches*numChannels*patchSize*patchSize +
			// Self-attention calculations
			numPatches*numPatches*headCount +
			// Additional buffer for processing
			embeddingLength*numPatches)
	case "llama4":
		// vision graph is computed independently in the same schedule
		// and is negligible compared to the worst case text graph
	}

	return weights, graphSize
}

// SupportsKVCacheType checks if the requested cache type is supported
func (f GGML) SupportsKVCacheType(cacheType string) bool {
	return slices.Contains([]string{"f16", "q8_0", "q4_0"}, cacheType)
}

// SupportsFlashAttention checks if the model supports flash attention
func (f GGML) SupportsFlashAttention() bool {
	_, isEmbedding := f.KV()[fmt.Sprintf("%s.pooling_type", f.KV().Architecture())]
	if isEmbedding {
		return false
	}

	// Check head counts match and are non-zero
	headCountK := f.KV().EmbeddingHeadCountK()
	headCountV := f.KV().EmbeddingHeadCountV()
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
