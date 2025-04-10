package ggml

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"reflect"
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

func (kv KV) HeadCounts() []uint64 {
	return kv.UintOrArrayAsArray("attention.head_count", kv.BlockCount(), 1)
}

func (kv KV) HeadCountKVs() []uint64 {
	return kv.UintOrArrayAsArray("attention.head_count_kv", kv.BlockCount(), 1)
}

func (kv KV) EmbeddingHeadCount() []uint64 {
	headCount := kv.HeadCounts()
	embeddingHeadCount := make([]uint64, len(headCount))
	for i, heads := range headCount {
		if heads == 0 {
			embeddingHeadCount[i] = 0
		} else {
			embeddingHeadCount[i] = kv.EmbeddingLength() / heads
		}
	}

	return embeddingHeadCount
}

func (kv KV) FillArrayOrDefault(key string, defaultValue []uint64) []uint64 {
	length := len(defaultValue)
	if v, ok := keyValueUntyped(kv, key); ok {
		switch v := v.(type) {
		case uint32:
			return FillArray(uint64(v), length)
		case uint64:
			return FillArray(v, length)
		case int32:
			return FillArray(uint64(v), length)
		default:
			slog.Warn("unsupported type", "key", key, "type", reflect.TypeOf(v))
		}
	}

	return defaultValue
}

func (kv KV) EmbeddingHeadCountK() []uint64 {
	return kv.FillArrayOrDefault("attention.key_length", kv.EmbeddingHeadCount())
}

func (kv KV) EmbeddingHeadCountV() []uint64 {
	return kv.FillArrayOrDefault("attention.value_length", kv.EmbeddingHeadCount())
}

func (kv KV) GQAMax() uint64 {
	heads := kv.HeadCounts()
	headsKV := kv.HeadCountKVs()
	if len(heads) != len(headsKV) {
		slog.Warn("head count and head count kv are not the same length")
		return 0
	}
	if len(heads) == 0 {
		slog.Warn("head count is empty")
		return 0
	}

	maxGQA := uint64(0)
	for i := range heads {
		head := heads[i]
		headKV := headsKV[i]
		if head == 0 || headKV == 0 {
			return 0
		}
		gqa := head / headKV
		if gqa > maxGQA {
			maxGQA = gqa
		}
	}

	return maxGQA
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

func (kv KV) Bool(key string, defaultValue ...bool) bool {
	return keyValue(kv, key, append(defaultValue, false)...)
}

func (kv KV) UintOrArrayAsArray(key string, n uint64, defaultSingleValue ...uint64) []uint64 {
	var singleValue *uint64
	if v, ok := keyValueUntyped(kv, key); ok {
		switch v := v.(type) {
		case *array:
			switch v.values[0].(type) {
			case int32, uint32, uint64:
				values, ok := AsUint64Array(v.values)
				if ok {
					return values
				}
			default:
				slog.Warn("unexpected array value type", "key", key, "type", reflect.TypeOf(v))
			}
		case uint32:
			val := uint64(v)
			singleValue = &val
		case int32:
			val := uint64(v)
			singleValue = &val
		}
	}
	if singleValue == nil {
		slog.Warn("falling back to default")
		singleValue = &defaultSingleValue[0]
	}

	values := make([]uint64, n)
	for i := range values {
		values[i] = *singleValue
	}

	return values
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

func (kv KV) Floats(key string, defaultValue ...[]float32) []float32 {
	r := keyValue(kv, key, &array{})
	s := make([]float32, r.size)
	for i := range r.size {
		s[i] = float32(r.values[i].(float32))
	}
	return s
}

func (kv KV) OllamaEngineRequired() bool {
	return slices.Contains([]string{
		"gemma3",
		"mistral3",
	}, kv.Architecture())
}

func keyValue[T string | uint32 | uint64 | float32 | *array | bool](kv KV, key string, defaultValue ...T) T {
	if val, ok := keyValueUntyped(kv, key); ok {
		return val.(T)
	}

	slog.Warn("key not found", "key", key, "default", defaultValue[0])
	return defaultValue[0]
}

func keyValueUntyped(kv KV, key string) (any, bool) {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key]; ok {
		return val, true
	}

	return nil, false
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
	switch t.Kind {
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
		return 2 + 2 + blockSize
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
		return 4 + blockSize + 2*blockSize/16
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

func (t Tensor) Type() string {
	return fileType(t.Kind).String()
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

func (f GGML) GraphSize(context, batch uint64, numParallel int, kvCacheType string) (kv []uint64, partialOffload, fullOffload uint64) {
	embedding := f.KV().EmbeddingLength()
	heads := f.KV().HeadCounts()
	headsKV := f.KV().HeadCountKVs()
	vocab := uint64(f.KV()["tokenizer.ggml.tokens"].(*array).size)

	embeddingHeads := f.KV().EmbeddingHeadCount()
	maxEmbeddingHeads, ok := MaxValue(embeddingHeads)
	if !ok {
		maxEmbeddingHeads = 1
		slog.Warn("failed to get max embedding heads")
	}
	embeddingHeadsK := f.KV().EmbeddingHeadCountK()
	maxEmbeddingHeadsK, ok := MaxValue(embeddingHeadsK)
	if !ok {
		maxEmbeddingHeadsK = 1
		slog.Warn("failed to get max embedding headsK")
	}
	embeddingHeadsV := f.KV().EmbeddingHeadCountV()

	layers := f.Tensors().GroupLayers()

	bytesPerElement := kvCacheBytesPerElement(kvCacheType)
	kv = make([]uint64, f.KV().BlockCount())
	for i := range kv {
		kv[i] = uint64(float64(context*(embeddingHeadsK[i]+embeddingHeadsV[i])*headsKV[i]) * bytesPerElement)
	}

	maxHeads, ok := MaxValue(heads)
	if !ok {
		maxHeads = 1
		slog.Warn("failed to get max heads")
	}
	maxHeadsKV, ok := MaxValue(headsKV)
	if !ok {
		maxHeadsKV = 1
		slog.Warn("failed to get max headsKV")
	}

	switch f.KV().Architecture() {
	case "llama":
		fullOffload = max(
			4*batch*(1+4*embedding+context*(1+maxHeads)),
			4*batch*(embedding+vocab),
		)

		partialOffload = 4 * batch * embedding
		partialOffload += max(
			4*batch*(1+embedding+max(context, embedding))+embedding*embedding*9/16+4*context*(batch*maxHeads+maxEmbeddingHeads*maxHeadsKV),
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
		)

		if ffnGateExpsWeight, ok := layers["blk.0"]["ffn_gate_exps.weight"]; ok {
			// mixtral 8x22b
			ff := uint64(f.KV()["llama.feed_forward_length"].(uint32))
			partialOffload = max(
				3*ffnGateExpsWeight.Size()+4*batch*(2*ff+maxHeadsKV+embedding+context+maxEmbeddingHeads*maxHeadsKV),
				4*(context*batch*maxHeads+context*maxEmbeddingHeads*maxHeadsKV+batch*1024+maxEmbeddingHeads*maxHeadsKV*batch),
			)
		} else if ffnGateWeight, ok := layers["blk.0"]["ffn_gate.0.weight"]; ok {
			// mixtral 8x7b
			ffnGateWeight1 := ffnGateWeight.Shape[1]
			fullOffload = 4 * batch * (2 + 3*embedding + context*(1+maxHeads) + 2*maxHeadsKV + ffnGateWeight1)
			partialOffload = max(
				4*batch*(3+maxEmbeddingHeads*maxHeadsKV+embedding+context*(1+maxHeads)+ffnGateWeight1)+(embedding*embedding+3*embedding*maxHeadsKV*ffnGateWeight1)*9/16,
				4*batch*(1+2*embedding+context*(1+maxHeads))+embedding*(6*context*maxHeadsKV/maxHeads+embedding*9/16),
			)
		}
	case "mllama":
		var visionTokens, tiles uint64 = 1601, 4

		crossAttentionLayers := f.KV().Uints("attention.cross_attention_layers")
		for i := range kv {
			if slices.Contains(crossAttentionLayers, uint32(i)) {
				kv[i] = headsKV[i] * (embeddingHeadsK[i] + embeddingHeadsV[i]) *
					4 * // sizeof(float32)
					visionTokens *
					tiles
			}
		}

		fullOffload = max(
			4*batch*(2+3*embedding+maxEmbeddingHeadsK*maxHeads+context*(1+maxHeads)),
			// vocab graph
			4*batch*(embedding+vocab),
		)

		var ropeFreqsCount uint64
		if ropeFreqs, ok := f.Tensors().GroupLayers()["rope_freqs"]; ok {
			if ropeFreqsWeights, ok := ropeFreqs["weights"]; ok {
				ropeFreqsCount = ropeFreqsWeights.parameters()
			}
		}

		partialOffload = max(
			4*(batch*
				(2*embedding+1+context*(1+maxHeads)+maxEmbeddingHeadsK*maxHeads)+
				ropeFreqsCount+
				maxEmbeddingHeadsK*context*maxHeadsKV),
			// vocab graph
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
		)
	case "gemma", "gemma2", "gemma3":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(2+context+context*maxHeads+2*embedding+2*maxEmbeddingHeadsK*maxHeads),
		)

		partialOffload = max(
			4*embedding*batch+embedding*vocab*105/128+4*vocab*batch,
			4*batch*(2*embedding+1+2*maxEmbeddingHeadsK*maxHeads+context+context*maxHeads)+
				4*maxEmbeddingHeadsK*context*8+
				embedding*embedding*maxEmbeddingHeadsK*maxHeads*9/16,
		)

		// Gemma2 also has sliding window attention but we only have an optimized implementation in the Ollama
		// engine. Gemma3 always uses the Ollama engine.
		if f.KV().Architecture() == "gemma3" {
			const gemma3GlobalCacheCount = 6
			slidingWindow := (uint64(numParallel) * uint64(f.KV().Uint("attention.sliding_window"))) + batch
			for i := range kv {
				// Every 6th layer is a global layer, which is the full context size that has already been set. The other
				// layers are the smaller local (sliding) layers.
				if (i+1)%gemma3GlobalCacheCount != 0 {
					kv[i] = uint64(float64(slidingWindow*(embeddingHeadsK[i]+embeddingHeadsV[i])*headsKV[i]) * bytesPerElement)
				}
			}
		}
	case "command-r":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(2+4*embedding+context*(1+maxHeads)),
		)

		partialOffload = max(
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
			4*batch*(1+2*embedding+context*(1+maxHeads))+4*embedding*context+embedding*embedding*9/16,
		)
	case "qwen2":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(1+2*embedding+context+context*maxHeads),
		)

		partialOffload = max(
			4*batch*(embedding+vocab)+embedding*vocab*105/128,
			4*(batch*(1+2*embedding+context*(1+maxHeads))+embedding*(1+context)),
		)
	case "phi2":
		fullOffload = max(
			4*batch*(embedding+vocab),
			4*batch*(1+4*embedding+context+context*maxHeads),
		)

		partialOffload = max(
			4*batch*(2*embedding+vocab)+embedding*vocab*105/128,
			4*batch*(2+3*embedding+context+context*maxHeads),
		)
	case "stablelm":
		fullOffload = 4 * batch * (context*(1+maxHeads) + 3*embedding + 2)
		partialOffload = max(
			4*batch*(vocab+2*embedding),
			fullOffload,
		)
	case "deepseek2":
		fullOffload = max(
			4*batch*(3*embedding+vocab),
			4*batch*(3*embedding+2+context*(1+maxHeadsKV)+2*maxEmbeddingHeadsK*maxHeadsKV),
		)

		partialOffload = max(
			4*batch*(3*embedding+vocab)+embedding*vocab*105/128,
			4*batch*(2*embedding+1+2*maxEmbeddingHeadsK*maxHeadsKV+context+context*maxHeadsKV)+4*maxEmbeddingHeadsK*context*maxHeadsKV+embedding*embedding*maxEmbeddingHeadsK*maxHeadsKV*9/16,
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
					context*maxHeads+
					maxEmbeddingHeadsK*maxHeads+
					qkvBias.Shape[0]),
			)

			partialOffload = max(
				partialOffload,
				4*batch*(1+
					2*embedding+
					maxEmbeddingHeadsK*maxHeads+
					context+
					context*maxHeads)+
					4*maxEmbeddingHeadsK*context+
					4*context*maxEmbeddingHeadsK+
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
	headCount := f.KV().HeadCounts()
	embeddingHeadCountK := f.KV().EmbeddingHeadCountK()
	embeddingHeadCountV := f.KV().EmbeddingHeadCountV()
	for i := range headCount {
		if embeddingHeadCountK[i] != embeddingHeadCountV[i] {
			return false
		}
	}
	return true
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

func AsUint64Array(v []any) ([]uint64, bool) {
	switch v[0].(type) {
	case uint32:
		values := make([]uint64, len(v))
		for i, v := range v {
			values[i] = uint64(v.(uint32))
		}
		return values, true
	case uint64:
		values := make([]uint64, len(v))
		for i, v := range v {
			values[i] = v.(uint64)
		}
		return values, true
	case int32:
		values := make([]uint64, len(v))
		for i, val := range v {
			val := val.(int32)
			if val < 0 {
				slog.Warn("negative value in int32 array", "value", val)
				return nil, false
			}
			values[i] = uint64(val)
		}
		return values, true
	}
	return nil, false
}

func MaxValue(values []uint64) (uint64, bool) {
	if len(values) == 0 {
		return 0, false
	}

	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max, true
}

func FillArray[T any](value T, n int) []T {
	values := make([]T, n)
	for i := range values {
		values[i] = value
	}
	return values
}
