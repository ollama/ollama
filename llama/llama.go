package llama

/*
#cgo CFLAGS: -std=c11
#cgo windows CFLAGS: -Wno-dll-attribute-on-redeclaration
#cgo CXXFLAGS: -std=c++17
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/common
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/vendor
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/tools/mtmd
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/src
#cgo CPPFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include

#include <stdlib.h>
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"

#include "sampling_ext.h"

extern bool llamaProgressCallback(float progress, void *user_data);
extern void llamaLog(int level, char* text, void* user_data);
*/
import "C"

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"runtime/cgo"
	"slices"
	"strings"
	"sync"
	"unsafe"

	_ "github.com/ollama/ollama/llama/llama.cpp/common"
	_ "github.com/ollama/ollama/llama/llama.cpp/src"
	_ "github.com/ollama/ollama/llama/llama.cpp/tools/mtmd"
	_ "github.com/ollama/ollama/llama/llama.cpp/tools/mtmd/models"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

func init() {
	C.llama_log_set(C.ggml_log_callback(C.llamaLog), nil)
}

//export llamaLog
func llamaLog(level C.int, text *C.char, _ unsafe.Pointer) {
	// slog levels zeros INFO and are multiples of 4
	if slog.Default().Enabled(context.TODO(), slog.Level(int(level-C.GGML_LOG_LEVEL_INFO)*4)) {
		fmt.Fprint(os.Stderr, C.GoString(text))
	}
}

func BackendInit() {
	ggml.OnceLoad()
	C.llama_backend_init()
}

type Devices struct {
	ml.DeviceID
	LlamaID uint64
}

func EnumerateGPUs() []Devices {
	var ids []Devices

	for i := range C.ggml_backend_dev_count() {
		device := C.ggml_backend_dev_get(i)

		switch C.ggml_backend_dev_type(device) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			var props C.struct_ggml_backend_dev_props
			C.ggml_backend_dev_get_props(device, &props)
			ids = append(ids, Devices{
				DeviceID: ml.DeviceID{
					ID:      C.GoString(props.id),
					Library: C.GoString(props.library),
				},
				LlamaID: uint64(i),
			})
		}
	}

	return ids
}

func GetModelArch(modelPath string) (string, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))

	gguf_ctx := C.gguf_init_from_file(mp, C.struct_gguf_init_params{no_alloc: true, ctx: (**C.struct_ggml_context)(C.NULL)})
	if gguf_ctx == nil {
		return "", errors.New("unable to load model file")
	}
	defer C.gguf_free(gguf_ctx)

	key := C.CString("general.architecture")
	defer C.free(unsafe.Pointer(key))
	arch_index := C.gguf_find_key(gguf_ctx, key)
	if int(arch_index) < 0 {
		return "", errors.New("unknown model architecture")
	}

	arch := C.gguf_get_val_str(gguf_ctx, arch_index)

	return C.GoString(arch), nil
}

type ContextParams struct {
	c C.struct_llama_context_params
}

func NewContextParams(numCtx int, batchSize int, numSeqMax int, threads int, flashAttention ml.FlashAttentionType, kvCacheType string) ContextParams {
	params := C.llama_context_default_params()
	params.n_ctx = C.uint(numCtx)
	params.n_batch = C.uint(batchSize * numSeqMax)
	params.n_ubatch = C.uint(batchSize)
	params.n_seq_max = C.uint(numSeqMax)
	params.n_threads = C.int(threads)
	params.n_threads_batch = params.n_threads
	params.embeddings = C.bool(true)
	switch flashAttention {
	case ml.FlashAttentionEnabled:
		params.flash_attn_type = int32(C.LLAMA_FLASH_ATTN_TYPE_ENABLED)
	case ml.FlashAttentionDisabled:
		params.flash_attn_type = int32(C.LLAMA_FLASH_ATTN_TYPE_DISABLED)
	case ml.FlashAttentionAuto:
		params.flash_attn_type = int32(C.LLAMA_FLASH_ATTN_TYPE_AUTO)
	}
	params.type_k = kvCacheTypeFromStr(strings.ToLower(kvCacheType))
	params.type_v = kvCacheTypeFromStr(strings.ToLower(kvCacheType))

	return ContextParams{c: params}
}

// kvCacheTypeFromStr converts a string cache type to the corresponding GGML type value
func kvCacheTypeFromStr(s string) C.enum_ggml_type {
	if s == "" {
		return C.GGML_TYPE_F16
	}

	switch s {
	case "q8_0":
		return C.GGML_TYPE_Q8_0
	case "q4_0":
		return C.GGML_TYPE_Q4_0
	default:
		return C.GGML_TYPE_F16
	}
}

type Context struct {
	c          *C.struct_llama_context
	numThreads int
}

var ErrKvCacheFull = errors.New("could not find a kv cache slot")

func (c *Context) Decode(batch *Batch) error {
	// Positive return values does not mean a fatal error, but rather a warning.
	//   0 - success
	//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
	// < 0 - error
	code := int(C.llama_decode(c.c, batch.c))

	if code < 0 {
		return fmt.Errorf("llama_decode failed with code %d", code)
	}

	if code > 0 {
		return ErrKvCacheFull
	}

	return nil
}

func (c *Context) Model() *Model {
	return &Model{c: C.llama_get_model(c.c)}
}

func (c *Context) KvCacheSeqAdd(seqId int, p0 int, p1 int, delta int) {
	C.llama_memory_seq_add(C.llama_get_memory(c.c), C.int(seqId), C.int(p0), C.int(p1), C.int(delta))
}

func (c *Context) KvCacheSeqRm(seqId int, p0 int, p1 int) bool {
	return bool(C.llama_memory_seq_rm(C.llama_get_memory(c.c), C.int(seqId), C.int(p0), C.int(p1)))
}

func (c *Context) KvCacheSeqCp(srcSeqId int, dstSeqId int, p0 int, p1 int) {
	C.llama_memory_seq_cp(C.llama_get_memory(c.c), C.int(srcSeqId), C.int(dstSeqId), C.int(p0), C.int(p1))
}

func (c *Context) KvCacheClear() {
	C.llama_memory_clear(C.llama_get_memory(c.c), true)
}

func (c *Context) KvCacheCanShift() bool {
	return bool(C.llama_memory_can_shift(C.llama_get_memory(c.c)))
}

// Get the embeddings for a sequence id
func (c *Context) GetEmbeddingsSeq(seqId int) []float32 {
	e := unsafe.Pointer(C.llama_get_embeddings_seq(c.c, C.int(seqId)))
	if e == nil {
		return nil
	}

	embeddings := make([]float32, c.Model().NEmbd())
	_ = copy(embeddings, unsafe.Slice((*float32)(e), c.Model().NEmbd()))
	return embeddings
}

func (c *Context) GetEmbeddingsIth(i int) []float32 {
	e := unsafe.Pointer(C.llama_get_embeddings_ith(c.c, C.int32_t(i)))
	if e == nil {
		return nil
	}

	embeddings := make([]float32, c.Model().NEmbd())
	_ = copy(embeddings, unsafe.Slice((*float32)(e), c.Model().NEmbd()))
	return embeddings
}

// GetLogitsIth gets the logits for the ith token
func (c *Context) GetLogitsIth(i int) []float32 {
	logits := unsafe.Pointer(C.llama_get_logits_ith(c.c, C.int32_t(i)))
	if logits == nil {
		return nil
	}

	vocabSize := c.Model().NumVocab()
	result := make([]float32, vocabSize)
	_ = copy(result, unsafe.Slice((*float32)(logits), vocabSize))
	return result
}

type ModelParams struct {
	Devices      []uint64
	NumGpuLayers int
	MainGpu      int
	UseMmap      bool
	TensorSplit  []float32
	Progress     func(float32)
	VocabOnly    bool
}

//export llamaProgressCallback
func llamaProgressCallback(progress C.float, userData unsafe.Pointer) C.bool {
	handle := *(*cgo.Handle)(userData)
	callback := handle.Value().(func(float32))
	callback(float32(progress))
	return true
}

func LoadModelFromFile(modelPath string, params ModelParams) (*Model, error) {
	cparams := C.llama_model_default_params()
	cparams.n_gpu_layers = C.int(params.NumGpuLayers)
	cparams.main_gpu = C.int32_t(params.MainGpu)
	cparams.use_mmap = C.bool(params.UseMmap)
	cparams.vocab_only = C.bool(params.VocabOnly)

	var devices []C.ggml_backend_dev_t
	for _, llamaID := range params.Devices {
		devices = append(devices, C.ggml_backend_dev_get(C.size_t(llamaID)))
	}
	if len(devices) > 0 {
		devices = append(devices, C.ggml_backend_dev_t(C.NULL))
		devicesData := &devices[0]

		var devicesPin runtime.Pinner
		devicesPin.Pin(devicesData)
		defer devicesPin.Unpin()

		cparams.devices = devicesData
	}

	if len(params.TensorSplit) > 0 {
		tensorSplitData := &params.TensorSplit[0]

		var tensorSplitPin runtime.Pinner
		tensorSplitPin.Pin(tensorSplitData)
		defer tensorSplitPin.Unpin()

		cparams.tensor_split = (*C.float)(unsafe.Pointer(tensorSplitData))
	}

	if params.Progress != nil {
		handle := cgo.NewHandle(params.Progress)
		defer handle.Delete()

		var handlePin runtime.Pinner
		handlePin.Pin(&handle)
		defer handlePin.Unpin()

		cparams.progress_callback = C.llama_progress_callback(C.llamaProgressCallback)
		cparams.progress_callback_user_data = unsafe.Pointer(&handle)
	}

	m := Model{c: C.llama_model_load_from_file(C.CString(modelPath), cparams)}
	if m.c == nil {
		return nil, fmt.Errorf("unable to load model: %s", modelPath)
	}

	return &m, nil
}

func FreeModel(model *Model) {
	C.llama_model_free(model.c)
}

func NewContextWithModel(model *Model, params ContextParams) (*Context, error) {
	c := Context{
		c:          C.llama_init_from_model(model.c, params.c),
		numThreads: int(params.c.n_threads),
	}
	if c.c == nil {
		return nil, errors.New("unable to create llama context")
	}

	return &c, nil
}

func (m *Model) NumVocab() int {
	return int(C.llama_vocab_n_tokens(m.Vocab()))
}

func (m *Model) TokenIsEog(token int) bool {
	return bool(C.llama_vocab_is_eog(m.Vocab(), C.llama_token(token)))
}

func (m *Model) AddBOSToken() bool {
	return bool(C.llama_vocab_get_add_bos(m.Vocab()))
}

func (m *Model) ApplyLoraFromFile(context *Context, loraPath string, scale float32, threads int) error {
	cLoraPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cLoraPath))

	loraAdapter := C.llama_adapter_lora_init(m.c, cLoraPath)
	if loraAdapter == nil {
		return errors.New("unable to load lora")
	}

	err := -1
	if loraAdapter != nil {
		err = int(C.llama_set_adapter_lora(context.c, loraAdapter, C.float(scale)))
	}
	if err != 0 {
		return errors.New("error applying lora from file")
	}

	return nil
}

func (m *Model) Vocab() *C.struct_llama_vocab {
	return C.llama_model_get_vocab(m.c)
}

type Batch struct {
	c         C.struct_llama_batch
	batchSize int
	maxSeq    int
	embedSize int
}

// Creates a new batch for either word tokens or image embeddings (if embedSize is non-zero).
// Batches cannot contain both types at the same time. batchSize is the maximum number of entries
// that can be added per sequence
func NewBatch(batchSize int, maxSeq int, embedSize int) (*Batch, error) {
	b := Batch{
		c:         C.llama_batch_init(C.int(batchSize*maxSeq), C.int(embedSize), C.int(maxSeq)),
		batchSize: batchSize,
		maxSeq:    maxSeq,
		embedSize: embedSize,
	}

	// Check to see if any of the allocations in llama_batch_init() failed
	nilPointer := (embedSize == 0 && b.c.token == nil) || (embedSize != 0 && b.c.embd == nil) ||
		b.c.pos == nil || b.c.n_seq_id == nil || b.c.seq_id == nil || b.c.logits == nil ||
		slices.Contains(unsafe.Slice(b.c.seq_id, b.allocSize()), nil)

	if nilPointer {
		C.llama_batch_free(b.c)
		return nil, fmt.Errorf("unable to allocate batch (batchSize=%v maxSeq=%v embedSize=%v)", batchSize, maxSeq, embedSize)
	}

	return &b, nil
}

func (b *Batch) Size() int {
	return b.batchSize
}

func (b *Batch) allocSize() int {
	return b.batchSize * b.maxSeq
}

func (b *Batch) NumTokens() int {
	return int(b.c.n_tokens)
}

func (b *Batch) IsEmbedding() bool {
	return b.embedSize != 0
}

// Add adds either a token or an image embedding to the batch depending on the type
// when the batch was initialized. The other argument will be ignored. Adds to the
// batch with the given position for the given sequence ids, and optionally instructs
// to include logits.
func (b *Batch) Add(token int, embed []float32, pos int, logits bool, seqIds ...int) {
	if !b.IsEmbedding() {
		unsafe.Slice(b.c.token, b.allocSize())[b.c.n_tokens] = C.llama_token(token)
	} else {
		copy(unsafe.Slice((*float32)(b.c.embd), b.allocSize()*b.embedSize)[int(b.c.n_tokens)*b.embedSize:], embed)
	}
	unsafe.Slice(b.c.pos, b.allocSize())[b.c.n_tokens] = C.llama_pos(pos)
	unsafe.Slice(b.c.n_seq_id, b.allocSize())[b.c.n_tokens] = C.int(len(seqIds))

	for i, s := range seqIds {
		unsafe.Slice((unsafe.Slice(b.c.seq_id, b.allocSize())[b.c.n_tokens]), C.int(len(seqIds)))[i] = C.int32_t(s)
	}

	if logits {
		unsafe.Slice(b.c.logits, b.allocSize())[b.c.n_tokens] = 1
	} else {
		unsafe.Slice(b.c.logits, b.allocSize())[b.c.n_tokens] = 0
	}

	b.c.n_tokens += 1
}

func (b *Batch) Clear() {
	b.c.n_tokens = 0
}

func (b *Batch) Free() {
	b.batchSize = 0
	C.llama_batch_free(b.c)
}

type Model struct {
	c *C.struct_llama_model
}

func (m *Model) TokenToPiece(token int) string {
	tokenLen := 12
	buf := make([]byte, tokenLen)
	tokenLen = int(C.llama_token_to_piece(
		m.Vocab(),
		C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(tokenLen),
		C.int32_t(0),
		C.bool(true),
	))
	if tokenLen < 0 {
		tokenLen = -tokenLen

		buf = make([]byte, tokenLen)
		C.llama_token_to_piece(
			m.Vocab(),
			C.int32_t(token),
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int32_t(tokenLen),
			C.int32_t(0),
			C.bool(true),
		)
	}
	return strings.TrimRight(string(buf), "\x00")
}

func (m *Model) Tokenize(text string, addSpecial bool, parseSpecial bool) ([]int, error) {
	maxTokens := len(text) + 2
	cTokens := make([]C.llama_token, maxTokens)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_tokenize(
		m.Vocab(),
		cText,
		C.int32_t(len(text)),
		&cTokens[0],
		C.int32_t(maxTokens),
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	// if the result is negative, reallocate and retry with the correct buffer size
	if result < 0 {
		maxTokens = int(-result)
		cTokens = make([]C.llama_token, maxTokens)
		result = C.llama_tokenize(
			m.Vocab(),
			cText,
			C.int32_t(len(text)),
			&cTokens[0],
			C.int32_t(maxTokens),
			C.bool(addSpecial),
			C.bool(parseSpecial),
		)
		if result < 0 {
			return nil, fmt.Errorf("tokenization failed, required %d tokens", -result)
		}
	}

	tokens := make([]int, result)
	for i := range result {
		tokens[i] = int(cTokens[i])
	}

	return tokens, nil
}

func (m *Model) NEmbd() int {
	return int(C.llama_model_n_embd(m.c))
}

// vision processing
type MtmdContext struct {
	c *C.struct_mtmd_context
}

func NewMtmdContext(llamaContext *Context, modelPath string) (*MtmdContext, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))
	// TODO: Support non-default params
	cp := C.mtmd_context_params_default()

	// NOTE: The model and projector embedding lengths are checked during init
	c := C.mtmd_init_from_file(mp, C.llama_get_model(llamaContext.c), cp)
	if c == nil {
		return nil, fmt.Errorf("unable to load mmtd model: %v", modelPath)
	}

	return &MtmdContext{c: c}, nil
}

func (c *MtmdContext) Free() {
	C.mtmd_free(c.c)
}

type MtmdChunk struct {
	Embed  []float32
	Tokens []int
}

func (c *MtmdContext) MultimodalTokenize(llamaContext *Context, data []byte) ([]MtmdChunk, error) {
	// Initialize the input chunks pointer
	ic := C.mtmd_input_chunks_init()
	defer C.mtmd_input_chunks_free(ic)

	// Initialize an empty text prompt so we can tokenize
	it := C.mtmd_input_text_init(C.mtmd_default_marker(), true, true)
	defer C.mtmd_input_text_free(it)

	// Initialize a bitmap with the image data
	bm := C.mtmd_helper_bitmap_init_from_buf(c.c, (*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	defer C.mtmd_bitmap_free(bm)

	// Tokenize the image
	if C.int32_t(0) != C.mtmd_tokenize(c.c, ic, it, &bm, 1) {
		return nil, errors.New("unable to tokenize mtmd embedding from image")
	}
	nChunks := C.mtmd_input_chunks_size(ic)
	numEmbed := llamaContext.Model().NEmbd()
	outChunks := make([]MtmdChunk, 0)
	for i := range int(nChunks) {
		chunk := C.mtmd_input_chunks_get(ic, C.size_t(i))
		numTokens := int(C.mtmd_input_chunk_get_n_tokens(chunk))
		slog.Debug("chunk tokens", "index", i, "numTokens", numTokens)

		if C.mtmd_input_chunk_get_type(chunk) == C.MTMD_INPUT_CHUNK_TYPE_TEXT {
			// If this is a text chunk, add the tokens
			cNumTokens := C.size_t(0)
			cTokens := C.mtmd_input_chunk_get_tokens_text(chunk, &cNumTokens)
			cTokensArr := unsafe.Slice(cTokens, int(cNumTokens))
			tokens := make([]int, int(cNumTokens))
			for j := range int(cNumTokens) {
				tokens[j] = int(cTokensArr[j])
			}
			outChunks = append(outChunks, MtmdChunk{Tokens: tokens})
		} else {
			// Otherwise, encode the image chunk to embeddings

			// Encode the chunk
			if C.int32_t(0) != C.mtmd_encode_chunk(c.c, chunk) {
				return nil, errors.New("unable to encode mtmd image chunk")
			}

			// Get the embeddings for this chunk
			chunkEmbed := make([][]float32, numTokens)
			chunkEmbd := C.mtmd_get_output_embd(c.c)
			if nil == chunkEmbd {
				return nil, errors.New("no mtmd image embedding")
			}

			// Extend the embedding array for each token
			s := unsafe.Slice((*float32)(chunkEmbd), numTokens*numEmbed)
			rows := make([]float32, len(s))
			copy(rows, s)
			for i := range numTokens {
				chunkEmbed[i] = rows[i*numEmbed : (i+1)*numEmbed]
			}
			for _, e := range chunkEmbed {
				outChunks = append(outChunks, MtmdChunk{Embed: e})
			}
		}
	}
	slog.Debug("image tokenization chunks", "totalChunks", len(outChunks))
	return outChunks, nil
}

func (c *Context) Synchronize() {
	C.llama_synchronize(c.c)
}

// sampling
// TODO: this is a temporary wrapper to allow calling C++ code from CGo
type SamplingContext struct {
	c *C.struct_common_sampler
}

type SamplingParams struct {
	TopK           int
	TopP           float32
	MinP           float32
	TypicalP       float32
	Temp           float32
	RepeatLastN    int
	PenaltyRepeat  float32
	PenaltyFreq    float32
	PenaltyPresent float32
	PenalizeNl     bool
	Seed           uint32
	Grammar        string
}

func NewSamplingContext(model *Model, params SamplingParams) (*SamplingContext, error) {
	var cparams C.struct_common_sampler_cparams
	cparams.top_k = C.int32_t(params.TopK)
	cparams.top_p = C.float(params.TopP)
	cparams.min_p = C.float(params.MinP)
	cparams.typical_p = C.float(params.TypicalP)
	cparams.temp = C.float(params.Temp)
	cparams.penalty_last_n = C.int32_t(params.RepeatLastN)
	cparams.penalty_repeat = C.float(params.PenaltyRepeat)
	cparams.penalty_freq = C.float(params.PenaltyFreq)
	cparams.penalty_present = C.float(params.PenaltyPresent)
	cparams.seed = C.uint32_t(params.Seed)

	grammar := C.CString(params.Grammar)
	defer C.free(unsafe.Pointer(grammar))

	cparams.grammar = grammar
	context := &SamplingContext{c: C.common_sampler_cinit(model.c, &cparams)}
	if context.c == nil {
		return nil, errors.New("unable to create sampling context")
	}

	runtime.SetFinalizer(context, func(s *SamplingContext) { C.common_sampler_cfree(s.c) })

	return context, nil
}

func (s *SamplingContext) Reset() {
	C.common_sampler_creset(s.c)
}

func (s *SamplingContext) Sample(llamaContext *Context, idx int) int {
	return int(C.common_sampler_csample(s.c, llamaContext.c, C.int(idx)))
}

func (s *SamplingContext) Accept(id int, applyGrammar bool) {
	C.common_sampler_caccept(s.c, C.llama_token(id), C.bool(applyGrammar))
}

// SchemaToGrammar converts the provided JSON schema to a grammar. It returns
// nil if the provided schema is invalid JSON or an invalid JSON schema.
func SchemaToGrammar(schema []byte) []byte {
	cStr := C.CString(string(schema))
	defer C.free(unsafe.Pointer(cStr))

	// Allocate buffer for grammar based on schema length but with upper bound
	maxLen := max(32768, min(1024*1024, len(schema)*4))
	buf := make([]byte, maxLen)

	// Call C function to convert schema to grammar
	n := C.schema_to_grammar(cStr, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(maxLen))
	if n == 0 {
		// preserve nil
		return nil
	}
	return buf[:n]
}

type TokenData struct {
	ID    int32
	Logit float32
}

type Grammar struct {
	c  *C.struct_llama_grammar
	mu sync.Mutex
}

func NewGrammar(grammar string, vocabIds []uint32, vocabValues []string, eogTokens []int32) *Grammar {
	cGrammar := C.CString(grammar)
	defer C.free(unsafe.Pointer(cGrammar))

	cTokens := make([]C.uint32_t, len(vocabIds))
	for i, token := range vocabIds {
		cTokens[i] = C.uint32_t(token)
	}

	cPieces := make([]*C.char, len(vocabValues))
	for i, piece := range vocabValues {
		cPieces[i] = C.CString(piece)
		defer C.free(unsafe.Pointer(cPieces[i]))
	}

	cEogTokens := make([]C.uint32_t, len(eogTokens))
	for i, token := range eogTokens {
		cEogTokens[i] = C.uint32_t(token)
	}

	g := C.grammar_init(cGrammar, unsafe.SliceData(cTokens), C.size_t(len(cTokens)), unsafe.SliceData(cPieces), unsafe.SliceData(cEogTokens), C.size_t(len(cEogTokens)))
	if g == nil {
		return nil
	}

	return &Grammar{c: g}
}

func (g *Grammar) Free() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.c != nil {
		C.grammar_free(g.c)
		g.c = nil
	}
}

func (g *Grammar) Apply(tokens []TokenData) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.c == nil {
		return
	}

	tds := make([]C.struct_llama_token_data, len(tokens))
	for i, token := range tokens {
		tds[i] = C.struct_llama_token_data{
			id:    C.int32_t(token.ID),
			logit: C.float(token.Logit),
			p:     C.float(0.0),
		}
	}
	tda := &C.llama_token_data_array{
		data:     (*C.struct_llama_token_data)(unsafe.Pointer(&tds[0])),
		size:     C.size_t(len(tokens)),
		selected: C.int64_t(-1),
		sorted:   C.bool(false),
	}
	var pinner runtime.Pinner
	pinner.Pin(&tds[0])
	defer pinner.Unpin()

	C.grammar_apply(g.c, tda)
	for i := range tokens {
		tokens[i].Logit = float32(tds[i].logit)
	}
}

func (g *Grammar) Accept(token int32) {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if grammar was freed
	if g.c == nil {
		return
	}

	C.grammar_accept(g.c, C.llama_token(token))
}
