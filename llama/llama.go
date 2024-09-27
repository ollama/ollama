package llama

/*
#cgo CFLAGS: -O2 -std=c11 -DGGML_BUILD=1 -DNDEBUG -DLOG_DISABLE_LOGS -DGGML_USE_LLAMAFILE
#cgo CXXFLAGS: -O2 -std=c++11 -DGGML_BUILD=1 -DNDEBUG -DLOG_DISABLE_LOGS -DGGML_USE_LLAMAFILE
#cgo amd64,avx CFLAGS: -mavx
#cgo amd64,avx CXXFLAGS: -mavx
#cgo amd64,avx2 CFLAGS: -mavx2 -mfma
#cgo amd64,avx2 CXXFLAGS: -mavx2 -mfma
#cgo amd64,f16c CFLAGS: -mf16c
#cgo amd64,f16c CXXFLAGS: -mf16c
#cgo amd64,fma CFLAGS: -mfma
#cgo amd64,fma CXXFLAGS: -mfma
#cgo avx CFLAGS: -mavx
#cgo avx CXXFLAGS: -mavx
#cgo avx2 CFLAGS: -mavx2 -mfma -mf16c
#cgo avx2 CXXFLAGS: -mavx2 -mfma -mf16c
#cgo cuda CFLAGS: -fPIE -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo cuda CFLAGS: -fPIE -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo cuda CXXFLAGS: -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo cuda CXXFLAGS: -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo cuda_v11 LDFLAGS: -lggml_cuda_v11 -L/usr/local/cuda-11/lib64
#cgo cuda_v12 LDFLAGS: -lggml_cuda_v12 -L/usr/local/cuda-12/lib64
#cgo darwin,amd64 CFLAGS: -Wno-incompatible-pointer-types-discards-qualifiers
#cgo darwin,amd64 CXXFLAGS: -Wno-incompatible-pointer-types-discards-qualifiers
#cgo darwin,amd64 LDFLAGS: -framework Foundation
#cgo darwin,amd64,avx2 CFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo darwin,amd64,avx2 CXXFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo darwin,amd64,avx2 LDFLAGS: -framework Accelerate
#cgo darwin,arm64 CFLAGS: -DGGML_USE_METAL -DGGML_USE_ACCELERATE -DGGML_METAL_EMBED_LIBRARY -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_BLAS
#cgo darwin,arm64 CXXFLAGS: -DGGML_USE_METAL -DGGML_USE_ACCELERATE -DGGML_METAL_EMBED_LIBRARY -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_BLAS
#cgo darwin,arm64 LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
#cgo linux CFLAGS: -D_GNU_SOURCE
#cgo linux CXXFLAGS: -D_GNU_SOURCE
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build/Linux/amd64
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build/Linux/amd64
#cgo linux,arm64 CFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA -D__ARM_FEATURE_MATMUL_INT8
#cgo linux,arm64 CXXFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA -D__ARM_FEATURE_MATMUL_INT8
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/build/Linux/arm64
#cgo linux,arm64,sve CFLAGS: -march=armv8.6-a+sve
#cgo linux,arm64,sve CXXFLAGS: -march=armv8.6-a+sve
#cgo linux,cuda LDFLAGS: -lcuda -lcudart -lcublas -lcublasLt -lpthread -ldl -lrt -lresolv
#cgo linux,rocm LDFLAGS: -L/opt/rocm/lib -lpthread -ldl -lrt -lresolv
#cgo rocm CFLAGS: -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm CXXFLAGS: -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm LDFLAGS: -L${SRCDIR} -lggml_rocm -lhipblas -lamdhip64 -lrocblas
#cgo windows CFLAGS: -Wno-discarded-qualifiers
#cgo windows CFLAGS: -Wno-discarded-qualifiers
#cgo windows LDFLAGS: -lmsvcrt
#cgo windows LDFLAGS: -lmsvcrt -static-libstdc++ -static-libgcc -static
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build/Windows/amd64
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build/Windows/amd64
#cgo windows,arm64 CFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
#cgo windows,arm64 CXXFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
#cgo windows,arm64 LDFLAGS: -L${SRCDIR}/build/Windows/arm64
#cgo windows,arm64 LDFLAGS: -L${SRCDIR}/build/Windows/arm64
#cgo windows,cuda LDFLAGS: -lcuda -lcudart -lcublas -lcublasLt
#cgo windows,rocm LDFLAGS: -lggml_rocm -lhipblas -lamdhip64 -lrocblas

#include <stdlib.h>
#include "llama.h"
#include "clip.h"
#include "llava.h"
#include "sampling_ext.h"

bool llamaProgressCallback(float progress, void *user_data);
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"runtime"
	"runtime/cgo"
	"strings"
	"unsafe"
)

var CpuFeatures = ""

func BackendInit() {
	C.llama_backend_init()
}

func PrintSystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

type ContextParams struct {
	c C.struct_llama_context_params
}

func NewContextParams(numCtx int, batchSize int, numSeqMax int, threads int, flashAttention bool) ContextParams {
	params := C.llama_context_default_params()
	params.n_ctx = C.uint(numCtx)
	params.n_batch = C.uint(batchSize)
	params.n_seq_max = C.uint(numSeqMax)
	params.n_threads = C.int(threads)
	params.n_threads_batch = params.n_threads
	params.embeddings = C.bool(true)
	params.flash_attn = C.bool(flashAttention)
	return ContextParams{c: params}
}

type Context struct {
	c          *C.struct_llama_context
	numThreads int
}

func (c *Context) KvCacheClear() {
	C.llama_kv_cache_clear(c.c)
}

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
		return fmt.Errorf("could not find a KV slot for the batch - try reducing the size of the batch or increase the context. code: %d", code)
	}

	return nil
}

func (c *Context) Model() *Model {
	return &Model{c: C.llama_get_model(c.c)}
}

func (c *Context) GetLogitsIth(i int) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(C.llama_get_logits_ith(c.c, C.int(i)))), c.Model().NumVocab())
}

func (c *Context) KvCacheSeqAdd(seqId int, p0 int, p1 int, delta int) {
	C.llama_kv_cache_seq_add(c.c, C.int(seqId), C.int(p0), C.int(p1), C.int(delta))
}

func (c *Context) KvCacheSeqRm(seqId int, p0 int, p1 int) bool {
	return bool(C.llama_kv_cache_seq_rm(c.c, C.int(seqId), C.int(p0), C.int(p1)))
}

func (c *Context) KvCacheSeqCp(srcSeqId int, dstSeqId int, p0 int, p1 int) {
	C.llama_kv_cache_seq_cp(c.c, C.int(srcSeqId), C.int(dstSeqId), C.int(p0), C.int(p1))
}

// Get the embeddings for a sequence id
func (c *Context) GetEmbeddingsSeq(seqId int) []float32 {
	embeddings := unsafe.Pointer(C.llama_get_embeddings_seq(c.c, C.int(seqId)))
	if embeddings == nil {
		return nil
	}

	return unsafe.Slice((*float32)(embeddings), c.Model().NEmbd())
}

func (c *Context) GetEmbeddingsIth(i int) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(C.llama_get_embeddings_ith(c.c, C.int32_t(i)))), c.Model().NEmbd())
}

type ModelParams struct {
	NumGpuLayers int
	MainGpu      int
	UseMmap      bool
	UseMlock     bool
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

func LoadModelFromFile(modelPath string, params ModelParams) *Model {
	cparams := C.llama_model_default_params()
	cparams.n_gpu_layers = C.int(params.NumGpuLayers)
	cparams.main_gpu = C.int32_t(params.MainGpu)
	cparams.use_mmap = C.bool(params.UseMmap)
	cparams.use_mlock = C.bool(params.UseMlock)
	cparams.vocab_only = C.bool(params.VocabOnly)

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

	return &Model{c: C.llama_load_model_from_file(C.CString(modelPath), cparams)}
}

func FreeModel(model *Model) {
	C.llama_free_model(model.c)
}

func NewContextWithModel(model *Model, params ContextParams) *Context {
	return &Context{
		c:          C.llama_new_context_with_model(model.c, params.c),
		numThreads: int(params.c.n_threads),
	}
}

func (m *Model) NumVocab() int {
	return int(C.llama_n_vocab(m.c))
}

func (m *Model) TokenIsEog(token int) bool {
	return bool(C.llama_token_is_eog(m.c, C.llama_token(token)))
}

func (m *Model) AddBOSToken() bool {
	return bool(C.llama_add_bos_token(m.c))
}

func (m *Model) ApplyLoraFromFile(context *Context, loraPath string, scale float32, threads int) error {
	cLoraPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cLoraPath))

	loraAdapter := C.llama_lora_adapter_init(m.c, cLoraPath)

	err := -1
	if loraAdapter != nil {
		err = int(C.llama_lora_adapter_set(context.c, loraAdapter, C.float(scale)))
	}
	if err != 0 {
		return errors.New("error applying lora from file")
	}

	return nil
}

type Batch struct {
	c         C.struct_llama_batch
	batchSize int
	embedSize int
}

// Creates a new batch for either word tokens if embed is 0 or
// image embeddings if embed is specified. Batches cannot contain
// both types at the same time
func NewBatch(nTokens int, embed int, maxSeq int) *Batch {
	return &Batch{
		c:         C.llama_batch_init(C.int(nTokens), C.int(embed), C.int(maxSeq)),
		batchSize: nTokens,
		embedSize: embed,
	}
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
func (b *Batch) Add(token int, embed []float32, pos int, seqIds []int, logits bool) {
	if !b.IsEmbedding() {
		unsafe.Slice(b.c.token, b.batchSize)[b.c.n_tokens] = C.llama_token(token)
	} else {
		copy(unsafe.Slice((*float32)(b.c.embd), b.batchSize*b.embedSize)[int(b.c.n_tokens)*b.embedSize:], embed)
	}
	unsafe.Slice(b.c.pos, b.batchSize)[b.c.n_tokens] = C.llama_pos(pos)
	unsafe.Slice(b.c.n_seq_id, b.batchSize)[b.c.n_tokens] = C.int(len(seqIds))

	for i, s := range seqIds {
		unsafe.Slice((unsafe.Slice(b.c.seq_id, b.batchSize)[b.c.n_tokens]), C.int(len(seqIds)))[i] = C.int32_t(s)
	}

	if logits {
		unsafe.Slice(b.c.logits, b.batchSize)[b.c.n_tokens] = 1
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
		m.c,
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
			m.c,
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
		m.c,
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
			m.c,
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
	return int(C.llama_n_embd(m.c))
}

func Quantize(infile, outfile string, ftype uint32) error {
	cinfile := C.CString(infile)
	defer C.free(unsafe.Pointer(cinfile))

	coutfile := C.CString(outfile)
	defer C.free(unsafe.Pointer(coutfile))

	params := C.llama_model_quantize_default_params()
	params.nthread = -1
	params.ftype = ftype

	if rc := C.llama_model_quantize(cinfile, coutfile, &params); rc != 0 {
		return fmt.Errorf("llama_model_quantize: %d", rc)
	}

	return nil
}

// llava
type ClipContext struct {
	c *C.struct_clip_ctx
}

func NewClipContext(modelPath string) *ClipContext {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))
	cc := C.clip_model_load(mp, 1)
	return &ClipContext{c: cc}
}

func (c *ClipContext) Free() {
	C.clip_free(c.c)
}

func NewLlavaImageEmbed(llamaContext *Context, clipContext *ClipContext, data []byte) [][]float32 {
	c := C.llava_image_embed_make_with_bytes(clipContext.c, C.int(llamaContext.numThreads), (*C.uchar)(unsafe.Pointer(&data[0])), C.int(len(data)))

	numTokens := int(c.n_image_pos)
	numEmbed := llamaContext.Model().NEmbd()

	s := unsafe.Slice((*float32)(c.embed), numEmbed*numTokens)

	embed := make([][]float32, numTokens)
	rows := make([]float32, len(s))
	copy(rows, s)

	for i := range embed {
		embed[i] = rows[i*numEmbed : (i+1)*numEmbed]
	}

	C.llava_image_embed_free(c)

	return embed
}

// sampling
// TODO: this is a temporary wrapper to allow calling C++ code from CGo
type SamplingContext struct {
	c *C.struct_gpt_sampler
}

type SamplingParams struct {
	TopK           int
	TopP           float32
	MinP           float32
	TfsZ           float32
	TypicalP       float32
	Temp           float32
	RepeatLastN    int
	PenaltyRepeat  float32
	PenaltyFreq    float32
	PenaltyPresent float32
	Mirostat       int
	MirostatTau    float32
	MirostatEta    float32
	PenalizeNl     bool
	Seed           uint32
	Grammar        string
}

func NewSamplingContext(model *Model, params SamplingParams) *SamplingContext {
	var cparams C.struct_gpt_sampler_cparams
	cparams.top_k = C.int32_t(params.TopK)
	cparams.top_p = C.float(params.TopP)
	cparams.min_p = C.float(params.MinP)
	cparams.tfs_z = C.float(params.TfsZ)
	cparams.typical_p = C.float(params.TypicalP)
	cparams.temp = C.float(params.Temp)
	cparams.penalty_last_n = C.int32_t(params.RepeatLastN)
	cparams.penalty_repeat = C.float(params.PenaltyRepeat)
	cparams.penalty_freq = C.float(params.PenaltyFreq)
	cparams.penalty_present = C.float(params.PenaltyFreq)
	cparams.mirostat = C.int32_t(params.Mirostat)
	cparams.mirostat_tau = C.float(params.MirostatTau)
	cparams.mirostat_eta = C.float(params.MirostatEta)
	cparams.penalize_nl = C.bool(params.PenalizeNl)
	cparams.seed = C.uint32_t(params.Seed)

	grammar := C.CString(params.Grammar)
	defer C.free(unsafe.Pointer(grammar))

	cparams.grammar = grammar
	context := &SamplingContext{c: C.gpt_sampler_cinit(model.c, &cparams)}
	runtime.SetFinalizer(context, func(s *SamplingContext) { C.gpt_sampler_cfree(s.c) })

	return context
}

func (s *SamplingContext) Reset() {
	C.gpt_sampler_creset(s.c)
}

func (s *SamplingContext) Sample(llamaContext *Context, idx int) int {
	return int(C.gpt_sampler_csample(s.c, llamaContext.c, C.int(idx)))
}

func (s *SamplingContext) Accept(id int, applyGrammar bool) {
	C.gpt_sampler_caccept(s.c, C.llama_token(id), C.bool(applyGrammar))
}
