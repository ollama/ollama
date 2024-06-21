package llama

/*
#cgo CFLAGS: -std=c11 -DNDEBUG -DLOG_DISABLE_LOGS
#cgo CXXFLAGS: -std=c++11 -DNDEBUG -DLOG_DISABLE_LOGS
#cgo darwin,arm64 CFLAGS: -DGGML_USE_METAL -DGGML_USE_ACCELERATE -DGGML_METAL_EMBED_LIBRARY -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo darwin,arm64 CXXFLAGS: -DGGML_USE_METAL -DGGML_USE_ACCELERATE -DGGML_METAL_EMBED_LIBRARY -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo darwin,arm64 LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
#cgo darwin,amd64 CFLAGS: -Wno-incompatible-pointer-types-discards-qualifiers
#cgo darwin,amd64 CXXFLAGS: -Wno-incompatible-pointer-types-discards-qualifiers
#cgo darwin,amd64 LDFLAGS: -framework Foundation
#cgo darwin,amd64,avx2 CFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo darwin,amd64,avx2 CXXFLAGS: -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
#cgo darwin,amd64,avx2 LDFLAGS: -framework Accelerate
#cgo linux CFLAGS: -D_GNU_SOURCE
#cgo linux CXXFLAGS: -D_GNU_SOURCE
#cgo windows CFLAGS: -Wno-discarded-qualifiers
#cgo windows LDFLAGS: -lmsvcrt
#cgo avx CFLAGS: -mavx
#cgo avx CXXFLAGS: -mavx
#cgo avx2 CFLAGS: -mavx2 -mfma
#cgo avx2 CXXFLAGS: -mavx2 -mfma
#cgo cuda CFLAGS: -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo cuda CXXFLAGS: -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm CFLAGS: -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm CXXFLAGS: -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm LDFLAGS: -L${SRCDIR} -lggml_hipblas -lhipblas -lamdhip64 -lrocblas
#cgo windows,cuda LDFLAGS: -L${SRCDIR} -lggml_cuda -lcuda -lcudart -lcublas -lcublasLt
#cgo windows,rocm LDFLAGS: -L${SRCDIR} -lggml_hipblas -lhipblas -lamdhip64 -lrocblas
#cgo linux,cuda LDFLAGS: -L${SRCDIR} -L/usr/local/cuda/lib64 -lggml_cuda -lcuda -lcudart -lcublas -lcublasLt -lpthread -ldl -lrt
#cgo linux,rocm LDFLAGS: -L/opt/rocm/lib

#include <stdlib.h>
#include "llama.h"
#include "clip.h"
#include "llava.h"
#include "sampling_ext.h"

bool llamaProgressCallback(float progress, void *user_data);
extern const char *ggml_metallib_start;
extern const char *ggml_metallib_end;
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

//go:embed ggml-common.h
var ggmlCommon string

//go:embed ggml-metal.metal
var ggmlMetal string

func init() {
	metal := strings.ReplaceAll(ggmlMetal, `#include "ggml-common.h"`, ggmlCommon)
	cMetal := C.CString(metal)
	C.ggml_metallib_start = cMetal
	C.ggml_metallib_end = (*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetal)) + uintptr(len(metal))))
}

func BackendInit() {
	C.llama_backend_init()
}

func PrintSystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

type ContextParams struct {
	c C.struct_llama_context_params
}

func NewContextParams(numCtx int, threads int, flashAttention bool) ContextParams {
	params := C.llama_context_default_params()
	params.n_ctx = C.uint(numCtx)
	params.n_threads = C.uint(runtime.NumCPU())
	params.n_threads_batch = params.n_threads
	params.embeddings = C.bool(true)
	params.flash_attn = C.bool(flashAttention)
	params.n_threads = C.uint(threads)
	return ContextParams{c: params}
}

type ModelParams struct {
	c C.struct_llama_model_params
}

//export llamaProgressCallback
func llamaProgressCallback(progress C.float, userData unsafe.Pointer) C.bool {
	handle := cgo.Handle(userData)
	callback := handle.Value().(func(float32))
	callback(float32(progress))
	return true
}

func NewModelParams(numGpuLayers int, mainGpu int, callback func(float32)) ModelParams {
	params := C.llama_model_default_params()
	params.n_gpu_layers = C.int(numGpuLayers)
	params.main_gpu = C.int32_t(mainGpu)

	handle := cgo.NewHandle(callback)
	params.progress_callback = C.llama_progress_callback(C.llamaProgressCallback)
	params.progress_callback_user_data = unsafe.Pointer(handle)
	runtime.SetFinalizer(&params, func(p *C.struct_llama_model_params) {
		handle.Delete()
	})

	return ModelParams{c: params}
}

type Context struct {
	c *C.struct_llama_context
}

func (c *Context) KvCacheClear() {
	C.llama_kv_cache_clear(c.c)
}

func (c *Context) Decode(batch Batch) error {
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

func (c *Context) SampleTokenGreedy(logits []float32) int {
	candidates := (*C.struct_llama_token_data)(C.malloc(C.size_t(len(logits)) * C.size_t(unsafe.Sizeof(C.struct_llama_token_data{}))))
	defer C.free(unsafe.Pointer(candidates))

	for i, logit := range logits {
		ptr := (*C.struct_llama_token_data)(unsafe.Pointer(uintptr(unsafe.Pointer(candidates)) + uintptr(i)*unsafe.Sizeof(C.struct_llama_token_data{})))
		ptr.id = C.int(i)
		ptr.logit = C.float(logit)
		ptr.p = 0.0
	}

	return int(C.llama_sample_token_greedy(c.c, &C.llama_token_data_array{
		data:   candidates,
		size:   C.size_t(len(logits)),
		sorted: C.bool(false),
	}))
}

func (c *Context) KvCacheSeqRm(seqId int, p0 int, p1 int) bool {
	return bool(C.llama_kv_cache_seq_rm(c.c, C.int(seqId), C.int(p0), C.int(p1)))
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

func LoadModelFromFile(modelPath string, params ModelParams) *Model {
	return &Model{c: C.llama_load_model_from_file(C.CString(modelPath), params.c)}
}

func NewContextWithModel(model *Model, params ContextParams) *Context {
	return &Context{c: C.llama_new_context_with_model(model.c, params.c)}
}

func (m *Model) NumVocab() int {
	return int(C.llama_n_vocab(m.c))
}

func (m *Model) TokenIsEog(token int) bool {
	return bool(C.llama_token_is_eog(m.c, C.llama_token(token)))
}

func (m *Model) ApplyLoraFromFile(loraPath string, scale float32, baseModelPath string, threads int) error {
	cLoraPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cLoraPath))

	var cBaseModelPath *C.char
	if baseModelPath != "" {
		cBaseModelPath = C.CString(baseModelPath)
	}

	code := int(C.llama_model_apply_lora_from_file(m.c, cLoraPath, C.float(scale), cBaseModelPath, C.int32_t(threads)))
	if code != 0 {
		return errors.New("error applying lora from file")
	}

	return nil
}

type Batch struct {
	c C.struct_llama_batch
}

func NewBatch(nTokens int, embd int, maxSeq int) Batch {
	return Batch{c: C.llama_batch_init(C.int(nTokens), C.int(embd), C.int(maxSeq))}
}

func (b *Batch) NumTokens() int {
	return int(b.c.n_tokens)
}

// Add adds a token to the batch with the given position for the given
// sequence ids, and optionally instructs to include logits.
func (b *Batch) Add(token int, pos int, seqIds []int, logits bool) {
	unsafe.Slice(b.c.token, 512)[b.c.n_tokens] = C.llama_token(token)
	unsafe.Slice(b.c.pos, 512)[b.c.n_tokens] = C.llama_pos(pos)
	unsafe.Slice(b.c.n_seq_id, 512)[b.c.n_tokens] = C.int(len(seqIds))

	for i, s := range seqIds {
		unsafe.Slice((unsafe.Slice(b.c.seq_id, 512)[b.c.n_tokens]), C.int(len(seqIds)))[i] = C.int32_t(s)
	}

	if logits {
		unsafe.Slice(b.c.logits, 512)[b.c.n_tokens] = 1
	}

	b.c.n_tokens += 1
}

func (b *Batch) Clear() {
	b.c.n_tokens = 0
}

func (b *Batch) Free() {
	C.llama_batch_free(b.c)
}

func BatchGetOne(tokens []int, pos0 int, seqId int) Batch {
	return Batch{c: C.llama_batch_get_one((*C.int)(unsafe.Pointer(&tokens[0])), C.int32_t(len(tokens)), C.int(pos0), C.int(seqId))}
}

type Model struct {
	c *C.struct_llama_model
}

func (m *Model) TokenToPiece(token int) string {
	buf := make([]byte, 12)
	C.llama_token_to_piece(
		m.c,
		C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(12),
		C.bool(true),
	)
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

	if result < 0 {
		return nil, fmt.Errorf("tokenization failed, required %d tokens", -result)
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

type LlavaContext struct {
	c *C.struct_llava_context
}

type LlavaImageEmbed struct {
	c *C.struct_llava_image_embed
}

func NewLlavaImageEmbed(clipContext *ClipContext, data []byte) *LlavaImageEmbed {
	return &LlavaImageEmbed{c: C.llava_image_embed_make_with_bytes(clipContext.c, C.int(runtime.NumCPU()), (*C.uchar)(unsafe.Pointer(&data[0])), C.int(len(data)))}
}

func LlavaEvalImageEmbed(llamaContext *Context, embed *LlavaImageEmbed, nBatch int, nPast *int) {
	C.llava_eval_image_embed(llamaContext.c, embed.c, C.int(nBatch), (*C.int)(unsafe.Pointer(nPast)))
}

// sampling
// TODO: this is a temporary wrapper to allow calling C++ code from CGo
type SamplingContext struct {
	c *C.struct_llama_sampling_context
}

type SamplingParams struct {
	TopK           int
	TopP           float32
	TfsZ           float32
	TypicalP       float32
	Temp           float32
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

func NewSamplingContext(params SamplingParams) *SamplingContext {
	var cparams C.struct_llama_sampling_cparams
	cparams.top_k = C.int32_t(params.TopK)
	cparams.top_p = C.float(params.TopP)
	cparams.tfs_z = C.float(params.TfsZ)
	cparams.typical_p = C.float(params.TypicalP)
	cparams.temp = C.float(params.Temp)
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
	return &SamplingContext{c: C.llama_sampling_cinit(&cparams)}
}

func (s *SamplingContext) Free() {
	C.llama_sampling_cfree(s.c)
}

func (s *SamplingContext) Reset() {
	C.llama_sampling_creset(s.c)
}

func (s *SamplingContext) Sample(ctxMain *Context, ctxConfig *Context, idx int) int {
	// TODO (jmorganca): handle nil for all args
	if ctxConfig == nil {
		return int(C.llama_sampling_csample(s.c, ctxMain.c, nil, C.int(idx)))
	}

	return int(C.llama_sampling_csample(s.c, ctxMain.c, ctxConfig.c, C.int(idx)))

}

func (s *SamplingContext) Accept(ctxMain *Context, id int, applyGrammar bool) {
	C.llama_sampling_caccept(s.c, ctxMain.c, C.llama_token(id), C.bool(applyGrammar))
}
