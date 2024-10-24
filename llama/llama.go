package llama

//go:generate make -j 8

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
#cgo cuda_jetpack5 LDFLAGS: -lggml_cuda_jetpack5 -L/usr/local/cuda-11/lib64
#cgo cuda_jetpack6 LDFLAGS: -lggml_cuda_jetpack6 -L/usr/local/cuda-12/lib64
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
#cgo linux,arm64 CFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
#cgo linux,arm64 CXXFLAGS: -D__aarch64__ -D__ARM_NEON -D__ARM_FEATURE_FMA
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/build/Linux/arm64
#cgo linux,arm64,sve CFLAGS: -march=armv8.6-a+sve
#cgo linux,arm64,sve CXXFLAGS: -march=armv8.6-a+sve
#cgo linux,cuda LDFLAGS: -lcuda -lcudart -lcublas -lcublasLt -lpthread -ldl -lrt -lresolv
#cgo linux,rocm LDFLAGS: -L/opt/rocm/lib -lpthread -ldl -lrt -lresolv
#cgo rocm CFLAGS: -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm CXXFLAGS: -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
#cgo rocm LDFLAGS: -L${SRCDIR} -lggml_rocm -lhipblas -lamdhip64 -lrocblas
#cgo windows CFLAGS: -Wno-discarded-qualifiers -D_WIN32_WINNT=0x602
#cgo windows CXXFLAGS: -D_WIN32_WINNT=0x602
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
#include "ggml.h"
#include "llava.h"
#include "mllama.h"

bool llamaProgressCallback(float progress, void *user_data);

typedef enum {COMP_UNKNOWN,COMP_GCC,COMP_CLANG} COMPILER;
COMPILER inline get_compiler() {
#if defined(__clang__)
	return COMP_CLANG;
#elif defined(__GNUC__)
	return COMP_GCC;
#else
	return UNKNOWN_COMPILER;
#endif
}
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"runtime/cgo"
	"slices"
	"strings"
	"unsafe"
)

var CpuFeatures = ""

func BackendInit() {
	C.llama_backend_init()
}

func PrintSystemInfo() string {
	var compiler string
	switch C.get_compiler() {
	case C.COMP_UNKNOWN:
		compiler = "cgo(unknown_compiler)"
	case C.COMP_GCC:
		compiler = "cgo(gcc)"
	case C.COMP_CLANG:
		compiler = "cgo(clang)"
	}
	return C.GoString(C.llama_print_system_info()) + compiler
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

func (c *Context) GetLogitsIth(i int) ([]float32, error) {
	logits := (*float32)(unsafe.Pointer(C.llama_get_logits_ith(c.c, C.int(i))))
	if logits == nil {
		return nil, errors.New("unable to get logits")
	}

	return unsafe.Slice(logits, c.Model().NumVocab()), nil
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

func (c *Context) KvCacheClear() {
	C.llama_kv_cache_clear(c.c)
}

func (c *Context) KvCacheDefrag() {
	C.llama_kv_cache_defrag(c.c)
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
	embeddings := unsafe.Pointer(C.llama_get_embeddings_ith(c.c, C.int32_t(i)))
	if embeddings == nil {
		return nil
	}

	return unsafe.Slice((*float32)(embeddings), c.Model().NEmbd())
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

func LoadModelFromFile(modelPath string, params ModelParams) (*Model, error) {
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

	m := Model{c: C.llama_load_model_from_file(C.CString(modelPath), cparams)}
	if m.c == nil {
		return nil, fmt.Errorf("unable to load model: %s", modelPath)
	}

	return &m, nil
}

func FreeModel(model *Model) {
	C.llama_free_model(model.c)
}

func NewContextWithModel(model *Model, params ContextParams) (*Context, error) {
	c := Context{
		c:          C.llama_new_context_with_model(model.c, params.c),
		numThreads: int(params.c.n_threads),
	}
	if c.c == nil {
		return nil, errors.New("unable to create llama context")
	}

	return &c, nil
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
	if loraAdapter == nil {
		return errors.New("unable to load lora")
	}

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

// vision processing
type ClipContext struct {
	c *C.struct_clip_ctx
}

func NewClipContext(llamaContext *Context, modelPath string) (*ClipContext, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))
	c := C.clip_model_load(mp, 1)
	if c == nil {
		return nil, fmt.Errorf("unable to load clip model: %v", modelPath)
	}

	projEmbedSize := int(C.clip_n_mmproj_embd(c))
	modelEmbedSize := llamaContext.Model().NEmbd()
	if projEmbedSize != modelEmbedSize {
		return nil, fmt.Errorf("projector embedding size (%d) does not match model (%d)", projEmbedSize, modelEmbedSize)
	}

	return &ClipContext{c: c}, nil
}

func (c *ClipContext) Free() {
	C.clip_free(c.c)
}

func (c *ClipContext) NewEmbed(llamaContext *Context, data []byte) ([][]float32, error) {
	l := C.llava_image_embed_make_with_bytes(c.c, C.int(llamaContext.numThreads), (*C.uchar)(unsafe.Pointer(&data[0])), C.int(len(data)))
	if l == nil {
		return nil, errors.New("unable to make llava embedding from image")
	}

	numTokens := int(l.n_image_pos)
	numEmbed := llamaContext.Model().NEmbd()

	s := unsafe.Slice((*float32)(l.embed), numEmbed*numTokens)

	embed := make([][]float32, numTokens)
	rows := make([]float32, len(s))
	copy(rows, s)

	for i := range embed {
		embed[i] = rows[i*numEmbed : (i+1)*numEmbed]
	}

	C.llava_image_embed_free(l)

	return embed, nil
}

type MllamaContext struct {
	c *C.struct_mllama_ctx
}

func NewMllamaContext(llamaContext *Context, modelPath string) (*MllamaContext, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))
	c := C.mllama_model_load(mp, 1)
	if c == nil {
		return nil, fmt.Errorf("unable to load mllama model: %v", modelPath)
	}

	projEmbedSize := int(C.mllama_n_embd(c))
	modelEmbedSize := llamaContext.Model().NEmbd()
	if projEmbedSize != modelEmbedSize {
		return nil, fmt.Errorf("projector embedding size (%d) does not match model (%d)", projEmbedSize, modelEmbedSize)
	}

	return &MllamaContext{c: c}, nil
}

func (m *MllamaContext) Free() {
	C.mllama_free(m.c)
}

func (m *MllamaContext) NewEmbed(llamaContext *Context, data []byte, aspectRatioId int) ([][]float32, error) {
	img := C.mllama_image_init()
	defer C.mllama_image_free(img)

	ok := bool(C.mllama_image_load_from_data(unsafe.Pointer(&data[0]), C.int(len(data)), 560, 560, 3, 4, C.int(aspectRatioId), img))
	if !ok {
		return nil, errors.New("unable to load mllama image data")
	}

	rows := make([]float32, m.EmbedSize(llamaContext))
	ok = bool(C.mllama_image_encode(m.c, C.int(llamaContext.numThreads), img, (*C.float)(unsafe.Pointer(&rows[0]))))
	if !ok {
		return nil, errors.New("unable to make mllama embedding from image")
	}

	embed := make([][]float32, 1)
	embed[0] = rows

	return embed, nil
}

func (m *MllamaContext) EmbedSize(llamaContext *Context) int {
	numTokens := int(C.mllama_n_positions(m.c) * C.mllama_n_tiles(m.c))
	numEmbed := llamaContext.Model().NEmbd()

	return numTokens * numEmbed
}

func (c *Context) SetCrossAttention(state bool) {
	C.llama_set_cross_attention(c.c, C.bool(state))
}

func (c *Context) Synchronize() {
	C.llama_synchronize(c.c)
}

// sampling
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

type SamplingContext struct {
	chain   *C.struct_llama_sampler
	grammar *C.struct_llama_sampler
}

func NewSamplingContext(model *Model, params SamplingParams) (*SamplingContext, error) {
	var s SamplingContext
	runtime.SetFinalizer(&s, func(s *SamplingContext) { s.free() })

	sparams := C.llama_sampler_chain_default_params()
	s.chain = C.llama_sampler_chain_init(sparams)

	grammar := C.CString(params.Grammar)
	defer C.free(unsafe.Pointer(grammar))
	root := C.CString("root")
	defer C.free(unsafe.Pointer(root))
	s.grammar = C.llama_sampler_init_grammar(model.c, grammar, root)

	C.llama_sampler_chain_add(s.chain,
		C.llama_sampler_init_penalties(
			C.llama_n_vocab(model.c),
			C.llama_token_eos(model.c),
			C.llama_token_nl(model.c),
			C.int32_t(params.RepeatLastN),
			C.float(params.PenaltyRepeat),
			C.float(params.PenaltyFreq),
			C.float(params.PenaltyPresent),
			C.bool(params.PenalizeNl),
			false))

	if params.Temp > 0 {
		switch params.Mirostat {
		case 0:
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_top_k(C.int32_t(params.TopK)))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_tail_free(C.float(params.TfsZ), 0))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_typical(C.float(params.TypicalP), 0))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_top_p(C.float(params.TopP), 0))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_min_p(C.float(params.MinP), 0))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_temp(C.float(params.Temp)))

			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_softmax())
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_dist(C.uint32_t(params.Seed)))
		case 1:
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_temp(C.float(params.Temp)))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_mirostat(C.llama_n_vocab(model.c),
				C.uint32_t(params.Seed), C.float(params.MirostatTau), C.float(params.MirostatEta), 100))
		case 2:
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_temp(C.float(params.Temp)))
			C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_mirostat_v2(C.uint32_t(params.Seed),
				C.float(params.MirostatTau), C.float(params.MirostatEta)))
		default:
			return nil, fmt.Errorf("sampling: unknown mirostat version: %v", params.Mirostat)
		}
	} else {
		C.llama_sampler_chain_add(s.chain, C.llama_sampler_init_greedy())
	}

	return &s, nil
}

func (s *SamplingContext) Sample(llamaContext *Context, idx int) (int, error) {
	logits, err := llamaContext.GetLogitsIth(idx)
	if err != nil {
		return 0, err
	}

	numVocab := llamaContext.Model().NumVocab()

	tokenData := make([]C.llama_token_data, numVocab)
	var tokenDataPin runtime.Pinner
	tokenDataPin.Pin(&tokenData[0])
	defer tokenDataPin.Unpin()

	for i := range tokenData {
		tokenData[i] = C.llama_token_data{id: C.llama_token(i), logit: C.float(logits[i])}
	}
	tokenDataArray := C.llama_token_data_array{data: &tokenData[0], size: C.size_t(len(tokenData)), selected: -1}

	C.llama_sampler_apply(s.chain, &tokenDataArray)

	id := tokenData[tokenDataArray.selected].id

	// Check if the selected token is allowed by the grammar
	// If it is allowed then return it, otherwise evaluate the grammar on all
	// tokens and resample (slow)
	tokenData[0] = C.llama_token_data{id: id, logit: 1}
	tokenDataArray = C.llama_token_data_array{data: &tokenData[0], size: 1, selected: -1}

	C.llama_sampler_apply(s.grammar, &tokenDataArray)
	if !math.IsInf(float64(tokenData[0].logit), -1) {
		return int(id), nil
	}

	for i := range tokenData {
		tokenData[i] = C.llama_token_data{id: C.llama_token(i), logit: C.float(logits[i])}
	}
	tokenDataArray = C.llama_token_data_array{data: &tokenData[0], size: C.size_t(len(tokenData)), selected: -1}

	C.llama_sampler_apply(s.grammar, &tokenDataArray)
	C.llama_sampler_apply(s.chain, &tokenDataArray)

	return int(tokenData[tokenDataArray.selected].id), nil
}

func (s *SamplingContext) Accept(id int, applyGrammar bool) {
	if applyGrammar {
		C.llama_sampler_accept(s.grammar, C.llama_token(id))
	}
	C.llama_sampler_accept(s.chain, C.llama_token(id))
}

func (s *SamplingContext) free() {
	if s != nil {
		C.llama_sampler_free(s.grammar)
		C.llama_sampler_free(s.chain)
	}
}
