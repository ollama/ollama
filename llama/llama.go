package llama

// #cgo darwin,arm64 CFLAGS: -std=c11 -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 CXXFLAGS: -std=c++11 -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64
// #cgo darwin,arm64 LDFLAGS: -ld_classic ${SRCDIR}/ggml-metal.o -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
// #cgo darwin,amd64 CFLAGS: -Wno-incompatible-pointer-types-discards-qualifiers
// #cgo darwin,amd64 CXXFLAGS: -std=c++11 -Wno-incompatible-pointer-types-discards-qualifiers
// #cgo darwin,amd64 LDFLAGS: -ld_classic -framework Foundation -framework Accelerate
// #cgo windows LDFLAGS: -lmsvcrt
// #cgo avx CFLAGS: -mavx
// #cgo avx CXXFLAGS: -mavx
// #cgo avx2 CFLAGS: -mavx -mavx2 -mfma
// #cgo avx2 CXXFLAGS: -mavx -mavx2 -mfma
// #cgo cuda CFLAGS: -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_MULTIPLATFORM -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
// #cgo cuda CXXFLAGS: -std=c++11 -DGGML_USE_CUDA -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_MULTIPLATFORM -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
// #cgo rocm CXXFLAGS: -std=c++11 -DGGML_USE_CUDA -DGGML_USE_HIPBLAS -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_MULTIPLATFORM -DGGML_CUDA_MMV_Y=1 -DGGML_BUILD=1
// #cgo windows,cuda LDFLAGS: -L. -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64" -lggml-cuda -lcuda -lcudart -lcublas -lcublasLt
// #cgo windows,rocm LDFLAGS: -L. -L"C:/Program Files/AMD/ROCm/5.7/lib" -lggml-hipblas -lhipblas -lamdhip64 -lrocblas
// #include <stdlib.h>
// #include "llama.h"
import "C"
import (
	"fmt"
	"runtime"
	"strings"
	"unsafe"

	"github.com/ollama/ollama/llm"
)

type Token int32
type Pos int32
type SeqId int32

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func PrintSystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func BackendInit() {
	C.llama_backend_init()
}

type ContextParams struct {
	c C.struct_llama_context_params
}

func NewContextParams() ContextParams {
	params := C.llama_context_default_params()
	params.seed = C.uint(1234)
	params.n_ctx = C.uint(2048)
	params.n_threads = C.uint(runtime.NumCPU())
	params.n_threads_batch = params.n_threads
	return ContextParams{c: params}
}

type ModelParams struct {
	c C.struct_llama_model_params
}

func NewModelParams() ModelParams {
	params := C.llama_model_default_params()
	params.n_gpu_layers = 999
	return ModelParams{c: params}
}

type Context struct {
	c *C.struct_llama_context
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
		return fmt.Errorf("could not find a KV slot for the batch - try reducing the size of the batch or increase the context. code: %d\n", code)
	}

	return nil
}

func (c *Context) getModel() *Model {
	return &Model{c: C.llama_get_model(c.c)}
}

func (c *Context) SampleTokenGreedy(batch Batch) Token {
	nv := c.getModel().NumVocab()

	// TODO(jmorganca): split this up into different functions
	candidates := (*C.struct_llama_token_data)(C.malloc(C.size_t(nv) * C.size_t(unsafe.Sizeof(C.struct_llama_token_data{}))))
	defer C.free(unsafe.Pointer(candidates))

	// get most recent logits
	logits := C.llama_get_logits_ith(c.c, C.int(batch.NumTokens()-1))
	for i := 0; i < int(nv); i++ {
		ptr := (*C.struct_llama_token_data)(unsafe.Pointer(uintptr(unsafe.Pointer(candidates)) + uintptr(i)*unsafe.Sizeof(C.struct_llama_token_data{})))
		ptr.id = C.int(i)
		ptr.logit = unsafe.Slice(logits, nv)[i]
		ptr.p = 0.0
	}

	return Token(C.llama_sample_token_greedy(c.c, &C.llama_token_data_array{
		data:   candidates,
		size:   C.size_t(nv),
		sorted: C.bool(false),
	}))
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

func (m *Model) TokenIsEog(token Token) bool {
	return bool(C.llama_token_is_eog(m.c, C.llama_token(token)))
}

type Batch struct {
	c C.struct_llama_batch
}

func NewBatch(nTokens int, nSeqs int, nCtx int) Batch {
	return Batch{c: C.llama_batch_init(C.int(nTokens), C.int(nSeqs), C.int(nCtx))}
}

func (b *Batch) NumTokens() int {
	return int(b.c.n_tokens)
}

func (b *Batch) Add(token Token, pos Pos, seqIds []SeqId, logits bool) {
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

type Model struct {
	c *C.struct_llama_model
}

func (m *Model) TokenToPiece(token Token) string {
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

func (m *Model) Tokenize(text string, maxTokens int, addSpecial bool, parseSpecial bool) ([]Token, error) {
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

	tokens := make([]Token, result)
	for i := 0; i < int(result); i++ {
		tokens[i] = Token(cTokens[i])
	}

	return tokens, nil
}

func Quantize(infile, outfile string, ftype llm.FileType) error {
	cinfile := C.CString(infile)
	defer C.free(unsafe.Pointer(cinfile))

	coutfile := C.CString(outfile)
	defer C.free(unsafe.Pointer(coutfile))

	params := C.llama_model_quantize_default_params()
	params.nthread = -1
	params.ftype = ftype.Value()

	if rc := C.llama_model_quantize(cinfile, coutfile, &params); rc != 0 {
		return fmt.Errorf("llama_model_quantize: %d", rc)
	}

	return nil
}
