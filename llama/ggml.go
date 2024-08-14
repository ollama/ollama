package llama

/*
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// This file largely follows the order in ggml.h to aid in updates
// TODO - consider breaking into a few files by API type?

type GGMLInitParams struct {
	c C.struct_ggml_init_params
}

// enum ggml_type {
type GGMLType uint

const (
	GGML_TYPE_F32  GGMLType = 0 + iota
	GGML_TYPE_F16           = 1
	GGML_TYPE_Q4_0          = 2
	GGML_TYPE_Q4_1          = 3
	// GGML_TYPE_Q4_2 = 4, support has been removed
	// GGML_TYPE_Q4_3 = 5, support has been removed
	GGML_TYPE_Q5_0     = 6
	GGML_TYPE_Q5_1     = 7
	GGML_TYPE_Q8_0     = 8
	GGML_TYPE_Q8_1     = 9
	GGML_TYPE_Q2_K     = 10
	GGML_TYPE_Q3_K     = 11
	GGML_TYPE_Q4_K     = 12
	GGML_TYPE_Q5_K     = 13
	GGML_TYPE_Q6_K     = 14
	GGML_TYPE_Q8_K     = 15
	GGML_TYPE_IQ2_XXS  = 16
	GGML_TYPE_IQ2_XS   = 17
	GGML_TYPE_IQ3_XXS  = 18
	GGML_TYPE_IQ1_S    = 19
	GGML_TYPE_IQ4_NL   = 20
	GGML_TYPE_IQ3_S    = 21
	GGML_TYPE_IQ2_S    = 22
	GGML_TYPE_IQ4_XS   = 23
	GGML_TYPE_I8       = 24
	GGML_TYPE_I16      = 25
	GGML_TYPE_I32      = 26
	GGML_TYPE_I64      = 27
	GGML_TYPE_F64      = 28
	GGML_TYPE_IQ1_M    = 29
	GGML_TYPE_BF16     = 30
	GGML_TYPE_Q4_0_4_4 = 31
	GGML_TYPE_Q4_0_4_8 = 32
	GGML_TYPE_Q4_0_8_8 = 33
	GGML_TYPE_COUNT    = iota
)

// precision
// enum ggml_prec {
type GGMLPrec uint

const (
	GGML_PREC_DEFAULT GGMLPrec = 0 + iota
	GGML_PREC_F32
)

// numa strategies
// enum ggml_numa_strategy {
type GGMLNumaStrategy uint

const (
	GGML_NUMA_STRATEGY_DISABLED GGMLNumaStrategy = 0 + iota
	GGML_NUMA_STRATEGY_DISTRIBUTE
	GGML_NUMA_STRATEGY_ISOLATE
	GGML_NUMA_STRATEGY_NUMACTL
	GGML_NUMA_STRATEGY_MIRROR
	GGML_NUMA_STRATEGY_COUNT
)

// available tensor operations:
// enum ggml_op {
type GGMLOp uint

const (
	GGML_OP_NONE GGMLOp = 0 + iota
	GGML_OP_DUP
	GGML_OP_ADD
	GGML_OP_ADD1
	GGML_OP_ACC
	GGML_OP_SUB
	GGML_OP_MUL
	GGML_OP_DIV
	GGML_OP_SQR
	GGML_OP_SQRT
	GGML_OP_LOG
	GGML_OP_SIN
	GGML_OP_COS
	GGML_OP_SUM
	GGML_OP_SUM_ROWS
	GGML_OP_MEAN
	GGML_OP_ARGMAX
	GGML_OP_REPEAT
	GGML_OP_REPEAT_BACK
	GGML_OP_CONCAT
	GGML_OP_SILU_BACK
	GGML_OP_NORM // normalize
	GGML_OP_RMS_NORM
	GGML_OP_RMS_NORM_BACK
	GGML_OP_GROUP_NORM
	GGML_OP_MUL_MAT
	GGML_OP_MUL_MAT_ID
	GGML_OP_OUT_PROD
	GGML_OP_SCALE
	GGML_OP_SET
	GGML_OP_CPY
	GGML_OP_CONT
	GGML_OP_RESHAPE
	GGML_OP_VIEW
	GGML_OP_PERMUTE
	GGML_OP_TRANSPOSE
	GGML_OP_GET_ROWS
	GGML_OP_GET_ROWS_BACK
	GGML_OP_DIAG
	GGML_OP_DIAG_MASK_INF
	GGML_OP_DIAG_MASK_ZERO
	GGML_OP_SOFT_MAX
	GGML_OP_SOFT_MAX_BACK
	GGML_OP_ROPE
	GGML_OP_ROPE_BACK
	GGML_OP_CLAMP
	GGML_OP_CONV_TRANSPOSE_1D
	GGML_OP_IM2COL
	GGML_OP_IM2COL_BACK
	GGML_OP_CONV_TRANSPOSE_2D
	GGML_OP_POOL_1D
	GGML_OP_POOL_2D
	GGML_OP_POOL_2D_BACK
	GGML_OP_UPSCALE // nearest interpolate
	GGML_OP_PAD
	GGML_OP_ARANGE
	GGML_OP_TIMESTEP_EMBEDDING
	GGML_OP_ARGSORT
	GGML_OP_LEAKY_RELU
	GGML_OP_FLASH_ATTN_EXT
	GGML_OP_FLASH_ATTN_BACK
	GGML_OP_SSM_CONV
	GGML_OP_SSM_SCAN
	GGML_OP_WIN_PART
	GGML_OP_WIN_UNPART
	GGML_OP_GET_REL_POS
	GGML_OP_ADD_REL_POS
	GGML_OP_RWKV_WKV
	GGML_OP_UNARY
	GGML_OP_MAP_UNARY
	GGML_OP_MAP_BINARY
	GGML_OP_MAP_CUSTOM1_F32
	GGML_OP_MAP_CUSTOM2_F32
	GGML_OP_MAP_CUSTOM3_F32
	GGML_OP_MAP_CUSTOM1
	GGML_OP_MAP_CUSTOM2
	GGML_OP_MAP_CUSTOM3
	GGML_OP_CROSS_ENTROPY_LOSS
	GGML_OP_CROSS_ENTROPY_LOSS_BACK
	GGML_OP_COUNT
)

// enum ggml_unary_op {
type GGMLUnaryOp uint

const (
	GGML_UNARY_OP_ABS GGMLUnaryOp = 0 + iota
	GGML_UNARY_OP_SGN
	GGML_UNARY_OP_NEG
	GGML_UNARY_OP_STEP
	GGML_UNARY_OP_TANH
	GGML_UNARY_OP_ELU
	GGML_UNARY_OP_RELU
	GGML_UNARY_OP_SIGMOID
	GGML_UNARY_OP_GELU
	GGML_UNARY_OP_GELU_QUICK
	GGML_UNARY_OP_SILU
	GGML_UNARY_OP_HARDSWISH
	GGML_UNARY_OP_HARDSIGMOID
	GGML_UNARY_OP_EXP
	GGML_UNARY_OP_COUNT
)

// enum ggml_op_pool {
type GGMLOpPool int

const (
	GGML_OP_POOL_MAX GGMLOpPool = 0 + iota
	GGML_OP_POOL_AVG
	GGML_OP_POOL_COUNT
)

// // sort rows
// enum ggml_sort_order {
type GGMLSortOrder int

const (
	GGML_SORT_ORDER_ASC GGMLSortOrder = 0 + iota
	GGML_SORT_ORDER_DESC
)

type GGMLContext struct {
	c *C.struct_ggml_context
}

// Note: GGML APIs do not currently have a mechanism to report errors
// gracefully, they simply log messages, and typically assert to crash the
// process, or in some cases return null pointers. This GetError pattern can be
// used to check for the null pointer case, and could be enhanced by upstreaming
// an "ernno" style error global which could be used to augment the error to the
// caller

func (c GGMLContext) GetError() error {
	if c.c == nil {
		return fmt.Errorf("context error") // TODO ggml errno
	}
	return nil
}

type GGMLScratch struct {
	c C.struct_ggml_scratch
}

type GGMLTensor struct {
	c *C.struct_ggml_tensor
}

func (c GGMLTensor) GetError() error {
	if c.c == nil {
		return fmt.Errorf("tensor error") // TODO ggml errno
	}
	return nil
}

type GGMLCGraph struct {
	c *C.struct_ggml_cgraph
}

func (c GGMLCGraph) GetError() error {
	if c.c == nil {
		return fmt.Errorf("graph error") // TODO ggml errno
	}
	return nil
}

type GGMLThreadPool struct {
	c *C.struct_ggml_threadpool
}

func (c GGMLThreadPool) GetError() error {
	if c.c == nil {
		return fmt.Errorf("threadpool error") // TODO ggml errno
	}
	return nil
}

type GGMLCPlan struct {
	c *C.struct_ggml_cplan
}

func (c GGMLCPlan) GetError() error {
	if c.c == nil {
		return fmt.Errorf("plan error") // TODO ggml errno
	}
	return nil
}

// TODO not sure if this is the right approach...
func LoadData(tensor GGMLTensor, data unsafe.Pointer, nBytes uintptr) {
	C.memcpy(tensor.c.data, data, (C.size_t)(nBytes))
}

// TODO looks unused outside of ggml lib
// bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);

// // misc

// void    ggml_time_init(void); // call this once at the beginning of the program
func GGMLTimeInit() {
	C.ggml_time_init()
}

// int64_t ggml_time_ms(void);
func GGMLTimeMs() uint64 {
	return (uint64)(C.ggml_time_ms())
}

// int64_t ggml_time_us(void);
func GGMLTimeUs() uint64 {
	return (uint64)(C.ggml_time_us())
}

// int64_t ggml_cycles(void);
func GGMLCycles() uint64 {
	return (uint64)(C.ggml_cycles())
}

// int64_t ggml_cycles_per_ms(void);
func GGMLCyclesPerMs() uint64 {
	return (uint64)(C.ggml_cycles_per_ms())
}

// TODO how do we want to tackle loading?
// // accepts a UTF-8 path, even on Windows
// FILE *  ggml_fopen(const char * fname, const char * mode);

// void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
func GGMLNumaInit(numa GGMLNumaStrategy) {
	C.ggml_numa_init((C.enum_ggml_numa_strategy)(numa))
}

// bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node
func GGMLIsNuma() bool {
	return (bool)(C.ggml_is_numa())
}

// TODO looks unused outside of ggml lib
// void    ggml_print_object (const struct ggml_object * obj);
// void    ggml_print_objects(const struct ggml_context * ctx);

// GGML_CALL int64_t ggml_nelements   (const struct ggml_tensor * tensor);
func GGMLNelements(tensor GGMLTensor) uint64 {
	return (uint64)(C.ggml_nelements(tensor.c))
}

// GGML_CALL int64_t ggml_nrows       (const struct ggml_tensor * tensor);
func GGMLNrows(tensor GGMLTensor) uint64 {
	return (uint64)(C.ggml_nrows(tensor.c))
}

// GGML_CALL size_t  ggml_nbytes      (const struct ggml_tensor * tensor);
func GGMLNbytes(tensor GGMLTensor) uintptr {
	return (uintptr)(C.ggml_nbytes(tensor.c))
}

// size_t  ggml_nbytes_pad  (const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
func GGMLNbytesPad(tensor GGMLTensor) uintptr {
	return (uintptr)(C.ggml_nbytes_pad(tensor.c))
}

// GGML_CALL int64_t ggml_blck_size(enum ggml_type type);
func GGMLBlckSize(gType GGMLType) uint64 {
	return (uint64)(C.ggml_blck_size((C.enum_ggml_type)(gType)))
}

// GGML_CALL size_t  ggml_type_size(enum ggml_type type); // size in bytes for all elements in a block
func GGMLTypeSize(gType GGMLType) uint64 {
	return (uint64)(C.ggml_type_size((C.enum_ggml_type)(gType)))
}

// GGML_CALL size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
func GGMLRowSize(gType GGMLType, ne int64) uintptr {
	return (uintptr)(C.ggml_row_size((C.enum_ggml_type)(gType), (C.int64_t)(ne)))
}

// GGML_DEPRECATED(
// double ggml_type_sizef(enum ggml_type type), // ggml_type_size()/ggml_blck_size() as float
// "use ggml_row_size() instead");

// GGML_CALL const char * ggml_type_name(enum ggml_type type);
func GGMLTypeName(gType GGMLType) string {
	return C.GoString(C.ggml_type_name((C.enum_ggml_type)(gType)))
}

// GGML_CALL const char * ggml_op_name  (enum ggml_op   op);
func GGMLOpName(op GGMLOp) string {
	return C.GoString(C.ggml_op_name((C.enum_ggml_op)(op)))
}

// const char * ggml_op_symbol(enum ggml_op   op);
func GGMLOpSymbol(op GGMLOp) string {
	return C.GoString(C.ggml_op_symbol((C.enum_ggml_op)(op)))
}

// const char * ggml_unary_op_name(enum ggml_unary_op op);
func GGMLUnaryOpName(op GGMLUnaryOp) string {
	return C.GoString(C.ggml_unary_op_name((C.enum_ggml_unary_op)(op)))
}

// GGML_CALL const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name
func GGMLOpDesc(t GGMLTensor) string {
	return C.GoString(C.ggml_op_desc(t.c))
}

// GGML_CALL size_t  ggml_element_size(const struct ggml_tensor * tensor);
func GGMLElementSize(tensor GGMLTensor) uintptr {
	return (uintptr)(C.ggml_element_size(tensor.c))
}

// GGML_CALL bool    ggml_is_quantized(enum ggml_type type);
func GGMLIsQuantized(gType GGMLType) bool {
	return (bool)(C.ggml_is_quantized((C.enum_ggml_type)(gType)))
}

// // TODO: temporary until model loading of ggml examples is refactored
// TODO unused outside of ggml lib
// enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);

// GGML_CALL bool ggml_is_transposed(const struct ggml_tensor * tensor);
func GGMLIsTransposed(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_transposed(tensor.c))
}

// GGML_CALL bool ggml_is_permuted  (const struct ggml_tensor * tensor);
func GGMLIsPermuted(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_permuted(tensor.c))
}

// GGML_CALL bool ggml_is_empty (const struct ggml_tensor * tensor);
func GGMLIsEmpty(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_empty(tensor.c))
}

// bool ggml_is_scalar    (const struct ggml_tensor * tensor);
func GGMLIsScalar(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_scalar(tensor.c))
}

// bool ggml_is_vector (const struct ggml_tensor * tensor);
func GGMLIsVector(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_vector(tensor.c))
}

// bool ggml_is_matrix (const struct ggml_tensor * tensor);
func GGMLIsMatrix(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_matrix(tensor.c))
}

// bool ggml_is_3d (const struct ggml_tensor * tensor);
func GGMLIs3d(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_3d(tensor.c))
}

// int  ggml_n_dims (const struct ggml_tensor * tensor); // returns 1 for scalars
func GGMLNDims(tensor GGMLTensor) int {
	return (int)(C.ggml_n_dims(tensor.c))
}

// GGML_CALL bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
func GGMLIsContiguous(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_contiguous(tensor.c))
}

// GGML_CALL bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
func GGMLIsContiguous0(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_contiguous_0(tensor.c))
}

// GGML_CALL bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
func GGMLIsContiguous1(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_contiguous_1(tensor.c))
}

// GGML_CALL bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
func GGMLIsContiguous2(tensor GGMLTensor) bool {
	return (bool)(C.ggml_is_contiguous_2(tensor.c))
}

// bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
func GGMLAreSameShape(t0, t1 GGMLTensor) bool {
	return (bool)(C.ggml_are_same_shape(t0.c, t1.c))
}

// bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
func GGMLAreSameStride(t0, t1 GGMLTensor) bool {
	return (bool)(C.ggml_are_same_stride(t0.c, t1.c))
}

// size_t ggml_tensor_overhead(void);
func GGMLTensorOverhead() uintptr {
	return (uintptr)(C.ggml_tensor_overhead())
}

// bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);
func GGMLValidateRowData(gType GGMLType, data unsafe.Pointer, nBytes uintptr) bool {
	return (bool)(C.ggml_validate_row_data((C.enum_ggml_type)(gType), data, (C.size_t)(nBytes)))
}

func NewGGMLInitParams(nBytes uintptr) GGMLInitParams {
	return GGMLInitParams{C.struct_ggml_init_params{
		mem_size: (C.size_t)(nBytes),
		no_alloc: (C.bool)(false),
	}}
}

// struct ggml_context * ggml_init(struct ggml_init_params params);
func GGMLInit(params GGMLInitParams) GGMLContext {
	return GGMLContext{C.ggml_init(params.c)}
}

// void ggml_free(struct ggml_context * ctx);
func GGMLFree(ctx GGMLContext) {
	C.ggml_free(ctx.c)
}

// size_t  ggml_used_mem(const struct ggml_context * ctx);
func GGMLUsedMem(ctx GGMLContext) uintptr {
	return (uintptr)(C.ggml_used_mem(ctx.c))
}

// size_t  ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);
func GGMLSetScratch(ctx GGMLContext, scratch GGMLScratch) uintptr {
	return (uintptr)(C.ggml_set_scratch(ctx.c, scratch.c))
}

// bool    ggml_get_no_alloc(struct ggml_context * ctx);
func GGMLGetNoAlloc(ctx GGMLContext) bool {
	return (bool)(C.ggml_get_no_alloc(ctx.c))
}

// void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
func GGMLSetNoAlloc(ctx GGMLContext, noAlloc bool) {
	C.ggml_set_no_alloc(ctx.c, C.bool(noAlloc))
}

// void *  ggml_get_mem_buffer     (const struct ggml_context * ctx);
func GGMLGetMemBuffer(ctx GGMLContext) unsafe.Pointer {
	return C.ggml_get_mem_buffer(ctx.c)
}

// size_t  ggml_get_mem_size       (const struct ggml_context * ctx);
func GGMLGetMemSize(ctx GGMLContext) uintptr {
	return (uintptr)(C.ggml_get_mem_size(ctx.c))
}

// size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);
func GGMLGetMaxTensorSize(ctx GGMLContext) uintptr {
	return (uintptr)(C.ggml_get_max_tensor_size(ctx.c))
}

// struct ggml_tensor * ggml_new_tensor(struct ggml_context * ctx,	enum   ggml_type type, int n_dims, const int64_t *ne);
func GGMLNewTensor(ctx GGMLContext, gType GGMLType, dims int, ne []int64) GGMLTensor {
	return GGMLTensor{C.ggml_new_tensor(ctx.c, (C.enum_ggml_type)(gType), C.int(dims), (*C.int64_t)(unsafe.Pointer(&ne[0])))}
}

// struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0);
func GGMLNewTensor1d(ctx GGMLContext, gType GGMLType, ne0 int64) GGMLTensor {
	return GGMLTensor{C.ggml_new_tensor_1d(ctx.c, (C.enum_ggml_type)(gType), (C.int64_t)(ne0))}
}

// struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx, enum ggml_type type,	int64_t ne0, int64_t ne1);
func GGMLNewTensor2d(ctx GGMLContext, gType GGMLType, ne0 int64, ne1 int64) GGMLTensor {
	return GGMLTensor{C.ggml_new_tensor_2d(ctx.c, (C.enum_ggml_type)(gType), (C.int64_t)(ne0), (C.int64_t)(ne1))}
}

// struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx, enum   ggml_type type,	int64_t ne0,	int64_t ne1,	int64_t ne2);
func GGMLNewTensor3d(ctx GGMLContext, gType GGMLType, ne0 int64, ne1 int64, ne2 int64) GGMLTensor {
	return GGMLTensor{C.ggml_new_tensor_3d(ctx.c, (C.enum_ggml_type)(gType), (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2))}
}

// struct ggml_tensor * ggml_new_tensor_4d(	struct ggml_context * ctx,	enum   ggml_type type,	int64_t ne0,	int64_t ne1,	int64_t ne2,	int64_t ne3);
func GGMLNewTensor4d(ctx GGMLContext, gType GGMLType, ne0 int64, ne1 int64, ne2 int64, ne3 int64) GGMLTensor {
	return GGMLTensor{C.ggml_new_tensor_4d(ctx.c, (C.enum_ggml_type)(gType), (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2), (C.int64_t)(ne3))}
}

// struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
func GGMLNewI32(ctx GGMLContext, value int32) GGMLTensor {
	return GGMLTensor{C.ggml_new_i32(ctx.c, (C.int32_t)(value))}
}

// struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
func GGMLNewF32(ctx GGMLContext, value float32) GGMLTensor {
	return GGMLTensor{C.ggml_new_f32(ctx.c, (C.float)(value))}
}

// struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
func GGMLDupTensor(ctx GGMLContext, src GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_dup_tensor(ctx.c, src.c)}
}

// struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);
func GGMLViewTensor(ctx GGMLContext, src GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_view_tensor(ctx.c, src.c)}
}

// struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
func GGMLGetFirstTensor(ctx GGMLContext) GGMLTensor {
	return GGMLTensor{C.ggml_get_first_tensor(ctx.c)}
}

// struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);
func GGMLGetNextTensor(ctx GGMLContext, tensor GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_get_next_tensor(ctx.c, tensor.c)}
}

// struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
func GGMLGetTensor(ctx GGMLContext, name string) GGMLTensor {
	cs := C.CString(name)
	defer C.free(unsafe.Pointer(cs))
	return GGMLTensor{C.ggml_get_tensor(ctx.c, cs)}
}

// struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
func GGMLSetZero(tensor GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_set_zero(tensor.c)}
}

// struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
func GGMLSetI32(tensor GGMLTensor, value int32) GGMLTensor {
	return GGMLTensor{C.ggml_set_i32(tensor.c, (C.int32_t)(value))}
}

// struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
func GGMLSetF32(tensor GGMLTensor, value float32) GGMLTensor {
	return GGMLTensor{C.ggml_set_f32(tensor.c, (C.float)(value))}
}

// void    ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);
func GGMLUnravelIndex(tensor GGMLTensor, i int64, i0, i1, i2, i3 *int64) {
	C.ggml_unravel_index(tensor.c, (C.int64_t)(i), (*C.int64_t)(i0), (*C.int64_t)(i1), (*C.int64_t)(i2), (*C.int64_t)(i3))
}

// int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
func GGMLGetI32_1d(tensor GGMLTensor, i int) int32 {
	return (int32)(C.ggml_get_i32_1d(tensor.c, (C.int)(i)))
}

// void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
func GGMLSetI32_1d(tensor GGMLTensor, i int, value int32) {
	C.ggml_set_i32_1d(tensor.c, (C.int)(i), (C.int32_t)(value))
}

// int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
func GGMLGetI32_nd(tensor GGMLTensor, i0, i1, i2, i3 int) int32 {
	return (int32)(C.ggml_get_i32_nd(tensor.c, (C.int)(i0), (C.int)(i1), (C.int)(i2), (C.int)(i3)))
}

// void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
func GGMLSetI32_nd(tensor GGMLTensor, i0, i1, i2, i3 int, value int32) {
	C.ggml_set_i32_nd(tensor.c, (C.int)(i0), (C.int)(i1), (C.int)(i2), (C.int)(i3), (C.int32_t)(value))
}

// float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
func GGMLGetF32_1d(tensor GGMLTensor, i int) float32 {
	return float32(C.ggml_get_f32_1d(tensor.c, (C.int)(i)))
}

// void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
func GGMLSetF32_1d(tensor GGMLTensor, i int, value float32) {
	C.ggml_set_f32_1d(tensor.c, (C.int)(i), (C.float)(value))
}

// float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
func GGMLGetF32_nd(tensor GGMLTensor, i0, i1, i2, i3 int) float32 {
	return float32(C.ggml_get_f32_nd(tensor.c, (C.int)(i0), (C.int)(i1), (C.int)(i2), (C.int)(i3)))
}

// void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
func GGMLSetF32_nd(tensor GGMLTensor, i0, i1, i2, i3 int, value float32) {
	C.ggml_set_f32_nd(tensor.c, (C.int)(i0), (C.int)(i1), (C.int)(i2), (C.int)(i3), (C.float)(value))
}

// void *  ggml_get_data    (const struct ggml_tensor * tensor);
func GGMLGetData(tensor GGMLTensor) unsafe.Pointer {
	return C.ggml_get_data(tensor.c)
}

// float * ggml_get_data_f32(const struct ggml_tensor * tensor);
func GGMLGetDataF32(tensor GGMLTensor) []float32 {
	data := C.ggml_get_data_f32(tensor.c)
	length := C.ggml_nelements(tensor.c)

	result := make([]float32, length)
	for i := range length {
		result[i] = (float32)(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(data)) + uintptr(i)*unsafe.Sizeof(C.float(0)))))
	}
	return result
}

// GGML_CALL enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
func GGMLGetUnaryOp(tensor GGMLTensor) GGMLUnaryOp {
	return (GGMLUnaryOp)(C.ggml_get_unary_op(tensor.c))
}

// const char * ggml_get_name(const struct ggml_tensor * tensor);
func GGMLGetName(tensor GGMLTensor) string {
	return C.GoString(C.ggml_get_name(tensor.c))
}

// struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name);
func GGMLSetName(tensor GGMLTensor, name string) GGMLTensor {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return GGMLTensor{C.ggml_set_name(tensor.c, cName)}
}

// struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt, ...);
// Use GGMLSetName(t, fmt.Sprintf(...))

//
// operations on tensors with backpropagation
//

// struct ggml_tensor * ggml_dup(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLDup(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_dup(ctx.c, a.c)}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_dup_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLDupInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_dup_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_add(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b);
func GGMLAdd(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_add(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_add_inplace(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLAddInplace(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_add_inplace(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_add_cast(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b,enum   ggml_type      type);
func GGMLAddCast(ctx GGMLContext, a GGMLTensor, b GGMLTensor, gType GGMLType) GGMLTensor {
	return GGMLTensor{C.ggml_add_cast(ctx.c, a.c, b.c, (C.enum_ggml_type)(gType))}
}

// struct ggml_tensor * ggml_add1(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLAdd1(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_add1(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_add1_inplace(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLAdd1Inplace(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_add1_inplace(ctx.c, a.c, b.c)}
}

// // dst = a
// // view(dst, nb1, nb2, nb3, offset) += b
// // return dst
// struct ggml_tensor * ggml_acc(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
func GGMLAcc(ctx GGMLContext, a GGMLTensor, b GGMLTensor, nb1, nb2, nb3, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_acc(ctx.c, a.c, b.c, (C.size_t)(nb1), (C.size_t)(nb2), (C.size_t)(nb3), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_acc_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
func GGMLAccInplace(ctx GGMLContext, a GGMLTensor, b GGMLTensor, nb1, nb2, nb3, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_acc_inplace(ctx.c, a.c, b.c, (C.size_t)(nb1), (C.size_t)(nb2), (C.size_t)(nb3), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_sub(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLSub(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sub(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_sub_inplace(struct ggml_context * ctx,struct ggml_tensor  * b);
func GGMLSubInplace(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sub_inplace(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_mul(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b);
func GGMLMul(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_mul(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_mul_inplace(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLMulInplace(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_mul_inplace(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_div(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLDiv(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_div(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_div_inplace(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLDivInplace(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_div_inplace(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_sqr(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSqr(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sqr(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sqr_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSqrInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sqr_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSqrt(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sqrt(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sqrt_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSqrtInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sqrt_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_log(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLLog(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_log(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_log_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLLogInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_log_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sin(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSin(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sin(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sin_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSinInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sin_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_cos(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLCos(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_cos(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_cos_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLCosInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_cos_inplace(ctx.c, a.c)}
}

// // return scalar
// struct ggml_tensor * ggml_sum(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSum(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sum(ctx.c, a.c)}
}

// // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
// struct ggml_tensor * ggml_sum_rows(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSumRows(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sum_rows(ctx.c, a.c)}
}

// // mean along rows
// struct ggml_tensor * ggml_mean(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLMean(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_mean(ctx.c, a.c)}
}

// // argmax along rows
// struct ggml_tensor * ggml_argmax(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLArgmax(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_argmax(ctx.c, a.c)}
}

// // if a is the same shape as b, and a is not parameter, return a
// // otherwise, return a new tensor: repeat(a) to fit in b
// struct ggml_tensor * ggml_repeat(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLRepeat(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_repeat(ctx.c, a.c, b.c)}
}

// // sums repetitions in a into shape of b
// struct ggml_tensor * ggml_repeat_back(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b);
func GGMLRepeatBack(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_repeat_back(ctx.c, a.c, b.c)}
}

// // concat a and b along dim
// // used in stable-diffusion
// struct ggml_tensor * ggml_concat(struct ggml_context * ctx,struct ggml_tensor  * a,struct ggml_tensor  * b,int                   dim);
func GGMLConcat(ctx GGMLContext, a, b GGMLTensor, dim int) GGMLTensor {
	return GGMLTensor{C.ggml_concat(ctx.c, a.c, b.c, (C.int)(dim))}
}

// struct ggml_tensor * ggml_abs(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLAbs(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_abs(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_abs_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLAbsInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_abs_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sgn(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSgn(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sgn(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sgn_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSgnInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sgn_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_neg(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLNeg(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_neg(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_neg_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLNegInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_neg_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_step(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLStep(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_step(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_step_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLStepInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_step_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_tanh(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLTanh(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_tanh(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_tanh_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLTanhInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_tanh_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_elu(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLElu(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_elu(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_elu_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLEluInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_elu_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_relu(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLRelu(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_relu(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_leaky_relu(struct ggml_context * ctx,struct ggml_tensor  * a, float negative_slope, bool inplace);
func GGMLLeakyRelu(ctx GGMLContext, a GGMLTensor, negative_slope float32, inplace bool) GGMLTensor {
	return GGMLTensor{C.ggml_leaky_relu(ctx.c, a.c, (C.float)(negative_slope), (C.bool)(inplace))}
}

// struct ggml_tensor * ggml_relu_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLReluInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_relu_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sigmoid(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSigmoid(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sigmoid(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_sigmoid_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLSigmoidInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_sigmoid_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_gelu(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLGelu(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_gelu(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_gelu_inplace(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLGeluInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_gelu_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_gelu_quick(struct ggml_context * ctx,struct ggml_tensor  * a);
func GGMLGeluQuick(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_gelu_quick(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_gelu_quick_inplace(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLGeluQuickInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_gelu_quick_inplace(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLSilu(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_silu(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_silu_inplace(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLSiluInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_silu_inplace(ctx.c, a.c)}
}

// // a - x
// // b - dy
// struct ggml_tensor * ggml_silu_back(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b);
func GGMLSiluBack(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_silu_back(ctx.c, a.c, b.c)}
}

// // hardswish(x) = x * relu6(x + 3) / 6
// struct ggml_tensor * ggml_hardswish(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLHardswish(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_hardswish(ctx.c, a.c)}
}

// // hardsigmoid(x) = relu6(x + 3) / 6
// struct ggml_tensor * ggml_hardsigmoid(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLHardsigmoid(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_hardsigmoid(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_exp(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLExp(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_exp(ctx.c, a.c)}
}

// struct ggml_tensor * ggml_exp_inplace(struct ggml_context * ctx, struct ggml_tensor  * a);
func GGMLExpInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_exp_inplace(ctx.c, a.c)}
}

// // normalize along rows
// struct ggml_tensor * ggml_norm(struct ggml_context * ctx, struct ggml_tensor  * a, float eps);
func GGMLNorm(ctx GGMLContext, a GGMLTensor, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_norm(ctx.c, a.c, (C.float)(eps))}
}

// struct ggml_tensor * ggml_norm_inplace(struct ggml_context * ctx,struct ggml_tensor  * a,float eps);
func GGMLNormInplace(ctx GGMLContext, a GGMLTensor, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_norm_inplace(ctx.c, a.c, (C.float)(eps))}
}

// struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor  * a, float eps);
func GGMLRmsNorm(ctx GGMLContext, a GGMLTensor, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_rms_norm(ctx.c, a.c, (C.float)(eps))}
}

// struct ggml_tensor * ggml_rms_norm_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, float eps);
func GGMLRmsNormInplace(ctx GGMLContext, a GGMLTensor, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_rms_norm_inplace(ctx.c, a.c, (C.float)(eps))}
}

// // group normalize along ne0*ne1*n_groups
// // used in stable-diffusion
// struct ggml_tensor * ggml_group_norm(struct ggml_context * ctx, struct ggml_tensor  * a, int n_groups, float eps);
func GGMLGroupNorm(ctx GGMLContext, a GGMLTensor, n_groups int, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_group_norm(ctx.c, a.c, (C.int)(n_groups), (C.float)(eps))}
}

// struct ggml_tensor * ggml_group_norm_inplace(struct ggml_context * ctx,struct ggml_tensor  * a,int n_groups, float eps);
func GGMLGroupNormInplace(ctx GGMLContext, a GGMLTensor, n_groups int, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_group_norm_inplace(ctx.c, a.c, (C.int)(n_groups), (C.float)(eps))}
}

// // a - x
// // b - dy
// struct ggml_tensor * ggml_rms_norm_back(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b, float                 eps);
func GGMLRmsNormBack(ctx GGMLContext, a, b GGMLTensor, eps float32) GGMLTensor {
	return GGMLTensor{C.ggml_rms_norm_back(ctx.c, a.c, b.c, (C.float)(eps))}
}

// // A: k columns, n rows => [ne03, ne02, n, k]
// // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
// // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
// struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b);
func GGMLMulMat(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_mul_mat(ctx.c, a.c, b.c)}
}

// // change the precision of a matrix multiplication
// // set to GGML_PREC_F32 for higher precision (useful for phi-2)
// void ggml_mul_mat_set_prec(struct ggml_tensor * a, enum ggml_prec prec);
func GGMLMulMatSetPrec(a GGMLTensor, prec GGMLPrec) {
	C.ggml_mul_mat_set_prec(a.c, (C.enum_ggml_prec)(prec))
}

// // indirect matrix multiplication
// struct ggml_tensor * ggml_mul_mat_id(struct ggml_context * ctx, struct ggml_tensor  * as, struct ggml_tensor  * b, struct ggml_tensor  * ids);
func GGMLMulMatId(ctx GGMLContext, as, b, ids GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_mul_mat_id(ctx.c, as.c, b.c, ids.c)}
}

// // A: m columns, n rows,
// // B: p columns, n rows,
// // result is m columns, p rows
// struct ggml_tensor * ggml_out_prod(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b);
func GGMLOutProd(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_out_prod(ctx.c, a.c, b.c)}
}

// //
// // operations on tensors without backpropagation
// //

// struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s);
func GGMLScale(ctx GGMLContext, a GGMLTensor, s float32) GGMLTensor {
	return GGMLTensor{C.ggml_scale(ctx.c, a.c, (C.float)(s))}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_scale_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, float                 s);
func GGMLScaleInplace(ctx GGMLContext, a GGMLTensor, s float32) GGMLTensor {
	return GGMLTensor{C.ggml_scale_inplace(ctx.c, a.c, (C.float)(s))}
}

// // b -> view(a,offset,nb1,nb2,3), return modified a
// struct ggml_tensor * ggml_set(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
func GGMLSet(ctx GGMLContext, a, b GGMLTensor, nb1, nb2, nb3, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_set(ctx.c, a.c, b.c, (C.size_t)(nb1), (C.size_t)(nb2), (C.size_t)(nb3), (C.size_t)(offset))}
}

// // b -> view(a,offset,nb1,nb2,3), return view(a)
// struct ggml_tensor * ggml_set_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
func GGMLSetInplace(ctx GGMLContext, a, b GGMLTensor, nb1, nb2, nb3, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_set_inplace(ctx.c, a.c, b.c, (C.size_t)(nb1), (C.size_t)(nb2), (C.size_t)(nb3), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_set_1d(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b, size_t offset);
func GGMLSet1d(ctx GGMLContext, a, b GGMLTensor, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_set_1d(ctx.c, a.c, b.c, (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_set_1d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
func GGMLSet1dInplace(ctx GGMLContext, a, b GGMLTensor, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_set_1d_inplace(ctx.c, a.c, b.c, (C.size_t)(offset))}
}

// // b -> view(a,offset,nb1,nb2,3), return modified a
// struct ggml_tensor * ggml_set_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);
func GGMLSet2d(ctx GGMLContext, a, b GGMLTensor, nb1, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_set_2d(ctx.c, a.c, b.c, (C.size_t)(nb1), (C.size_t)(offset))}
}

// // b -> view(a,offset,nb1,nb2,3), return view(a)
// struct ggml_tensor * ggml_set_2d_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b, size_t nb1, size_t offset);
func GGMLSet2dInplace(ctx GGMLContext, a, b GGMLTensor, nb1, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_set_2d_inplace(ctx.c, a.c, b.c, (C.size_t)(nb1), (C.size_t)(offset))}
}

// // a -> b, return view(b)
// struct ggml_tensor * ggml_cpy(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b);
func GGMLCpy(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_cpy(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_cast( struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type);
func GGMLCast(ctx GGMLContext, a GGMLTensor, gType GGMLType) GGMLTensor {
	return GGMLTensor{C.ggml_cast(ctx.c, a.c, (C.enum_ggml_type)(gType))}
}

// // make contiguous
// struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a);
func GGMLCont(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_cont(ctx.c, a.c)}
}

// // make contiguous, with new shape
// struct ggml_tensor * ggml_cont_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0);
func GGMLCont1d(ctx GGMLContext, a GGMLTensor, ne0 int64) GGMLTensor {
	return GGMLTensor{C.ggml_cont_1d(ctx.c, a.c, (C.int64_t)(ne0))}
}

// struct ggml_tensor * ggml_cont_2d(struct ggml_context * ctx, struct ggml_tensor  * a, int64_t ne0, int64_t ne1);
func GGMLCont2d(ctx GGMLContext, a GGMLTensor, ne0, ne1 int64) GGMLTensor {
	return GGMLTensor{C.ggml_cont_2d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1))}
}

// struct ggml_tensor * ggml_cont_3d(struct ggml_context * ctx, struct ggml_tensor  * a, int64_t ne0, int64_t ne1, int64_t ne2);
func GGMLCont3d(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2 int64) GGMLTensor {
	return GGMLTensor{C.ggml_cont_3d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2))}
}

// struct ggml_tensor * ggml_cont_4d(struct ggml_context * ctx, struct ggml_tensor  * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
func GGMLCont4d(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2, ne3 int64) GGMLTensor {
	return GGMLTensor{C.ggml_cont_4d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2), (C.int64_t)(ne3))}
}

// struct ggml_tensor * ggml_reshape(	struct ggml_context * ctx,	struct ggml_tensor  * a,	struct ggml_tensor  * b);
func GGMLReshape(ctx GGMLContext, a GGMLTensor, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_reshape(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_reshape_1d(	struct ggml_context * ctx,	struct ggml_tensor  * a,	int64_t               ne0);
func GGMLReshape1d(ctx GGMLContext, a GGMLTensor, ne0 int64) GGMLTensor {
	return GGMLTensor{C.ggml_reshape_1d(ctx.c, a.c, (C.int64_t)(ne0))}
}

// struct ggml_tensor * ggml_reshape_2d(	struct ggml_context * ctx,	struct ggml_tensor  * a,	int64_t               ne0,	int64_t               ne1);
func GGMLReshape2d(ctx GGMLContext, a GGMLTensor, ne0, ne1 int64) GGMLTensor {
	return GGMLTensor{C.ggml_reshape_2d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1))}
}

// struct ggml_tensor * ggml_reshape_3d(	struct ggml_context * ctx,	struct ggml_tensor  * a,	int64_t               ne0,int64_t               ne1,	int64_t               ne2);
func GGMLReshape3d(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2 int64) GGMLTensor {
	return GGMLTensor{C.ggml_reshape_3d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2))}
}

// struct ggml_tensor * ggml_reshape_4d(	struct ggml_context * ctx,	struct ggml_tensor  * a,	int64_t               ne0,	int64_t               ne1,	int64_t               ne2,	int64_t               ne3);
func GGMLReshape4d(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2, ne3 int64) GGMLTensor {
	return GGMLTensor{C.ggml_reshape_4d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2), (C.int64_t)(ne3))}
}

// offset in bytes
// struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, size_t offset);
func GGMLView1d(ctx GGMLContext, a GGMLTensor, ne0 int64, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_view_1d(ctx.c, a.c, (C.int64_t)(ne0), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_view_2d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int64_t               ne0,
//	int64_t               ne1,
//	size_t                nb1, // row stride in bytes
//	size_t                offset);
func GGMLView2d(ctx GGMLContext, a GGMLTensor, ne0, ne1 int64, nb1, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_view_2d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.size_t)(nb1), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_view_3d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int64_t               ne0,
//	int64_t               ne1,
//	int64_t               ne2,
//	size_t                nb1, // row   stride in bytes
//	size_t                nb2, // slice stride in bytes
//	size_t                offset);
func GGMLView3d(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2 int64, nb1, nb2, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_view_3d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2), (C.size_t)(nb1), (C.size_t)(nb2), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_view_4d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int64_t               ne0,
//	int64_t               ne1,
//	int64_t               ne2,
//	int64_t               ne3,
//	size_t                nb1, // row   stride in bytes
//	size_t                nb2, // slice stride in bytes
//	size_t                nb3,
//	size_t                offset);
func GGMLView4d(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2, ne3 int64, nb1, nb2, nb3, offset uintptr) GGMLTensor {
	return GGMLTensor{C.ggml_view_4d(ctx.c, a.c, (C.int64_t)(ne0), (C.int64_t)(ne1), (C.int64_t)(ne2), (C.int64_t)(ne3), (C.size_t)(nb1), (C.size_t)(nb2), (C.size_t)(nb3), (C.size_t)(offset))}
}

// struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a, int axis0,	int axis1, int axis2, int axis3);
func GGMLPermute(ctx GGMLContext, a GGMLTensor, axis0, axis1, axis2, axis3 int) GGMLTensor {
	return GGMLTensor{C.ggml_permute(ctx.c, a.c, (C.int)(axis0), (C.int)(axis1), (C.int)(axis2), (C.int)(axis3))}
}

// alias for ggml_permute(ctx, a, 1, 0, 2, 3)
// struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a);
func GGMLTranspose(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_transpose(ctx.c, a.c)}
}

// // supports 3D: a->ne[2] == b->ne[1]
// struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
func GGMLGetRows(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_get_rows(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_get_rows_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c);
func GGMLGetRowsBack(ctx GGMLContext, a, b, c GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_get_rows_back(ctx.c, a.c, b.c, c.c)}
}

// struct ggml_tensor * ggml_diag(struct ggml_context * ctx, struct ggml_tensor * a);
func GGMLDiag(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_diag(ctx.c, a.c)}
}

// // set elements above the diagonal to -INF
// struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
func GGMLDiagMaskInf(ctx GGMLContext, a GGMLTensor, n_past int) GGMLTensor {
	return GGMLTensor{C.ggml_diag_mask_inf(ctx.c, a.c, (C.int)(n_past))}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_diag_mask_inf_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
func GGMLDiagMaskInfInplace(ctx GGMLContext, a GGMLTensor, n_past int) GGMLTensor {
	return GGMLTensor{C.ggml_diag_mask_inf_inplace(ctx.c, a.c, (C.int)(n_past))}
}

// // set elements above the diagonal to 0
// struct ggml_tensor * ggml_diag_mask_zero(struct ggml_context * ctx, struct ggml_tensor * a, int n_past);
func GGMLDiagMaskZero(ctx GGMLContext, a GGMLTensor, n_past int) GGMLTensor {
	return GGMLTensor{C.ggml_diag_mask_zero(ctx.c, a.c, (C.int)(n_past))}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_diag_mask_zero_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, int n_past);
func GGMLDiagMaskZeroInplace(ctx GGMLContext, a GGMLTensor, n_past int) GGMLTensor {
	return GGMLTensor{C.ggml_diag_mask_zero_inplace(ctx.c, a.c, (C.int)(n_past))}
}

// struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a);
func GGMLSoftMax(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_soft_max(ctx.c, a.c)}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_soft_max_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
func GGMLSoftMaxInplace(ctx GGMLContext, a GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_soft_max_inplace(ctx.c, a.c)}
}

// // fused soft_max(a*scale + mask*(ALiBi slope))
// // mask is optional
// // max_bias = 0.0f for no ALiBi
// struct ggml_tensor * ggml_soft_max_ext(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * mask, float scale, float max_bias);
func GGMLSoftMaxExt(ctx GGMLContext, a, mask GGMLTensor, scale, max_bias float32) GGMLTensor {
	return GGMLTensor{C.ggml_soft_max_ext(ctx.c, a.c, mask.c, (C.float)(scale), (C.float)(max_bias))}
}

// struct ggml_tensor * ggml_soft_max_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
func GGMLSoftMaxBack(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_soft_max_back(ctx.c, a.c, b.c)}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_soft_max_back_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
func GGMLSoftMaxBackInplace(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_soft_max_back_inplace(ctx.c, a.c, b.c)}
}

// // rotary position embedding
// // if (mode & 1) - skip n_past elements (NOT SUPPORTED)
// // if (mode & GGML_ROPE_TYPE_NEOX) - GPT-NeoX style
// //
// // b is an int32 vector with size a->ne[2], it contains the positions
// struct ggml_tensor * ggml_rope(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b, int n_dims, int mode);
func GGMLRope(ctx GGMLContext, a, b GGMLTensor, n_dims, mode int) GGMLTensor {
	return GGMLTensor{C.ggml_rope(ctx.c, a.c, b.c, (C.int)(n_dims), (C.int)(mode))}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_rope_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, struct ggml_tensor  * b, int n_dims, int mode);
func GGMLRopeInplace(ctx GGMLContext, a, b GGMLTensor, n_dims, mode int) GGMLTensor {
	return GGMLTensor{C.ggml_rope_inplace(ctx.c, a.c, b.c, (C.int)(n_dims), (C.int)(mode))}
}

// // custom RoPE
// // c is freq factors (e.g. phi3-128k), (optional)
// struct ggml_tensor * ggml_rope_ext(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * b,
//	struct ggml_tensor  * c,
//	int                   n_dims,
//	int                   mode,
//	int                   n_ctx_orig,
//	float                 freq_base,
//	float                 freq_scale,
//	float                 ext_factor,
//	float                 attn_factor,
//	float                 beta_fast,
//	float                 beta_slow);
func GGMLRopeExt(ctx GGMLContext, a, b, c GGMLTensor, n_dims, mode, n_ctx_orig int, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow float32) GGMLTensor {
	return GGMLTensor{C.ggml_rope_ext(ctx.c, a.c, b.c, c.c, (C.int)(n_dims), (C.int)(mode), (C.int)(n_ctx_orig), (C.float)(freq_base), (C.float)(freq_scale), (C.float)(ext_factor), (C.float)(attn_factor), (C.float)(beta_fast), (C.float)(beta_slow))}
}

// // in-place, returns view(a)
// struct ggml_tensor * ggml_rope_ext_inplace(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * b,
//	struct ggml_tensor  * c,
//	int                   n_dims,
//	int                   mode,
//	int                   n_ctx_orig,
//	float                 freq_base,
//	float                 freq_scale,
//	float                 ext_factor,
//	float                 attn_factor,
//	float                 beta_fast,
//	float                 beta_slow);
func GGMLRopeExtInplace(ctx GGMLContext, a, b, c GGMLTensor, n_dims, mode, n_ctx_orig int, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow float32) GGMLTensor {
	return GGMLTensor{C.ggml_rope_ext_inplace(ctx.c, a.c, b.c, c.c, (C.int)(n_dims), (C.int)(mode), (C.int)(n_ctx_orig), (C.float)(freq_base), (C.float)(freq_scale), (C.float)(ext_factor), (C.float)(attn_factor), (C.float)(beta_fast), (C.float)(beta_slow))}
}

// GGML_DEPRECATED(struct ggml_tensor * ggml_rope_custom(
// 		struct ggml_context * ctx,
// 		struct ggml_tensor  * a,
// 		struct ggml_tensor  * b,
// 		int                   n_dims,
// 		int                   mode,
// 		int                   n_ctx_orig,
// 		float                 freq_base,
// 		float                 freq_scale,
// 		float                 ext_factor,
// 		float                 attn_factor,
// 		float                 beta_fast,
// 		float                 beta_slow),
// 	"use ggml_rope_ext instead");

// GGML_DEPRECATED(struct ggml_tensor * ggml_rope_custom_inplace(
// 		struct ggml_context * ctx,
// 		struct ggml_tensor  * a,
// 		struct ggml_tensor  * b,
// 		int                   n_dims,
// 		int                   mode,
// 		int                   n_ctx_orig,
// 		float                 freq_base,
// 		float                 freq_scale,
// 		float                 ext_factor,
// 		float                 attn_factor,
// 		float                 beta_fast,
// 		float                 beta_slow),
// 	"use ggml_rope_ext_inplace instead");

// // compute correction dims for YaRN RoPE scaling
// GGML_CALL void ggml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);
func GGMLRopeYarnCorrDims(n_dims, n_ctx_orig int, freq_base, beta_fast, beta_slow float32, dims [2]float32) {
	C.ggml_rope_yarn_corr_dims((C.int)(n_dims), (C.int)(n_ctx_orig), (C.float)(freq_base), (C.float)(beta_fast), (C.float)(beta_slow), (*C.float)(unsafe.Pointer(&dims[0])))
}

// // rotary position embedding backward, i.e compute dx from dy
// // a - dy
// struct ggml_tensor * ggml_rope_back(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * b,
//	struct ggml_tensor  * c,
//	int                   n_dims,
//	int                   mode,
//	int                   n_ctx_orig,
//	float                 freq_base,
//	float                 freq_scale,
//	float                 ext_factor,
//	float                 attn_factor,
//	float                 beta_fast,
//	float                 beta_slow);
func GGMLRopeBack(ctx GGMLContext, a, b, c GGMLTensor, n_dims, mode, n_ctx_orig int, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow float32) GGMLTensor {
	return GGMLTensor{C.ggml_rope_back(ctx.c, a.c, b.c, c.c, (C.int)(n_dims), (C.int)(mode), (C.int)(n_ctx_orig), (C.float)(freq_base), (C.float)(freq_scale), (C.float)(ext_factor), (C.float)(attn_factor), (C.float)(beta_fast), (C.float)(beta_slow))}
}

// // clamp
// // in-place, returns view(a)
// struct ggml_tensor * ggml_clamp(struct ggml_context * ctx, struct ggml_tensor * a, float min, float max);
func GGMLClamp(ctx GGMLContext, a GGMLTensor, min, max float32) GGMLTensor {
	return GGMLTensor{C.ggml_clamp(ctx.c, a.c, (C.float)(min), (C.float)(max))}
}

// // im2col
// // converts data into a format that effectively results in a convolution when combined with matrix multiplication
// struct ggml_tensor * ggml_im2col(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,  // convolution kernel
//	struct ggml_tensor  * b,  // data
//	int                   s0, // stride dimension 0
//	int                   s1, // stride dimension 1
//	int                   p0, // padding dimension 0
//	int                   p1, // padding dimension 1
//	int                   d0, // dilation dimension 0
//	int                   d1, // dilation dimension 1
//	bool                  is_2D,
//	enum ggml_type        dst_type);
func GGMLIm2col(ctx GGMLContext, a, b GGMLTensor, s0, s1, p0, p1, d0, d1 int, is_2D bool, dst_type GGMLType) GGMLTensor {
	return GGMLTensor{C.ggml_im2col(ctx.c, a.c, b.c, (C.int)(s0), (C.int)(s1), (C.int)(p0), (C.int)(p1), (C.int)(d0), (C.int)(d1), (C.bool)(is_2D), (C.enum_ggml_type)(dst_type))}
}

// struct ggml_tensor * ggml_im2col_back(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,  // convolution kernel
//	struct ggml_tensor  * b,  // gradient of im2col output
//	int64_t             * ne, // shape of im2col input
//	int                   s0, // stride dimension 0
//	int                   s1, // stride dimension 1
//	int                   p0, // padding dimension 0
//	int                   p1, // padding dimension 1
//	int                   d0, // dilation dimension 0
//	int                   d1, // dilation dimension 1
//	bool                  is_2D);
func GGMLIm2colBack(ctx GGMLContext, a, b GGMLTensor, ne []int64, s0, s1, p0, p1, d0, d1 int, is_2D bool) GGMLTensor {
	return GGMLTensor{C.ggml_im2col_back(ctx.c, a.c, b.c, (*C.int64_t)(unsafe.Pointer(&ne[0])), (C.int)(s0), (C.int)(s1), (C.int)(p0), (C.int)(p1), (C.int)(d0), (C.int)(d1), (C.bool)(is_2D))}
}

// struct ggml_tensor * ggml_conv_depthwise_2d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,  // convolution kernel
//	struct ggml_tensor  * b,  // data
//	int                  s0,  // stride dimension 0
//	int                  s1,  // stride dimension 1
//	int                  p0,  // padding dimension 0
//	int                  p1,  // padding dimension 1
//	int                  d0,  // dilation dimension 0
//	int                  d1); // dilation dimension 1
func GGMLConvDepthwise2d(ctx GGMLContext, a, b GGMLTensor, s0, s1, p0, p1, d0, d1 int) GGMLTensor {
	return GGMLTensor{C.ggml_conv_depthwise_2d(ctx.c, a.c, b.c, (C.int)(s0), (C.int)(s1), (C.int)(p0), (C.int)(p1), (C.int)(d0), (C.int)(d1))}
}

// struct ggml_tensor * ggml_conv_1d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,   // convolution kernel
//	struct ggml_tensor  * b,   // data
//	int                   s0,  // stride
//	int                   p0,  // padding
//	int                   d0); // dilation
func GGMLConv1d(ctx GGMLContext, a, b GGMLTensor, s0, p0, d0 int) GGMLTensor {
	return GGMLTensor{C.ggml_conv_1d(ctx.c, a.c, b.c, (C.int)(s0), (C.int)(p0), (C.int)(d0))}
}

// // conv_1d with padding = half
// // alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
// struct ggml_tensor* ggml_conv_1d_ph(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,  // convolution kernel
//	struct ggml_tensor  * b,  // data
//	int                   s,  // stride
//	int                   d); // dilation
func GGMLConv1dPh(ctx GGMLContext, a, b GGMLTensor, s, d int) GGMLTensor {
	return GGMLTensor{C.ggml_conv_1d_ph(ctx.c, a.c, b.c, (C.int)(s), (C.int)(d))}
}

// struct ggml_tensor * ggml_conv_transpose_1d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,   // convolution kernel
//	struct ggml_tensor  * b,   // data
//	int                   s0,  // stride
//	int                   p0,  // padding
//	int                   d0); // dilation
func GGMLConvTranspose1d(ctx GGMLContext, a, b GGMLTensor, s0, p0, d0 int) GGMLTensor {
	return GGMLTensor{C.ggml_conv_transpose_1d(ctx.c, a.c, b.c, (C.int)(s0), (C.int)(p0), (C.int)(d0))}
}

// struct ggml_tensor * ggml_conv_2d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,   // convolution kernel
//	struct ggml_tensor  * b,   // data
//	int                   s0,  // stride dimension 0
//	int                   s1,  // stride dimension 1
//	int                   p0,  // padding dimension 0
//	int                   p1,  // padding dimension 1
//	int                   d0,  // dilation dimension 0
//	int                   d1); // dilation dimension 1
func GGMLConv2d(ctx GGMLContext, a, b GGMLTensor, s0, s1, p0, p1, d0, d1 int) GGMLTensor {
	return GGMLTensor{C.ggml_conv_2d(ctx.c, a.c, b.c, (C.int)(s0), (C.int)(s1), (C.int)(p0), (C.int)(p1), (C.int)(d0), (C.int)(d1))}
}

// // kernel size is a->ne[0] x a->ne[1]
// // stride is equal to kernel size
// // padding is zero
// // example:
// // a:     16   16    3  768
// // b:   1024 1024    3    1
// // res:   64   64  768    1
// // used in sam
// struct ggml_tensor * ggml_conv_2d_sk_p0(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * b);
func GGMLConv2dSkP0(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_conv_2d_sk_p0(ctx.c, a.c, b.c)}
}

// // kernel size is a->ne[0] x a->ne[1]
// // stride is 1
// // padding is half
// // example:
// // a:      3    3    256  256
// // b:     64   64    256    1
// // res:   64   64    256    1
// // used in sam
// struct ggml_tensor * ggml_conv_2d_s1_ph(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * b);
func GGMLConv2dS1Ph(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_conv_2d_s1_ph(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_conv_transpose_2d_p0(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * b,
//	int                   stride);
func GGMLConvTranspose2dp0(ctx GGMLContext, a, b GGMLTensor, stride int) GGMLTensor {
	return GGMLTensor{C.ggml_conv_transpose_2d_p0(ctx.c, a.c, b.c, (C.int)(stride))}
}

// struct ggml_tensor * ggml_pool_1d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	enum ggml_op_pool     op,
//	int                   k0, // kernel size
//	int                   s0, // stride
//	int                   p0); // padding
func GGMLPool1d(ctx GGMLContext, a GGMLTensor, op GGMLOpPool, k0, s0, p0 int) GGMLTensor {
	return GGMLTensor{C.ggml_pool_1d(ctx.c, a.c, (C.enum_ggml_op_pool)(op), (C.int)(k0), (C.int)(s0), (C.int)(p0))}
}

// // the result will have 2*p0 padding for the first dimension
// // and 2*p1 padding for the second dimension
// struct ggml_tensor * ggml_pool_2d(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	enum ggml_op_pool     op,
//	int                   k0,
//	int                   k1,
//	int                   s0,
//	int                   s1,
//	float                 p0,
//	float                 p1);
func GGMLPool2d(ctx GGMLContext, a GGMLTensor, op GGMLOpPool, k0, k1, s0, s1 int, p0, p1 float32) GGMLTensor {
	return GGMLTensor{C.ggml_pool_2d(ctx.c, a.c, (C.enum_ggml_op_pool)(op), (C.int)(k0), (C.int)(k1), (C.int)(s0), (C.int)(s1), (C.float)(p0), (C.float)(p1))}
}

// struct ggml_tensor * ggml_pool_2d_back(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * af, // "a"/input used in forward pass
//	enum ggml_op_pool     op,
//	int                   k0,
//	int                   k1,
//	int                   s0,
//	int                   s1,
//	float                 p0,
//	float                 p1);
func GGMLPool2dBack(ctx GGMLContext, a, af GGMLTensor, op GGMLOpPool, k0, k1, s0, s1 int, p0, p1 float32) GGMLTensor {
	return GGMLTensor{C.ggml_pool_2d_back(ctx.c, a.c, af.c, (C.enum_ggml_op_pool)(op), (C.int)(k0), (C.int)(k1), (C.int)(s0), (C.int)(s1), (C.float)(p0), (C.float)(p1))}
}

// // nearest interpolate
// // multiplies ne0 and ne1 by scale factor
// // used in stable-diffusion
// struct ggml_tensor * ggml_upscale(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int                   scale_factor);
func GGMLUpscale(ctx GGMLContext, a GGMLTensor, scale_factor int) GGMLTensor {
	return GGMLTensor{C.ggml_upscale(ctx.c, a.c, (C.int)(scale_factor))}
}

// // nearest interpolate
// // nearest interpolate to specified dimensions
// // used in tortoise.cpp
// struct ggml_tensor * ggml_upscale_ext(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int                   ne0,
//	int                   ne1,
//	int                   ne2,
//	int                   ne3);
func GGMLUpscaleExt(ctx GGMLContext, a GGMLTensor, ne0, ne1, ne2, ne3 int) GGMLTensor {
	return GGMLTensor{C.ggml_upscale_ext(ctx.c, a.c, (C.int)(ne0), (C.int)(ne1), (C.int)(ne2), (C.int)(ne3))}
}

// // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
// struct ggml_tensor * ggml_pad(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int                  p0,
//	int                  p1,
//	int                  p2,
//	int                  p3);
func GGMLPad(ctx GGMLContext, a GGMLTensor, p0, p1, p2, p3 int) GGMLTensor {
	return GGMLTensor{C.ggml_pad(ctx.c, a.c, (C.int)(p0), (C.int)(p1), (C.int)(p2), (C.int)(p3))}
}

// // Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
// // timesteps: [N,]
// // return: [N, dim]
// struct ggml_tensor * ggml_timestep_embedding(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * timesteps,
//	int                   dim,
//	int                   max_period);
func GGMLTimestepEmbedding(ctx GGMLContext, timesteps GGMLTensor, dim, max_period int) GGMLTensor {
	return GGMLTensor{C.ggml_timestep_embedding(ctx.c, timesteps.c, (C.int)(dim), (C.int)(max_period))}
}

// struct ggml_tensor * ggml_argsort(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_sort_order order);
func GGMLArgsort(ctx GGMLContext, a GGMLTensor, order GGMLSortOrder) GGMLTensor {
	return GGMLTensor{C.ggml_argsort(ctx.c, a.c, (C.enum_ggml_sort_order)(order))}
}

// struct ggml_tensor * ggml_arange(struct ggml_context * ctx, float start, float stop, float step);
func GGMLArange(ctx GGMLContext, start, stop, step float32) GGMLTensor {
	return GGMLTensor{C.ggml_arange(ctx.c, (C.float)(start), (C.float)(stop), (C.float)(step))}
}

// // top k elements per row
// struct ggml_tensor * ggml_top_k(struct ggml_context * ctx, struct ggml_tensor  * a, int k);
func GGMLTopK(ctx GGMLContext, a GGMLTensor, k int) GGMLTensor {
	return GGMLTensor{C.ggml_top_k(ctx.c, a.c, (C.int)(k))}
}

// #define GGML_KQ_MASK_PAD 32

// // q:    [n_embd, n_batch,     n_head,    1]
// // k:    [n_embd, n_kv,        n_head_kv, 1]
// // v:    [n_embd, n_kv,        n_head_kv, 1] !! not transposed !!
// // mask: [n_kv,   n_batch_pad, 1,         1] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
// // res:  [n_embd, n_head,      n_batch,   1] !! permuted !!
// struct ggml_tensor * ggml_flash_attn_ext(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * q,
//	struct ggml_tensor  * k,
//	struct ggml_tensor  * v,
//	struct ggml_tensor  * mask,
//	float                 scale,
//	float                 max_bias,
//	float                 logit_softcap);
func GGMLFlashAttnExt(ctx GGMLContext, q, k, v, mask GGMLTensor, scale, max_bias, logit_softcap float32) GGMLTensor {
	return GGMLTensor{C.ggml_flash_attn_ext(ctx.c, q.c, k.c, v.c, mask.c, (C.float)(scale), (C.float)(max_bias), (C.float)(logit_softcap))}
}

// void ggml_flash_attn_ext_set_prec(struct ggml_tensor * a, enum ggml_prec prec);
func GGMLFlashAttnExtSetPrec(a GGMLTensor, prec GGMLPrec) {
	C.ggml_flash_attn_ext_set_prec(a.c, (C.enum_ggml_prec)(prec))
}

// // TODO: needs to be adapted to ggml_flash_attn_ext
// struct ggml_tensor * ggml_flash_attn_back(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * q,
//	struct ggml_tensor  * k,
//	struct ggml_tensor  * v,
//	struct ggml_tensor  * d,
//	bool                  masked);
func GGMLFlashAttnBack(ctx GGMLContext, q, k, v, d GGMLTensor, masked bool) GGMLTensor {
	return GGMLTensor{C.ggml_flash_attn_back(ctx.c, q.c, k.c, v.c, d.c, (C.bool)(masked))}
}

// struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx, struct ggml_tensor  * sx, struct ggml_tensor  * c);
func GGMLSsmConf(ctx GGMLContext, sx, c GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_ssm_conv(ctx.c, sx.c, c.c)}
}

// struct ggml_tensor * ggml_ssm_scan(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * s,
//	struct ggml_tensor  * x,
//	struct ggml_tensor  * dt,
//	struct ggml_tensor  * A,
//	struct ggml_tensor  * B,
//	struct ggml_tensor  * C);
func GGMLSsmScan(ctx GGMLContext, s, x, dt, a, b, c GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_ssm_scan(ctx.c, s.c, x.c, dt.c, a.c, b.c, c.c)}
}

// // partition into non-overlapping windows with padding if needed
// // example:
// // a:   768   64   64    1
// // w:    14
// // res: 768   14   14    25
// // used in sam
// struct ggml_tensor * ggml_win_part(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int                   w);
func GGMLWinPart(ctx GGMLContext, a GGMLTensor, w int) GGMLTensor {
	return GGMLTensor{C.ggml_win_part(ctx.c, a.c, (C.int)(w))}
}

// // reverse of ggml_win_part
// // used in sam
// struct ggml_tensor * ggml_win_unpart(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int                   w0,
//	int                   h0,
//	int                   w);
func GGMLWinUnpart(ctx GGMLContext, a GGMLTensor, w0, h0, w int) GGMLTensor {
	return GGMLTensor{C.ggml_win_unpart(ctx.c, a.c, (C.int)(w0), (C.int)(h0), (C.int)(w))}
}

// struct ggml_tensor * ggml_unary(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op);
func GGMLUnary(ctx GGMLContext, a GGMLTensor, op GGMLUnaryOp) GGMLTensor {
	return GGMLTensor{C.ggml_unary(ctx.c, a.c, (C.enum_ggml_unary_op)(op))}
}

// struct ggml_tensor * ggml_unary_inplace(struct ggml_context * ctx, struct ggml_tensor  * a, enum ggml_unary_op op);
func GGMLUnaryInplace(ctx GGMLContext, a GGMLTensor, op GGMLUnaryOp) GGMLTensor {
	return GGMLTensor{C.ggml_unary_inplace(ctx.c, a.c, (C.enum_ggml_unary_op)(op))}
}

// // used in sam
// struct ggml_tensor * ggml_get_rel_pos(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	int                   qh,
//	int                   kh);
func GGMLGetRelPos(ctx GGMLContext, a GGMLTensor, qh, kh int) GGMLTensor {
	return GGMLTensor{C.ggml_get_rel_pos(ctx.c, a.c, (C.int)(qh), (C.int)(kh))}
}

// // used in sam
// struct ggml_tensor * ggml_add_rel_pos(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * pw,
//	struct ggml_tensor  * ph);
func GGMLAddRelPos(ctx GGMLContext, a, pw, ph GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_add_rel_pos(ctx.c, a.c, pw.c, ph.c)}
}

// struct ggml_tensor * ggml_add_rel_pos_inplace(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * a,
//	struct ggml_tensor  * pw,
//	struct ggml_tensor  * ph);
func GGMLAddRelPosInplace(ctx GGMLContext, a, pw, ph GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_add_rel_pos_inplace(ctx.c, a.c, pw.c, ph.c)}
}

// struct ggml_tensor * ggml_rwkv_wkv(
//
//	struct ggml_context * ctx,
//	struct ggml_tensor  * k,
//	struct ggml_tensor  * v,
//	struct ggml_tensor  * r,
//	struct ggml_tensor  * tf,
//	struct ggml_tensor  * td,
//	struct ggml_tensor  * state);
func GGMLRwkvWkv(ctx GGMLContext, k, v, r, tf, td, state GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_rwkv_wkv(ctx.c, k.c, v.c, r.c, tf.c, td.c, state.c)}
}

// deprecated (v1) custom operators omitted

// // custom operators v2

// TODO all of these appear unused outside of ggml lib
//
// typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata);
// typedef void (*ggml_custom2_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);
// typedef void (*ggml_custom3_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);
//
// #define GGML_N_TASKS_MAX -1
//
// struct ggml_tensor * ggml_map_custom1(
// 		struct ggml_context   * ctx,
// 		struct ggml_tensor    * a,
// 		ggml_custom1_op_t       fun,
// 		int                     n_tasks,
// 		void                  * userdata);
//
// struct ggml_tensor * ggml_map_custom1_inplace(
// 		struct ggml_context   * ctx,
// 		struct ggml_tensor    * a,
// 		ggml_custom1_op_t       fun,
// 		int                     n_tasks,
// 		void                  * userdata);
//
// struct ggml_tensor * ggml_map_custom2(
// 		struct ggml_context   * ctx,
// 		struct ggml_tensor    * a,
// 		struct ggml_tensor    * b,
// 		ggml_custom2_op_t       fun,
// 		int                     n_tasks,
// 		void                  * userdata);
//
// struct ggml_tensor * ggml_map_custom2_inplace(
// 		struct ggml_context   * ctx,
// 		struct ggml_tensor    * a,
// 		struct ggml_tensor    * b,
// 		ggml_custom2_op_t       fun,
// 		int                     n_tasks,
// 		void                  * userdata);
//
// struct ggml_tensor * ggml_map_custom3(
// 		struct ggml_context   * ctx,
// 		struct ggml_tensor    * a,
// 		struct ggml_tensor    * b,
// 		struct ggml_tensor    * c,
// 		ggml_custom3_op_t       fun,
// 		int                     n_tasks,
// 		void                  * userdata);
//
// struct ggml_tensor * ggml_map_custom3_inplace(
// 		struct ggml_context   * ctx,
// 		struct ggml_tensor    * a,
// 		struct ggml_tensor    * b,
// 		struct ggml_tensor    * c,
// 		ggml_custom3_op_t       fun,
// 		int                     n_tasks,
// 		void                  * userdata);

// // loss function

// struct ggml_tensor * ggml_cross_entropy_loss(
//
//	struct ggml_context         * ctx,
//	struct ggml_tensor          * a,
//	struct ggml_tensor          * b);
func GGMLCrossEntropyLoss(ctx GGMLContext, a, b GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_cross_entropy_loss(ctx.c, a.c, b.c)}
}

// struct ggml_tensor * ggml_cross_entropy_loss_back(
//
//	struct ggml_context         * ctx,
//	struct ggml_tensor          * a,
//	struct ggml_tensor          * b,
//	struct ggml_tensor          * c);
func GGMLCrossEntropyLossBack(ctx GGMLContext, a, b, c GGMLTensor) GGMLTensor {
	return GGMLTensor{C.ggml_cross_entropy_loss_back(ctx.c, a.c, b.c, c.c)}
}

//
// automatic differentiation
//

// void ggml_set_param(	struct ggml_context * ctx,	struct ggml_tensor  * tensor);
func GGMLSetParam(ctx GGMLContext, tensor GGMLTensor) {
	C.ggml_set_param(ctx.c, tensor.c)
}

// void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
func GGMLBuildForwardExpand(cgraph GGMLCGraph, tensor GGMLTensor) {
	C.ggml_build_forward_expand(cgraph.c, tensor.c)
}

// void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep);
func GGMLBuildBackwardExpand(ctx GGMLContext, gf GGMLCGraph, gb GGMLCGraph, keep bool) {
	C.ggml_build_backward_expand(ctx.c, gf.c, gb.c, (C.bool)(keep))
}

// struct ggml_cgraph * ggml_new_graph         (struct ggml_context * ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
func GGMLNewGraph(ctx GGMLContext) GGMLCGraph {
	return GGMLCGraph{C.ggml_new_graph(ctx.c)}
}

// struct ggml_cgraph * ggml_new_graph_custom  (struct ggml_context * ctx, size_t size, bool grads);
func GGMLNewGraphCustom(ctx GGMLContext, size int, grads bool) GGMLCGraph {
	return GGMLCGraph{C.ggml_new_graph_custom(ctx.c, (C.size_t)(size), (C.bool)(grads))}
}

// struct ggml_cgraph * ggml_graph_dup         (struct ggml_context * ctx, struct ggml_cgraph * cgraph);
func GGMLGraphDup(ctx GGMLContext, cgraph GGMLCGraph) GGMLCGraph {
	return GGMLCGraph{C.ggml_graph_dup(ctx.c, cgraph.c)}
}

// struct ggml_cgraph   ggml_graph_view        (struct ggml_cgraph * cgraph, int i0, int i1);
func GGMLGraphView(cgraph GGMLCGraph, i0 int, i1 int) GGMLCGraph {
	c := C.ggml_graph_view(cgraph.c, (C.int)(i0), (C.int)(i1))
	return GGMLCGraph{&c}
}

// void ggml_graph_cpy (struct ggml_cgraph * src, struct ggml_cgraph * dst);
func GGMLGraphCpy(src GGMLCGraph, dst GGMLCGraph) {
	C.ggml_graph_cpy(src.c, dst.c)
}

// void ggml_graph_reset (struct ggml_cgraph * cgraph);  // zero grads
func GGMLGraphReset(cgraph GGMLCGraph) {
	C.ggml_graph_reset(cgraph.c)
}

// void ggml_graph_clear (struct ggml_cgraph * cgraph);
func GGMLGraphClear(cgraph GGMLCGraph) {
	C.ggml_graph_clear(cgraph.c)
}

// size_t ggml_graph_overhead(void);
func GGMLGraphOverhead() int {
	return int(C.ggml_graph_overhead())
}

// size_t ggml_graph_overhead_custom(size_t size, bool grads);
func GGMLGraphOverheadCustom(size int, grads bool) uintptr {
	return (uintptr)(C.ggml_graph_overhead_custom((C.size_t)(size), (C.bool)(grads)))
}

// struct ggml_cplan ggml_graph_plan (const struct ggml_cgraph * cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);
func GGMLGraphPlan(cgraph GGMLCGraph, nThreads int, threadpool GGMLThreadPool) GGMLCPlan {
	c := C.ggml_graph_plan(cgraph.c, (C.int)(nThreads), threadpool.c)
	return GGMLCPlan{&c}
}

// enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
func GGMLGraphCompute(cgraph GGMLCGraph, cplan GGMLCPlan) int {
	return int(C.ggml_graph_compute(cgraph.c, cplan.c))
}

// enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
func GGMLGraphComputeWithCtx(ctx GGMLContext, cgraph GGMLCGraph, nThreads int) int {
	return int(C.ggml_graph_compute_with_ctx(ctx.c, cgraph.c, (C.int)(nThreads)))
}

// struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);
func GGMLGraphGetTensor(cgraph GGMLCGraph, name string) GGMLTensor {
	return GGMLTensor{C.ggml_graph_get_tensor(cgraph.c, C.CString(name))}
}

// TODO how do we want to adapt these?  Maybe useful to adapt for debug/eval tooling
// void                 ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
// struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
// void ggml_graph_print(const struct ggml_cgraph * cgraph);
// void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
