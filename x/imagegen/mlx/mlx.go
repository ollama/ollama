//go:build mlx

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

// Use generated wrappers instead of direct MLX headers
#include "mlx.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Forward declare cpu_stream
static mlx_stream cpu_stream();

// Cached default GPU stream for all ops
static mlx_stream _default_stream = {0};
static mlx_stream _cpu_stream = {0};

static inline mlx_stream default_stream() {
    if (_default_stream.ctx == NULL) {
        _default_stream = mlx_default_gpu_stream_new();
    }
    return _default_stream;
}

static inline void set_default_stream(mlx_stream s) {
    _default_stream = s;
}

// CPU stream for file loading (Load primitive only runs on CPU)
static inline mlx_stream cpu_stream() {
    if (_cpu_stream.ctx == NULL) {
        _cpu_stream = mlx_default_cpu_stream_new();
    }
    return _cpu_stream;
}

// CGO noescape/nocallback hints to reduce CGO overhead
// noescape: pointers won't escape, no heap allocation needed
// nocallback: function won't call back into Go
*/
import "C"
import (
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// Dtype represents MLX data types
type Dtype int

const (
	DtypeBool      Dtype = C.MLX_BOOL
	DtypeUint8     Dtype = C.MLX_UINT8
	DtypeUint16    Dtype = C.MLX_UINT16
	DtypeUint32    Dtype = C.MLX_UINT32
	DtypeUint64    Dtype = C.MLX_UINT64
	DtypeInt8      Dtype = C.MLX_INT8
	DtypeInt16     Dtype = C.MLX_INT16
	DtypeInt32     Dtype = C.MLX_INT32
	DtypeInt64     Dtype = C.MLX_INT64
	DtypeFloat16   Dtype = C.MLX_FLOAT16
	DtypeFloat32   Dtype = C.MLX_FLOAT32
	DtypeFloat64   Dtype = C.MLX_FLOAT64
	DtypeBFloat16  Dtype = C.MLX_BFLOAT16
	DtypeComplex64 Dtype = C.MLX_COMPLEX64
)

// String implements fmt.Stringer for Dtype
func (d Dtype) String() string {
	switch d {
	case DtypeBool:
		return "bool"
	case DtypeUint8:
		return "u8"
	case DtypeUint16:
		return "u16"
	case DtypeUint32:
		return "u32"
	case DtypeUint64:
		return "u64"
	case DtypeInt8:
		return "i8"
	case DtypeInt16:
		return "i16"
	case DtypeInt32:
		return "i32"
	case DtypeInt64:
		return "i64"
	case DtypeFloat16:
		return "f16"
	case DtypeFloat32:
		return "f32"
	case DtypeFloat64:
		return "f64"
	case DtypeBFloat16:
		return "bf16"
	case DtypeComplex64:
		return "c64"
	default:
		return "unknown"
	}
}

// Memory Management:
//
// All arrays are automatically tracked for cleanup. On Eval(), non-kept arrays are freed.
//
//	x := mlx.Matmul(input, weight)   // x is tracked for cleanup
//	mlx.Keep(x)                       // mark x as persistent
//	mlx.Eval(x)                       // eval + free non-kept arrays
//
// Use Keep() for arrays that should persist (weights, caches).
// Use Free() to mark a kept array for cleanup on next Eval().
//
// Note: Not goroutine-safe. Use from a single goroutine.

// Array wraps an MLX array handle.
// Arrays are freed via Eval() cleanup (deterministic) or GC (fallback).
type Array struct {
	c     C.mlx_array
	freed bool // Prevents double-free
	kept  bool // If true, survives Eval() cleanup
}

// arrays tracks all live arrays. On Eval(), non-kept arrays are freed.
// Not goroutine-safe.
var arrays = make([]*Array, 0, 4096)

// evalHandles is a pre-allocated slice for passing arrays to MLX eval.
var evalHandles = make([]C.mlx_array, 0, 64)

// arrayPool reduces allocations for intermediate arrays
var arrayPool = sync.Pool{
	New: func() any { return &Array{} },
}

func newArray(array C.mlx_array) *Array {
	// In compiled closures, MLX manages memory - skip Go tracking
	if InClosureCallback() {
		return &Array{c: array}
	}

	// Use pooled Array struct for efficiency
	a := arrayPool.Get().(*Array)
	a.c = array
	a.freed = false
	a.kept = false

	// Track in global list
	arrays = append(arrays, a)

	return a
}

// Collect uses reflection to find all *Array fields in a struct (recursively).
// Use this to automatically gather model weights, cache state, etc.
func Collect(v any) []*Array {
	var arrays []*Array
	seen := make(map[uintptr]bool)
	collect(reflect.ValueOf(v), &arrays, seen)
	return arrays
}

func collect(v reflect.Value, arrays *[]*Array, seen map[uintptr]bool) {
	if !v.IsValid() {
		return
	}

	// Handle pointers
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return
		}
		// Avoid infinite loops
		ptr := v.Pointer()
		if seen[ptr] {
			return
		}
		seen[ptr] = true

		// Check if it's *Array
		if arr, ok := v.Interface().(*Array); ok {
			if arr != nil && arr.c.ctx != nil {
				*arrays = append(*arrays, arr)
			}
			return
		}
		collect(v.Elem(), arrays, seen)
		return
	}

	// Handle structs
	if v.Kind() == reflect.Struct {
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			if field.CanInterface() {
				collect(field, arrays, seen)
			}
		}
		return
	}

	// Handle slices
	if v.Kind() == reflect.Slice {
		for i := 0; i < v.Len(); i++ {
			collect(v.Index(i), arrays, seen)
		}
		return
	}

	// Handle maps
	if v.Kind() == reflect.Map {
		for _, key := range v.MapKeys() {
			collect(v.MapIndex(key), arrays, seen)
		}
		return
	}

	// Handle interfaces
	if v.Kind() == reflect.Interface {
		if !v.IsNil() {
			collect(v.Elem(), arrays, seen)
		}
		return
	}
}

// FreeStruct releases all *Array fields in a struct (recursively).
// Use this to free model weights when unloading a model.
func FreeStruct(v any) {
	for _, arr := range Collect(v) {
		arr.Free()
	}
}

// Keep marks arrays to persist across Eval() cleanup.
// Kept arrays will NOT be freed when Eval() runs cleanup.
func Keep(arrays ...*Array) {
	for _, a := range arrays {
		if a != nil {
			a.kept = true
		}
	}
}

// cleanup frees non-kept arrays and compacts the live array list.
// Returns number of arrays freed.
func cleanup() int {
	freed := 0
	n := 0
	for _, a := range arrays {
		if a.kept {
			arrays[n] = a
			n++
		} else if a.c.ctx != nil && !a.freed {
			C.mlx_array_free(a.c)
			a.c.ctx = nil
			arrayPool.Put(a)
			freed++
		}
	}
	arrays = arrays[:n]
	return freed
}

// DebugArrays prints summary info about all tracked arrays.
func DebugArrays() {
	var totalBytes int64
	var keptCount, unkeptCount int
	for _, a := range arrays {
		if a.kept {
			keptCount++
		} else {
			unkeptCount++
		}
		totalBytes += a.Nbytes()
	}
	fmt.Printf("[DEBUG] Arrays: %d kept, %d unkept, %.2f GB total\n",
		keptCount, unkeptCount, float64(totalBytes)/(1024*1024*1024))
}

// DebugArraysVerbose prints detailed info about all tracked arrays, sorted by size.
func DebugArraysVerbose(topN int) {
	type arrayInfo struct {
		shape []int32
		dtype Dtype
		bytes int64
		kept  bool
	}

	var infos []arrayInfo
	var totalBytes int64
	for _, a := range arrays {
		bytes := a.Nbytes()
		infos = append(infos, arrayInfo{
			shape: a.Shape(),
			dtype: a.Dtype(),
			bytes: bytes,
			kept:  a.kept,
		})
		totalBytes += bytes
	}

	// Sort by size descending
	for i := 0; i < len(infos)-1; i++ {
		for j := i + 1; j < len(infos); j++ {
			if infos[j].bytes > infos[i].bytes {
				infos[i], infos[j] = infos[j], infos[i]
			}
		}
	}

	fmt.Printf("[DEBUG] %d arrays, %.2f GB total:\n", len(infos), float64(totalBytes)/(1024*1024*1024))
	for i, info := range infos {
		if i >= topN {
			break
		}
		keptStr := ""
		if info.kept {
			keptStr = " [kept]"
		}
		fmt.Printf("  %3d. %8.2f MB  %v %v%s\n",
			i+1, float64(info.bytes)/(1024*1024), info.shape, info.dtype, keptStr)
	}
}

// Eval synchronously evaluates arrays and cleans up non-kept arrays.
// Outputs are automatically kept (survive cleanup). Returns them for chaining.
func Eval(outputs ...*Array) []*Array {
	// Keep outputs so cleanup doesn't free them
	for _, o := range outputs {
		if o != nil {
			o.kept = true
		}
	}

	// Cleanup non-kept arrays
	cleanup()

	// Then evaluate
	if len(outputs) > 0 {
		evalHandles = evalHandles[:0]
		for _, o := range outputs {
			if o != nil {
				evalHandles = append(evalHandles, o.c)
			}
		}
		if len(evalHandles) > 0 {
			vec := C.mlx_vector_array_new_data(&evalHandles[0], C.size_t(len(evalHandles)))
			C.mlx_eval(vec)
			C.mlx_vector_array_free(vec)
		}
	}
	return outputs
}

// AsyncEval dispatches async evaluation and cleans up non-kept arrays.
// Outputs are automatically kept (survive cleanup).
func AsyncEval(outputs ...*Array) {
	// Keep outputs so cleanup doesn't free them
	for _, o := range outputs {
		if o != nil {
			o.kept = true
		}
	}

	// Cleanup non-kept arrays
	cleanup()

	// Then dispatch async eval
	if len(outputs) > 0 {
		evalHandles = evalHandles[:0]
		for _, o := range outputs {
			if o != nil {
				evalHandles = append(evalHandles, o.c)
			}
		}
		if len(evalHandles) > 0 {
			vec := C.mlx_vector_array_new_data(&evalHandles[0], C.size_t(len(evalHandles)))
			C.mlx_async_eval(vec)
			C.mlx_vector_array_free(vec)
		}
	}
}

// Sync waits for all async operations to complete (no cleanup).
func Sync() {
	C.mlx_synchronize(C.default_stream())
}

// Free marks this array for cleanup on the next Eval().
// The array is not immediately freed - cleanup happens during Eval().
//
// Pattern for loops:
//
//	oldLatents.Free()     // mark for cleanup
//	mlx.Eval(newLatents)  // frees old, evals new
func (a *Array) Free() {
	if a != nil {
		a.kept = false
	}
}

// Eval evaluates this single array and runs cleanup.
func (a *Array) Eval() *Array {
	Eval(a)
	return a
}

// Valid returns true if the array hasn't been freed.
func (a *Array) Valid() bool {
	return a != nil && a.c.ctx != nil
}

// Kept returns true if the array is marked to survive Eval() cleanup.
func (a *Array) Kept() bool {
	return a != nil && a.kept
}

func int32ToCInt(s []int32) *C.int {
	if len(s) == 0 {
		return nil
	}
	return (*C.int)(unsafe.Pointer(&s[0]))
}

// NewArray creates a new MLX array from float32 data
func NewArray(data []float32, shape []int32) *Array {
	handle := C.mlx_array_new_data(
		unsafe.Pointer(&data[0]),
		int32ToCInt(shape),
		C.int(len(shape)),
		C.MLX_FLOAT32,
	)
	return newArray(handle)
}

// NewArrayInt32 creates a new MLX array from int32 data
func NewArrayInt32(data []int32, shape []int32) *Array {
	handle := C.mlx_array_new_data(
		unsafe.Pointer(&data[0]),
		int32ToCInt(shape),
		C.int(len(shape)),
		C.MLX_INT32,
	)
	return newArray(handle)
}

// NewArrayFloat32 creates a new float32 array from data
func NewArrayFloat32(data []float32, shape []int32) *Array {
	return NewArray(data, shape)
}

// Zeros creates an array of zeros with optional dtype (default float32)
func Zeros(shape []int32, dtype ...Dtype) *Array {
	res := C.mlx_array_new()
	dt := DtypeFloat32
	if len(dtype) > 0 {
		dt = dtype[0]
	}
	C.mlx_zeros(&res, int32ToCInt(shape), C.size_t(len(shape)), C.mlx_dtype(dt), C.default_stream())
	return newArray(res)
}

// ZerosLike creates a zeros array with the same dtype as a.
// If shape is provided, uses that shape; otherwise uses a's shape.
func ZerosLike(a *Array, shape ...int32) *Array {
	res := C.mlx_array_new()
	if len(shape) == 0 {
		C.mlx_zeros_like(&res, a.c, C.default_stream())
	} else {
		dtype := a.Dtype()
		C.mlx_zeros(&res, int32ToCInt(shape), C.size_t(len(shape)), C.mlx_dtype(dtype), C.default_stream())
	}
	return newArray(res)
}

// Ones creates an array of ones
func Ones(shape ...int32) *Array {
	res := C.mlx_array_new()
	C.mlx_ones(&res, int32ToCInt(shape), C.size_t(len(shape)), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}

// Full creates an array filled with a value
func Full(value float32, shape ...int32) *Array {
	vals := C.mlx_array_new_float(C.float(value))
	res := C.mlx_array_new()
	C.mlx_full(&res, int32ToCInt(shape), C.size_t(len(shape)), vals, C.MLX_FLOAT32, C.default_stream())
	C.mlx_array_free(vals)
	return newArray(res)
}

// Arange creates a range of values
func Arange(start, stop, step float32) *Array {
	res := C.mlx_array_new()
	C.mlx_arange(&res, C.double(start), C.double(stop), C.double(step), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}

// Linspace creates evenly spaced values
func Linspace(start, stop float32, steps int32) *Array {
	res := C.mlx_array_new()
	C.mlx_linspace(&res, C.double(start), C.double(stop), C.int(steps), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}

// ============ Math Operations ============

// Add adds two arrays element-wise
func Add(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_add(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// AddRaw is like Add - kept for API compatibility (now identical to Add)
func AddRaw(a, b *Array) *Array {
	return Add(a, b)
}

// Sub subtracts two arrays element-wise
func Sub(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_subtract(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Mul multiplies two arrays element-wise
func Mul(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Div divides two arrays element-wise
func Div(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_divide(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Matmul performs matrix multiplication
func Matmul(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_matmul(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// AddMM computes: result = beta*c + alpha*(a @ b)
// This fuses bias addition with matmul into a single op.
func AddMM(c, a, b *Array, alpha, beta float32) *Array {
	res := C.mlx_array_new()
	C.mlx_addmm(&res, c.c, a.c, b.c, C.float(alpha), C.float(beta), C.default_stream())
	return newArray(res)
}

// Linear performs matrix multiplication: a @ weight
func Linear(a, weight *Array) *Array {
	return Matmul(a, weight)
}

// Sqrt computes element-wise square root
func Sqrt(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sqrt(&res, a.c, C.default_stream())
	return newArray(res)
}

// RSqrt computes element-wise reciprocal square root
func RSqrt(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_rsqrt(&res, a.c, C.default_stream())
	return newArray(res)
}

// Erf computes element-wise error function
func Erf(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_erf(&res, a.c, C.default_stream())
	return newArray(res)
}

// Exp computes element-wise exponential
func Exp(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_exp(&res, a.c, C.default_stream())
	return newArray(res)
}

// Log computes element-wise natural logarithm
func Log(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_log(&res, a.c, C.default_stream())
	return newArray(res)
}

// Sin computes element-wise sine
func Sin(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sin(&res, a.c, C.default_stream())
	return newArray(res)
}

// Cos computes element-wise cosine
func Cos(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_cos(&res, a.c, C.default_stream())
	return newArray(res)
}

// Neg negates the array
func Neg(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_negative(&res, a.c, C.default_stream())
	return newArray(res)
}

// Abs computes element-wise absolute value
func Abs(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_abs(&res, a.c, C.default_stream())
	return newArray(res)
}

// Square computes element-wise square
func Square(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_square(&res, a.c, C.default_stream())
	return newArray(res)
}

// Pow raises a to the power of b element-wise
func Pow(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_power(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Max computes element-wise maximum
func Max(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_maximum(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// Min computes element-wise minimum
func Min(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_minimum(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// scalarWithDtype creates a scalar array matching the dtype of a (critical for graph fusion!)
func scalarWithDtype(s float32, a *Array) C.mlx_array {
	// Create float32 scalar, then cast to match input dtype
	f32 := C.mlx_array_new_float(C.float(s))
	dtype := a.Dtype()
	if dtype == DtypeFloat32 {
		return f32 // No cast needed
	}
	// Cast to match input dtype
	casted := C.mlx_array_new()
	C.mlx_astype(&casted, f32, C.mlx_dtype(dtype), C.default_stream())
	C.mlx_array_free(f32)
	return casted
}

// AddScalar adds a scalar to an array (matches dtype for graph fusion)
func AddScalar(a *Array, s float32) *Array {
	scalar := scalarWithDtype(s, a)
	res := C.mlx_array_new()
	C.mlx_add(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// MulScalar multiplies an array by a scalar (matches dtype for graph fusion)
func MulScalar(a *Array, s float32) *Array {
	scalar := scalarWithDtype(s, a)
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// DivScalar divides an array by a scalar (matches dtype for graph fusion)
func DivScalar(a *Array, s float32) *Array {
	scalar := scalarWithDtype(s, a)
	res := C.mlx_array_new()
	C.mlx_divide(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// DivScalarInt divides an int array by an int scalar (regular division, may return float)
func DivScalarInt(a *Array, s int32) *Array {
	scalar := C.mlx_array_new_int(C.int(s))
	res := C.mlx_array_new()
	C.mlx_divide(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// FloorDivideScalar performs integer floor division (a // s), preserving int dtype
func FloorDivideScalar(a *Array, s int32) *Array {
	scalar := C.mlx_array_new_int(C.int(s))
	res := C.mlx_array_new()
	C.mlx_floor_divide(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// ============ Reduction Operations ============

// Sum reduces along an axis
func Sum(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_sum_axis(&res, a.c, C.int(axis), C._Bool(keepdims), C.default_stream())
	return newArray(res)
}

// SumAll reduces the entire array to a scalar
func SumAll(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sum(&res, a.c, false, C.default_stream())
	return newArray(res)
}

// Mean reduces along an axis
func Mean(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_mean_axis(&res, a.c, C.int(axis), C._Bool(keepdims), C.default_stream())
	return newArray(res)
}

// MeanAll reduces the entire array to a scalar
func MeanAll(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_mean(&res, a.c, false, C.default_stream())
	return newArray(res)
}

// Var computes variance along an axis
func Var(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_var_axis(&res, a.c, C.int(axis), C._Bool(keepdims), 0, C.default_stream())
	return newArray(res)
}

// Argmax returns indices of maximum values along an axis
func Argmax(a *Array, axis int, keepdims bool) *Array {
	res := C.mlx_array_new()
	C.mlx_argmax_axis(&res, a.c, C.int(axis), C._Bool(keepdims), C.default_stream())
	return newArray(res)
}

// ArgmaxAll returns the index of the maximum element (flattened).
// Triggers cleanup of non-kept arrays.
func ArgmaxAll(a *Array) int32 {
	cleanup()
	// Flatten, then argmax with keepdims=false
	flat := C.mlx_array_new()
	C.mlx_flatten(&flat, a.c, 0, -1, C.default_stream())
	res := C.mlx_array_new()
	C.mlx_argmax(&res, flat, false, C.default_stream())
	C.mlx_array_eval(res)
	var val C.int32_t
	C.mlx_array_item_int32(&val, res)
	C.mlx_array_free(flat)
	C.mlx_array_free(res)
	return int32(val)
}

// Reshape reshapes the array
func Reshape(a *Array, shape ...int32) *Array {
	res := C.mlx_array_new()
	C.mlx_reshape(&res, a.c, int32ToCInt(shape), C.size_t(len(shape)), C.default_stream())
	return newArray(res)
}

// Transpose permutes the dimensions
func Transpose(a *Array, axes ...int) *Array {
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	res := C.mlx_array_new()
	C.mlx_transpose_axes(&res, a.c, &cAxes[0], C.size_t(len(axes)), C.default_stream())
	return newArray(res)
}

// AsStrided creates a view with custom strides. Useful for fusing reshape+transpose.
func AsStrided(a *Array, shape []int32, strides []int64, offset int64) *Array {
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	cStrides := make([]C.int64_t, len(strides))
	for i, s := range strides {
		cStrides[i] = C.int64_t(s)
	}
	res := C.mlx_array_new()
	C.mlx_as_strided(&res, a.c, &cShape[0], C.size_t(len(shape)), &cStrides[0], C.size_t(len(strides)), C.size_t(offset), C.default_stream())
	return newArray(res)
}

// ExpandDims adds a dimension at the specified axis
func ExpandDims(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_expand_dims(&res, a.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Squeeze removes a dimension at the specified axis
func Squeeze(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_squeeze_axis(&res, a.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Flatten flattens the array to 1D
func Flatten(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_flatten(&res, a.c, 0, -1, C.default_stream())
	return newArray(res)
}

// FlattenRange flattens consecutive axes from start_axis to end_axis (intermediates)
func FlattenRange(a *Array, startAxis, endAxis int) *Array {
	res := C.mlx_array_new()
	C.mlx_flatten(&res, a.c, C.int(startAxis), C.int(endAxis), C.default_stream())
	return newArray(res)
}

// View reinterprets the array with a new dtype (no data copy)
func View(a *Array, dtype int) *Array {
	res := C.mlx_array_new()
	C.mlx_view(&res, a.c, C.mlx_dtype(dtype), C.default_stream())
	return newArray(res)
}

// Contiguous returns a contiguous copy of the array (row-major)
func Contiguous(a *Array) *Array {
	res := C.mlx_array_new()
	// Use allow_col=false to force row-major contiguous layout
	C.mlx_contiguous(&res, a.c, false, C.default_stream())
	return newArray(res)
}

// Clip clips values to [min, max]. Pass nil for no bound on that side.
func Clip(a *Array, aMin, aMax *Array) *Array {
	res := C.mlx_array_new()
	var minH, maxH C.mlx_array
	if aMin != nil {
		minH = aMin.c
	}
	if aMax != nil {
		maxH = aMax.c
	}
	C.mlx_clip(&res, a.c, minH, maxH, C.default_stream())
	return newArray(res)
}

// ClipScalar clips array values using scalar bounds (matches dtype for graph fusion)
// Pass math.NaN() or set hasMin/hasMax to false for unbounded
func ClipScalar(a *Array, minVal, maxVal float32, hasMin, hasMax bool) *Array {
	var minArr, maxArr C.mlx_array
	if hasMin {
		minArr = scalarWithDtype(minVal, a)
	}
	if hasMax {
		maxArr = scalarWithDtype(maxVal, a)
	}
	res := C.mlx_array_new()
	C.mlx_clip(&res, a.c, minArr, maxArr, C.default_stream())
	if hasMin {
		C.mlx_array_free(minArr)
	}
	if hasMax {
		C.mlx_array_free(maxArr)
	}
	return newArray(res)
}

// GreaterEqual returns element-wise a >= b
func GreaterEqual(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_greater_equal(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// LessArray returns element-wise a < b
func LessArray(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_less(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// LogicalAnd returns element-wise a && b
func LogicalAnd(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_logical_and(&res, a.c, b.c, C.default_stream())
	return newArray(res)
}

// AllClose returns true if all elements of a and b are within tolerance.
// Uses rtol (relative tolerance) and atol (absolute tolerance):
// |a - b| <= atol + rtol * |b|
func AllClose(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_allclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(false), C.default_stream())
	return newArray(res)
}

// AllCloseEqualNaN is like AllClose but treats NaN as equal to NaN.
func AllCloseEqualNaN(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_allclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(true), C.default_stream())
	return newArray(res)
}

// ArrayEqual returns true if arrays have same shape and all elements are equal.
func ArrayEqual(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_array_equal(&res, a.c, b.c, C.bool(false), C.default_stream())
	return newArray(res)
}

// ArrayEqualNaN is like ArrayEqual but treats NaN as equal to NaN.
func ArrayEqualNaN(a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_array_equal(&res, a.c, b.c, C.bool(true), C.default_stream())
	return newArray(res)
}

// IsClose returns element-wise bool array indicating if values are within tolerance.
// |a - b| <= atol + rtol * |b|
func IsClose(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_isclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(false), C.default_stream())
	return newArray(res)
}

// IsCloseEqualNaN is like IsClose but treats NaN as equal to NaN.
func IsCloseEqualNaN(a, b *Array, rtol, atol float64) *Array {
	res := C.mlx_array_new()
	C.mlx_isclose(&res, a.c, b.c, C.double(rtol), C.double(atol), C.bool(true), C.default_stream())
	return newArray(res)
}

// ReduceMax reduces array to max value over all dimensions.
func ReduceMax(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_max(&res, a.c, C.bool(false), C.default_stream())
	return newArray(res)
}

// ArangeInt creates an array with values from start to stop with step and specified dtype
func ArangeInt(start, stop, step int32, dtype Dtype) *Array {
	res := C.mlx_array_new()
	C.mlx_arange(&res, C.double(start), C.double(stop), C.double(step), C.mlx_dtype(dtype), C.default_stream())
	return newArray(res)
}

// Concatenate concatenates arrays along an axis
func Concatenate(arrays []*Array, axis int) *Array {
	handles := make([]C.mlx_array, len(arrays))
	for i, arr := range arrays {
		handles[i] = arr.c
	}
	vec := C.mlx_vector_array_new_data(&handles[0], C.size_t(len(handles)))
	res := C.mlx_array_new()
	C.mlx_concatenate_axis(&res, vec, C.int(axis), C.default_stream())
	C.mlx_vector_array_free(vec)
	return newArray(res)
}

// Concat is a convenience function to concatenate two arrays
func Concat(a, b *Array, axis int) *Array {
	return Concatenate([]*Array{a, b}, axis)
}

// Slice slices the array
func Slice(a *Array, start, stop []int32) *Array {
	n := len(start)
	cStart := make([]C.int, n)
	cStop := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = 1 // Default stride of 1
	}
	res := C.mlx_array_new()
	C.mlx_slice(&res, a.c, &cStart[0], C.size_t(n), &cStop[0], C.size_t(n), &cStrides[0], C.size_t(n), C.default_stream())
	return newArray(res)
}

// SliceStride slices with start:stop:stride like Python a[start:stop:stride]
func SliceStride(a *Array, start, stop, strides []int32) *Array {
	cStart := make([]C.int, len(start))
	cStop := make([]C.int, len(stop))
	cStrides := make([]C.int, len(strides))
	for i := range start {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = C.int(strides[i])
	}
	res := C.mlx_array_new()
	C.mlx_slice(&res, a.c, &cStart[0], C.size_t(len(start)), &cStop[0], C.size_t(len(stop)), &cStrides[0], C.size_t(len(strides)), C.default_stream())
	return newArray(res)
}

// Tile repeats the array along each dimension
func Tile(a *Array, reps []int32) *Array {
	res := C.mlx_array_new()
	C.mlx_tile(&res, a.c, int32ToCInt(reps), C.size_t(len(reps)), C.default_stream())
	return newArray(res)
}

// BroadcastTo broadcasts an array to a given shape
func BroadcastTo(a *Array, shape []int32) *Array {
	res := C.mlx_array_new()
	C.mlx_broadcast_to(&res, a.c, int32ToCInt(shape), C.size_t(len(shape)), C.default_stream())
	return newArray(res)
}

// ============ Neural Network Operations ============

// Softmax computes softmax along an axis
func Softmax(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_softmax_axis(&res, a.c, C.int(axis), false, C.default_stream())
	return newArray(res)
}

// Take gathers elements along an axis using indices
func Take(a *Array, indices *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_take_axis(&res, a.c, indices.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Argsort returns indices that would sort the array along an axis
func Argsort(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_argsort_axis(&res, a.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Sigmoid computes element-wise sigmoid
func Sigmoid(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_sigmoid(&res, a.c, C.default_stream())
	return newArray(res)
}

// ReLU computes element-wise ReLU: max(0, x)
func ReLU(a *Array) *Array {
	// ReLU = maximum(x, 0) - mlx-c doesn't have mlx_relu, but we can use maximum
	zero := C.mlx_array_new_float(0.0)
	res := C.mlx_array_new()
	C.mlx_maximum(&res, a.c, zero, C.default_stream())
	C.mlx_array_free(zero)
	return newArray(res)
}

// SiLU computes element-wise SiLU (Swish): x * sigmoid(x)
func SiLU(a *Array) *Array {
	// SiLU = x * sigmoid(x)
	sig := C.mlx_array_new()
	C.mlx_sigmoid(&sig, a.c, C.default_stream())
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, sig, C.default_stream())
	C.mlx_array_free(sig)
	return newArray(res)
}

// GELU computes element-wise GELU (Gaussian Error Linear Unit)
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
func GELU(a *Array) *Array {
	sqrt2 := C.mlx_array_new_float(1.4142135623730951)
	scaled := C.mlx_array_new()
	C.mlx_divide(&scaled, a.c, sqrt2, C.default_stream())
	erfd := C.mlx_array_new()
	C.mlx_erf(&erfd, scaled, C.default_stream())
	one := C.mlx_array_new_float(1.0)
	erfdPlusOne := C.mlx_array_new()
	C.mlx_add(&erfdPlusOne, erfd, one, C.default_stream())
	half := C.mlx_array_new_float(0.5)
	halfErfdPlusOne := C.mlx_array_new()
	C.mlx_multiply(&halfErfdPlusOne, half, erfdPlusOne, C.default_stream())
	res := C.mlx_array_new()
	C.mlx_multiply(&res, a.c, halfErfdPlusOne, C.default_stream())
	C.mlx_array_free(sqrt2)
	C.mlx_array_free(scaled)
	C.mlx_array_free(erfd)
	C.mlx_array_free(one)
	C.mlx_array_free(erfdPlusOne)
	C.mlx_array_free(half)
	C.mlx_array_free(halfErfdPlusOne)
	return newArray(res)
}

// Tanh computes element-wise tanh
func Tanh(a *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_tanh(&res, a.c, C.default_stream())
	return newArray(res)
}

// RMSNorm computes RMS normalization using mlx.fast
func RMSNorm(x, weight *Array, eps float32) *Array {
	res := C.mlx_array_new()
	C.mlx_fast_rms_norm(&res, x.c, weight.c, C.float(eps), C.default_stream())
	return newArray(res)
}

// RMSNormNoWeight applies RMS normalization without a weight
// x * rsqrt(mean(x^2) + eps)
// Uses mlx_fast_rms_norm with ones weight for f32 accumulation precision
func RMSNormNoWeight(x *Array, eps float32) *Array {
	// Create weight of ones matching last dimension
	lastDim := x.Shape()[len(x.Shape())-1]
	ones := AsType(Full(1.0, lastDim), x.Dtype())
	return RMSNorm(x, ones, eps)
}

// LayerNorm applies layer normalization without learnable params
// (x - mean) / sqrt(var + eps)
func LayerNorm(x *Array, eps float32) *Array {
	return LayerNormWithWeightBias(x, nil, nil, eps)
}

// LayerNormWithWeightBias computes layer normalization using mlx.fast
// weight and bias can be nil for elementwise_affine=False
func LayerNormWithWeightBias(x, weight, bias *Array, eps float32) *Array {
	res := C.mlx_array_new()
	var wc, bc C.mlx_array
	if weight != nil {
		wc = weight.c
	}
	if bias != nil {
		bc = bias.c
	}
	C.mlx_fast_layer_norm(&res, x.c, wc, bc, C.float(eps), C.default_stream())
	return newArray(res)
}

// RoPE applies rotary position embeddings using mlx.fast
func RoPE(x *Array, dims int, traditional bool, base, scale float32, offset int) *Array {
	res := C.mlx_array_new()
	optBase := C.mlx_optional_float{value: C.float(base), has_value: true}
	C.mlx_fast_rope(&res, x.c, C.int(dims), C._Bool(traditional), optBase, C.float(scale), C.int(offset), C.mlx_array{}, C.default_stream())
	return newArray(res)
}

// RoPEWithFreqs applies rotary position embeddings with custom frequencies (for YaRN)
// freqs is required - use RoPE() if you don't have custom frequencies
func RoPEWithFreqs(x, freqs *Array, dims int, traditional bool, scale float32, offset int) *Array {
	res := C.mlx_array_new()
	optBase := C.mlx_optional_float{has_value: false} // No base when using freqs
	C.mlx_fast_rope(&res, x.c, C.int(dims), C._Bool(traditional), optBase, C.float(scale), C.int(offset), freqs.c, C.default_stream())
	return newArray(res)
}

// ============ Indexing ============

// EmbeddingLookup performs embedding lookup (gathers from table)
// table: [vocab_size, hidden_size], indices: [batch, seq_len]
// returns: [batch, seq_len, hidden_size]
func EmbeddingLookup(table, indices *Array) *Array {
	return Take(table, indices, 0)
}

// Gather gathers elements using indices - simplified to use take axis 0
func Gather(a, indices *Array) *Array {
	return Take(a, indices, 0)
}

// ============ Array Properties ============

// Ndim returns the number of dimensions
func (a *Array) Ndim() int {
	return int(C.mlx_array_ndim(a.c))
}

// Size returns the total number of elements
func (a *Array) Size() int {
	return int(C.mlx_array_size(a.c))
}

// IsContiguous returns whether the array's data is contiguous in memory.
// Non-contiguous arrays (e.g., from SliceStride) must call Contiguous() before Data().
func (a *Array) IsContiguous() bool {
	var res C.bool
	C._mlx_array_is_contiguous(&res, a.c)
	return bool(res)
}

// Dim returns the size of a dimension
func (a *Array) Dim(axis int) int32 {
	return int32(C.mlx_array_dim(a.c, C.int(axis)))
}

// Shape returns the shape as a slice
func (a *Array) Shape() []int32 {
	ndim := a.Ndim()
	shape := make([]int32, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = a.Dim(i)
	}
	return shape
}

// IsValid returns true if the array hasn't been freed
func (a *Array) IsValid() bool {
	return a != nil && a.c.ctx != nil
}

// Dtype returns the data type
func (a *Array) Dtype() Dtype {
	return Dtype(C.mlx_array_dtype(a.c))
}

// Nbytes returns the total size in bytes
func (a *Array) Nbytes() int64 {
	return int64(a.Size()) * a.Dtype().ItemSize()
}

// ItemSize returns the size in bytes of one element for this dtype
func (d Dtype) ItemSize() int64 {
	switch d {
	case DtypeBool, DtypeUint8, DtypeInt8:
		return 1
	case DtypeUint16, DtypeInt16, DtypeFloat16, DtypeBFloat16:
		return 2
	case DtypeUint32, DtypeInt32, DtypeFloat32:
		return 4
	case DtypeUint64, DtypeInt64, DtypeFloat64, DtypeComplex64:
		return 8
	default:
		return 4
	}
}

// ============ Data Access ============

// Data copies the float32 data out of the array.
// Note: For non-contiguous arrays (e.g., from SliceStride), call Contiguous() first.
// Note: Arrays of other dtypes (bf16, f16, etc) are automatically converted to float32.
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) Data() []float32 {
	cleanup()
	size := a.Size()
	if size == 0 {
		return nil
	}

	arr := a
	if a.Dtype() != DtypeFloat32 {
		arr = AsType(a, DtypeFloat32)
		arr.Eval()
		// Cast array will be cleaned up on next Eval
	}

	ptr := C.mlx_array_data_float32(arr.c)
	if ptr == nil {
		return nil
	}
	data := make([]float32, size)
	copy(data, unsafe.Slice((*float32)(unsafe.Pointer(ptr)), size))
	return data
}

// Item returns the scalar value from a 0-dimensional array.
// Converts to float32 if necessary. Triggers cleanup.
func (a *Array) Item() float32 {
	data := a.Data() // Data() calls cleanup()
	if len(data) == 0 {
		return 0
	}
	return data[0]
}

// DataInt32 copies the int32 data out of the array.
// Note: For non-contiguous arrays (e.g., from SliceStride), call Contiguous() first.
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) DataInt32() []int32 {
	cleanup()
	size := a.Size()
	if size == 0 {
		return nil
	}
	ptr := C.mlx_array_data_int32(a.c)
	if ptr == nil {
		return nil
	}
	data := make([]int32, size)
	copy(data, unsafe.Slice((*int32)(unsafe.Pointer(ptr)), size))
	return data
}

// ItemInt32 gets a single scalar value efficiently (no array copy).
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) ItemInt32() int32 {
	cleanup()
	var val C.int32_t
	C.mlx_array_item_int32(&val, a.c)
	return int32(val)
}

// Bytes copies the raw bytes out of the array without type conversion.
// Works with common dtypes (float32, int32, uint32, uint8).
// For non-contiguous arrays, call Contiguous() first.
// Note: Triggers cleanup of non-kept arrays.
func (a *Array) Bytes() []byte {
	cleanup()
	nbytes := a.Nbytes()
	if nbytes == 0 {
		return nil
	}

	// Get raw pointer based on dtype
	var ptr unsafe.Pointer
	switch a.Dtype() {
	case DtypeFloat32:
		ptr = unsafe.Pointer(C.mlx_array_data_float32(a.c))
	case DtypeInt32:
		ptr = unsafe.Pointer(C.mlx_array_data_int32(a.c))
	case DtypeUint32:
		ptr = unsafe.Pointer(C.mlx_array_data_uint32(a.c))
	case DtypeUint8:
		ptr = unsafe.Pointer(C.mlx_array_data_uint8(a.c))
	default:
		// For other types (bf16, f16, etc), convert to float32
		arr := AsType(a, DtypeFloat32)
		arr.Eval()
		ptr = unsafe.Pointer(C.mlx_array_data_float32(arr.c))
		nbytes = arr.Nbytes()
	}

	if ptr == nil {
		return nil
	}
	data := make([]byte, nbytes)
	copy(data, unsafe.Slice((*byte)(ptr), nbytes))
	return data
}

// ============ Utility ============

// String returns a string representation
func (a *Array) String() string {
	shape := a.Shape()
	size := a.Size()
	if size <= 20 {
		data := a.Data()
		return fmt.Sprintf("Array(shape=%v, data=%v)", shape, data)
	}
	return fmt.Sprintf("Array(shape=%v, size=%d)", shape, size)
}

// ============ Safetensors Support ============

// NewArrayFromBytes creates an array from raw bytes (for safetensors)
func NewArrayFromBytes(data []byte, shape []int32, dtype Dtype) *Array {
	cData := unsafe.Pointer(&data[0])
	intShape := make([]C.int, len(shape))
	for i, s := range shape {
		intShape[i] = C.int(s)
	}
	handle := C.mlx_array_new_data(cData, &intShape[0], C.int(len(shape)), C.mlx_dtype(dtype))
	return newArray(handle)
}

// ============ Device Control ============

// SetDefaultDeviceGPU sets the default device to GPU (Metal)
func SetDefaultDeviceGPU() {
	dev := C.mlx_device_new_type(C.MLX_GPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
}

// SetDefaultDeviceCPU sets the default device to CPU
func SetDefaultDeviceCPU() {
	dev := C.mlx_device_new_type(C.MLX_CPU, 0)
	C.mlx_set_default_device(dev)
	C.mlx_device_free(dev)
}

// MetalIsAvailable returns true if Metal GPU is available
func MetalIsAvailable() bool {
	var available C._Bool
	C.mlx_metal_is_available(&available)
	return bool(available)
}

// MetalStartCapture starts a GPU trace capture to the given file path.
// The path must not already exist. Run with MTL_CAPTURE_ENABLED=1 env var.
// Open the resulting .gputrace file in Xcode for analysis.
func MetalStartCapture(path string) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	C.mlx_metal_start_capture(cPath)
}

// MetalStopCapture stops the current GPU trace capture.
func MetalStopCapture() {
	C.mlx_metal_stop_capture()
}

// GPUIsAvailable returns true if any GPU (Metal or CUDA) is available
func GPUIsAvailable() bool {
	// On Linux with CUDA build, GPU is available
	// On macOS, check Metal availability
	if MetalIsAvailable() {
		return true
	}
	// CUDA is available if we compiled with CUDA support (Linux)
	return runtime.GOOS == "linux"
}

// GetDefaultDeviceType returns the current default device (0=CPU, 1=GPU)
func GetDefaultDeviceType() int {
	var dev C.mlx_device
	C.mlx_get_default_device(&dev)
	var devType C.mlx_device_type
	C.mlx_device_get_type(&devType, dev)
	C.mlx_device_free(dev)
	return int(devType)
}

// Synchronize waits for all GPU operations to complete
func Synchronize() {
	C.mlx_synchronize(C.default_stream())
}

// ScaledDotProductAttention computes optimized attention using GPU kernel
// Q, K, V should be [batch, heads, seq, head_dim]
func ScaledDotProductAttention(q, k, v *Array, scale float32, causalMask bool) *Array {
	res := C.mlx_array_new()
	maskMode := "" // empty string for no mask
	if causalMask {
		maskMode = "causal"
	}
	cMaskMode := C.CString(maskMode)
	defer C.free(unsafe.Pointer(cMaskMode))
	C.mlx_fast_scaled_dot_product_attention(&res, q.c, k.c, v.c, C.float(scale), cMaskMode, C.mlx_array{}, C.mlx_array{}, C.default_stream())
	return newArray(res)
}

// ScaledDotProductAttentionWithSinks computes attention with sinks support
// maskMode: "causal", "sliding_window", or "" for none
// mask: optional attention mask array (nil for none)
// sinks: attention sinks array (nil for none)
func ScaledDotProductAttentionWithSinks(q, k, v *Array, scale float32, maskMode string, mask, sinks *Array) *Array {
	res := C.mlx_array_new()
	cMaskMode := C.CString(maskMode)
	defer C.free(unsafe.Pointer(cMaskMode))
	var maskH, sinksH C.mlx_array
	if mask != nil {
		maskH = mask.c
	}
	if sinks != nil {
		sinksH = sinks.c
	}
	C.mlx_fast_scaled_dot_product_attention(&res, q.c, k.c, v.c, C.float(scale), cMaskMode, maskH, sinksH, C.default_stream())
	return newArray(res)
}

// ============ Native Safetensors Loading ============

// SafetensorsFile represents a loaded safetensors file
type SafetensorsFile struct {
	arrays   C.mlx_map_string_to_array
	metadata C.mlx_map_string_to_string
}

// LoadSafetensorsNative loads a safetensors file using MLX's optimized loader
// Note: Uses CPU stream because Load primitive only runs on CPU
func LoadSafetensorsNative(path string) (*SafetensorsFile, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var arrays C.mlx_map_string_to_array
	var metadata C.mlx_map_string_to_string
	if C.mlx_load_safetensors(&arrays, &metadata, cPath, C.cpu_stream()) != 0 {
		return nil, fmt.Errorf("failed to load safetensors: %s", path)
	}
	return &SafetensorsFile{arrays: arrays, metadata: metadata}, nil
}

// Get retrieves a tensor by name
func (s *SafetensorsFile) Get(name string) *Array {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var arr C.mlx_array
	if C.mlx_map_string_to_array_get(&arr, s.arrays, cName) != 0 {
		return nil
	}
	if arr.ctx == nil {
		return nil
	}
	return newArray(arr)
}

// Set replaces a tensor in the map (like Python's weights[k] = v)
func (s *SafetensorsFile) Set(name string, arr *Array) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.mlx_map_string_to_array_insert(s.arrays, cName, arr.c)
}

// Count returns the number of tensors (not directly available, would need iterator)
func (s *SafetensorsFile) Count() int {
	// mlx-c doesn't have a direct count - would need to iterate
	return 0
}

// Free releases the safetensors file
func (s *SafetensorsFile) Free() {
	C.mlx_map_string_to_array_free(s.arrays)
	C.mlx_map_string_to_string_free(s.metadata)
}

// SaveSafetensors saves arrays to a safetensors file using MLX's native implementation.
// This correctly handles all dtypes including uint32 for quantized weights.
func SaveSafetensors(path string, arrays map[string]*Array) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	// Create the map
	cArrays := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(cArrays)

	// Add each array to the map
	for name, arr := range arrays {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(cArrays, cName, arr.c)
		C.free(unsafe.Pointer(cName))
	}

	// Create empty metadata (optional)
	cMeta := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(cMeta)

	// Save
	if C.mlx_save_safetensors(cPath, cArrays, cMeta) != 0 {
		return fmt.Errorf("failed to save safetensors: %s", path)
	}
	return nil
}

// ============ NPY Loading ============

// LoadNpy loads a numpy array from an npy file
// Note: Uses CPU stream because Load primitive only runs on CPU
func LoadNpy(path string) (*Array, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var arr C.mlx_array
	if C.mlx_load(&arr, cPath, C.cpu_stream()) != 0 {
		return nil, fmt.Errorf("failed to load npy: %s", path)
	}
	if arr.ctx == nil {
		return nil, fmt.Errorf("failed to load npy: %s", path)
	}
	return newArray(arr), nil
}

// ============ Slice Update ============

// SliceUpdate updates a slice of the array with new values
func SliceUpdate(a, update *Array, start, stop []int32) *Array {
	n := len(start)
	cStart := make([]C.int, n)
	cStop := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = 1 // Default stride of 1
	}
	res := C.mlx_array_new()
	C.mlx_slice_update(&res, a.c, update.c, &cStart[0], C.size_t(n), &cStop[0], C.size_t(n), &cStrides[0], C.size_t(n), C.default_stream())
	return newArray(res)
}

// SliceUpdateInplace updates a slice and returns a new array.
// Note: Despite the name, this is NOT in-place - MLX arrays are immutable.
// The caller must use the returned value.
func SliceUpdateInplace(a, update *Array, start, stop []int32) *Array {
	return SliceUpdate(a, update, start, stop)
}

// ============ Optimized Operations ============

// SampleArgmax gets the last logit position and returns argmax (fused operation)
func SampleArgmax(logits *Array) int32 {
	result := Argmax(logits, -1, false)
	return result.ItemInt32()
}

// ArgmaxKeepArray returns argmax as an Array (for pipelining, no sync)
// This is like mlx-lm's sampler that returns y as an array, not .item()
func ArgmaxKeepArray(logits *Array) *Array {
	// For greedy decoding: logits shape is [1, 1, vocab]
	// We want argmax over vocab dimension, return shape []
	return Argmax(logits, -1, false)
}

// RandomState is the global PRNG state, analogous to mx.random.state in Python.
// It's a slice containing a single key array. Random functions use and update this state.
//
// Thread safety: Protected by randomStateMu, mimicking Python's GIL behavior.
// All random functions that use global state acquire this lock.
var RandomState = []*Array{nil}
var randomStateMu sync.Mutex

var mlxInitialized bool
var mlxInitError error

// InitMLX initializes the MLX library by dynamically loading libmlxc.
// This must be called before using any MLX functions.
// Returns an error if the library cannot be loaded.
func InitMLX() error {
	if mlxInitialized {
		return mlxInitError
	}

	// Try to load the MLX dynamic library
	ret := C.mlx_dynamic_init()
	if ret != 0 {
		errMsg := C.GoString(C.mlx_dynamic_error())
		mlxInitError = fmt.Errorf("failed to initialize MLX: %s", errMsg)
		return mlxInitError
	}

	// Initialize all function pointers via dlsym
	handle := C.mlx_get_handle()
	ret = C.mlx_load_functions(handle)
	if ret != 0 {
		mlxInitError = fmt.Errorf("failed to load MLX function symbols")
		return mlxInitError
	}

	mlxInitialized = true
	mlxInitError = nil
	return nil
}

// IsMLXAvailable returns whether MLX was successfully initialized
func IsMLXAvailable() bool {
	return mlxInitialized && mlxInitError == nil
}

// GetMLXInitError returns any error that occurred during MLX initialization
func GetMLXInitError() error {
	return mlxInitError
}

func init() {
	// Initialize MLX dynamic library first
	if err := InitMLX(); err != nil {
		// Don't panic in init - let the caller handle the error
		// Store the error for later retrieval
		mlxInitError = err
		return
	}

	// Lock main goroutine to OS thread for CUDA context stability.
	// CUDA contexts are bound to threads; Go can migrate goroutines between threads.
	runtime.LockOSThread()
	RandomState[0] = RandomKey(uint64(time.Now().UnixMilli()))
	Keep(RandomState[0]) // Global state should persist
}

// RandomKey creates a PRNG key from a seed
func RandomKey(seed uint64) *Array {
	var res C.mlx_array
	C.mlx_random_key(&res, C.uint64_t(seed))
	return newArray(res)
}

// RandomSplit splits a PRNG key into two new keys
func RandomSplit(key *Array) (*Array, *Array) {
	var key1, key2 C.mlx_array
	C.mlx_random_split(&key1, &key2, key.c, C.default_stream())
	return newArray(key1), newArray(key2)
}

// RandomCategoricalWithKey samples from categorical distribution using provided key.
func RandomCategoricalWithKey(logits, key *Array, axis int, numSamples int) *Array {
	res := C.mlx_array_new()
	C.mlx_random_categorical_num_samples(&res, logits.c, C.int(axis), C.int(numSamples), key.c, C.default_stream())
	return newArray(res)
}

// RandomCategorical samples using global RandomState.
// For simple scripts - production code should use RandomCategoricalWithKey with explicit key management.
func RandomCategorical(logits *Array, axis int, numSamples int) *Array {
	randomStateMu.Lock()
	oldKey := RandomState[0]
	key1, key2 := RandomSplit(oldKey)
	Keep(key1) // key1 becomes the new global state
	oldKey.Free()
	RandomState[0] = key1
	randomStateMu.Unlock()
	return RandomCategoricalWithKey(logits, key2, axis, numSamples)
}

// RandomNormal creates a random normal (Gaussian) tensor in float32
func RandomNormal(shape []int32, seed uint64) *Array {
	return RandomNormalWithDtype(shape, seed, DtypeFloat32)
}

// RandomNormalWithDtype creates a random normal (Gaussian) tensor with specified dtype
func RandomNormalWithDtype(shape []int32, seed uint64, dtype Dtype) *Array {
	key := RandomKey(seed)
	res := C.mlx_array_new()
	C.mlx_random_normal(&res, int32ToCInt(shape), C.size_t(len(shape)), C.mlx_dtype(dtype), 0.0, 1.0, key.c, C.default_stream())
	return newArray(res)
}

// RandomUniform generates uniform random values in [0, 1) with the given shape
func RandomUniform(shape []int32, seed uint64) *Array {
	key := RandomKey(seed)
	low := C.mlx_array_new_float(0.0)
	high := C.mlx_array_new_float(1.0)
	res := C.mlx_array_new()
	C.mlx_random_uniform(&res, low, high, int32ToCInt(shape), C.size_t(len(shape)), C.MLX_FLOAT32, key.c, C.default_stream())
	C.mlx_array_free(low)
	C.mlx_array_free(high)
	return newArray(res)
}

// Conv2d performs 2D convolution
// input: [N, H, W, C], weight: [O, kH, kW, C]  (MLX uses NHWC layout)
// Returns: [N, H', W', O]
func Conv2d(input, weight *Array, stride, padding int32) *Array {
	res := C.mlx_array_new()
	C.mlx_conv2d(&res, input.c, weight.c, C.int(stride), C.int(stride), C.int(padding), C.int(padding), 1, 1, 1, C.default_stream())
	return newArray(res)
}

// Conv3d performs 3D convolution
// input: [N, D, H, W, C], weight: [O, kD, kH, kW, C]  (MLX uses NDHWC layout)
// Returns: [N, D', H', W', O]
func Conv3d(input, weight *Array, strideD, strideH, strideW, padD, padH, padW int32) *Array {
	res := C.mlx_array_new()
	C.mlx_conv3d(&res, input.c, weight.c, C.int(strideD), C.int(strideH), C.int(strideW), C.int(padD), C.int(padH), C.int(padW), 1, 1, 1, 1, C.default_stream())
	return newArray(res)
}

// ============ Compilation Control ============

// EnableCompile enables global compilation/graph fusion
func EnableCompile() {
	C.mlx_enable_compile()
}

// DisableCompile disables global compilation
func DisableCompile() {
	C.mlx_disable_compile()
}

// SetCompileMode sets the compile mode
// 0=disabled, 1=no_simplify, 2=no_fuse, 3=enabled
func SetCompileMode(mode int) {
	C.mlx_set_compile_mode(C.mlx_compile_mode(mode))
}

// ============ Stream Control ============

// Stream represents an MLX execution stream
type Stream struct {
	c C.mlx_stream
}

// NewStream creates a new execution stream on the default device
func NewStream() *Stream {
	var dev C.mlx_device
	C.mlx_get_default_device(&dev)
	stream := C.mlx_stream_new_device(dev)
	C.mlx_device_free(dev)
	return &Stream{c: stream}
}

// Free releases the stream
func (s *Stream) Free() {
	if s.c.ctx != nil {
		C.mlx_stream_free(s.c)
		s.c.ctx = nil
	}
}

// SetDefaultStream sets the default stream for operations
func SetDefaultStream(s *Stream) {
	C.mlx_set_default_stream(s.c)
	C.set_default_stream(s.c) // Also update our cached stream
}

// GetDefaultStream returns the current default stream
func GetDefaultStream() *Stream {
	var stream C.mlx_stream
	var dev C.mlx_device
	C.mlx_get_default_device(&dev)
	C.mlx_get_default_stream(&stream, dev)
	C.mlx_device_free(dev)
	return &Stream{c: stream}
}

// SynchronizeStream waits for all operations on the stream to complete
func SynchronizeStream(s *Stream) {
	C.mlx_synchronize(s.c)
}

// ============ Metal Memory Control ============

// MetalGetCacheMemory returns the current cache memory usage in bytes
func MetalGetCacheMemory() uint64 {
	var size C.size_t
	C.mlx_get_cache_memory(&size)
	return uint64(size)
}

// MetalGetPeakMemory returns the peak memory usage in bytes
func MetalGetPeakMemory() uint64 {
	var size C.size_t
	C.mlx_get_peak_memory(&size)
	return uint64(size)
}

// MetalResetPeakMemory resets the peak memory counter
func MetalResetPeakMemory() {
	C.mlx_reset_peak_memory()
}

// MetalSetWiredLimit sets the wired memory limit and returns the previous limit
// This keeps tensors pinned in GPU memory for faster access
func MetalSetWiredLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_wired_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// MetalGetActiveMemory returns the current active memory usage in bytes
func MetalGetActiveMemory() uint64 {
	var size C.size_t
	C.mlx_get_active_memory(&size)
	return uint64(size)
}

// ClearCache clears the MLX memory cache
func ClearCache() {
	C.mlx_clear_cache()
}

// SetCacheLimit sets the free cache limit in bytes
// Setting to 0 disables caching (useful for memory-constrained generation)
// Returns the previous cache limit
func SetCacheLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_cache_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// SetMemoryLimit sets the overall memory limit in bytes
// This is a guideline for maximum memory during graph evaluation.
// When Metal is available, defaults to 1.5x the max recommended working set.
// Returns the previous memory limit
func SetMemoryLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_memory_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// GetMemoryLimit returns the current memory limit in bytes
func GetMemoryLimit() uint64 {
	var size C.size_t
	C.mlx_get_memory_limit(&size)
	return uint64(size)
}

// ============ MoE Operations ============

// GatherMM performs gather matrix multiplication for MoE
// a: input, b: weight matrices
// lhsIndices, rhsIndices: optional expert selection indices (nil for none)
func GatherMM(a, b *Array, lhsIndices, rhsIndices *Array, sortedIndices bool) *Array {
	var lhs, rhs C.mlx_array
	if lhsIndices != nil {
		lhs = lhsIndices.c
	}
	if rhsIndices != nil {
		rhs = rhsIndices.c
	}
	res := C.mlx_array_new()
	C.mlx_gather_mm(&res, a.c, b.c, lhs, rhs, C._Bool(sortedIndices), C.default_stream())
	return newArray(res)
}

// GatherQMM performs quantized gather matrix multiplication for MoE
// Used for MXFP4 and other quantized MoE inference
func GatherQMM(x, w, scales *Array, biases, lhsIndices, rhsIndices *Array, transpose bool, groupSize, bits int, mode string, sortedIndices bool) *Array {
	var b, lhs, rhs C.mlx_array
	if biases != nil {
		b = biases.c
	}
	if lhsIndices != nil {
		lhs = lhsIndices.c
	}
	if rhsIndices != nil {
		rhs = rhsIndices.c
	}
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	res := C.mlx_array_new()
	C.mlx_gather_qmm(&res, x.c, w.c, scales.c, b, lhs, rhs, C._Bool(transpose), optGroupSize, optBits, cMode, C._Bool(sortedIndices), C.default_stream())
	return newArray(res)
}

// ============ Quantization ============

// Quantize quantizes weights to specified bits per element.
// Returns (quantized_weights, scales, biases).
// groupSize: number of elements quantized together (default 64)
// bits: bits per element, 2, 4, or 8 (default 4)
// mode: "affine" (default), "mxfp4", or "mxfp8"
// Note: mxfp8 mode returns nil biases (only weights and scales)
func Quantize(w *Array, groupSize, bits int, mode string) (weights, scales, biases *Array) {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	res := C.mlx_vector_array_new()
	C.mlx_quantize(&res, w.c, optGroupSize, optBits, cMode, C.default_stream())

	// Result is a vector of arrays: [weights, scales, biases?]
	// mxfp8 mode returns only 2 elements (no biases)
	vecSize := int(C.mlx_vector_array_size(res))
	var w0, w1, w2 C.mlx_array
	C.mlx_vector_array_get(&w0, res, 0)
	C.mlx_vector_array_get(&w1, res, 1)
	if vecSize >= 3 {
		C.mlx_vector_array_get(&w2, res, 2)
	}
	C.mlx_vector_array_free(res)

	if vecSize >= 3 {
		return newArray(w0), newArray(w1), newArray(w2)
	}
	return newArray(w0), newArray(w1), nil
}

// Dequantize reconstructs weights from quantized form.
// groupSize: number of elements quantized together (default 64)
// bits: bits per element, 2, 4, or 8 (default 4)
// mode: "affine" (default) or "mxfp4"
func Dequantize(w, scales, biases *Array, groupSize, bits int, mode string) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	optDtype := C.mlx_optional_dtype{has_value: false}

	var b C.mlx_array
	if biases != nil {
		b = biases.c
	}

	res := C.mlx_array_new()
	C.mlx_dequantize(&res, w.c, scales.c, b, optGroupSize, optBits, cMode, optDtype, C.default_stream())
	return newArray(res)
}

// QuantizedMatmul performs matrix multiplication with quantized weights.
// x: input tensor [batch..., in_features]
// w: quantized weights
// scales, biases: from Quantize
// transpose: if true, compute x @ w.T (typical for Linear layers)
// groupSize, bits, mode: must match what was used in Quantize
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}

	var b C.mlx_array
	if biases != nil {
		b = biases.c
	}

	res := C.mlx_array_new()
	C.mlx_quantized_matmul(&res, x.c, w.c, scales.c, b, C._Bool(transpose), optGroupSize, optBits, cMode, C.default_stream())
	return newArray(res)
}

// ============ Sorting and Top-K ============

// TopK returns the k largest elements along an axis
func TopK(a *Array, k int, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_topk_axis(&res, a.c, C.int(k), C.int(axis), C.default_stream())
	return newArray(res)
}

// Argpartition returns indices for partial sort (k-th smallest first)
func Argpartition(a *Array, kth int, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_argpartition_axis(&res, a.c, C.int(kth), C.int(axis), C.default_stream())
	return newArray(res)
}

// TakeAlongAxis takes elements from array using indices along axis
func TakeAlongAxis(a, indices *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_take_along_axis(&res, a.c, indices.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// PutAlongAxis puts values into array at indices along axis
func PutAlongAxis(a, indices, values *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_put_along_axis(&res, a.c, indices.c, values.c, C.int(axis), C.default_stream())
	return newArray(res)
}

// Cumsum computes cumulative sum along an axis
func Cumsum(a *Array, axis int) *Array {
	res := C.mlx_array_new()
	C.mlx_cumsum(&res, a.c, C.int(axis), false, false, C.default_stream())
	return newArray(res)
}

// Where selects elements: condition ? a : b
func Where(condition, a, b *Array) *Array {
	res := C.mlx_array_new()
	C.mlx_where(&res, condition.c, a.c, b.c, C.default_stream())
	return newArray(res)
}

// LessScalar returns element-wise a < scalar
func LessScalar(a *Array, s float32) *Array {
	scalar := C.mlx_array_new_float(C.float(s))
	res := C.mlx_array_new()
	C.mlx_less(&res, a.c, scalar, C.default_stream())
	C.mlx_array_free(scalar)
	return newArray(res)
}

// FullDtype creates an array filled with a value with specific dtype
func FullDtype(value float32, dtype Dtype, shape ...int32) *Array {
	intShape := make([]C.int, len(shape))
	for i, s := range shape {
		intShape[i] = C.int(s)
	}
	vals := C.mlx_array_new_float(C.float(value))
	res := C.mlx_array_new()
	C.mlx_full(&res, &intShape[0], C.size_t(len(shape)), vals, C.mlx_dtype(dtype), C.default_stream())
	C.mlx_array_free(vals)
	return newArray(res)
}

// AsType casts an array to a different dtype
func AsType(a *Array, dtype Dtype) *Array {
	res := C.mlx_array_new()
	C.mlx_astype(&res, a.c, C.mlx_dtype(dtype), C.default_stream())
	return newArray(res)
}

// ToBFloat16 casts an array to bfloat16
func ToBFloat16(a *Array) *Array {
	return AsType(a, DtypeBFloat16)
}

// ============ VibeVoice Helper Functions ============

// NewScalarArray creates a true 0-dimensional scalar array from a float32 value
func NewScalarArray(value float32) *Array {
	return newArray(C.mlx_array_new_float(C.float(value)))
}

// Global random seed counter for RandN
var randnSeedCounter uint64 = uint64(time.Now().UnixNano())

// RandN creates an array of random samples from a standard normal distribution
func RandN(shape []int32) *Array {
	// Use incrementing seed for unique random values each call
	seed := atomic.AddUint64(&randnSeedCounter, 1)
	return RandomNormal(shape, seed)
}

// Pad pads an array with zeros
// paddings: [before_0, after_0, before_1, after_1, ...] for each dimension
func Pad(a *Array, paddings []int32) *Array {
	numAxes := len(paddings) / 2
	// Convert to low/high pairs
	lowPad := make([]C.int, numAxes)
	highPad := make([]C.int, numAxes)
	for i := 0; i < numAxes; i++ {
		lowPad[i] = C.int(paddings[i*2])
		highPad[i] = C.int(paddings[i*2+1])
	}
	zero := C.mlx_array_new_float(0.0)
	res := C.mlx_array_new()
	// mlx_pad takes axes, low, high arrays
	axes := make([]C.int, numAxes)
	for i := 0; i < numAxes; i++ {
		axes[i] = C.int(i)
	}
	cMode := C.CString("constant")
	defer C.free(unsafe.Pointer(cMode))
	C.mlx_pad(&res, a.c, &axes[0], C.size_t(numAxes), &lowPad[0], C.size_t(numAxes), &highPad[0], C.size_t(numAxes), zero, cMode, C.default_stream())
	C.mlx_array_free(zero)
	return newArray(res)
}

// Conv1d performs 1D convolution
// x: [B, L, Cin], weight: [Cout, K, Cin] (MLX uses NLC layout)
// bias: optional (nil for no bias)
func Conv1d(x, weight *Array, bias *Array, stride int32) *Array {
	res := C.mlx_array_new()
	C.mlx_conv1d(&res, x.c, weight.c, C.int(stride), C.int(0), C.int(1), 1, C.default_stream())
	// Apply bias if provided
	if bias != nil {
		biased := C.mlx_array_new()
		C.mlx_add(&biased, res, bias.c, C.default_stream())
		C.mlx_array_free(res)
		return newArray(biased)
	}
	return newArray(res)
}

// ConvTranspose1d performs transposed 1D convolution
// x: [B, L, Cin], weight: [Cout, K, Cin] (MLX uses NLC layout)
// bias: optional (nil for no bias)
func ConvTranspose1d(x, weight *Array, bias *Array, stride int32) *Array {
	res := C.mlx_array_new()
	// stride, padding, dilation, output_padding, groups
	C.mlx_conv_transpose1d(&res, x.c, weight.c, C.int(stride), 0, 1, 0, 1, C.default_stream())
	// Apply bias if provided
	if bias != nil {
		biased := C.mlx_array_new()
		C.mlx_add(&biased, res, bias.c, C.default_stream())
		C.mlx_array_free(res)
		return newArray(biased)
	}
	return newArray(res)
}

// DepthwiseConv1d performs depthwise 1D convolution (groups=Cin)
// x: [B, L, C], weight: [1, K, C] (groups = C)
// bias: optional (nil for no bias)
func DepthwiseConv1d(x, weight *Array, bias *Array) *Array {
	// Get number of input channels for groups
	shape := x.Shape()
	groups := int(shape[len(shape)-1])
	res := C.mlx_array_new()
	C.mlx_conv1d(&res, x.c, weight.c, 1, 0, 1, C.int(groups), C.default_stream())
	// Apply bias if provided
	if bias != nil {
		biased := C.mlx_array_new()
		C.mlx_add(&biased, res, bias.c, C.default_stream())
		C.mlx_array_free(res)
		return newArray(biased)
	}
	return newArray(res)
}

// SliceAxis extracts a slice along a specific axis
func SliceAxis(a *Array, axis int, start, stop int32) *Array {
	shape := a.Shape()

	// Build start and stop indices for all dimensions
	starts := make([]int32, len(shape))
	stops := make([]int32, len(shape))
	for i := range shape {
		if i == axis {
			starts[i] = start
			stops[i] = stop
		} else {
			starts[i] = 0
			stops[i] = shape[i]
		}
	}

	return Slice(a, starts, stops)
}

// Tri creates a lower triangular matrix
func Tri(n, m int32, k int) *Array {
	res := C.mlx_array_new()
	C.mlx_tri(&res, C.int(n), C.int(m), C.int(k), C.MLX_FLOAT32, C.default_stream())
	return newArray(res)
}
