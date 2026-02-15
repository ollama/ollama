//go:build mlx

package mlx

/*
#include "mlx.h"
#include <stdlib.h>

// Forward declaration for Go callback
extern int goClosureCallback(mlx_vector_array* res, mlx_vector_array input, void* payload);

// Destructor for payload (Go handle)
extern void goClosureDestructor(void* payload);
*/
import "C"
import (
	"runtime/cgo"
	"sync"
	"unsafe"
)

// inClosureCallback is set to true during closure callback execution.
var inClosureCallback bool
var closureCallbackMu sync.Mutex

// InClosureCallback returns true if we're currently executing inside a closure callback.
func InClosureCallback() bool {
	closureCallbackMu.Lock()
	defer closureCallbackMu.Unlock()
	return inClosureCallback
}

// CompiledFunc is a compiled MLX function that can be called efficiently.
// All intermediate arrays during execution stay inside MLX - only inputs
// and outputs cross the Go boundary.
type CompiledFunc struct {
	closure  C.mlx_closure
	compiled C.mlx_closure
}

// ClosureFunc is the signature for functions that can be compiled.
// It takes a slice of input arrays and returns a slice of output arrays.
type ClosureFunc func(inputs []*Array) []*Array

// Compile compiles a Go function into an optimized MLX closure.
// The function is traced once during compilation, then subsequent calls
// run the optimized graph without creating Go intermediate arrays.
//
// Example:
//
//	compiled := mlx.Compile(func(inputs []*mlx.Array) []*mlx.Array {
//	    a, b := inputs[0], inputs[1]
//	    c := mlx.Add(a, b)
//	    d := mlx.Mul(c, c)
//	    return []*mlx.Array{d}
//	})
//	defer compiled.Free()
//
//	result := compiled.Call(x, y)[0]
func Compile(fn ClosureFunc) *CompiledFunc {
	return CompileShapeless(fn, false)
}

// CompileShapeless compiles with optional shapeless mode.
// If shapeless=true, the function works for any input shape after tracing.
func CompileShapeless(fn ClosureFunc, shapeless bool) *CompiledFunc {
	// Create a cgo.Handle to prevent the Go function from being GC'd
	handle := cgo.NewHandle(fn)

	// Create the closure from the Go callback
	closure := C.mlx_closure_new_func_payload(
		(*[0]byte)(C.goClosureCallback),
		unsafe.Pointer(handle),
		(*[0]byte)(C.goClosureDestructor),
	)

	// Compile the closure
	compiled := C.mlx_closure_new()
	C.mlx_compile(&compiled, closure, C.bool(shapeless))

	return &CompiledFunc{
		closure:  closure,
		compiled: compiled,
	}
}

// Call invokes the compiled function with the given inputs.
func (cf *CompiledFunc) Call(inputs ...*Array) []*Array {
	// Pack inputs into vector
	inputVec := C.mlx_vector_array_new()
	for _, arr := range inputs {
		C.mlx_vector_array_append_value(inputVec, arr.c)
	}

	// Apply compiled closure
	outputVec := C.mlx_vector_array_new()
	C.mlx_closure_apply(&outputVec, cf.compiled, inputVec)
	C.mlx_vector_array_free(inputVec)

	// Unpack outputs
	numOutputs := int(C.mlx_vector_array_size(outputVec))
	outputs := make([]*Array, numOutputs)
	for i := 0; i < numOutputs; i++ {
		var arr C.mlx_array
		C.mlx_vector_array_get(&arr, outputVec, C.size_t(i))
		outputs[i] = newArray(arr)
	}
	C.mlx_vector_array_free(outputVec)

	return outputs
}

// CallEval invokes the compiled function and evaluates the results.
func (cf *CompiledFunc) CallEval(inputs ...*Array) []*Array {
	outputs := cf.Call(inputs...)
	Eval(outputs...)
	return outputs
}

// Free releases the compiled function resources.
func (cf *CompiledFunc) Free() {
	C.mlx_closure_free(cf.compiled)
	C.mlx_closure_free(cf.closure)
}

// borrowArray wraps a C array WITHOUT setting up GC cleanup.
// Use this for arrays we don't own (e.g., borrowed references in callbacks).
func borrowArray(array C.mlx_array) *Array {
	return &Array{c: array}
}

//export goClosureCallback
func goClosureCallback(res *C.mlx_vector_array, input C.mlx_vector_array, payload unsafe.Pointer) C.int {
	// Set flag to disable AddCleanup during callback
	closureCallbackMu.Lock()
	inClosureCallback = true
	closureCallbackMu.Unlock()
	defer func() {
		closureCallbackMu.Lock()
		inClosureCallback = false
		closureCallbackMu.Unlock()
	}()

	// Recover the Go function from the handle
	handle := cgo.Handle(payload)
	fn := handle.Value().(ClosureFunc)

	// Convert input vector to Go slice - use borrowArray since MLX owns these
	numInputs := int(C.mlx_vector_array_size(input))
	inputs := make([]*Array, numInputs)
	for i := 0; i < numInputs; i++ {
		var arr C.mlx_array
		C.mlx_vector_array_get(&arr, input, C.size_t(i))
		inputs[i] = borrowArray(arr) // Don't set up cleanup - MLX owns these
	}

	// Call the Go function
	outputs := fn(inputs)

	// Build output vector
	*res = C.mlx_vector_array_new()
	for _, arr := range outputs {
		C.mlx_vector_array_append_value(*res, arr.c)
	}

	return 0
}

//export goClosureDestructor
func goClosureDestructor(payload unsafe.Pointer) {
	handle := cgo.Handle(payload)
	handle.Delete()
}
