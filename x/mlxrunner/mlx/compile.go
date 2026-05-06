package mlx

// #include <stdlib.h>
// #include "generated.h"
//
// extern int closureCallback(mlx_vector_array* res, mlx_vector_array input, void* payload);
// extern void closureDestructor(void* payload);
import "C"

import (
	"log/slog"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// CompileFunc is the signature of a function that can be compiled.
type CompileFunc func(inputs ...*Array) []*Array

// CompileOption configures Compile behavior.
type CompileOption func(*compileConfig)

type compileConfig struct {
	shapeless bool
}

// Shapeless traces the function once against symbolic shapes so the compiled
// graph accepts any input shape afterwards. Without this option, MLX re-traces
// on each new (shape, dtype) combination and caches each specialization.
func Shapeless() CompileOption {
	return func(c *compileConfig) { c.shapeless = true }
}

// Compile returns a compiled version of fn. When called during another
// compile's trace, fn is inlined directly so outer compiles can fuse through
// inner ones.
//
// Compiled functions must not have side effects outside of the function. Do
// not access data other than the arguments passed in (either Go data or MLX
// arrays) unless it is a constant.
func Compile(name string, fn CompileFunc, opts ...CompileOption) CompileFunc {
	var cfg compileConfig
	for _, o := range opts {
		o(&cfg)
	}

	var closure C.mlx_closure
	var once sync.Once

	return func(inputs ...*Array) []*Array {
		if tracing {
			return fn(inputs...)
		}

		once.Do(func() {
			payload := (*cgo.Handle)(C.malloc(C.size_t(unsafe.Sizeof(cgo.Handle(0)))))
			*payload = cgo.NewHandle(fn)
			src := C.mlx_closure_new_func_payload(
				(*[0]byte)(C.closureCallback),
				unsafe.Pointer(payload),
				(*[0]byte)(C.closureDestructor),
			)
			defer C.mlx_closure_free(src)

			closure = C.mlx_closure_new()
			mlxCheck(name+": compile failed", func() C.int {
				return C.mlx_compile(&closure, src, C.bool(cfg.shapeless))
			})
		})

		inVec := C.mlx_vector_array_new()
		defer C.mlx_vector_array_free(inVec)
		for _, in := range inputs {
			C.mlx_vector_array_append_value(inVec, in.ctx)
		}

		outVec := C.mlx_vector_array_new()
		defer C.mlx_vector_array_free(outVec)
		mlxCheck(name+": closure apply failed", func() C.int {
			return C.mlx_closure_apply(&outVec, closure, inVec)
		})

		n := int(C.mlx_vector_array_size(outVec))
		outputs := make([]*Array, n)
		for i := range n {
			outputs[i] = New(name)
			C.mlx_vector_array_get(&outputs[i].ctx, outVec, C.size_t(i))
		}
		return outputs
	}
}

// Compile1 compiles a unary function. See Compile.
func Compile1(name string, fn func(*Array) *Array, opts ...CompileOption) func(*Array) *Array {
	cf := Compile(name, func(in ...*Array) []*Array {
		return []*Array{fn(in[0])}
	}, opts...)
	return func(a *Array) *Array {
		return cf(a)[0]
	}
}

// Compile2 compiles a binary function. See Compile.
func Compile2(name string, fn func(*Array, *Array) *Array, opts ...CompileOption) func(*Array, *Array) *Array {
	cf := Compile(name, func(in ...*Array) []*Array {
		return []*Array{fn(in[0], in[1])}
	}, opts...)
	return func(a, b *Array) *Array {
		return cf(a, b)[0]
	}
}

// Compile3 compiles a ternary function. See Compile.
func Compile3(name string, fn func(*Array, *Array, *Array) *Array, opts ...CompileOption) func(*Array, *Array, *Array) *Array {
	cf := Compile(name, func(in ...*Array) []*Array {
		return []*Array{fn(in[0], in[1], in[2])}
	}, opts...)
	return func(a, b, c *Array) *Array {
		return cf(a, b, c)[0]
	}
}

// tracing is true while a compile callback is running. Since MLX is
// single-threaded at this level a plain Go bool suffices.
var tracing bool

// traceScratch collects arrays created during a compile trace so they can be
// freed as a group when the callback returns.
var traceScratch []*Array

//export closureCallback
func closureCallback(res *C.mlx_vector_array, input C.mlx_vector_array, payload unsafe.Pointer) (rc C.int) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("mlx closure callback panicked", "panic", r)
			rc = 1
		}
	}()

	handle := *(*cgo.Handle)(payload)
	fn := handle.Value().(CompileFunc)

	// When tracing, we track all of the intermediates that are created and free them separately at the end of
	// the process. This will give the effect of a single op - inputs are owned by the original caller (via
	// the MLX layer) and outputs are transferred back to MLX to create a new Go side tensor.
	if tracing {
		panic("mlx: nested compile trace")
	}
	tracing = true
	traceScratch = nil
	defer func() {
		for _, a := range traceScratch {
			if a.pinned.Load() > 0 {
				panic("mlx: traced array was pinned during compilation")
			}
			if a.Valid() {
				C.mlx_array_free(a.ctx)
				a.ctx.ctx = nil
			}
		}
		tracing = false
		traceScratch = nil
	}()

	n := int(C.mlx_vector_array_size(input))
	inputs := make([]*Array, n)
	for i := range n {
		a := New("")
		C.mlx_vector_array_get(&a.ctx, input, C.size_t(i))
		inputs[i] = a
	}

	outputs := fn(inputs...)

	var arrPtr *C.mlx_array
	if len(outputs) > 0 {
		handles := make([]C.mlx_array, len(outputs))
		for i, out := range outputs {
			handles[i] = out.ctx
		}
		arrPtr = &handles[0]
	}
	C.mlx_vector_array_set_data(res, arrPtr, C.size_t(len(outputs)))
	return 0
}

//export closureDestructor
func closureDestructor(payload unsafe.Pointer) {
	handle := *(*cgo.Handle)(payload)
	handle.Delete()
	C.free(payload)
}
