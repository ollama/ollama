package mlx

// #include <stdlib.h>
// #include "generated.h"
//
// extern int mlxClosureCallback(mlx_vector_array* res, mlx_vector_array input, void* payload);
// extern void mlxClosureDestructor(void* payload);
import "C"

import (
	"log/slog"
	"runtime"
	"runtime/cgo"
	"unsafe"
)

// ClosureFunc is the signature of a function that can be compiled.
type ClosureFunc func(inputs []*Array) []*Array

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

// CompiledFunc is a compiled MLX function. Compiled functions are intended to
// live for the program's lifetime; the underlying MLX closure is released
// automatically via runtime.AddCleanup when the CompiledFunc becomes
// unreachable.
type CompiledFunc struct {
	name    string
	closure C.mlx_closure
}

// Compile wraps fn as a compiled MLX closure. name is a human-readable tag
// used in error messages; it has no effect on execution. MLX traces fn
// lazily on the first Call for each distinct (shape, dtype).
func Compile(name string, fn ClosureFunc, opts ...CompileOption) *CompiledFunc {
	var cfg compileConfig
	for _, o := range opts {
		o(&cfg)
	}

	// The payload is a C-allocated slot holding a cgo.Handle wrapping fn.
	// Using C memory avoids cgo's rule against passing Go memory that
	// contains unpinned Go pointers; the slot is freed by the destructor
	// when MLX releases its last reference.
	payload := (*cgo.Handle)(C.malloc(C.size_t(unsafe.Sizeof(cgo.Handle(0)))))
	*payload = cgo.NewHandle(fn)
	src := C.mlx_closure_new_func_payload(
		(*[0]byte)(C.mlxClosureCallback),
		unsafe.Pointer(payload),
		(*[0]byte)(C.mlxClosureDestructor),
	)
	// mlx_compile moves fn into the compiled closure, so src is freed
	// either way. The compiled closure keeps the payload alive via its
	// own shared_ptr until its mlx_closure_free runs the destructor.
	defer C.mlx_closure_free(src)

	compiled := C.mlx_closure_new()
	clearLastError()
	if rc := C.mlx_compile(&compiled, src, C.bool(cfg.shapeless)); rc != 0 {
		msg := lastError()
		if msg == "" {
			msg = "mlx_compile failed"
		}
		panic("mlx: " + name + ": " + msg)
	}

	cf := &CompiledFunc{name: name, closure: compiled}
	runtime.AddCleanup(cf, func(c C.mlx_closure) { C.mlx_closure_free(c) }, cf.closure)
	return cf
}

// Call invokes the compiled function with the given inputs. Returned outputs
// participate in the normal Pin/Sweep lifecycle.
func (cf *CompiledFunc) Call(inputs ...*Array) []*Array {
	inVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(inVec)
	for _, in := range inputs {
		C.mlx_vector_array_append_value(inVec, in.ctx)
	}

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	clearLastError()
	if rc := C.mlx_closure_apply(&outVec, cf.closure, inVec); rc != 0 {
		msg := lastError()
		if msg == "" {
			msg = "mlx_closure_apply failed"
		}
		panic("mlx: " + cf.name + ": " + msg)
	}

	n := int(C.mlx_vector_array_size(outVec))
	outputs := make([]*Array, n)
	for i := range n {
		outputs[i] = New(cf.name)
		C.mlx_vector_array_get(&outputs[i].ctx, outVec, C.size_t(i))
	}
	return outputs
}

// Compile1 compiles a unary function and returns a plain callable. The
// underlying closure is built on first call so package-level declarations
// work even before MLX's dynamic library is loaded.
//
// If invoked while another compile is tracing, the original fn is called
// directly so its ops are inlined into the outer trace rather than applied
// as a nested compiled closure. This matches upstream mlx's @mx.compile
// decorator and lets outer compiles fuse through inner ones.
func Compile1(name string, fn func(*Array) *Array, opts ...CompileOption) func(*Array) *Array {
	var cf *CompiledFunc
	return func(a *Array) *Array {
		if tracing {
			return fn(a)
		}
		if cf == nil {
			cf = Compile(name, func(in []*Array) []*Array {
				return []*Array{fn(in[0])}
			}, opts...)
		}
		return cf.Call(a)[0]
	}
}

// Compile2 compiles a binary function. See Compile1.
func Compile2(name string, fn func(*Array, *Array) *Array, opts ...CompileOption) func(*Array, *Array) *Array {
	var cf *CompiledFunc
	return func(a, b *Array) *Array {
		if tracing {
			return fn(a, b)
		}
		if cf == nil {
			cf = Compile(name, func(in []*Array) []*Array {
				return []*Array{fn(in[0], in[1])}
			}, opts...)
		}
		return cf.Call(a, b)[0]
	}
}

// Compile3 compiles a ternary function. See Compile1.
func Compile3(name string, fn func(*Array, *Array, *Array) *Array, opts ...CompileOption) func(*Array, *Array, *Array) *Array {
	var cf *CompiledFunc
	return func(a, b, c *Array) *Array {
		if tracing {
			return fn(a, b, c)
		}
		if cf == nil {
			cf = Compile(name, func(in []*Array) []*Array {
				return []*Array{fn(in[0], in[1], in[2])}
			}, opts...)
		}
		return cf.Call(a, b, c)[0]
	}
}

//export mlxClosureCallback
func mlxClosureCallback(res *C.mlx_vector_array, input C.mlx_vector_array, payload unsafe.Pointer) (rc C.int) {
	// Recover panics so they don't unwind into C (which is UB). MLX
	// overwrites the user's error message with a generic one after any
	// non-zero return, so log the original panic and let the caller see
	// a failed Call via the non-zero rc. Registered first so it is
	// outermost and catches panics from any subsequent code, including
	// the handle lookup and type assertion.
	defer func() {
		if r := recover(); r != nil {
			slog.Error("mlx closure callback panicked", "panic", r)
			rc = 1
		}
	}()

	handle := *(*cgo.Handle)(payload)
	fn := handle.Value().(ClosureFunc)

	// Route arrays produced during fn through traceScratch. They are
	// symbolic tracing handles that MLX captures into the compiled graph;
	// our wrappers must be freed before returning or we leak a handle and
	// a refcount per traced op.
	prevTracing := tracing
	prevScratch := traceScratch
	tracing = true
	traceScratch = nil
	defer func() {
		for _, a := range traceScratch {
			if a.pinned > 0 {
				panic("mlx: traced array was pinned during compilation")
			}
			if a.Valid() {
				C.mlx_array_free(a.ctx)
				a.ctx.ctx = nil
			}
		}
		tracing = prevTracing
		traceScratch = prevScratch
	}()

	// Each mlx_vector_array_get populates a caller-owned handle; route it
	// into traceScratch so it is freed when the callback returns.
	n := int(C.mlx_vector_array_size(input))
	inputs := make([]*Array, n)
	for i := range n {
		a := New("")
		C.mlx_vector_array_get(&a.ctx, input, C.size_t(i))
		inputs[i] = a
	}

	outputs := fn(inputs)

	// Populate the output vector via set_data, which handles any initial
	// state of *res (null or previously allocated) per mlx-c convention.
	// Our wrappers remain independent and are freed via traceScratch.
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

//export mlxClosureDestructor
func mlxClosureDestructor(payload unsafe.Pointer) {
	handle := *(*cgo.Handle)(payload)
	handle.Delete()
	C.free(payload)
}
