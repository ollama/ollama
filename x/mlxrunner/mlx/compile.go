package mlx

// #include "generated.h"
// int goClosureFunc(mlx_vector_array*, mlx_vector_array, void*);
// void goClosureDestructor(void*);
import "C"

import (
	"runtime/cgo"
	"unsafe"
)

type Closure struct {
	ctx C.mlx_closure
}

func (c Closure) Call(inputs []*Array) []*Array {
	inputsVector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(inputsVector)

	for _, input := range inputs {
		C.mlx_vector_array_append_value(inputsVector, input.ctx)
	}

	outputsVector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outputsVector)

	C.mlx_closure_apply(&outputsVector, c.ctx, inputsVector)

	outputs := make([]*Array, int(C.mlx_vector_array_size(outputsVector)))
	for i := range outputs {
		t := New("", inputs...)
		C.mlx_vector_array_get(&t.ctx, outputsVector, C.size_t(i))
		outputs[i] = t
	}

	return outputs
}

func Compile(fn func([]*Array) []*Array, shapeless bool) *Closure {
	closure := C.mlx_closure_new_func_payload(
		(*[0]byte)(C.goClosureFunc),
		unsafe.Pointer(cgo.NewHandle(fn)),
		(*[0]byte)(C.goClosureDestructor),
	)

	compiled := C.mlx_closure_new()
	C.mlx_compile(&compiled, closure, C.bool(shapeless))
	return &Closure{ctx: compiled}
}

//export goClosureFunc
func goClosureFunc(outputsVector *C.mlx_vector_array, inputsVector C.mlx_vector_array, payload unsafe.Pointer) C.int {
	handle := cgo.Handle(payload)
	fn := handle.Value().(func([]*Array) []*Array)

	inputs := make([]*Array, int(C.mlx_vector_array_size(inputsVector)))
	for i := range inputs {
		t := New("")
		C.mlx_vector_array_get(&t.ctx, inputsVector, C.size_t(i))
		inputs[i] = t
	}

	var outputs []C.mlx_array
	for _, output := range fn(inputs) {
		outputs = append(outputs, output.ctx)
	}

	C.mlx_vector_array_set_data(outputsVector, unsafe.SliceData(outputs), C.size_t(len(outputs)))
	return 0
}

//export goClosureDestructor
func goClosureDestructor(payload unsafe.Pointer) {
	cgo.Handle(payload).Delete()
}
