package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
//go:generate sh -c "go run generator/main.go -output=. ./dist/include/mlx/c/*.h"

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/dist/include
// #cgo LDFLAGS: -L${SRCDIR}/dist/lib -lstdc++
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
// #include "generated.h"
import "C"

import (
	"unsafe"
)

func doEval(outputs []*Tensor, async bool) {
	vectorData := make([]C.mlx_array, 0, len(outputs))
	for _, output := range outputs {
		if output.Valid() {
			vectorData = append(vectorData, output.ctx)
		}
	}

	vector := C.mlx_vector_array_new_data(unsafe.SliceData(vectorData), C.size_t(len(vectorData)))
	defer C.mlx_vector_array_free(vector)

	if async {
		C.mlx_async_eval(vector)
	} else {
		C.mlx_eval(vector)
	}
}

func AsyncEval(outputs ...*Tensor) {
	doEval(outputs, true)
}

func Eval(outputs ...*Tensor) {
	doEval(outputs, false)
}
