//go:build mlx

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

func doEval(outputs []*Array, async bool) {
	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	for _, output := range outputs {
		if output.Valid() {
			C.mlx_vector_array_append_value(vector, output.ctx)
		}
	}

	if async {
		C.mlx_async_eval(vector)
	} else {
		C.mlx_eval(vector)
	}
}

func AsyncEval(outputs ...*Array) {
	doEval(outputs, true)
}

func Eval(outputs ...*Array) {
	doEval(outputs, false)
}
