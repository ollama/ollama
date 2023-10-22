package main

/*
#cgo CFLAGS: -Ofast -std=c11 -fPIC -Wno-deprecated-declarations -Wno-unused-but-set-variable
#cgo CPPFLAGS: -Ofast -Wall -Wextra -Wno-unused-function -Wno-unused-variable -Wno-deprecated-declarations -Wno-unused-but-set-variable -DNDEBUG -DGGML_USE_K_QUANTS
#cgo CXXFLAGS: -std=c++11 -fPIC
#cgo darwin CPPFLAGS:  -DGGML_USE_ACCELERATE
#cgo darwin,arm64 CPPFLAGS: -DGGML_USE_METAL
#cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#include <stdlib.h>
#include "main.h"
*/
import "C"
import (
	"fmt"
	"os"
	"unsafe"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("No prompt provided")
		return
	}

	prompt := C.CString(os.Args[1])
	defer C.free(unsafe.Pointer(prompt))
	model := C.CString("./model.bin")
	defer C.free(unsafe.Pointer(model))

	C.generate(model, prompt)
}
