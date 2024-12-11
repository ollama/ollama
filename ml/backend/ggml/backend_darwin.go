package ggml

//go:generate sh -c "echo \"// Code generated $(date). DO NOT EDIT.\n\" >ggml-metal-embed.metal"
//go:generate sh -c "sed -e '/#include \"ggml-common.h\"/r ggml-common.h' -e '/#include \"ggml-common.h\"/d' ggml-metal.metal >>ggml-metal-embed.metal"

// #cgo arm64 CPPFLAGS: -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY -DGGML_USE_ACCELERATE -DGGML_METAL_NDEBUG
// #cgo arm64 LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
// #include "ggml-metal.h"
import "C"

func newBackend() *C.struct_ggml_backend {
	return C.ggml_backend_metal_init()
}
