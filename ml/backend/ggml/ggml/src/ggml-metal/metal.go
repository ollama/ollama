package metal

// #cgo CPPFLAGS: -DGGML_METAL_EMBED_LIBRARY -I${SRCDIR}/.. -I${SRCDIR}/../../include
// #cgo LDFLAGS: -framework Metal -framework MetalKit
import "C"
