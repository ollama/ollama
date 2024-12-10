package metal

// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../include
// #cgo CPPFLAGS: -DGGML_METAL_EMBED_LIBRARY
// #cgo LDFLAGS: -framework Metal -framework MetalKit -framework Accelerate
import "C"
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/ggml-blas"
