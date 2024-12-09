package cuda

// #cgo cuda_v11 LDFLAGS: -L. -lggml_cuda_v11
// #cgo cuda_v12 LDFLAGS: -L. -lggml_cuda_v12
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcublasLt
import "C"
