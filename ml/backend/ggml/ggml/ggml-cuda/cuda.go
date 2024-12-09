package cuda

// #cgo cuda_v11 LDFLAGS: -L. -lggml_cuda_v11
// #cgo cuda_v12 LDFLAGS: -L. -lggml_cuda_v12
// #cgo cuda_v11 cuda_v12 LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcublasLt
// #cgo rocm LDFLAGS: -L. -lggml_rocm -L/opt/rocm/lib -lhipblas -lamdhip64 -lrocblas
import "C"
