//go:build cuda
// +build cuda

package llm

/*
#cgo CFLAGS: -O3 -std=c11 -fPIC -DNDEBUG -pthread -march=native -mtune=native
#cgo CXXFLAGS: -O3 -std=c++11 -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -pthread -march=native -mtune=native -DGGML_USE_K_QUANTS
#cgo linux CFLAGS: -DGGML_USE_CUBLAS -I/usr/local/cuda/include
#cgo linux CXXFLAGS: -DGGML_USE_CUBLAS -I/usr/local/cuda/include
#cgo linux LDFLAGS:  -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -lcublas -lcublasLt -lcudart -lculibos -L/usr/local/cuda/targets/x86_64-linux/lib
*/
import "C"
