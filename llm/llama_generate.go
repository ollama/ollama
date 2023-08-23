//go:build !darwin
// +build !darwin

package llm

//go:generate cmake -S llama.cpp/ggml -B llama.cpp/ggml/build
//go:generate cmake --build llama.cpp/ggml/build --target server --config Release
