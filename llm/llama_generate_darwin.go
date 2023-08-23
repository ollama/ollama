//go:build darwin
// +build darwin

package llm

//go:generate cmake -S llama.cpp/ggml -B llama.cpp/ggml/build/gpu -DLLAMA_METAL=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build llama.cpp/ggml/build/gpu --target server --config Release
