package llm

//go:generate cmake -S llama.cpp/ggml -B llama.cpp/ggml/build/cpu
//go:generate cmake --build llama.cpp/ggml/build/cpu --target server --config Release
