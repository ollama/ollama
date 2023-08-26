package llm

//go:generate cmake -S ggml -B ggml/build/cpu
//go:generate cmake --build ggml/build/cpu --target server --config Release
