//go:build darwin
// +build darwin

package llm

//go:generate cmake -S ggml -B ggml/build -DLLAMA_METAL=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build --target server --config Release
