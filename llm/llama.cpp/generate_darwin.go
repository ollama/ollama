//go:build darwin
// +build darwin

package llm

//go:generate git submodule update --init --recursive
//go:generate sh ggml_patch/apply.sh
//go:generate cmake -S ggml -B ggml/build/gpu -DLLAMA_METAL=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build/gpu --target server --config Release
