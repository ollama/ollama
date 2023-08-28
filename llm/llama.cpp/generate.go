package llm

//go:generate git submodule update --init --recursive
//go:generate sh ggml_patch/apply.sh
//go:generate cmake -S ggml -B ggml/build/cpu
//go:generate cmake --build ggml/build/cpu --target server --config Release
