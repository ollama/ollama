//go:build !darwin
// +build !darwin

package llm

//go:generate cmake -S ggml -B ggml/build
//go:generate cmake --build ggml/build --target server --config Release
