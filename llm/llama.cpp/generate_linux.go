//go:build linux
// +build linux

package llm

//go:generate git submodule init
//go:generate git submodule update --force ggml gguf
//go:generate git -C ggml apply ../ggml_patch/0001-add-detokenize-endpoint.patch
//go:generate git -C ggml apply ../ggml_patch/0002-34B-model-support.patch
//go:generate sh generate_linux.sh
