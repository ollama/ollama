package llm

//go:generate -command cmake-toolchain cmake --toolchain ../cmake/$GOOS-$GOARCH-toolchain.cmake -DCMAKE_POLICY_DEFAULT_CMP0077=NEW

//go:generate git submodule init

//go:generate git submodule update --force ggml
//go:generate -command git-apply git -C ggml apply
//go:generate git-apply ../ggml_patch/0001-add-detokenize-endpoint.patch
//go:generate git-apply ../ggml_patch/0002-34B-model-support.patch
//go:generate cmake-toolchain -S ggml -B ggml/build/cpu
//go:generate cmake --build ggml/build/cpu --target server --config Release

//go:generate git submodule update --force gguf
//go:generate cmake-toolchain -S gguf -B gguf/build/cpu
//go:generate cmake --build gguf/build/cpu --target server --config Release
