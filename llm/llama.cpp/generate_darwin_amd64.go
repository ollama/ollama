package llm

//go:generate git submodule init
//go:generate git submodule update --force ggml
//go:generate git -C ggml apply ../ggml_patch/0001-add-detokenize-endpoint.patch
//go:generate git -C ggml apply ../ggml_patch/0002-34B-model-support.patch
//go:generate cmake --fresh -S ggml -B ggml/build/cpu -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64
//go:generate cmake --build ggml/build/cpu --target server --config Release
