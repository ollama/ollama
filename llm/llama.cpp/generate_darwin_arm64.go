package llm

//go:generate git submodule init
//go:generate git submodule update --force ggml gguf
//go:generate git -C ggml apply ../ggml_patch/0001-add-detokenize-endpoint.patch
//go:generate git -C ggml apply ../ggml_patch/0002-34B-model-support.patch
//go:generate git -C ggml apply ../ggml_patch/0003-metal-fix-synchronization-in-new-matrix-multiplicati.patch
//go:generate git -C ggml apply ../ggml_patch/0004-metal-add-missing-barriers-for-mul-mat-2699.patch
//go:generate cmake --fresh -S ggml -B ggml/build/gpu -DLLAMA_METAL=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
//go:generate cmake --build ggml/build/gpu --target server --config Release
//go:generate cmake -S gguf -B gguf/build/gpu -DLLAMA_METAL=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
//go:generate cmake --build gguf/build/gpu --target server --config Release
