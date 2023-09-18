package llm

//go:generate git submodule init

//go:generate git submodule update --force ggml
//go:generate -command git-apply git -C ggml apply
//go:generate git-apply ../ggml_patch/0001-add-detokenize-endpoint.patch
//go:generate git-apply ../ggml_patch/0002-34B-model-support.patch
//go:generate git-apply ../ggml_patch/0003-metal-fix-synchronization-in-new-matrix-multiplicati.patch
//go:generate git-apply ../ggml_patch/0004-metal-add-missing-barriers-for-mul-mat-2699.patch
//go:generate cmake -S ggml -B ggml/build/metal -DLLAMA_METAL=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
//go:generate cmake --build ggml/build/metal --target server --config Release

//go:generate git submodule update --force gguf
//go:generate cmake -S gguf -B gguf/build/metal -DLLAMA_METAL=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
//go:generate cmake --build gguf/build/metal --target server --config Release
