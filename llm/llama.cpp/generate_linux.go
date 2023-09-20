package llm

//go:generate git submodule init

//go:generate git submodule update --force ggml
//go:generate -command git-apply git -C ggml apply
//go:generate git-apply ../ggml_patch/0001-add-detokenize-endpoint.patch
//go:generate git-apply ../ggml_patch/0002-34B-model-support.patch
//go:generate git-apply ../ggml_patch/0005-ggml-support-CUDA-s-half-type-for-aarch64-1455-2670.patch

//go:generate cmake -S ggml -B ggml/build/cpu -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build/cpu --target server --config Release

//go:generate git submodule update --force gguf
//go:generate cmake -S gguf -B gguf/build/cpu -DLLAMA_K_QUANTS=on
//go:generate cmake --build gguf/build/cpu --target server --config Release

//go:generate cmake -S ggml -B ggml/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build/cuda --target server --config Release
//go:generate cmake -S gguf -B gguf/build/cuda -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
//go:generate cmake --build gguf/build/cuda --target server --config Release

//go:generate cp --dereference ${CUBLAS_PATH} ggml/build/cuda/bin/libcublas.so.11
//go:generate cp --dereference ${CUBLAS_PATH} gguf/build/cuda/bin/libcublas.so.11
//go:generate cp --dereference ${CUDART_PATH} ggml/build/cuda/bin/libcudart.so.11.0
//go:generate cp --dereference ${CUDART_PATH} gguf/build/cuda/bin/libcudart.so.11.0
//go:generate cp --dereference ${CUBLASLT_PATH} ggml/build/cuda/bin/libcublasLt.so.11
//go:generate cp --dereference ${CUBLASLT_PATH} gguf/build/cuda/bin/libcublasLt.so.11
