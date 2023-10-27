package llm

//go:generate git submodule init

//go:generate git submodule update --force ggml
//go:generate git -C ggml apply ../patches/0001-add-detokenize-endpoint.patch
//go:generate git -C ggml apply ../patches/0002-34B-model-support.patch
//go:generate cmake -S ggml -B ggml/build/cpu -DLLAMA_K_QUANTS=on
//go:generate cmake --build ggml/build/cpu --target server --config Release
//go:generate cmd /c move ggml\build\cpu\bin\Release\server.exe ggml\build\cpu\bin\Release\ollama-runner.exe

//go:generate git submodule update --force gguf
//go:generate git -C gguf apply ../patches/0001-update-default-log-target.patch
//go:generate cmake -S gguf -B gguf/build/cpu -DLLAMA_K_QUANTS=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off
//go:generate cmake --build gguf/build/cpu --target server --config Release
//go:generate cmd /c move gguf\build\cpu\bin\Release\server.exe gguf\build\cpu\bin\Release\ollama-runner.exe
