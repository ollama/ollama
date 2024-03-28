export EXTRA_LIBS='-lvulkan'
export OLLAMA_CUSTOM_CPU_DEFS="-DLLAMA_VULKAN=on -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=on -DLLAMA_FMA=on -DLLAMA_AVX512_VBMI=on -DLLAMA_AVX512_VNNI=on -DLLAMA_F16C=on"
rm -rf llm/llama.cpp/build
go generate ./...
go build .
killall ollama
