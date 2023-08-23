package llm

// Remove existing directories.
//go:generate rm -rf llama.cpp
//go:generate git clone -b master-9e232f0 --single-branch https://github.com/ggerganov/llama.cpp.git llama.cpp/ggml

//go:generate cmake -S llama.cpp/ggml -B llama.cpp/ggml/build/cpu
//go:generate cmake --build llama.cpp/ggml/build/cpu --target server --config Release
