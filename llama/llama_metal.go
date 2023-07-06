//go:build metal
package llama

//go:generate cmake -S . -B build --fresh -DLLAMA_METAL=on
//go:generate cmake --build build
