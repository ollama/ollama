#!/bin/bash

cmake -B build
make -C build
mv build/llama_cpp_metal llm/llama_cpp_gpu
mv build/llama_cpp llm/llama_cpp
