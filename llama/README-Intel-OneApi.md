# `For Intel OneApi`

This document recode process of merge ggml-sycl from [llama.cpp](https://github.com/ggml-org/llama.cpp). to support Intel-Gpu.

Only tested in windows and intel integrated Graphics Card.

A portable package in https://github.com/chnxq/ollama/releases/tag/chnxq%2Fv0.0.1a 

# `develope config`

## Pre-request
Install Intel-OneApi: from [OneApiBaseToolkit](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-base-toolkit/2025.html#inpage-nav-1-1)

Other ref: [SYCL](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) document for detail.

(default install in C:\Program Files (x86)\Intel\oneAPI\) for next example.

## Compile the CPU & GPU dynamic libraries

on windows powershell,establish environmental variables:
```shell
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```

build libraries:
on ollama root directory:
```shell
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
```
```shell
cmake --build build --config Release -j 
```

build go src:
```shell
go build -o ollama.exe
```
## Run
on ollama root directory

```shell
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'

set OLLAMA_INTEL_GPU=true
set OLLAMA_INTEL_GPU_SYCL=true
set OLLAMA_NUM_GPU=999
set SYCL_CACHE_PERSISTENT=1
set OLLAMA_LIBRARY_PATH=./build/lib/ollama
# run ollama server
.\ollama.exe serve
```
or use shell ollama-intel-gpu.bat:
```shell
set OLLAMA_INTEL_GPU=true
set OLLAMA_INTEL_GPU_SYCL=true
set OLLAMA_NUM_GPU=64
set SYCL_CACHE_PERSISTENT=1
set OLLAMA_LIBRARY_PATH=./build/lib/ollama
set ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
set PATH=%PATH%;%ONEAPI_ROOT%\2025.1\bin;./build/lib/ollama;

.\ollama.exe serve
```
note:
  1. set OLLAMA_NUM_GPU=xxx
     xxx: It needs to be manually set. According to the number of model layers that the video memory can load.for example my T140 has 16G shared video memory,i set it to 64.
  2. Next 2 env is necessary if use sycl for discover intel gpu:
     set OLLAMA_INTEL_GPU=true
     set OLLAMA_INTEL_GPU_SYCL=true
  3. When use pure CPU inference,a known bug need to delete ggml_sycl library temporay.

```shell
# run ollama test client
.\ollama.exe run deepseek-r1:1.5b --verbose
```
