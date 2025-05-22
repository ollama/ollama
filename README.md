

# 1 Pre-release version can run various models supported by the main branch of Ollama. only tested for windows .
    test env: 
        Intel(R) Core(TM) Ultra 9 285H 
        Windows11 24H2

## Explanation of environment variables:

    OLLAMA_INTEL_GPU : true = use intel gpu; false normal ollama
    
    OLLAMA_INTEL_IF_TYPE : SYCL = use SYCL of ggml lib to find gpu device and auto parse memory size; ONEAPI = use level zero lib to find gpu device (can't get memory size)

    OLLAMA_NUM_GPU : when use OLLAMA_INTEL_IF_TYPE=ONEAPI,  this param is max number of layers offload to gpu; default = 64. if use OLLAMA_INTEL_IF_TYPE=SYCL, layers offload to gpu is auto calculate. 

## Run Ollama server
```shell
.\ollama-intel-gpu.bat
```

## Run client

```shell
.\ollama.exe run gemma3:12b --verbose
```


# 2 For Intel OneApi developer

This document recode process of merge ggml-sycl from [llama.cpp](https://github.com/ggml-org/llama.cpp). to support Intel-Gpu.

Only tested in windows and intel integrated Graphics Card.

A portable package in https://github.com/chnxq/ollama/releases

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
set OLLAMA_INTEL_IF_TYPE=SYCL
set OLLAMA_NUM_GPU=999
set SYCL_CACHE_PERSISTENT=1
set OLLAMA_LIBRARY_PATH=./build/lib/ollama
# run ollama server
.\ollama.exe serve
```
or use shell ollama-intel-gpu.bat:
```shell
set OLLAMA_INTEL_GPU=true
set OLLAMA_INTEL_IF_TYPE=SYCL
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
   set OLLAMA_INTEL_IF_TYPE=SYCL (env OLLAMA_INTEL_IF_TYPE is used in go and c code of ollama and llama.cpp,same name as build param)
3. When use pure CPU inference,a known bug need to delete ggml_sycl library temporay.

```shell
# run ollama test client 1
.\ollama.exe run deepseek-r1:1.5b --verbose
```

```shell
# run ollama test client 2
.\ollama.exe run qwen3:4b-fp16 --verbose
```

```shell
# run ollama test client 3
.\ollama.exe run gemma3:12b --verbose
```
