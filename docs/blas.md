# Linear Algebra Libraries

llama.cpp provides support for non GPU enabled hosts.
If you don't have a GPU, you can also get some of this acceleration.
One common way is to utilize linear algebra software to accelerate math operations.

## Intel MKL

[llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/README.md#intel-onemkl) provides details on how to configure llama.cpp build with intel mkl.

On intel cpus, intel MKL (math kernel library) is available to use.
MKL speeds up linear algebra and optimizies the math required.

One can follow the directions for installing mkl on your platform choice.

For linux, you can then:

```bash
source /opt/intel/oneapi/setvars.sh
OLLAMA_CUSTOM_CPU_DEFS="-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=Intel10_64lp -DLLAMA_NATIVE=ON" go generate ./...
go build .
```
