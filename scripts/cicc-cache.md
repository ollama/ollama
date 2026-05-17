# cicc-cache

Content-addressed cache for NVIDIA's `cicc` device compiler, used to deduplicate
redundant compilations during CUDA builds.

## Problem

When building for N CUDA architectures, nvcc preprocesses each `.cu` source file
once per architecture, then invokes `cicc` (the device compiler) on each result.
For most source files, the preprocessed output (`.cpp1.ii`) is identical across
architectures because the code doesn't use `__CUDA_ARCH__` guards. Without
caching, cicc compiles the same input N times.

For the ollama CUDA 13 preset (12 architectures), this means ~7 out of 12
invocations per source file are redundant (5 arches produce genuinely different
output due to `cuda_fp8.hpp` having `__CUDA_ARCH__ >= 890` guards).

## How it works

The script replaces `/usr/local/cuda/nvvm/bin/cicc` with a wrapper. The real
cicc is moved to `cicc.real`.

1. Parse cicc's command line to find the input file (`*.cpp1.ii`) and all output
   path flags (`-o`, `--gen_c_file_name`, `--stub_file_name`,
   `--gen_device_file_name`, `--module_id_file_name`).
2. Hash the input file with md5sum.
3. If a cache entry exists for that hash, copy the cached outputs to the expected
   paths and exit.
4. Otherwise, run `cicc.real` with the original arguments, then store the outputs
   in the cache.

If the input file or output flags can't be parsed, the wrapper falls through to
`cicc.real` unchanged.

## cicc output files

cicc produces 5 output files per invocation:

| Flag                     | Description                          |
|--------------------------|--------------------------------------|
| `-o`                     | PTX assembly (`.ptx`)                |
| `--gen_c_file_name`      | Generated C file (`.cudafe1.c`)      |
| `--stub_file_name`       | Stub file (`.cudafe1.stub.c`)        |
| `--gen_device_file_name` | Device code (`.cudafe1.gpu`)         |
| `--module_id_file_name`  | Module ID (`.module_id`)             |

All 5 must be cached and restored together for correctness.

## nvcc intermediate file naming

nvcc's pipeline produces numbered intermediate files:

- `.cpp1.ii` -- Preprocessed device code, input to `cicc`.
- `.cpp2.i` / `.cpp3.i` -- Internal cudafe intermediates (not relevant here).
- `.cpp4.ii` -- Preprocessed host code, input to `cudafe++` then the host
  compiler. Not seen by `cicc`.

## Installation (in Dockerfile)

```dockerfile
COPY scripts/cicc-cache /usr/local/cuda/nvvm/bin/cicc-cache
RUN chmod +x /usr/local/cuda/nvvm/bin/cicc-cache \
    && mv /usr/local/cuda/nvvm/bin/cicc /usr/local/cuda/nvvm/bin/cicc.real \
    && mv /usr/local/cuda/nvvm/bin/cicc-cache /usr/local/cuda/nvvm/bin/cicc
```

## Configuration

- `CICC_CACHE_DIR` -- Override cache directory (default: `/tmp/cicc-cache`).

## Limitations

- Only caches within a single build. The cache lives in `/tmp` which is ephemeral
  in Docker. Cross-build caching would require mounting the cache directory.
- Hash is based on the preprocessed input file only, not on cicc flags like
  `-arch`. This is correct because nvcc preprocesses once per arch, and
  arch-specific differences (like `__CUDA_ARCH__` expansion) appear in the
  preprocessed output.

## Related changes

- `llama/patches/0037-ggml-cuda-remove-compile-time-__CUDA_ARCH__-from-NO_.patch`
  removes `__CUDA_ARCH__` from the `NO_DEVICE_CODE` macro in `common.cuh`, which
  previously caused every architecture to produce different preprocessed output
  for files using that macro.
