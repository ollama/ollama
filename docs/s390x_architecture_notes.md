# s390x Architecture Notes for Vector Processing

## Overview

This document summarizes the `s390x` vector-processing facilities that are most relevant to the `ollama-s390x` project, with an emphasis on machine learning inference workloads. On IBM z/Architecture, vector execution is provided through the Vector Facility and its enhancement levels. These facilities expose fixed-width vector registers and instruction sets that can accelerate data-parallel operations commonly found in tensor math, quantized kernels, reductions, data movement, and preprocessing.

For inference-oriented software such as Ollama, these capabilities matter in two ways:

- They define the practical SIMD-style execution model available on `s390x`.
- They influence how portable low-level kernels must be written when adapting implementations originally optimized for `x86_64`, `ARM64`, or GPU backends.

The sections below describe the base Vector Facility, Vector-Enhancements Facility 1, Vector-Enhancements Facility 2, and the portability implications for CPU-only inference.

## Vector Facility for z/Architecture

The base Vector Facility introduces a SIMD-like programming model for `z/Architecture`.

### Core characteristics

- Provides **32 vector registers**, each **128 bits** wide.
- Supports instructions that operate on **fixed-sized vectors** containing **1 to 16 elements**, depending on element width.
- Enables **parallel element processing**, where multiple elements in a vector are handled as part of a single instruction.
- An instruction is considered complete **only after all vector elements have been processed**.

This execution model is important for inference kernels because many operations in machine learning are naturally data parallel, including:

- dot products
- elementwise arithmetic
- quantized dequantization steps
- activation functions
- masking and comparison operations
- tensor layout conversion and packing

### Instruction organization

The Vector Facility includes **23 different instructions** operating on fixed-sized vectors. These instructions establish the baseline vector execution semantics and register model used by later enhancement facilities.

Although the architectural model is vector-oriented, software still needs to map algorithm structure carefully onto:

- element width
- lane count
- alignment behavior
- load/store patterns
- reduction strategy

These details strongly affect realized performance in inference code.

### Vector Facility Integer

The integer subset of the Vector Facility includes:

- **28 instructions**
- additional usability through **extended mnemonics**

This subset is especially relevant to machine learning inference because integer-heavy execution is common in:

- quantized matrix multiplication
- accumulation over low-precision weights or activations
- bitwise masking
- index generation
- packing and unpacking tensor fragments

Extended mnemonics improve readability and make low-level implementations easier to maintain, especially when building architecture-specific kernels or reviewing compiler output.

### Vector Facility String

The string subset includes:

- **5 instructions** for string processing

A notable architectural detail is:

- all string instructions **except `VECTOR ISOLATE STRING`** produce an **element index** in the **rightmost bits of the leftmost doubleword**

While these instructions are not primarily designed for tensor arithmetic, they can still be relevant for:

- tokenization-adjacent preprocessing
- substring or delimiter scanning
- byte-oriented parsing
- search-style operations in support code

For systems software and runtime support paths, these instructions may provide efficient byte-lane search behavior without requiring scalar fallback loops.

### Vector Facility Floating-Point

The floating-point subset supports binary floating-point operations in:

- **short format**
- **long format**
- **extended format**

It also provides a **single element control bit** intended to improve performance in cases where operating on one element is preferable to full-vector behavior.

This matters for inference because floating-point execution on `s390x` may be used for:

- `fp32` accumulation
- mixed-precision conversion paths
- normalization steps
- scalar tail handling
- fallback implementations when integer quantization paths are unavailable or unsuitable

The single-element control capability is particularly useful when balancing correctness and throughput in kernels that mix vectorized bulk processing with scalar-like edge handling.

## Vector-Enhancements Facility 1

Vector-Enhancements Facility 1 is available on models that already implement the base Vector Facility. It extends the original instruction set with additional logical, arithmetic, and floating-point capabilities.

### Availability

- Present on systems that support the **Vector Facility**
- Extends the base vector model rather than replacing it

### New instructions

Vector-Enhancements Facility 1 adds the following instructions:

- `Vector Bit Permute`
- `Vector MULTIPLY SUM LOGICAL`
- `Vector NOT EXCLUSIVE OR`
- `Vector NAND`
- `Vector FP MAXIMUM`
- `Vector FP MINIMUM`

### Practical relevance to inference

These additions are useful in several machine learning contexts:

- **`Vector Bit Permute`**
  - useful for bit-level rearrangement
  - relevant to packed formats, lookup-table style transforms, and low-bit quantization support

- **`Vector MULTIPLY SUM LOGICAL`**
  - potentially valuable for multiply-accumulate-like integer workflows
  - relevant to quantized inference and packed logical arithmetic patterns

- **`Vector NOT EXCLUSIVE OR`** and **`Vector NAND`**
  - useful for mask construction, predicate manipulation, and compact logical transforms
  - can support optimized control-flow elimination in vectorized kernels

- **`Vector FP MAXIMUM`** and **`Vector FP MINIMUM`**
  - directly relevant to reduction-style operations
  - useful for clamping, activation support, range tracking, and numerically bounded transforms

### Enhanced population count

Vector-Enhancements Facility 1 also enhances:

- `Vector POPULATION COUNT`

The enhancement adds support for these element sizes:

- **halfword**
- **word**
- **doubleword**

This is useful for:

- bit-density analysis
- sparse mask handling
- packed-state inspection
- binary or low-bit model support paths
- internal runtime bookkeeping where bit counts are performance-sensitive

## Vector-Enhancements Facility 2

Vector-Enhancements Facility 2 extends Vector-Enhancements Facility 1 and adds both new instructions and targeted performance-oriented improvements.

### Scope of enhancement

This facility improves support for:

- **little-endian format**
- **vector shifting**
- **substring search**
- **conversion support for short-format arithmetic**

These areas are especially relevant when porting software from ecosystems where little-endian assumptions are common, which includes most modern ML kernel implementations originally developed for `x86_64` and `ARM64`.

### New instructions

Vector-Enhancements Facility 2 introduces the following instructions:

- `VLBR`
- `VLER`
- `VLLEBRZ`
- `VLEBRH`
- `VLEBRF`
- `VLEBRG`
- `VLBRREP`
- `VSTBR`
- `VSTER`
- `VSTEBRH`
- `VSTEBRF`
- `VSTEBRG`
- `VSLD`
- `VSRD`
- `VSTRS`

These instructions improve the efficiency of several common low-level tasks:

- endian-aware vector loads and stores
- replicated or rearranged load patterns
- double-length shift operations
- substring search and scan behavior
- conversion and movement of short-format data

### Alternate forms

Vector-Enhancements Facility 2 also provides alternate forms for:

- `VSL`
- `VSRA`
- `VSRL`
- `VCFPS`
- `VCSFP`
- `VCLFP`

These alternate forms are important because they can reduce instruction count or improve mapping for common compiler-generated patterns, especially in code that performs:

- shifts across vector boundaries
- arithmetic and logical right shifts
- floating-point conversion between formats
- short-format arithmetic preparation and normalization

### Relevance to machine learning inference

For inference workloads, Vector-Enhancements Facility 2 is particularly valuable because it addresses several practical bottlenecks encountered during porting and optimization:

- **Little-endian handling**
  - many ML formats, tensor layouts, and serialization paths assume little-endian behavior
  - improved endian-aware vector instructions reduce the need for manual byte shuffling

- **Vector shifting**
  - useful for packing, unpacking, sliding-window operations, and quantized data alignment
  - important in kernels that assemble partial products or manipulate sub-byte representations

- **Substring search**
  - less relevant to dense tensor math, but useful in tokenizer, parser, and runtime support code

- **Short-format conversion**
  - important when moving between compact storage formats and compute-friendly formats
  - relevant to mixed-precision and quantized inference pipelines

## Portability Considerations

Porting vectorized code from `x86_64` to `s390x` requires careful attention to architectural differences. Even when both platforms provide SIMD-style execution, the programming model, instruction semantics, and performance characteristics are not interchangeable.

### Risks when translating from `x86_64` SIMD

Common risks include:

- **Instruction-set mismatch**
  - `x86_64` SIMD code may rely on SSE, AVX, AVX2, or AVX-512 idioms that have no direct one-to-one equivalent on `s390x`

- **Different lane semantics**
  - operations that appear conceptually similar may differ in lane width assumptions, saturation behavior, reduction support, or mask handling

- **Endian assumptions**
  - code written with implicit little-endian expectations may require explicit adaptation on `s390x`

- **Load/store behavior differences**
  - alignment expectations, byte ordering, and shuffle strategies may need redesign rather than direct translation

- **Compiler intrinsic portability**
  - architecture-specific intrinsics are generally not portable across SIMD families
  - hand-tuned kernels often need separate implementations

- **Reduction and horizontal operation differences**
  - dot products, horizontal sums, min/max reductions, and population-count-based logic may need different decomposition strategies

- **Tail processing**
  - kernels optimized for one vector width or masking model may require different scalar or partial-vector cleanup logic

### Performance considerations for CPU-only inference

For `ollama-s390x`, CPU-only inference performance depends on more than simply enabling vector instructions.

Important considerations include:

- **Memory bandwidth and cache locality**
  - inference kernels are often limited by data movement rather than arithmetic throughput
  - tensor blocking and layout choices remain critical

- **Quantized arithmetic efficiency**
  - integer vector support is highly relevant, but performance depends on how well quantized formats map onto available instructions

- **Instruction selection quality**
  - compiler-generated vector code may not match the quality of architecture-aware handwritten kernels
  - profiling is necessary to determine whether intrinsics or assembly are justified

- **Endian conversion overhead**
  - if model data or intermediate layouts require frequent byte-order adaptation, conversion cost can erode vectorization gains

- **Mixed scalar/vector execution**
  - some inference paths include scalar control logic, irregular indexing, or small tails that reduce effective SIMD utilization

- **Kernel portability trade-offs**
  - a portable generic implementation may be easier to maintain but leave performance on the table
  - an `s390x`-specific kernel may improve throughput but increase maintenance burden

- **Workload sensitivity**
  - token generation, prompt processing, embedding, and preprocessing may stress different parts of the CPU
  - not all stages benefit equally from vector acceleration

### Practical guidance

When targeting `s390x` for inference:

- prefer architecture-aware benchmarking over assumptions based on `x86_64`
- validate endian-sensitive code paths explicitly
- isolate vectorized kernels behind clean abstractions
- keep scalar fallbacks for correctness and portability
- profile real inference workloads, not only microbenchmarks
- prioritize kernels with high arithmetic intensity and frequent reuse

## Summary

The `s390x` vector architecture provides a capable 128-bit vector execution model through the base Vector Facility and its enhancement levels. For machine learning inference, the most relevant strengths are:

- fixed-width vector parallelism through **32 128-bit registers**
- integer, string, and floating-point vector instruction subsets
- enhancement facilities that improve bit manipulation, reductions, endian-aware data movement, shifting, and conversion
- a viable foundation for CPU-only optimization in architecture-specific inference kernels

At the same time, successful use in `ollama-s390x` depends on careful portability work. Direct translation from `x86_64` SIMD is risky, and performance must be validated with workload-specific profiling, especially for quantized and mixed-precision inference paths.