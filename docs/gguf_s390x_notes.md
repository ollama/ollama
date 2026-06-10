# GGUF Format Documentation for s390x Architecture

## Overview

This document provides comprehensive information about the GGUF (GPT-Generated Unified Format) file format and its implementation considerations for IBM Z and LinuxONE s390x architecture, which uses Big-Endian byte ordering.

## Table of Contents

1. [Endianness Considerations](#endianness-considerations)
2. [Quantization Formats](#quantization-formats)
3. [Model File Conversion](#model-file-conversion)
4. [SIMD Implementation](#simd-implementation)
5. [Architecture-Specific Notes](#architecture-specific-notes)

---

## Endianness Considerations

### Background

**Byte Order Fundamentals:**
- **Little-Endian**: Least significant byte stored at the lowest memory address (most common in modern CPUs)
- **Big-Endian**: Most significant byte stored at the lowest memory address (used by IBM Z and LinuxONE)

### Architecture Differences

| Architecture Type | Byte Order | Examples |
|------------------|------------|----------|
| Most CPU Architectures | Little-Endian | x86, x86_64, ARM (typically), RISC-V |
| IBM Z / LinuxONE | Big-Endian | s390x |

### Historical Context

**Original llama.cpp Design:**
- Built with Little-Endian byte order as the primary assumption
- Pre-quantized GGUF v2/v3 binaries on HuggingFace compiled using Little-Endian byte ordering
- These binaries are **incompatible** with s390x architecture without conversion

**Resolution:**
- Endianness issues have been fixed in llama.cpp
- Big-Endian byte order support was introduced in **GGUFv3**
- llama.cpp can now load Big-Endian byte order models natively

### Working with s390x

To use GGUF models on s390x architecture, you have two options:

1. **Compile Locally**: Build models from source on s390x hardware
2. **Use Big-Endian Binaries**: Look for repositories with "BE" (Big-Endian) suffix in their names

**Example:**
```bash
# Look for repositories like:
# model-name-gguf-BE
# model-name-q4_0-BE.gguf
```

---

## Quantization Formats

GGUF supports multiple quantization formats that balance model size, inference speed, and accuracy. Below are the formats relevant to s390x implementation:

### Q4_8 Quantization Format

**Description:**
- Mixed precision quantization format
- Optimized for reduced memory footprint while maintaining reasonable accuracy

**Technical Details:**
- **Weights**: Quantized to `block_q4_8` (4-bit quantization)
- **Activations**: Quantized to `block_q8_8` (8-bit quantization)

**Use Cases:**
- Memory-constrained environments
- Faster inference with acceptable quality trade-off
- Good balance between size and performance

**Block Structure:**
```
block_q4_8:
  - 4 bits per weight value
  - Grouped in blocks for efficient processing
  
block_q8_8:
  - 8 bits per activation value
  - Higher precision for intermediate computations
```

### Q8_8 Quantization Format

**Description:**
- Uniform 8-bit quantization for both weights and activations
- Higher precision than Q4_8 with larger memory footprint

**Technical Details:**
- **Weights**: Quantized to `block_q8_8` (8-bit quantization)
- **Activations**: Quantized to `block_q8_8` (8-bit quantization)

**Use Cases:**
- Better accuracy requirements
- Systems with more available memory
- When quality is prioritized over size

**Advantages:**
- More consistent precision throughout the model
- Reduced quantization error compared to Q4_8
- Better suited for tasks requiring higher fidelity

### FP16 (16-bit Floating Point)

**Description:**
- Half-precision floating-point format
- Native support without quantization/dequantization overhead

**Technical Details:**
- **Weights**: FP16 data type (IEEE 754 half-precision)
- **Activations**: FP16 data type
- **Direct Computation**: Dot products computed directly without upscaling or conversion

**Advantages:**
- No quantization artifacts
- Direct hardware support on many accelerators
- Simpler implementation (no dequantization required)

**Memory Requirements:**
- 2 bytes per parameter
- Larger than quantized formats but smaller than FP32

**Performance Characteristics:**
```
FP16 Benefits:
✓ No conversion overhead
✓ Direct dot product computation
✓ Hardware acceleration support
✓ Better numerical stability than lower precision
```

### BF16 (Brain Float 16)

**Status:** ⚠️ **DOCUMENTATION PENDING**

**Description:**
- Alternative 16-bit floating-point format
- Developed by Google Brain for machine learning workloads

**Known Characteristics:**
- Same exponent range as FP32 (8 bits)
- Reduced mantissa precision (7 bits vs 10 bits in FP16)
- Better dynamic range than FP16

**TODO:**
```
[ ] Research BF16 support in GGUF format
[ ] Determine s390x hardware support for BF16
[ ] Document conversion requirements
[ ] Identify performance characteristics on s390x
[ ] Compare with FP16 for s390x workloads
```

**Placeholder for Future Documentation:**
- Hardware support on s390x
- Conversion procedures
- Performance benchmarks
- Use case recommendations
- Implementation details

---

## Model File Conversion

### GGUFv3 and Big-Endian Support

**Key Milestone:**
- Big-Endian byte order support introduced in **GGUFv3**
- llama.cpp can now load Big-Endian models natively

### Conversion Requirements

**When Conversion is NOT Needed:**
- Using GGUFv3 or later format
- Model already in Big-Endian byte order
- Building from source on s390x

**When Conversion IS Needed:**
- Using pre-quantized Little-Endian binaries from HuggingFace
- Migrating from GGUFv2 to GGUFv3
- Converting between different quantization formats

### Conversion Process

```bash
# Example conversion workflow (conceptual)
# 1. Download Little-Endian model
wget https://huggingface.co/model/model-q4_0.gguf

# 2. Convert to Big-Endian (if needed)
# Use llama.cpp conversion tools with appropriate flags

# 3. Verify byte order
# Check GGUF header for endianness marker
```

**Best Practices:**
1. Always verify byte order after conversion
2. Test inference on small inputs before production use
3. Keep original files as backup
4. Document conversion parameters used

---

## SIMD Implementation

### Single Instruction Multiple Data (SIMD)

**Status:** ✅ Can be implemented relatively easily on s390x

**Overview:**
- SIMD allows parallel processing of multiple data elements
- Critical for efficient matrix operations in neural networks
- s390x architecture has SIMD capabilities

**s390x SIMD Features:**
- Vector facility support
- Parallel processing capabilities
- Optimized for Big-Endian operations

**Implementation Considerations:**
```
SIMD on s390x:
✓ Vector instructions available
✓ Relatively straightforward implementation
✓ Can leverage existing llama.cpp SIMD abstractions
✓ Performance gains for matrix operations
```

**Optimization Opportunities:**
- Vectorized dot products
- Parallel quantization/dequantization
- Batch processing of activations
- Efficient memory access patterns

---

## Architecture-Specific Notes

### s390x Compatibility Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| GGUFv3 Support | ✅ Supported | Native Big-Endian support |
| Q4_8 Quantization | ✅ Supported | Requires Big-Endian format |
| Q8_8 Quantization | ✅ Supported | Requires Big-Endian format |
| FP16 | ✅ Supported | Direct computation available |
| BF16 | ⚠️ TBD | Documentation pending |
| SIMD | ✅ Supported | Relatively easy implementation |

### Performance Considerations

**Memory Access:**
- Big-Endian byte order is native to s390x
- No conversion overhead for properly formatted models
- Efficient cache utilization with aligned data structures

**Compute Efficiency:**
- SIMD instructions available for parallel processing
- Vector facility optimizations
- Potential for hardware-specific tuning

### Debugging Tips

**Endianness Issues:**
```bash
# Check GGUF file byte order
hexdump -C model.gguf | head -n 20

# Look for GGUF magic number and version
# Verify endianness marker in header
```

**Common Issues:**
1. Loading Little-Endian models on s390x → Conversion needed
2. Incorrect quantization format → Verify model metadata
3. Performance degradation → Check SIMD utilization

---

## References

### Related Documentation
- [llama.cpp GGUF Specification](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- [s390x Architecture Guide](https://www.ibm.com/docs/en/linux-on-systems)
- IBM Z and LinuxONE documentation

### Version History
- **GGUFv2**: Little-Endian only
- **GGUFv3**: Big-Endian support added ✅

### Contributing

If you have additional information about:
- BF16 support on s390x
- Performance benchmarks
- Optimization techniques
- Conversion tools

Please contribute to this documentation.

---

## Appendix: Quick Reference

### Byte Order Check
```python
import sys
print(f"System byte order: {sys.byteorder}")
# s390x should report: 'big'
```

### Quantization Format Selection Guide

| Priority | Recommended Format | Reason |
|----------|-------------------|--------|
| Maximum Quality | FP16 | No quantization loss |
| Balanced | Q8_8 | Good quality, moderate size |
| Minimum Size | Q4_8 | Smallest footprint |
| TBD | BF16 | Pending investigation |

---

**Last Updated:** 2026-06-10  
**Status:** Living document - BF16 section pending research