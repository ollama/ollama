# Critical Issues Backlog

This document tracks critical technical issues preventing Ollama from running on IBM mainframe (s390x) architecture.

---

## 1. Endianness Incompatibility

### Issue Description
Endianness incompatibility prevents Large Language Models (LLMs) from running locally on IBM mainframes with s390x architecture.

### Technical Details
- **Architecture**: IBM mainframes use s390x architecture, which is big-endian
- **Problem**: Most LLM model files and inference engines are designed for little-endian systems (x86_64, ARM)
- **Impact**: Model weights, tensors, and binary data structures are stored in little-endian format
- **Consequence**: When loaded on big-endian s390x systems, the byte order is incorrect, causing:
  - Corrupted model weights
  - Invalid tensor operations
  - Incorrect inference results
  - Runtime crashes and segmentation faults

### Root Cause
The underlying issue stems from:
1. Model serialization formats (GGUF, SafeTensors, PyTorch) defaulting to little-endian byte order
2. Lack of endianness detection and conversion in model loading routines
3. Binary compatibility assumptions in llama.cpp and related inference libraries

### Required Solution
- Implement endianness detection at runtime
- Add byte-swapping routines for model loading on big-endian systems
- Ensure all tensor operations handle endianness correctly
- Test model conversion and inference pipelines on s390x architecture

---

## 2. llama.cpp Runtime Issues

### Issue Description
Significant struggles encountered when attempting to run llama.cpp on IBM mainframe infrastructure.

### Technical Details
- **Component**: llama.cpp inference engine
- **Platform**: s390x architecture / IBM mainframe
- **Status**: Unable to successfully execute llama.cpp runtime

### Known Problems
1. **Build Failures**: Compilation issues specific to s390x architecture
2. **Runtime Crashes**: Segmentation faults and unexpected terminations
3. **Performance Issues**: Suboptimal or non-functional execution even when builds succeed
4. **Architecture-Specific Code**: x86/ARM-specific optimizations and intrinsics not compatible with s390x

### Dependencies Affected
- Core inference engine (llama.cpp)
- GGML tensor library
- SIMD optimizations
- Memory management routines

### Investigation Needed
- Detailed error logs from build attempts
- Runtime debugging on s390x systems
- Architecture-specific code paths that need porting
- Alternative inference engines compatible with big-endian systems

---

## 3. Jupyter Notebook Container Access

### Issue Description
Unable to run Jupyter notebook on the container that accesses the mainframe, preventing interactive development and testing workflows.

### Technical Details
- **Component**: Jupyter Notebook server
- **Environment**: Container accessing IBM mainframe
- **Impact**: Blocks interactive development, testing, and demonstration capabilities

### Specific Problems
1. **Container Configuration**: Jupyter server fails to start or bind to accessible ports
2. **Network Access**: Connectivity issues between container and mainframe resources
3. **Authentication**: Potential authentication/authorization barriers
4. **Resource Constraints**: Container may lack necessary resources or permissions

### Development Impact
- Cannot perform interactive model testing
- Unable to create demonstration notebooks
- Blocks exploratory data analysis workflows
- Prevents documentation of working examples

### Required Investigation
- Container logs and error messages
- Network configuration and port mappings
- Jupyter server configuration
- Container resource allocation and permissions
- Alternative approaches for interactive development on mainframe

---

## Priority Assessment

| Issue | Severity | Priority | Blocking |
|-------|----------|----------|----------|
| Endianness Incompatibility | Critical | High | Yes - Core functionality |
| llama.cpp Runtime Issues | Critical | High | Yes - Inference engine |
| Jupyter Notebook Access | High | Medium | No - Development tooling |

## Next Steps

1. **Immediate**: Focus on endianness compatibility research and solutions
2. **Short-term**: Debug and resolve llama.cpp runtime issues on s390x
3. **Medium-term**: Establish working Jupyter notebook environment for development

---

*Last Updated: 2026-06-08*