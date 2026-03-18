// DirectML GGML backend for NPU/GPU acceleration on Windows
//
// This backend uses Microsoft's DirectML API (built on Direct3D 12) to accelerate
// tensor operations on GPUs and NPUs. DirectML provides hardware-agnostic ML
// acceleration across Intel, AMD, NVIDIA, and Qualcomm devices on Windows.
//
// Initial implementation supports FP32 and FP16 operations. Quantized formats
// (Q4_0, Q8_0, etc.) are dequantized on CPU before transfer to device.

#include "ggml-directml.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
// DXCore headers for NPU/MCDM device discovery. We dynamically load dxcore.dll
// so the binary runs on systems without it. Include <initguid.h> first to get
// GUID definitions inline (we don't link dxcore.lib).
#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// DirectML SDK header (from Windows SDK)
// ---------------------------------------------------------------------------

// DirectML.h provides all DML interfaces: IDMLDevice1, IDMLCompiledOperator,
// IDMLOperatorInitializer, IDMLBindingTable, IDMLCommandRecorder, etc.
// We dynamically load DirectML.dll so the binary can run on systems without it.

#include <DirectML.h>

#include <functional>
#include <unordered_map>

typedef HRESULT (WINAPI *PFN_DMLCreateDevice1)(
    ID3D12Device *d3d12Device, DML_CREATE_DEVICE_FLAGS flags,
    DML_FEATURE_LEVEL minFeatureLevel, REFIID riid, void **ppv);

static HMODULE                 s_dml_module = nullptr;
static PFN_DMLCreateDevice1    s_DMLCreateDevice1 = nullptr;

static bool dml_load_library() {
    if (s_dml_module) return true;
    s_dml_module = LoadLibraryW(L"DirectML.dll");
    if (!s_dml_module) {
        GGML_LOG_DEBUG("%s: DirectML.dll not found\n", __func__);
        return false;
    }
    s_DMLCreateDevice1 = (PFN_DMLCreateDevice1)GetProcAddress(s_dml_module, "DMLCreateDevice1");
    if (!s_DMLCreateDevice1) {
        // Fall back to DMLCreateDevice (older API)
        GGML_LOG_DEBUG("%s: DMLCreateDevice1 not found, DirectML too old\n", __func__);
        FreeLibrary(s_dml_module);
        s_dml_module = nullptr;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// D3D12 Compute Shader Infrastructure (for custom HLSL ops)
// ---------------------------------------------------------------------------

// D3DCompile loaded dynamically from d3dcompiler_47.dll
typedef HRESULT (WINAPI *PFN_D3DCompile)(
    LPCVOID pSrcData, SIZE_T SrcDataSize, LPCSTR pSourceName,
    const void * pDefines, void * pInclude,
    LPCSTR pEntrypoint, LPCSTR pTarget,
    UINT Flags1, UINT Flags2,
    ID3DBlob ** ppCode, ID3DBlob ** ppErrorMsgs);

static PFN_D3DCompile s_D3DCompile = nullptr;

static bool dml_load_d3dcompiler() {
    static bool tried = false;
    static bool loaded = false;
    if (tried) return loaded;
    tried = true;

    HMODULE mod = LoadLibraryW(L"d3dcompiler_47.dll");
    if (!mod) {
        GGML_LOG_DEBUG("%s: d3dcompiler_47.dll not found\n", __func__);
        return false;
    }
    s_D3DCompile = (PFN_D3DCompile)GetProcAddress(mod, "D3DCompile");
    loaded = (s_D3DCompile != nullptr);
    return loaded;
}

// ---------------------------------------------------------------------------
// DXCore dynamic loading (for NPU/MCDM device discovery)
// ---------------------------------------------------------------------------
// DXCore can enumerate compute-only devices (NPUs) that DXGI cannot see.
// Loaded dynamically so the binary runs on systems without dxcore.dll.

typedef HRESULT (WINAPI *PFN_DXCoreCreateAdapterFactory)(REFIID riid, void **ppvFactory);

static HMODULE                         s_dxcore_module = nullptr;
static PFN_DXCoreCreateAdapterFactory  s_DXCoreCreateAdapterFactory = nullptr;

static bool dml_load_dxcore() {
    static bool tried = false;
    if (tried) return (s_dxcore_module != nullptr);
    tried = true;

    s_dxcore_module = LoadLibraryW(L"dxcore.dll");
    if (!s_dxcore_module) {
        GGML_LOG_DEBUG("%s: dxcore.dll not found — NPU discovery unavailable\n", __func__);
        return false;
    }
    s_DXCoreCreateAdapterFactory = (PFN_DXCoreCreateAdapterFactory)
        GetProcAddress(s_dxcore_module, "DXCoreCreateAdapterFactory");
    if (!s_DXCoreCreateAdapterFactory) {
        GGML_LOG_DEBUG("%s: DXCoreCreateAdapterFactory not found in dxcore.dll\n", __func__);
        FreeLibrary(s_dxcore_module);
        s_dxcore_module = nullptr;
        return false;
    }
    GGML_LOG_INFO("%s: dxcore.dll loaded — NPU discovery available\n", __func__);
    return true;
}

// HLSL compute shader sources

// Element-wise ADD with broadcasting: dst[i] = src0[i] + src1[i % n_src1]
static const char * s_hlsl_add = R"(
cbuffer Params : register(b0) {
    uint n_elements;
    uint n_src1;
    uint ne00;
    uint src0_off;
    uint src1_off;
    uint dst_off;
};
RWByteAddressBuffer buf_src0 : register(u0);
RWByteAddressBuffer buf_src1 : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_elements) return;
    uint col = dtid.x % ne00;
    uint src1_idx = col + (dtid.x / ne00 % (n_src1 / ne00)) * ne00;
    float a = asfloat(buf_src0.Load(src0_off + dtid.x * 4));
    float b = asfloat(buf_src1.Load(src1_off + src1_idx * 4));
    buf_dst.Store(dst_off + dtid.x * 4, asuint(a + b));
}
)";

// Element-wise MUL with broadcasting: dst[i] = src0[i] * src1[i % n_src1]
// Supports row broadcasting where src1 has fewer rows than src0.
static const char * s_hlsl_mul = R"(
cbuffer Params : register(b0) {
    uint n_elements;  // total elements in dst
    uint n_src1;      // total elements in src1 (for broadcast modulo)
    uint ne00;        // row width (innermost dim)
    uint src0_off;
    uint src1_off;
    uint dst_off;
};
RWByteAddressBuffer buf_src0 : register(u0);
RWByteAddressBuffer buf_src1 : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_elements) return;
    // For broadcasting: element index within the row is the same,
    // but the row index wraps around src1's row count
    uint col = dtid.x % ne00;
    uint src1_idx = col + (dtid.x / ne00 % (n_src1 / ne00)) * ne00;
    float a = asfloat(buf_src0.Load(src0_off + dtid.x * 4));
    float b = asfloat(buf_src1.Load(src1_off + src1_idx * 4));
    buf_dst.Store(dst_off + dtid.x * 4, asuint(a * b));
}
)";

// SCALE: dst[i] = src0[i] * scale_factor
static const char * s_hlsl_scale = R"(
cbuffer Params : register(b0) {
    uint n_elements;
    uint scale_bits;  // float reinterpreted as uint
    uint src_off;
    uint dst_off;
};
RWByteAddressBuffer buf_src : register(u0);
RWByteAddressBuffer buf_dst : register(u1);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_elements) return;
    float x = asfloat(buf_src.Load(src_off + dtid.x * 4));
    float s = asfloat(scale_bits);
    buf_dst.Store(dst_off + dtid.x * 4, asuint(x * s));
}
)";

// SOFT_MAX: numerically stable softmax over the innermost dimension
// Applies: val = scale * src[i] + mask[i]  (mask is optional, from src[1])
// Each workgroup processes one row. Uses shared memory reduction.
static const char * s_hlsl_soft_max = R"(
cbuffer Params : register(b0) {
    uint ne00;        // row width (innermost dim)
    uint nrows;       // total number of rows in dst
    uint scale_bits;  // scale factor as float bits
    uint has_mask;    // 1 if mask present, 0 otherwise
    uint src_off;     // byte offset into src buffer
    uint mask_off;    // byte offset into mask buffer (if has_mask)
    uint dst_off;     // byte offset into dst buffer
    uint mask_nb1;    // mask row stride in bytes
    uint mask_nrows;  // total rows in mask (for wrapping)
};
RWByteAddressBuffer buf_src  : register(u0);
RWByteAddressBuffer buf_mask : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
groupshared float sdata[256];
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= nrows) return;
    uint t = tid.x;
    float scale = asfloat(scale_bits);
    uint base_s = src_off + row * ne00 * 4;
    uint base_d = dst_off + row * ne00 * 4;
    uint mask_row_base = mask_off + (row % mask_nrows) * mask_nb1;

    // Pass 1: compute scaled+masked values and find max
    float max_val = -3.402823466e+38;
    for (uint i = t; i < ne00; i += 256) {
        float v = asfloat(buf_src.Load(base_s + i * 4)) * scale;
        if (has_mask) {
            v += asfloat(buf_mask.Load(mask_row_base + i * 4));
        }
        max_val = max(max_val, v);
    }
    sdata[t] = max_val;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sdata[t] = max(sdata[t], sdata[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    max_val = sdata[0];

    // Pass 2: compute sum of exp(val - max)
    float sum_exp = 0.0;
    for (uint i = t; i < ne00; i += 256) {
        float v = asfloat(buf_src.Load(base_s + i * 4)) * scale;
        if (has_mask) {
            v += asfloat(buf_mask.Load(mask_row_base + i * 4));
        }
        sum_exp += exp(v - max_val);
    }
    sdata[t] = sum_exp;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sdata[t] += sdata[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_sum = 1.0 / sdata[0];

    // Pass 3: write normalized values
    for (uint i = t; i < ne00; i += 256) {
        float v = asfloat(buf_src.Load(base_s + i * 4)) * scale;
        if (has_mask) {
            v += asfloat(buf_mask.Load(mask_row_base + i * 4));
        }
        buf_dst.Store(base_d + i * 4, asuint(exp(v - max_val) * inv_sum));
    }
}
)";

static const char * s_hlsl_silu = R"(
cbuffer Params : register(b0) {
    uint n_elements;
    uint src_off;
    uint dst_off;
};
RWByteAddressBuffer buf_src : register(u0);
RWByteAddressBuffer buf_dst : register(u1);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_elements) return;
    float x = asfloat(buf_src.Load(src_off + dtid.x * 4));
    buf_dst.Store(dst_off + dtid.x * 4, asuint(x / (1.0 + exp(-x))));
}
)";

static const char * s_hlsl_rms_norm = R"(
cbuffer Params : register(b0) {
    uint ne00;
    uint nrows;
    uint eps_bits;
    uint src_off;
    uint dst_off;
};
RWByteAddressBuffer buf_src : register(u0);
RWByteAddressBuffer buf_dst : register(u1);
groupshared float sdata[256];
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= nrows) return;
    uint t = tid.x;
    uint base_s = src_off + row * ne00 * 4;
    uint base_d = dst_off + row * ne00 * 4;
    float sum = 0.0;
    for (uint i = t; i < ne00; i += 256) {
        float v = asfloat(buf_src.Load(base_s + i * 4));
        sum += v * v;
    }
    sdata[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sdata[t] += sdata[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float scale = rsqrt(sdata[0] / (float)ne00 + asfloat(eps_bits));
    for (uint i = t; i < ne00; i += 256) {
        float v = asfloat(buf_src.Load(base_s + i * 4));
        buf_dst.Store(base_d + i * 4, asuint(v * scale));
    }
}
)";

static const char * s_hlsl_diag_mask_inf = R"(
cbuffer Params : register(b0) {
    uint ne00;
    uint ne01;
    int n_past;
    uint n_total;
    uint src_off;
    uint dst_off;
};
RWByteAddressBuffer buf_src : register(u0);
RWByteAddressBuffer buf_dst : register(u1);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_total) return;
    uint col = dtid.x % ne00;
    uint row = (dtid.x / ne00) % ne01;
    float val = asfloat(buf_src.Load(src_off + dtid.x * 4));
    if ((int)col > n_past + (int)row) val = asfloat(0xFF800000u);
    buf_dst.Store(dst_off + dtid.x * 4, asuint(val));
}
)";

// SET_ROWS: scatter-write F32 source rows into F16 destination at indexed positions.
// Each thread handles 2 adjacent elements (one F16 pair = one uint32 write).
// ne00 must be even (always true for embedding dimensions).
static const char * s_hlsl_set_rows_f16 = R"(
cbuffer Params : register(b0) {
    uint ne00;      // row width (elements)
    uint ne01;      // num source rows
    uint ne02;      // batch dim 2
    uint ne03;      // batch dim 3
    uint ne11;      // src1 broadcast dim 1
    uint ne12;      // src1 broadcast dim 2
    uint nb1_dst;   // dst row stride (bytes)
    uint nb2_dst;   // dst dim2 stride (bytes)
    uint nb3_dst;   // dst dim3 stride (bytes)
    uint nb10;      // src1 dim0 stride (bytes)
    uint nb11;      // src1 dim1 stride (bytes)
    uint nb12;      // src1 dim2 stride (bytes)
    uint src0_off;  // byte offset in src0 buffer
    uint src1_off;  // byte offset in src1 buffer
    uint dst_off;   // byte offset in dst buffer
    uint n_pairs;   // total F16 pairs = (ne00/2) * ne01 * ne02 * ne03
};
RWByteAddressBuffer buf_src0 : register(u0);
RWByteAddressBuffer buf_src1 : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_pairs) return;

    uint half_ne00 = ne00 / 2;
    uint idx = dtid.x;
    uint i00_pair = idx % half_ne00;
    idx /= half_ne00;
    uint i01 = idx % ne01;
    idx /= ne01;
    uint i02 = idx % ne02;
    uint i03 = idx / ne02;

    // Broadcast index lookup
    uint i10 = i01;
    uint i11 = i02 % ne11;
    uint i12 = i03 % ne12;

    // Read destination row index (supports I32 and I64 — just load low 32 bits)
    uint src1_byte = src1_off + i10 * nb10 + i11 * nb11 + i12 * nb12;
    int row_idx = asint(buf_src1.Load(src1_byte));

    // Read 2 F32 values from src0 (contiguous F32 layout)
    uint i00_base = i00_pair * 2;
    uint src0_byte = src0_off + (i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00_base) * 4;
    float v0 = asfloat(buf_src0.Load(src0_byte));
    float v1 = asfloat(buf_src0.Load(src0_byte + 4));

    // Convert to F16 and pack into one uint32
    uint packed = f32tof16(v0) | (f32tof16(v1) << 16);

    // Write to dst at scattered row position
    uint dst_byte = dst_off + (uint)row_idx * nb1_dst + i02 * nb2_dst + i03 * nb3_dst + i00_base * 2;
    buf_dst.Store(dst_byte, packed);
}
)";

// SET_ROWS: scatter-write F32 source rows into F32 destination at indexed positions.
// Each thread handles 1 element.
static const char * s_hlsl_set_rows_f32 = R"(
cbuffer Params : register(b0) {
    uint ne00;      // row width (elements)
    uint ne01;      // num source rows
    uint ne02;      // batch dim 2
    uint ne03;      // batch dim 3
    uint ne11;      // src1 broadcast dim 1
    uint ne12;      // src1 broadcast dim 2
    uint nb1_dst;   // dst row stride (bytes)
    uint nb2_dst;   // dst dim2 stride (bytes)
    uint nb3_dst;   // dst dim3 stride (bytes)
    uint nb10;      // src1 dim0 stride (bytes)
    uint nb11;      // src1 dim1 stride (bytes)
    uint nb12;      // src1 dim2 stride (bytes)
    uint src0_off;  // byte offset in src0 buffer
    uint src1_off;  // byte offset in src1 buffer
    uint dst_off;   // byte offset in dst buffer
    uint n_total;   // total elements = ne00 * ne01 * ne02 * ne03
};
RWByteAddressBuffer buf_src0 : register(u0);
RWByteAddressBuffer buf_src1 : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_total) return;

    uint idx = dtid.x;
    uint i00 = idx % ne00;
    idx /= ne00;
    uint i01 = idx % ne01;
    idx /= ne01;
    uint i02 = idx % ne02;
    uint i03 = idx / ne02;

    uint i10 = i01;
    uint i11 = i02 % ne11;
    uint i12 = i03 % ne12;

    uint src1_byte = src1_off + i10 * nb10 + i11 * nb11 + i12 * nb12;
    int row_idx = asint(buf_src1.Load(src1_byte));

    uint src0_byte = src0_off + (i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00) * 4;
    float val = asfloat(buf_src0.Load(src0_byte));

    uint dst_byte = dst_off + (uint)row_idx * nb1_dst + i02 * nb2_dst + i03 * nb3_dst + i00 * 4;
    buf_dst.Store(dst_byte, asuint(val));
}
)";

// Mixed-precision matrix multiply: A (F16) × B (F32) → C (F32)
// Implements GGML MUL_MAT: C[m, n] = sum_k A[k, m] * B[k, n]
// GGML layout: ne[0] is innermost. Element [i0, i1] at flat index i1 * ne[0] + i0.
// A shape [K, M]: a[k, m] at flat (m * K + k), stored as F16
// B shape [K, N]: b[k, n] at flat (n * K + k), stored as F32
// C shape [M, N]: c[m, n] at flat (n * M + m), stored as F32
// A may have fewer batches than B (broadcasting: a_batch_count <= batch_count).
// Naive per-element implementation for correctness testing (no shared memory).
static const char * s_hlsl_mul_mat_f16_f32 = R"(
cbuffer Params : register(b0) {
    uint M;            // a->ne[1], output ne[0]
    uint N;            // b->ne[1], output ne[1]
    uint K;            // a->ne[0] = b->ne[0], contraction dim
    uint batch_count;  // total output batches
    uint a_batch_count;// A batches (for broadcasting, typically 1 for weights)
    uint a_off;        // byte offset of A in buffer
    uint b_off;        // byte offset of B in buffer
    uint c_off;        // byte offset of C in buffer
};
RWByteAddressBuffer buf_a : register(u0);   // F16 weights
RWByteAddressBuffer buf_b : register(u1);   // F32 input
RWByteAddressBuffer buf_c : register(u2);   // F32 output

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    // Each thread computes one output element C[m, n, batch]
    uint idx = dtid.x;
    uint total = M * N * batch_count;
    if (idx >= total) return;

    uint m = idx % M;
    uint rem = idx / M;
    uint n = rem % N;
    uint batch = rem / N;

    uint a_batch_idx = batch % a_batch_count;
    uint a_base = a_off + a_batch_idx * M * K * 2;  // F16 = 2 bytes
    uint b_base = b_off + batch * K * N * 4;         // F32 = 4 bytes

    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        // A[k, m] at flat offset (m * K + k), F16
        uint a_addr = a_base + (m * K + k) * 2;
        uint a_word = buf_a.Load(a_addr & ~3u);
        uint a_shift = (a_addr & 2u) ? 16 : 0;
        float a_val = f16tof32((a_word >> a_shift) & 0xFFFF);

        // B[k, n] at flat offset (n * K + k), F32
        uint b_addr = b_base + (n * K + k) * 4;
        float b_val = asfloat(buf_b.Load(b_addr));

        acc += a_val * b_val;
    }

    // C[m, n] at flat offset (n * M + m), F32
    uint c_addr = c_off + (batch * M * N + n * M + m) * 4;
    buf_c.Store(c_addr, asuint(acc));
}
)";

// F32 matrix multiply: A (F32) × B (F32) → C (F32)
// Same as above but with F32 A tensor.
// Naive per-element implementation for correctness testing.
static const char * s_hlsl_mul_mat_f32_f32 = R"(
cbuffer Params : register(b0) {
    uint M;
    uint N;
    uint K;
    uint batch_count;
    uint a_batch_count;
    uint a_off;
    uint b_off;
    uint c_off;
};
RWByteAddressBuffer buf_a : register(u0);
RWByteAddressBuffer buf_b : register(u1);
RWByteAddressBuffer buf_c : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    uint total = M * N * batch_count;
    if (idx >= total) return;

    uint m = idx % M;
    uint rem = idx / M;
    uint n = rem % N;
    uint batch = rem / N;

    uint a_batch_idx = batch % a_batch_count;
    uint a_base = a_off + a_batch_idx * M * K * 4;  // F32 = 4 bytes
    uint b_base = b_off + batch * K * N * 4;

    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        float a_val = asfloat(buf_a.Load(a_base + (m * K + k) * 4));
        float b_val = asfloat(buf_b.Load(b_base + (n * K + k) * 4));
        acc += a_val * b_val;
    }

    uint c_addr = c_off + (batch * M * N + n * M + m) * 4;
    buf_c.Store(c_addr, asuint(acc));
}
)";

// CONT: strided-to-contiguous copy (implements GGML_OP_CONT / GGML_OP_DUP)
// Each thread copies one element from a strided source to contiguous destination.
// Source strides nb00..nb03 are in *bytes* (divided by 4 to get float indices).
static const char * s_hlsl_cont = R"(
cbuffer Params : register(b0) {
    uint ne0;       // dst innermost dim
    uint ne1;       // dst dim 1
    uint ne2;       // dst dim 2
    uint ne3;       // dst dim 3
    uint nb00;      // src stride dim 0 (bytes)
    uint nb01;      // src stride dim 1 (bytes)
    uint nb02;      // src stride dim 2 (bytes)
    uint nb03;      // src stride dim 3 (bytes)
    uint src_off;   // byte offset into src buffer
    uint dst_off;   // byte offset into dst buffer
    uint n_total;   // total elements
};
RWByteAddressBuffer buf_src : register(u0);
RWByteAddressBuffer buf_dst : register(u1);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_total) return;
    uint idx = dtid.x;
    uint i0 = idx % ne0; idx /= ne0;
    uint i1 = idx % ne1; idx /= ne1;
    uint i2 = idx % ne2;
    uint i3 = idx / ne2;
    uint src_byte = src_off + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    float val = asfloat(buf_src.Load(src_byte));
    buf_dst.Store(dst_off + dtid.x * 4, asuint(val));
}
)";

// ROPE: Rotary Position Embedding (normal + neox modes)
// Each thread processes one pair of elements.
// Normal mode: pairs are adjacent (x[2i], x[2i+1])
// Neox mode: pairs are split-half (x[i], x[i + n_dims/2])
static const char * s_hlsl_rope = R"(
cbuffer Params : register(b0) {
    uint ne0;           // head dimension
    uint ne1;           // number of heads
    uint ne2;           // sequence length
    uint n_dims;        // dims to apply rope to (must be even, <= ne0)
    uint mode;          // 0=normal, 2=neox
    uint src_off;       // byte offset of src0
    uint dst_off;       // byte offset of dst
    uint pos_off;       // byte offset of position indices (src1, int32)
    uint freq_base_bits;   // float as uint bits
    uint freq_scale_bits;  // float as uint bits
    uint ne3;           // batch dim
    uint nb00;          // src stride dim 0 (bytes)
    uint nb01;          // src stride dim 1 (bytes)
    uint nb02;          // src stride dim 2 (bytes)
    uint nb03;          // src stride dim 3 (bytes)
};
RWByteAddressBuffer buf_src  : register(u0);
RWByteAddressBuffer buf_pos  : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    // Each thread handles one pair: (i0_pair, i1, i2, i3)
    uint n_pairs = (n_dims / 2) * ne1 * ne2 * ne3;
    if (dtid.x >= n_pairs) return;

    uint idx = dtid.x;
    uint i_pair = idx % (n_dims / 2); idx /= (n_dims / 2);
    uint i1 = idx % ne1; idx /= ne1;
    uint i2 = idx % ne2;
    uint i3 = idx / ne2;

    // Position index for this sequence position
    int pos = asint(buf_pos.Load(pos_off + i2 * 4));

    // Compute theta for this pair
    float freq_base = asfloat(freq_base_bits);
    float freq_scale = asfloat(freq_scale_bits);
    float theta = (float)pos * freq_scale * pow(freq_base, -2.0 * (float)i_pair / (float)n_dims);
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    bool is_neox = (mode & 2) != 0;

    // Source element indices for this pair
    uint i0_a, i0_b;
    if (is_neox) {
        i0_a = i_pair;
        i0_b = i_pair + n_dims / 2;
    } else {
        i0_a = i_pair * 2;
        i0_b = i_pair * 2 + 1;
    }

    // Read source values using strides
    uint sa = src_off + i0_a * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    uint sb = src_off + i0_b * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    float x0 = asfloat(buf_src.Load(sa));
    float x1 = asfloat(buf_src.Load(sb));

    // Rotate
    float r0 = x0 * cos_t - x1 * sin_t;
    float r1 = x0 * sin_t + x1 * cos_t;

    // Write to contiguous dest
    uint da = dst_off + (((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0_a) * 4;
    uint db = dst_off + (((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0_b) * 4;
    buf_dst.Store(da, asuint(r0));
    buf_dst.Store(db, asuint(r1));
}
)";

// ROPE passthrough: copy elements outside n_dims range untouched
// This handles the remaining channels (i0 >= n_dims) that ROPE doesn't rotate.
static const char * s_hlsl_rope_passthrough = R"(
cbuffer Params : register(b0) {
    uint ne0;
    uint ne1;
    uint ne2;
    uint n_dims;
    uint src_off;
    uint dst_off;
    uint ne3;
    uint nb00;
    uint nb01;
    uint nb02;
    uint nb03;
    uint n_total;   // (ne0 - n_dims) * ne1 * ne2 * ne3
};
RWByteAddressBuffer buf_src : register(u0);
RWByteAddressBuffer buf_dst : register(u1);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_total) return;
    uint pass_width = ne0 - n_dims;
    uint idx = dtid.x;
    uint ip = idx % pass_width; idx /= pass_width;
    uint i1 = idx % ne1; idx /= ne1;
    uint i2 = idx % ne2;
    uint i3 = idx / ne2;
    uint i0 = n_dims + ip;
    uint s = src_off + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    float val = asfloat(buf_src.Load(s));
    uint d = dst_off + (((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0) * 4;
    buf_dst.Store(d, asuint(val));
}
)";

// GLU/SWIGLU: dst[i] = silu(gate[i]) * up[i]
// Single-tensor mode: first half = gate, second half = up (or swapped)
// Split mode (src1 present): src0 = gate, src1 = up
static const char * s_hlsl_swiglu = R"(
cbuffer Params : register(b0) {
    uint nc;        // output row width (ne0/2 for single, ne0 for split)
    uint nrows;     // total output rows
    uint swapped;   // boolean: swap gate/up halves
    uint has_src1;  // boolean: split mode (src1 is separate up tensor)
    uint ne00;      // src0 row width (original ne0)
    uint src0_off;  // byte offset of src0
    uint src1_off;  // byte offset of src1 (only if has_src1)
    uint dst_off;   // byte offset of dst
};
RWByteAddressBuffer buf_src0 : register(u0);
RWByteAddressBuffer buf_src1 : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint total = nc * nrows;
    if (dtid.x >= total) return;
    uint col = dtid.x % nc;
    uint row = dtid.x / nc;
    float gate_val, up_val;
    if (has_src1) {
        gate_val = asfloat(buf_src0.Load(src0_off + (row * ne00 + col) * 4));
        up_val   = asfloat(buf_src1.Load(src1_off + (row * ne00 + col) * 4));
    } else {
        uint gate_col = swapped ? col + nc : col;
        uint up_col   = swapped ? col : col + nc;
        gate_val = asfloat(buf_src0.Load(src0_off + (row * ne00 + gate_col) * 4));
        up_val   = asfloat(buf_src0.Load(src0_off + (row * ne00 + up_col) * 4));
    }
    // silu(gate) * up
    float silu_gate = gate_val / (1.0 + exp(-gate_val));
    buf_dst.Store(dst_off + dtid.x * 4, asuint(silu_gate * up_val));
}
)";

// GET_ROWS: gather rows from src0 using integer indices from src1
// src0 is the embedding table, src1 is indices (I32)
// Each thread copies one element of one gathered row.
static const char * s_hlsl_get_rows = R"(
cbuffer Params : register(b0) {
    uint ne00;      // row width (elements)
    uint ne01;      // number of rows to gather
    uint ne02;      // batch dim 2
    uint src0_off;  // byte offset of src0 (embedding table)
    uint src1_off;  // byte offset of src1 (indices, I32)
    uint dst_off;   // byte offset of dst
    uint nb01;      // src0 row stride (bytes)
    uint n_total;   // ne00 * ne01 * ne02
    uint src0_is_f16; // 1 if src0 is F16, 0 if F32
};
RWByteAddressBuffer buf_src0 : register(u0);
RWByteAddressBuffer buf_src1 : register(u1);
RWByteAddressBuffer buf_dst  : register(u2);
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= n_total) return;
    uint idx = dtid.x;
    uint i0 = idx % ne00; idx /= ne00;
    uint i1 = idx % ne01;
    uint i2 = idx / ne01;
    // Read row index from src1
    int row_idx = asint(buf_src1.Load(src1_off + (i2 * ne01 + i1) * 4));
    // Read from src0 at the indexed row
    float val;
    if (src0_is_f16) {
        uint byte_addr = src0_off + (uint)row_idx * nb01 + i0 * 2;
        uint word = buf_src0.Load(byte_addr & ~3u);
        uint shift = (byte_addr & 2u) ? 16 : 0;
        val = f16tof32((word >> shift) & 0xFFFF);
    } else {
        val = asfloat(buf_src0.Load(src0_off + (uint)row_idx * nb01 + i0 * 4));
    }
    // Write to contiguous dst
    buf_dst.Store(dst_off + dtid.x * 4, asuint(val));
}
)";

// ---------------------------------------------------------------------------
// DXGI/PDH memory helpers (declared in ggml-impl.h, defined in mem_dxgi_pdh.cpp)
// ---------------------------------------------------------------------------
// ggml_dxgi_pdh_init() -> int
// ggml_dxgi_pdh_get_device_memory(const char* luid, size_t *free, size_t *total, bool is_integrated_gpu) -> int

// Helper: convert LUID to the hex string format expected by ggml_dxgi_pdh_get_device_memory
static std::string luid_to_string(const LUID & luid) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%08x%08x", (unsigned)luid.HighPart, (unsigned)luid.LowPart);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Operator compilation cache
// ---------------------------------------------------------------------------

// Cache key: uniquely identifies a DML operator configuration.
// Two dispatches with the same key are guaranteed to use the same compiled operator.
struct dml_op_cache_key {
    ggml_op    op;
    ggml_type  src0_type;
    ggml_type  src1_type;
    ggml_type  dst_type;
    int64_t    src0_ne[4];
    int64_t    src1_ne[4];
    int64_t    dst_ne[4];
    int32_t    op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // raw op_params

    bool operator==(const dml_op_cache_key & other) const {
        return op == other.op &&
               src0_type == other.src0_type &&
               src1_type == other.src1_type &&
               dst_type == other.dst_type &&
               memcmp(src0_ne, other.src0_ne, sizeof(src0_ne)) == 0 &&
               memcmp(src1_ne, other.src1_ne, sizeof(src1_ne)) == 0 &&
               memcmp(dst_ne, other.dst_ne, sizeof(dst_ne)) == 0 &&
               memcmp(op_params, other.op_params, sizeof(op_params)) == 0;
    }
};

struct dml_op_cache_key_hash {
    size_t operator()(const dml_op_cache_key & k) const {
        // FNV-1a hash over the raw bytes of the key
        size_t h = 14695981039346656037ULL;
        const uint8_t * data = reinterpret_cast<const uint8_t *>(&k);
        for (size_t i = 0; i < sizeof(k); i++) {
            h ^= data[i];
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct dml_cached_op {
    ComPtr<IDMLCompiledOperator> compiled;
    ComPtr<ID3D12Resource>       persistent_resource; // persistent resource if needed
    ComPtr<ID3D12Resource>       temp_resource;       // temporary resource if needed
};

// Build a cache key from a tensor node
static dml_op_cache_key dml_make_cache_key(const struct ggml_tensor * node) {
    dml_op_cache_key key = {};
    key.op = node->op;
    key.dst_type = node->type;
    memcpy(key.dst_ne, node->ne, sizeof(key.dst_ne));

    if (node->src[0]) {
        key.src0_type = node->src[0]->type;
        memcpy(key.src0_ne, node->src[0]->ne, sizeof(key.src0_ne));
    }
    if (node->src[1]) {
        key.src1_type = node->src[1]->type;
        memcpy(key.src1_ne, node->src[1]->ne, sizeof(key.src1_ne));
    }

    static_assert(sizeof(key.op_params) <= GGML_MAX_OP_PARAMS, "op_params size mismatch");
    memcpy(key.op_params, node->op_params, sizeof(key.op_params));
    return key;
}

// ---------------------------------------------------------------------------
// Per-device context
// ---------------------------------------------------------------------------

// Forward declaration — defined after the struct
static HRESULT dml_create_buffer(ID3D12Device * device, size_t size,
                                  D3D12_HEAP_TYPE heap_type,
                                  D3D12_RESOURCE_FLAGS flags,
                                  D3D12_RESOURCE_STATES initial_state,
                                  ComPtr<ID3D12Resource> & out);

struct ggml_dml_device_info {
    int               index;
    std::string       name;         // short name: "DirectML0"
    std::string       description;  // e.g. "Intel(R) AI Boost"
    std::string       pci_id;       // PCI bus id or empty
    std::string       luid_str;     // LUID as hex string for DXGI/PDH lookups
    bool              is_npu;       // true if classified as NPU/accelerator
    bool              integrated;   // true if shares system memory
    bool              has_compute_shaders; // true for GPUs with HLSL support, false for MCDM/NPU
    bool              is_core_device;      // true for Core/MCDM devices (COMPUTE queue only)
    size_t            total_memory;
    size_t            free_memory;
    LUID              adapter_luid;

    // Persistent D3D12/DML state for compute
    ComPtr<ID3D12Device>              d3d_device;
    ComPtr<IDMLDevice1>               dml_device;
    ComPtr<ID3D12CommandQueue>        cmd_queue;
    ComPtr<ID3D12CommandAllocator>    cmd_allocator;
    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    ComPtr<ID3D12Fence>               fence;
    uint64_t                          fence_value = 0;
    HANDLE                            fence_event = nullptr;
    ComPtr<IDMLCommandRecorder>       dml_cmd_recorder;

    // Descriptor heap for DML binding tables
    ComPtr<ID3D12DescriptorHeap>      desc_heap;
    UINT                              desc_size = 0;
    static const UINT                 DESC_HEAP_SIZE = 1024; // descriptors

    // Operator compilation cache — avoids recompiling the same op configuration
    std::unordered_map<dml_op_cache_key, dml_cached_op, dml_op_cache_key_hash> op_cache;

    // Per-device shared staging buffers for set_tensor / get_tensor.
    // Allocated lazily on first use to avoid massive upfront allocations.
    ComPtr<ID3D12Resource> staging_upload;
    ComPtr<ID3D12Resource> staging_readback;
    size_t                 staging_upload_size   = 0;
    size_t                 staging_readback_size = 0;

    // Ensure the upload staging buffer is at least `size` bytes.
    // Returns the D3D12 resource pointer, or nullptr on failure.
    ID3D12Resource * ensure_upload_staging(size_t size) {
        if (staging_upload && staging_upload_size >= size) {
            return staging_upload.Get();
        }
        staging_upload.Reset();
        staging_upload_size = 0;
        HRESULT hr = dml_create_buffer(d3d_device.Get(), size,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ, staging_upload);
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: failed to create upload staging (%zu bytes): 0x%08x\n",
                            __func__, size, (unsigned)hr);
            return nullptr;
        }
        staging_upload_size = size;
        return staging_upload.Get();
    }

    // Ensure the readback staging buffer is at least `size` bytes.
    ID3D12Resource * ensure_readback_staging(size_t size) {
        if (staging_readback && staging_readback_size >= size) {
            return staging_readback.Get();
        }
        staging_readback.Reset();
        staging_readback_size = 0;
        HRESULT hr = dml_create_buffer(d3d_device.Get(), size,
            D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST, staging_readback);
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: failed to create readback staging (%zu bytes): 0x%08x\n",
                            __func__, size, (unsigned)hr);
            return nullptr;
        }
        staging_readback_size = size;
        return staging_readback.Get();
    }

    // Custom compute shader state (for ops not natively supported by DML)
    ComPtr<ID3D12RootSignature> compute_root_sig;
    ComPtr<ID3D12PipelineState> pso_add;
    ComPtr<ID3D12PipelineState> pso_mul;
    ComPtr<ID3D12PipelineState> pso_scale;
    ComPtr<ID3D12PipelineState> pso_soft_max;
    ComPtr<ID3D12PipelineState> pso_silu;
    ComPtr<ID3D12PipelineState> pso_rms_norm;
    ComPtr<ID3D12PipelineState> pso_diag_mask_inf;
    ComPtr<ID3D12PipelineState> pso_set_rows_f16;
    ComPtr<ID3D12PipelineState> pso_set_rows_f32;
    ComPtr<ID3D12PipelineState> pso_mul_mat_f16_f32;
    ComPtr<ID3D12PipelineState> pso_mul_mat_f32_f32;
    ComPtr<ID3D12PipelineState> pso_cont;
    ComPtr<ID3D12PipelineState> pso_rope;
    ComPtr<ID3D12PipelineState> pso_rope_passthrough;
    ComPtr<ID3D12PipelineState> pso_swiglu;
    ComPtr<ID3D12PipelineState> pso_get_rows;
    bool compute_shaders_ready = false;

    // Compile an HLSL compute shader and create a PSO
    ComPtr<ID3D12PipelineState> compile_shader(const char * hlsl, const char * name) {
        ComPtr<ID3DBlob> code, errors;
        HRESULT hr = s_D3DCompile(hlsl, strlen(hlsl), name, nullptr, nullptr,
                                    "main", "cs_5_1", 0, 0, &code, &errors);
        if (FAILED(hr)) {
            const char * msg = errors ? (const char *)errors->GetBufferPointer() : "unknown";
            GGML_LOG_ERROR("DirectML: HLSL compile '%s' failed: %s\n", name, msg);
            return nullptr;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = compute_root_sig.Get();
        pd.CS.pShaderBytecode = code->GetBufferPointer();
        pd.CS.BytecodeLength = code->GetBufferSize();
        ComPtr<ID3D12PipelineState> pso;
        hr = d3d_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("DirectML: CreateComputePipelineState '%s' failed: 0x%08x\n", name, (unsigned)hr);
            return nullptr;
        }
        return pso;
    }

    bool init_compute_shaders() {
        if (compute_shaders_ready) return true;
        if (!dml_load_d3dcompiler()) return false;

        // Create root signature: 16 root constants (b0) + 3 root UAVs (u0, u1, u2)
        D3D12_ROOT_PARAMETER params[4] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.ShaderRegister = 0;
        params[0].Constants.RegisterSpace = 0;
        params[0].Constants.Num32BitValues = 16;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        for (int i = 0; i < 3; i++) {
            params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            params[1 + i].Descriptor.ShaderRegister = i;
            params[1 + i].Descriptor.RegisterSpace = 0;
            params[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC rsd = {};
        rsd.NumParameters = 4;
        rsd.pParameters = params;

        ComPtr<ID3DBlob> sig_blob, sig_err;
        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                                   &sig_blob, &sig_err);
        if (FAILED(hr)) {
            GGML_LOG_ERROR("DirectML: SerializeRootSignature failed: 0x%08x\n", (unsigned)hr);
            return false;
        }
        hr = d3d_device->CreateRootSignature(0, sig_blob->GetBufferPointer(),
                                               sig_blob->GetBufferSize(),
                                               IID_PPV_ARGS(&compute_root_sig));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("DirectML: CreateRootSignature failed: 0x%08x\n", (unsigned)hr);
            return false;
        }

        pso_add = compile_shader(s_hlsl_add, "add");
        pso_mul = compile_shader(s_hlsl_mul, "mul");
        pso_scale = compile_shader(s_hlsl_scale, "scale");
        pso_soft_max = compile_shader(s_hlsl_soft_max, "soft_max");
        pso_silu = compile_shader(s_hlsl_silu, "silu");
        pso_rms_norm = compile_shader(s_hlsl_rms_norm, "rms_norm");
        pso_diag_mask_inf = compile_shader(s_hlsl_diag_mask_inf, "diag_mask_inf");
        pso_set_rows_f16 = compile_shader(s_hlsl_set_rows_f16, "set_rows_f16");
        pso_set_rows_f32 = compile_shader(s_hlsl_set_rows_f32, "set_rows_f32");
        pso_mul_mat_f16_f32 = compile_shader(s_hlsl_mul_mat_f16_f32, "mul_mat_f16_f32");
        pso_mul_mat_f32_f32 = compile_shader(s_hlsl_mul_mat_f32_f32, "mul_mat_f32_f32");
        pso_cont = compile_shader(s_hlsl_cont, "cont");
        pso_rope = compile_shader(s_hlsl_rope, "rope");
        pso_rope_passthrough = compile_shader(s_hlsl_rope_passthrough, "rope_passthrough");
        pso_swiglu = compile_shader(s_hlsl_swiglu, "swiglu");
        pso_get_rows = compile_shader(s_hlsl_get_rows, "get_rows");

        compute_shaders_ready = (pso_add && pso_mul && pso_scale && pso_soft_max &&
                                  pso_silu && pso_rms_norm && pso_diag_mask_inf &&
                                  pso_set_rows_f16 && pso_set_rows_f32 &&
                                  pso_mul_mat_f16_f32 && pso_mul_mat_f32_f32 &&
                                  pso_cont && pso_rope && pso_rope_passthrough &&
                                  pso_swiglu && pso_get_rows);
        if (compute_shaders_ready) {
            GGML_LOG_INFO("DirectML: compute shaders compiled successfully\n");
        }
        return compute_shaders_ready;
    }

    // Execute the current command list and wait for completion
    void execute_and_wait() {
        cmd_list->Close();
        ID3D12CommandList * lists[] = { cmd_list.Get() };
        cmd_queue->ExecuteCommandLists(1, lists);
        fence_value++;
        cmd_queue->Signal(fence.Get(), fence_value);
        fence->SetEventOnCompletion(fence_value, fence_event);
        WaitForSingleObject(fence_event, INFINITE);
    }

    // Reset command allocator and command list for new recording
    void reset_cmd_list() {
        cmd_allocator->Reset();
        cmd_list->Reset(cmd_allocator.Get(), nullptr);
    }
};

struct ggml_dml_context {
    std::vector<ggml_dml_device_info> devices;

    // Initialize D3D12 command queue, fence, DML command recorder, descriptor
    // heap on a device_info that already has d3d_device and dml_device set.
    // Returns true on success.
    bool init_device_state(ggml_dml_device_info & info) {
        HRESULT hr;
        auto * d3d_device = info.d3d_device.Get();
        auto * dml_device = info.dml_device.Get();

        // Core/MCDM devices only support COMPUTE and COPY queues.
        // GPU devices use DIRECT queues.
        D3D12_COMMAND_LIST_TYPE cmd_type = info.is_core_device
            ? D3D12_COMMAND_LIST_TYPE_COMPUTE
            : D3D12_COMMAND_LIST_TYPE_DIRECT;

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = cmd_type;
        hr = d3d_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&info.cmd_queue));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateCommandQueue(%s) failed: 0x%08x\n", __func__,
                           info.is_core_device ? "COMPUTE" : "DIRECT", (unsigned)hr);
            return false;
        }
        hr = d3d_device->CreateCommandAllocator(
            cmd_type, IID_PPV_ARGS(&info.cmd_allocator));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateCommandAllocator failed: 0x%08x\n", __func__, (unsigned)hr);
            return false;
        }
        hr = d3d_device->CreateCommandList(0, cmd_type,
            info.cmd_allocator.Get(), nullptr, IID_PPV_ARGS(&info.cmd_list));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateCommandList failed: 0x%08x\n", __func__, (unsigned)hr);
            return false;
        }
        info.cmd_list->Close();

        hr = d3d_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&info.fence));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateFence failed: 0x%08x\n", __func__, (unsigned)hr);
            return false;
        }
        info.fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        hr = dml_device->CreateCommandRecorder(IID_PPV_ARGS(&info.dml_cmd_recorder));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateCommandRecorder failed: 0x%08x\n", __func__, (unsigned)hr);
            return false;
        }

        D3D12_DESCRIPTOR_HEAP_DESC dhd = {};
        dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        dhd.NumDescriptors = ggml_dml_device_info::DESC_HEAP_SIZE;
        dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = d3d_device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(&info.desc_heap));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateDescriptorHeap failed: 0x%08x\n", __func__, (unsigned)hr);
            return false;
        }
        info.desc_size = d3d_device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return true;
    }

    // Finalize: refine memory info, log, and set up D3D12/DML state for each device.
    // Call after all adapters have been added to `devices`.
    void finalize_devices() {
        ggml_dxgi_pdh_init();

        // Remove devices where D3D12/DML state init fails (iterate in reverse)
        for (int i = (int)devices.size() - 1; i >= 0; i--) {
            auto & info = devices[i];

            // Refine memory via DXGI/PDH
            size_t pdh_free = 0, pdh_total = 0;
            if (ggml_dxgi_pdh_get_device_memory(info.luid_str.c_str(),
                    &pdh_free, &pdh_total, info.integrated) == 0) {
                if (pdh_total > 0) {
                    info.total_memory = pdh_total;
                    info.free_memory = pdh_free;
                }
            }

            if (!init_device_state(info)) {
                GGML_LOG_ERROR("%s: device %d (%s) D3D12/DML init failed, removing\n",
                               __func__, info.index, info.description.c_str());
                devices.erase(devices.begin() + i);
                continue;
            }
        }

        // Check OLLAMA_DML_SIMULATE_NPU=1 to force DML-native-only path on all devices
        // (disables HLSL compute shaders, routes ops through DML native or CPU fallback)
        {
            const char * sim = getenv("OLLAMA_DML_SIMULATE_NPU");
            if (sim && strcmp(sim, "1") == 0) {
                for (auto & d : devices) {
                    GGML_LOG_INFO("%s: OLLAMA_DML_SIMULATE_NPU=1: forcing DML-native path for %s\n",
                                  __func__, d.description.c_str());
                    d.has_compute_shaders = false;
                }
            }
        }

        // Re-index devices and log
        for (int i = 0; i < (int)devices.size(); i++) {
            devices[i].index = i;
            devices[i].name = "DirectML" + std::to_string(i);
            auto & info = devices[i];
            GGML_LOG_INFO("%s: device %d: %s (%s, %.1f MiB, %s%s)\n",
                          __func__, info.index, info.description.c_str(), info.name.c_str(),
                          info.total_memory / (1024.0 * 1024.0),
                          info.is_npu ? "npu" : (info.integrated ? "integrated" : "discrete"),
                          info.has_compute_shaders ? "" : ", no HLSL");
        }
    }

    // Sort devices: NPU first, then discrete GPU, then integrated GPU.
    void sort_devices() {
        std::stable_sort(devices.begin(), devices.end(),
            [](const ggml_dml_device_info & a, const ggml_dml_device_info & b) {
                // NPU > discrete GPU > integrated GPU
                auto rank = [](const ggml_dml_device_info & d) -> int {
                    if (d.is_npu)      return 0; // highest priority
                    if (!d.integrated) return 1; // discrete GPU
                    return 2;                    // integrated GPU
                };
                return rank(a) < rank(b);
            });
    }

    // Enumerate adapters via DXCore (finds NPUs that DXGI cannot see).
    // Returns true if at least one device was found.
    bool init_dxcore() {
        if (!dml_load_dxcore()) return false;

        ComPtr<IDXCoreAdapterFactory> factory;
        HRESULT hr = s_DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: DXCoreCreateAdapterFactory failed: 0x%08x\n",
                           __func__, (unsigned)hr);
            return false;
        }

        ComPtr<IDXCoreAdapterList> adapter_list;

        // Try the modern CreateAdapterListByWorkload API first (requires
        // IDXCoreAdapterFactory1) — it can filter by workload type and
        // hardware type in a single call.
        ComPtr<IDXCoreAdapterFactory1> factory1;
        hr = factory->QueryInterface(IID_PPV_ARGS(&factory1));
        if (SUCCEEDED(hr) && factory1) {
            hr = factory1->CreateAdapterListByWorkload(
                DXCoreWorkload::MachineLearning,
                DXCoreRuntimeFilterFlags::D3D12,
                static_cast<DXCoreHardwareTypeFilterFlags>(
                    static_cast<uint32_t>(DXCoreHardwareTypeFilterFlags::NPU) |
                    static_cast<uint32_t>(DXCoreHardwareTypeFilterFlags::GPU)),
                IID_PPV_ARGS(&adapter_list));
            if (SUCCEEDED(hr)) {
                GGML_LOG_INFO("%s: DXCore CreateAdapterListByWorkload(ML, D3D12, NPU|GPU) found %u adapters\n",
                              __func__, adapter_list->GetAdapterCount());
            }
        }

        // Fall back to the legacy CreateAdapterList with D3D12_CORE_COMPUTE attribute
        if (!adapter_list) {
            const GUID attributes[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
            hr = factory->CreateAdapterList(1, attributes, IID_PPV_ARGS(&adapter_list));
            if (FAILED(hr) || !adapter_list) {
                GGML_LOG_ERROR("%s: DXCore CreateAdapterList failed: 0x%08x\n",
                               __func__, (unsigned)hr);
                return false;
            }
            GGML_LOG_INFO("%s: DXCore CreateAdapterList(D3D12_CORE_COMPUTE) found %u adapters\n",
                          __func__, adapter_list->GetAdapterCount());
        }

        uint32_t count = adapter_list->GetAdapterCount();
        for (uint32_t i = 0; i < count; i++) {
            ComPtr<IDXCoreAdapter> adapter;
            hr = adapter_list->GetAdapter(i, IID_PPV_ARGS(&adapter));
            if (FAILED(hr)) continue;

            if (!adapter->IsValid()) continue;

            // Get adapter description
            char desc_utf8[256] = {};
            size_t desc_size = 0;
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
                adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &desc_size);
                if (desc_size > 0 && desc_size <= sizeof(desc_utf8) - 1) {
                    adapter->GetProperty(DXCoreAdapterProperty::DriverDescription,
                                         desc_size, desc_utf8);
                }
            }

            // Get LUID (required for D3D12 device creation and memory queries)
            LUID adapter_luid = {};
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::InstanceLuid)) {
                adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid,
                                     sizeof(adapter_luid), &adapter_luid);
            }

            // Classify hardware type using definitive DXCore attributes
            bool is_npu = adapter->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU);
            bool is_gpu = adapter->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU);

            // Check D3D12 runtime support attributes
            bool has_d3d12_graphics     = adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);
            bool has_d3d12_core_compute = adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
            bool has_d3d12_generic_ml   = adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);

            // Get driver version
            uint64_t driver_version = 0;
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverVersion)) {
                adapter->GetProperty(DXCoreAdapterProperty::DriverVersion,
                                     sizeof(driver_version), &driver_version);
            }

            // Get KMD model version (WDDM version indicator)
            uint32_t kmd_model = 0;
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::KmdModelVersion)) {
                adapter->GetProperty(DXCoreAdapterProperty::KmdModelVersion,
                                     sizeof(kmd_model), &kmd_model);
            }

            // Check if hardware (not software/WARP)
            bool is_hardware = false;
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware)) {
                adapter->GetProperty(DXCoreAdapterProperty::IsHardware,
                                     sizeof(is_hardware), &is_hardware);
            }
            if (!is_hardware) {
                GGML_LOG_INFO("%s: DXCore adapter %u: \"%s\" — skipped (software)\n",
                              __func__, i, desc_utf8);
                continue;
            }

            // Get integrated flag
            bool is_integrated = false;
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsIntegrated)) {
                adapter->GetProperty(DXCoreAdapterProperty::IsIntegrated,
                                     sizeof(is_integrated), &is_integrated);
            }

            // Get memory info
            size_t dedicated_mem = 0, shared_mem = 0;
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::DedicatedAdapterMemory)) {
                adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory,
                                     sizeof(dedicated_mem), &dedicated_mem);
            }
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::SharedSystemMemory)) {
                adapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory,
                                     sizeof(shared_mem), &shared_mem);
            }

            GGML_LOG_INFO("%s: DXCore adapter %u: \"%s\" %s%s, dedicated=%.1f MiB, shared=%.1f MiB, LUID=%s\n"
                          "    D3D12 attrs: graphics=%d core_compute=%d generic_ml=%d, "
                          "driver=0x%llx, kmd_model=%u%s\n",
                          __func__, i, desc_utf8,
                          is_npu ? "NPU" : (is_gpu ? "GPU" : "unknown"),
                          is_integrated ? " (integrated)" : "",
                          dedicated_mem / (1024.0 * 1024.0),
                          shared_mem / (1024.0 * 1024.0),
                          luid_to_string(adapter_luid).c_str(),
                          has_d3d12_graphics, has_d3d12_core_compute, has_d3d12_generic_ml,
                          (unsigned long long)driver_version, kmd_model,
                          is_hardware ? "" : " [software]");

            // Create D3D12 device — strategy depends on adapter type.
            // Non-graphics adapters (NPU/MCDM): use DXCore directly with Core/Generic feature levels.
            // Graphics adapters (GPU): try DXGI first (FL 11_0), then DXCore fallback.
            ComPtr<ID3D12Device> d3d_device;
            bool core_device = false;  // tracks whether device was created at a Core/Generic FL

            if (!has_d3d12_graphics) {
                // NPU/MCDM path: skip DXGI entirely, try Core/Generic feature levels only.
                // MS docs: requesting FL 9.x–12.x never returns a Core device.
                struct { D3D_FEATURE_LEVEL fl; const char * name; } levels[] = {
                    { D3D_FEATURE_LEVEL_1_0_GENERIC, "1_0_GENERIC" },
                    { D3D_FEATURE_LEVEL_1_0_CORE, "1_0_CORE" },
                };
                for (auto & lvl : levels) {
                    hr = D3D12CreateDevice(adapter.Get(), lvl.fl, IID_PPV_ARGS(&d3d_device));
                    if (SUCCEEDED(hr)) {
                        GGML_LOG_INFO("%s: DXCore adapter %u: D3D12 via DXCore (FL %s, non-graphics)\n",
                                      __func__, i, lvl.name);
                        core_device = true;
                        break;
                    }
                    GGML_LOG_INFO("%s: DXCore adapter %u: D3D12CreateDevice(FL %s) failed: 0x%08x\n",
                                  __func__, i, lvl.name, (unsigned)hr);
                }
            } else {
                // GPU path: try DXGI lookup first (FL 11_0), then DXCore with all feature levels.
                {
                    ComPtr<IDXGIFactory4> dxgi_factory;
                    hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
                    if (SUCCEEDED(hr)) {
                        ComPtr<IDXGIAdapter1> dxgi_adapter;
                        hr = dxgi_factory->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&dxgi_adapter));
                        if (SUCCEEDED(hr)) {
                            hr = D3D12CreateDevice(dxgi_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                   IID_PPV_ARGS(&d3d_device));
                            if (SUCCEEDED(hr)) {
                                GGML_LOG_INFO("%s: DXCore adapter %u: D3D12 via DXGI LUID (FL 11_0)\n",
                                              __func__, i);
                            }
                        }
                    }
                }

                // DXCore fallback for graphics adapters
                if (!d3d_device) {
                    struct { D3D_FEATURE_LEVEL fl; const char * name; bool core; } levels[] = {
                        { D3D_FEATURE_LEVEL_11_0, "11_0", false },
                        { D3D_FEATURE_LEVEL_1_0_CORE, "1_0_CORE", true },
                        { D3D_FEATURE_LEVEL_1_0_GENERIC, "1_0_GENERIC", true },
                    };
                    for (auto & lvl : levels) {
                        hr = D3D12CreateDevice(adapter.Get(), lvl.fl, IID_PPV_ARGS(&d3d_device));
                        if (SUCCEEDED(hr)) {
                            GGML_LOG_INFO("%s: DXCore adapter %u: D3D12 via DXCore (FL %s)\n",
                                          __func__, i, lvl.name);
                            core_device = lvl.core;
                            break;
                        }
                        GGML_LOG_INFO("%s: DXCore adapter %u: D3D12CreateDevice(FL %s) failed: 0x%08x\n",
                                      __func__, i, lvl.name, (unsigned)hr);
                    }
                }
            }

            if (!d3d_device) {
                if (has_d3d12_generic_ml) {
                    // Generic ML-only NPU: DXCore reports the adapter but D3D12
                    // cannot create a device for it. DirectML requires ID3D12Device
                    // so this NPU is not usable through the DirectML API.
                    // Such devices may be accessible via ONNX Runtime DML EP or
                    // Windows.AI.MachineLearning which use internal driver paths.
                    GGML_LOG_INFO("%s: DXCore adapter %u: Generic ML-only NPU — "
                                  "D3D12 device creation unsupported, skipping "
                                  "(NPU may be usable via ONNX Runtime DML EP)\n",
                                  __func__, i);
                } else {
                    GGML_LOG_INFO("%s: DXCore adapter %u: all D3D12 device creation "
                                  "attempts failed (0x%08x), skipping\n",
                                  __func__, i, (unsigned)hr);
                }
                continue;
            }

            // Create DirectML device
            ComPtr<IDMLDevice1> dml_device;
            hr = s_DMLCreateDevice1(d3d_device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                    DML_FEATURE_LEVEL_1_0, IID_PPV_ARGS(&dml_device));
            if (FAILED(hr)) {
                GGML_LOG_INFO("%s: DXCore adapter %u: DMLCreateDevice1 failed: 0x%08x\n",
                              __func__, i, (unsigned)hr);
                continue;
            }

            GGML_LOG_INFO("%s: DXCore adapter %u: D3D12 + DirectML device created\n", __func__, i);

            ggml_dml_device_info info = {};
            info.index = (int)devices.size();
            info.adapter_luid = adapter_luid;
            info.luid_str = luid_to_string(adapter_luid);
            info.description = desc_utf8;
            info.name = "DirectML" + std::to_string(info.index);
            info.is_npu = is_npu;
            info.integrated = is_integrated;
            info.has_compute_shaders = !is_npu && has_d3d12_graphics;
            info.is_core_device = core_device;
            info.total_memory = dedicated_mem ? dedicated_mem : shared_mem;
            info.free_memory = info.total_memory;
            info.d3d_device = d3d_device;
            info.dml_device = dml_device;

            devices.push_back(std::move(info));
        }

        return !devices.empty();
    }

    // Enumerate adapters via DXGI (original path — cannot find MCDM/NPU devices).
    bool init_dxgi() {
        ComPtr<IDXGIFactory4> factory;
        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            GGML_LOG_ERROR("%s: CreateDXGIFactory1 failed: 0x%08x\n", __func__, (unsigned)hr);
            return false;
        }

        for (UINT i = 0; ; i++) {
            ComPtr<IDXGIAdapter1> adapter;
            hr = factory->EnumAdapters1(i, &adapter);
            if (hr == DXGI_ERROR_NOT_FOUND) break;

            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);

            char name_utf8[256] = {};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                name_utf8, sizeof(name_utf8) - 1, nullptr, nullptr);
            GGML_LOG_INFO("%s: DXGI adapter %u: \"%s\" "
                          "DedicatedVideoMem=%.1f MiB, DedicatedSysMem=%.1f MiB, "
                          "SharedSysMem=%.1f MiB, Flags=0x%x, VendorId=0x%04x, DeviceId=0x%04x\n",
                          __func__, i, name_utf8,
                          desc.DedicatedVideoMemory / (1024.0 * 1024.0),
                          desc.DedicatedSystemMemory / (1024.0 * 1024.0),
                          desc.SharedSystemMemory / (1024.0 * 1024.0),
                          desc.Flags, desc.VendorId, desc.DeviceId);

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                GGML_LOG_INFO("%s:   -> skipped (software adapter)\n", __func__);
                continue;
            }

            ComPtr<ID3D12Device> d3d_device;
            hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&d3d_device));
            if (FAILED(hr)) {
                GGML_LOG_INFO("%s:   -> skipped (D3D12CreateDevice failed: 0x%08x)\n",
                              __func__, (unsigned)hr);
                continue;
            }

            ComPtr<IDMLDevice1> dml_device;
            hr = s_DMLCreateDevice1(d3d_device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                    DML_FEATURE_LEVEL_1_0, IID_PPV_ARGS(&dml_device));
            if (FAILED(hr)) {
                GGML_LOG_INFO("%s:   -> skipped (DMLCreateDevice1 failed: 0x%08x)\n",
                              __func__, (unsigned)hr);
                continue;
            }
            GGML_LOG_INFO("%s:   -> D3D12 + DirectML device created successfully\n", __func__);

            ggml_dml_device_info info = {};
            info.index = (int)devices.size();
            info.adapter_luid = desc.AdapterLuid;
            info.luid_str = luid_to_string(desc.AdapterLuid);

            char desc_utf8[256] = {};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                desc_utf8, sizeof(desc_utf8) - 1, nullptr, nullptr);
            info.description = desc_utf8;
            info.name = "DirectML" + std::to_string(info.index);

            info.total_memory = (size_t)(desc.DedicatedVideoMemory
                                         ? desc.DedicatedVideoMemory
                                         : desc.SharedSystemMemory);
            info.free_memory = info.total_memory;
            info.integrated = (desc.DedicatedVideoMemory == 0);
            info.is_npu = (desc.DedicatedVideoMemory == 0 &&
                           desc.SharedSystemMemory > 0 &&
                           desc.DedicatedSystemMemory == 0);
            info.has_compute_shaders = !info.is_npu;
            info.is_core_device = false;  // DXGI path only creates FL 11_0 devices
            info.d3d_device = d3d_device;
            info.dml_device = dml_device;

            devices.push_back(std::move(info));
        }

        return !devices.empty();
    }

    bool init() {
        if (!dml_load_library()) {
            return false;
        }

        // Try DXCore first — it can discover NPU/MCDM devices that DXGI cannot see.
        bool found = init_dxcore();

        // Fall back to DXGI if DXCore found nothing (older Windows, no dxcore.dll, etc.)
        if (!found) {
            GGML_LOG_INFO("%s: DXCore found no devices, falling back to DXGI enumeration\n", __func__);
            found = init_dxgi();
        }

        if (!found) return false;

        // Sort: NPU first, then discrete GPU, then integrated GPU
        sort_devices();

        // Initialize D3D12/DML state, refine memory, log final device list
        finalize_devices();

        return !devices.empty();
    }
};

// Singleton device context
static ggml_dml_context & dml_ctx() {
    static ggml_dml_context ctx;
    static std::once_flag once;
    std::call_once(once, [&] { ctx.init(); });
    return ctx;
}

// ---------------------------------------------------------------------------
// Backend buffer (D3D12 GPU resources)
// ---------------------------------------------------------------------------

// Sentinel base pointer for GGML allocator. D3D12 GPU resources have no
// CPU-visible address, so we return a sentinel. The allocator assigns
// tensor->data = base + offset; we recover the offset via arithmetic.
static void * const dml_ptr_base = (void *)(uintptr_t)0x1000;

static uint64_t dml_tensor_offset(const struct ggml_tensor * tensor) {
    return (uint8_t *)tensor->data - (uint8_t *)dml_ptr_base;
}

// Helper: create a D3D12 committed resource
static HRESULT dml_create_buffer(ID3D12Device * device, size_t size,
                                  D3D12_HEAP_TYPE heap_type,
                                  D3D12_RESOURCE_FLAGS flags,
                                  D3D12_RESOURCE_STATES initial_state,
                                  ComPtr<ID3D12Resource> & out) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap_type;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size > 0 ? size : 1; // D3D12 doesn't allow 0-byte buffers
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    return device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd, initial_state, nullptr,
        IID_PPV_ARGS(&out));
}

struct ggml_dml_buffer_context {
    int                    device_index;
    ComPtr<ID3D12Resource> gpu_resource;      // DEFAULT heap (GPU-local)
    size_t                 size;

    // CPU staging for quantized tensor dequantization.
    // Quantized data arrives in chunks (128KB) but must be dequantized as complete
    // tensors. This map accumulates raw quantized bytes per tensor until complete,
    // then dequantizes to F16 and uploads to GPU.
    struct quant_staging {
        std::vector<uint8_t> data;  // accumulated raw quantized bytes
        size_t               native_size; // total expected size in native format
        size_t               received;    // bytes received so far
    };
    std::unordered_map<const ggml_tensor *, quant_staging> quant_stage;
};

static void ggml_backend_dml_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_dml_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_dml_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *)dml_ptr_base;
}

static void ggml_backend_dml_buffer_memset(ggml_backend_buffer_t buffer,
                                            struct ggml_tensor * tensor,
                                            uint8_t value, size_t offset, size_t size) {
    // Quantized tensors are stored as F16 on device — memset would write the
    // wrong number of bytes (native quantized size vs F16 size).
    if (ggml_is_quantized(tensor->type)) {
        GGML_LOG_ERROR("DirectML memset_tensor on quantized tensor '%s' (type=%s) — "
                        "device stores F16, memset would corrupt layout!\n",
                        tensor->name, ggml_type_name(tensor->type));
        GGML_ABORT("DirectML memset_tensor on quantized tensor is invalid with F16-expanded device storage");
    }

    auto * ctx = (ggml_dml_buffer_context *)buffer->context;
    auto & dev = dml_ctx().devices[ctx->device_index];
    uint64_t buf_offset = dml_tensor_offset(tensor) + offset;

    // Use per-device lazy upload staging buffer
    ID3D12Resource * staging = dev.ensure_upload_staging(size);
    if (!staging) return;

    void * mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    HRESULT hr = staging->Map(0, &read_range, &mapped);
    if (SUCCEEDED(hr)) {
        memset(mapped, value, size);
        D3D12_RANGE write_range = { 0, (SIZE_T)size };
        staging->Unmap(0, &write_range);

        dev.reset_cmd_list();
        dev.cmd_list->CopyBufferRegion(ctx->gpu_resource.Get(), buf_offset,
                                        staging, 0, size);
        dev.execute_and_wait();
    }
}

static void ggml_backend_dml_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                struct ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_dml_buffer_context *)buffer->context;
    auto & dev = dml_ctx().devices[ctx->device_index];

    // For quantized types, accumulate chunks on CPU and dequantize when complete.
    // Data arrives in ~128KB chunks with advancing offsets in native quantized format.
    // We buffer everything, then dequantize (quant → F32 → F16) and upload to GPU.
    if (ggml_is_quantized(tensor->type)) {
        const size_t native_size = ggml_nbytes(tensor);
        auto & stage = ctx->quant_stage[tensor];

        // Initialize staging on first chunk
        if (stage.data.empty()) {
            stage.data.resize(native_size);
            stage.native_size = native_size;
            stage.received = 0;
        }

        // Accumulate this chunk
        GGML_ASSERT(offset + size <= native_size);
        memcpy(stage.data.data() + offset, data, size);
        stage.received += size;

        // If all data received, dequantize and upload
        if (stage.received >= native_size) {
            const int64_t n_elements = ggml_nelements(tensor);
            const size_t f16_size = n_elements * sizeof(ggml_fp16_t);

            // Dequantize: quant → F32
            std::vector<float> f32_buf(n_elements);
            const auto * traits = ggml_get_type_traits(tensor->type);
            if (traits && traits->to_float) {
                traits->to_float(stage.data.data(), f32_buf.data(), n_elements);
            } else {
                GGML_LOG_ERROR("DirectML: no dequantize function for type %s\n",
                                ggml_type_name(tensor->type));
                ctx->quant_stage.erase(tensor);
                return;
            }

            // Convert F32 → F16
            std::vector<ggml_fp16_t> f16_buf(n_elements);
            ggml_fp32_to_fp16_row(f32_buf.data(), f16_buf.data(), n_elements);

            // Upload F16 data to GPU
            uint64_t buf_offset = dml_tensor_offset(tensor);
            ID3D12Resource * staging = dev.ensure_upload_staging(f16_size);
            if (staging) {
                void * mapped = nullptr;
                D3D12_RANGE read_range = {0, 0};
                if (SUCCEEDED(staging->Map(0, &read_range, &mapped))) {
                    memcpy(mapped, f16_buf.data(), f16_size);
                    D3D12_RANGE write_range = { 0, (SIZE_T)f16_size };
                    staging->Unmap(0, &write_range);

                    dev.reset_cmd_list();
                    dev.cmd_list->CopyBufferRegion(ctx->gpu_resource.Get(), buf_offset,
                                                    staging, 0, f16_size);
                    dev.execute_and_wait();
                }
            }

            // Free staging memory
            ctx->quant_stage.erase(tensor);
        }
        return;
    }

    // Direct copy for F32/F16 data
    uint64_t buf_offset = dml_tensor_offset(tensor) + offset;

    // Use per-device lazy upload staging buffer
    ID3D12Resource * staging = dev.ensure_upload_staging(size);
    if (!staging) return;

    void * mapped = nullptr;
    D3D12_RANGE read_range = {0, 0};
    HRESULT hr = staging->Map(0, &read_range, &mapped);
    if (SUCCEEDED(hr)) {
        memcpy(mapped, data, size);
        D3D12_RANGE write_range = { 0, (SIZE_T)size };
        staging->Unmap(0, &write_range);

        dev.reset_cmd_list();
        dev.cmd_list->CopyBufferRegion(ctx->gpu_resource.Get(), buf_offset,
                                        staging, 0, size);
        dev.execute_and_wait();
    }
}

static void ggml_backend_dml_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const struct ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    // Quantized tensors are stored as F16 on device — returning raw F16 bytes
    // to a caller expecting quantized bytes is a contract violation.
    if (ggml_is_quantized(tensor->type)) {
        GGML_LOG_ERROR("DirectML get_tensor on quantized tensor '%s' (type=%s) — "
                        "device stores F16, caller expects quantized bytes!\n",
                        tensor->name, ggml_type_name(tensor->type));
        GGML_ABORT("DirectML get_tensor on quantized tensor is invalid with F16-expanded device storage");
    }

    auto * ctx = (ggml_dml_buffer_context *)buffer->context;
    auto & dev = dml_ctx().devices[ctx->device_index];
    uint64_t buf_offset = dml_tensor_offset(tensor) + offset;

    // Use per-device lazy readback staging buffer
    ID3D12Resource * staging = dev.ensure_readback_staging(size);
    if (!staging) return;

    dev.reset_cmd_list();
    dev.cmd_list->CopyBufferRegion(staging, 0,
                                    ctx->gpu_resource.Get(), buf_offset, size);
    dev.execute_and_wait();

    void * mapped = nullptr;
    D3D12_RANGE read_range = { 0, (SIZE_T)size };
    HRESULT hr = staging->Map(0, &read_range, &mapped);
    if (SUCCEEDED(hr)) {
        memcpy(data, mapped, size);
        D3D12_RANGE write_range = {0, 0};
        staging->Unmap(0, &write_range);
    }
}

static void ggml_backend_dml_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_dml_buffer_context *)buffer->context;
    auto & dev = dml_ctx().devices[ctx->device_index];

    // Clear in chunks via upload staging to avoid huge staging allocations.
    // 64 MiB chunk size balances transfer efficiency vs memory use.
    const size_t chunk_size = 64 * 1024 * 1024;
    ID3D12Resource * staging = dev.ensure_upload_staging(std::min(ctx->size, chunk_size));
    if (!staging) return;

    size_t remaining = ctx->size;
    size_t buf_offset = 0;
    while (remaining > 0) {
        size_t xfer = std::min(remaining, chunk_size);

        void * mapped = nullptr;
        D3D12_RANGE read_range = {0, 0};
        HRESULT hr = staging->Map(0, &read_range, &mapped);
        if (FAILED(hr)) break;
        memset(mapped, value, xfer);
        D3D12_RANGE write_range = { 0, (SIZE_T)xfer };
        staging->Unmap(0, &write_range);

        dev.reset_cmd_list();
        dev.cmd_list->CopyBufferRegion(ctx->gpu_resource.Get(), buf_offset,
                                        staging, 0, xfer);
        dev.execute_and_wait();

        buf_offset += xfer;
        remaining -= xfer;
    }
}

static struct ggml_backend_buffer_i ggml_backend_dml_buffer_iface = {
    /* .free_buffer   = */ ggml_backend_dml_buffer_free,
    /* .get_base      = */ ggml_backend_dml_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_dml_buffer_memset,
    /* .set_tensor    = */ ggml_backend_dml_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_dml_buffer_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ggml_backend_dml_buffer_clear,
    /* .reset         = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend buffer type
// ---------------------------------------------------------------------------

struct ggml_dml_buft_context {
    int device_index;
};

static const char * ggml_backend_dml_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "DirectML";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_dml_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = (ggml_dml_buft_context *)buft->context;
    auto & dev = dml_ctx().devices[buft_ctx->device_index];
    auto * ctx = new ggml_dml_buffer_context{};
    ctx->device_index = buft_ctx->device_index;
    ctx->size = size;

    // GPU-local buffer (DEFAULT heap) with UAV flag for DML
    HRESULT hr = dml_create_buffer(dev.d3d_device.Get(), size,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON, ctx->gpu_resource);
    if (FAILED(hr)) {
        GGML_LOG_ERROR("%s: failed to create GPU buffer (%zu bytes): 0x%08x\n",
                        __func__, size, (unsigned)hr);
        delete ctx;
        return nullptr;
    }

    // Staging buffers are allocated lazily per-device in set_tensor/get_tensor to avoid
    // excessive memory usage. Large buffers like KV cache don't need full-size
    // staging since data is transferred incrementally per-tensor.

    return ggml_backend_buffer_init(buft, ggml_backend_dml_buffer_iface, ctx, size);
}

static size_t ggml_backend_dml_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    // D3D12 requires 256-byte alignment for constant buffers, 16 for others
    return 256;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_dml_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    // Cap buffers at < 4 GB so that all HLSL byte offsets (uint32_t) are safe.
    // The GGML allocator will split tensors across multiple buffers if needed.
    return (1ull << 32) - 256;
    GGML_UNUSED(buft);
}

static bool ggml_backend_dml_buft_is_host(ggml_backend_buffer_type_t buft) {
    // GPU-resident buffers — scheduler must use set_tensor/get_tensor
    return false;
    GGML_UNUSED(buft);
}

// For quantized tensors, we store the raw quantized bytes on the GPU.
// Dequantization happens in compute shaders (MUL_MAT) or on CPU fallback.
static size_t ggml_backend_dml_buft_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                     const struct ggml_tensor * tensor) {
    // Quantized types are dequantized to F16 during upload, so allocate F16 size
    if (ggml_is_quantized(tensor->type)) {
        return ggml_nelements(tensor) * sizeof(ggml_fp16_t);
    }
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static struct ggml_backend_buffer_type_i ggml_backend_dml_buft_iface = {
    /* .get_name      = */ ggml_backend_dml_buft_get_name,
    /* .alloc_buffer  = */ ggml_backend_dml_buft_alloc_buffer,
    /* .get_alignment = */ ggml_backend_dml_buft_get_alignment,
    /* .get_max_size  = */ ggml_backend_dml_buft_get_max_size,
    /* .get_alloc_size = */ ggml_backend_dml_buft_get_alloc_size,
    /* .is_host       = */ ggml_backend_dml_buft_is_host,
    /* .noalloc_buffer = */ nullptr,
};

// Per-device buffer type instances
static ggml_backend_buffer_type ggml_dml_buft_instances[GGML_DML_MAX_DEVICES] = {};
static ggml_dml_buft_context    ggml_dml_buft_contexts[GGML_DML_MAX_DEVICES] = {};

// ---------------------------------------------------------------------------
// Backend (stream/compute)
// ---------------------------------------------------------------------------

struct ggml_dml_backend_context {
    int device_index;
};

static const char * ggml_backend_dml_get_name(ggml_backend_t backend) {
    return "DirectML";
    GGML_UNUSED(backend);
}

static void ggml_backend_dml_free(ggml_backend_t backend) {
    auto * ctx = (ggml_dml_backend_context *)backend->context;
    delete ctx;
}

static void ggml_backend_dml_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_dml_backend_context *)backend->context;
    auto & dev = dml_ctx().devices[ctx->device_index];
    if (dev.fence->GetCompletedValue() < dev.fence_value) {
        dev.fence->SetEventOnCompletion(dev.fence_value, dev.fence_event);
        WaitForSingleObject(dev.fence_event, INFINITE);
    }
}

// ---------------------------------------------------------------------------
// DML operator dispatch helpers
// ---------------------------------------------------------------------------

// Helper: get the D3D12 resource backing a tensor's buffer
static ID3D12Resource * dml_get_tensor_resource(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return nullptr;
    auto * buf_ctx = (ggml_dml_buffer_context *)tensor->buffer->context;
    return buf_ctx->gpu_resource.Get();
}

// Helper: map GGML type to DML tensor data type
// Quantized types are stored as raw bytes on GPU — only F32 and F16 have DML types.
static DML_TENSOR_DATA_TYPE ggml_type_to_dml(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return DML_TENSOR_DATA_TYPE_FLOAT32;
        case GGML_TYPE_F16:  return DML_TENSOR_DATA_TYPE_FLOAT16;
        default:
            // Quantized types are dequantized to F16 during set_tensor upload
            if (ggml_is_quantized(t)) return DML_TENSOR_DATA_TYPE_FLOAT16;
            return DML_TENSOR_DATA_TYPE_FLOAT32;
    }
}

// Helper: element size for a DML data type
static size_t dml_type_size(DML_TENSOR_DATA_TYPE dt) {
    switch (dt) {
        case DML_TENSOR_DATA_TYPE_FLOAT32: return 4;
        case DML_TENSOR_DATA_TYPE_FLOAT16: return 2;
        default: return 4;
    }
}

// Metadata for building DML tensor descriptions
struct dml_tensor_info {
    UINT              sizes[4];
    UINT              strides[4];
    DML_TENSOR_DATA_TYPE data_type;
    UINT64            buffer_offset;
    UINT64            total_size;
    ID3D12Resource *  resource;
};

static dml_tensor_info dml_make_tensor_info(const struct ggml_tensor * t) {
    dml_tensor_info info = {};
    info.data_type = ggml_type_to_dml(t->type);
    info.resource = dml_get_tensor_resource(t);
    info.buffer_offset = dml_tensor_offset(t);
    // Total buffer size for DML tensor (elements × element type size)
    info.total_size = ggml_nelements(t) * dml_type_size(info.data_type);

    // Map GGML ne[3,2,1,0] -> DML sizes[0,1,2,3]  (NCHW: N=ne[3], C=ne[2], H=ne[1], W=ne[0])
    info.sizes[0] = (UINT)t->ne[3];
    info.sizes[1] = (UINT)t->ne[2];
    info.sizes[2] = (UINT)t->ne[1];
    info.sizes[3] = (UINT)t->ne[0];

    // Compute packed strides (in element counts, not bytes)
    info.strides[3] = 1;
    info.strides[2] = info.sizes[3];
    info.strides[1] = info.strides[2] * info.sizes[2];
    info.strides[0] = info.strides[1] * info.sizes[1];

    return info;
}

// Helper: broadcast a tensor's sizes to match a target shape (for element-wise ops)
static dml_tensor_info dml_broadcast_tensor_info(const struct ggml_tensor * t,
                                                   const UINT target_sizes[4]) {
    dml_tensor_info info = dml_make_tensor_info(t);
    // Adjust sizes and strides for broadcasting
    for (int i = 0; i < 4; i++) {
        if (info.sizes[i] == 1 && target_sizes[i] != 1) {
            info.sizes[i] = target_sizes[i];
            info.strides[i] = 0; // broadcast: stride 0 means replicate
        }
    }
    return info;
}

// Execute a compiled DML operator
static bool dml_execute_op(ggml_dml_device_info & dev,
                            IDMLCompiledOperator * compiled_op,
                            const std::vector<DML_BINDING_DESC> & input_descs,
                            const DML_BINDING_DESC & output_desc,
                            ID3D12Resource * temp_resource = nullptr,
                            ID3D12Resource * persistent_resource = nullptr) {
    auto bp = compiled_op->GetBindingProperties();

    if (bp.RequiredDescriptorCount > ggml_dml_device_info::DESC_HEAP_SIZE) {
        GGML_LOG_ERROR("DirectML: compiled op needs %u descriptors, heap has %u\n",
                       bp.RequiredDescriptorCount, ggml_dml_device_info::DESC_HEAP_SIZE);
        return false;
    }

    // Create binding table using the device's descriptor heap
    DML_BINDING_TABLE_DESC btd = {};
    btd.Dispatchable = compiled_op;
    btd.CPUDescriptorHandle = dev.desc_heap->GetCPUDescriptorHandleForHeapStart();
    btd.GPUDescriptorHandle = dev.desc_heap->GetGPUDescriptorHandleForHeapStart();
    btd.SizeInDescriptors = bp.RequiredDescriptorCount;

    ComPtr<IDMLBindingTable> binding_table;
    HRESULT hr = dev.dml_device->CreateBindingTable(&btd, IID_PPV_ARGS(&binding_table));
    if (FAILED(hr)) {
        GGML_LOG_ERROR("DirectML: CreateBindingTable failed: 0x%08x\n", (unsigned)hr);
        return false;
    }

    // Bind inputs (caller provides DML_BINDING_DESC entries, including NONE for optional slots)
    binding_table->BindInputs((UINT)input_descs.size(), input_descs.data());

    // Bind output
    binding_table->BindOutputs(1, &output_desc);

    // Bind temporary resource if needed
    if (bp.TemporaryResourceSize > 0) {
        if (!temp_resource) {
            GGML_LOG_ERROR("DirectML: exec needs %llu temp bytes but no temp resource provided!\n",
                           (unsigned long long)bp.TemporaryResourceSize);
            return false;
        }
        DML_BUFFER_BINDING temp_bind = {};
        temp_bind.Buffer = temp_resource;
        temp_bind.SizeInBytes = bp.TemporaryResourceSize;
        DML_BINDING_DESC temp_desc = { DML_BINDING_TYPE_BUFFER, &temp_bind };
        binding_table->BindTemporaryResource(&temp_desc);
    }

    // Bind persistent resource if needed
    if (bp.PersistentResourceSize > 0) {
        if (!persistent_resource) {
            GGML_LOG_ERROR("DirectML: exec needs %llu persistent bytes but no persistent resource!\n",
                           (unsigned long long)bp.PersistentResourceSize);
            return false;
        }
        DML_BUFFER_BINDING persist_bind = {};
        persist_bind.Buffer = persistent_resource;
        persist_bind.SizeInBytes = bp.PersistentResourceSize;
        DML_BINDING_DESC persist_desc = { DML_BINDING_TYPE_BUFFER, &persist_bind };
        binding_table->BindPersistentResource(&persist_desc);
    }

    // Record and execute
    dev.reset_cmd_list();
    ID3D12DescriptorHeap * heaps[] = { dev.desc_heap.Get() };
    dev.cmd_list->SetDescriptorHeaps(1, heaps);
    dev.dml_cmd_recorder->RecordDispatch(dev.cmd_list.Get(), compiled_op, binding_table.Get());

    // UAV barrier for output resource (extract from output_desc if it's a buffer binding)
    if (output_desc.Type == DML_BINDING_TYPE_BUFFER && output_desc.Desc) {
        auto * out_buf = static_cast<const DML_BUFFER_BINDING *>(output_desc.Desc);
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = out_buf->Buffer;
        dev.cmd_list->ResourceBarrier(1, &barrier);
    }

    dev.execute_and_wait();
    return true;
}

// Compile and initialize a DML operator (one-time cost, to be cached later)
// Returns a dml_cached_op with compiled operator, persistent and temp resources.
static dml_cached_op dml_compile_op(ggml_dml_device_info & dev,
                                     const DML_OPERATOR_DESC & op_desc) {
    dml_cached_op result = {};

    ComPtr<IDMLOperator> dml_op;
    HRESULT hr = dev.dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&dml_op));
    if (FAILED(hr)) {
        GGML_LOG_ERROR("DirectML: CreateOperator failed: 0x%08x\n", (unsigned)hr);
        return result;
    }

    ComPtr<IDMLCompiledOperator> compiled;
    hr = dev.dml_device->CompileOperator(dml_op.Get(),
        DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, IID_PPV_ARGS(&compiled));
    if (FAILED(hr)) {
        GGML_LOG_ERROR("DirectML: CompileOperator failed: 0x%08x\n", (unsigned)hr);
        return result;
    }

    // Get the compiled operator's execution binding properties
    auto exec_bp = compiled->GetBindingProperties();

    // Pre-allocate persistent + temp resources based on execution binding properties
    if (exec_bp.PersistentResourceSize > 0) {
        dml_create_buffer(dev.d3d_device.Get(), exec_bp.PersistentResourceSize,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, result.persistent_resource);
    }
    if (exec_bp.TemporaryResourceSize > 0) {
        dml_create_buffer(dev.d3d_device.Get(), exec_bp.TemporaryResourceSize,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, result.temp_resource);
    }

    // Initialize the operator (one-time GPU work)
    ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator * ops[] = { compiled.Get() };
    hr = dev.dml_device->CreateOperatorInitializer(1, ops, IID_PPV_ARGS(&initializer));
    if (FAILED(hr)) {
        GGML_LOG_ERROR("DirectML: CreateOperatorInitializer failed: 0x%08x\n", (unsigned)hr);
        return result; // Return with compiled but uninitialized
    }

    auto init_bp = initializer->GetBindingProperties();

    // Skip initialization dispatch if the initializer doesn't need anything
    if (init_bp.RequiredDescriptorCount == 0 &&
        init_bp.TemporaryResourceSize == 0 &&
        !result.persistent_resource) {
        result.compiled = compiled;
        return result;
    }

    // Create temp buffer for initialization if needed
    ComPtr<ID3D12Resource> init_temp;
    if (init_bp.TemporaryResourceSize > 0) {
        dml_create_buffer(dev.d3d_device.Get(), init_bp.TemporaryResourceSize,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, init_temp);
    }

    // Create binding table for initialization
    UINT init_desc_count = init_bp.RequiredDescriptorCount;
    if (init_desc_count == 0) init_desc_count = 1; // need at least 1 for binding table

    DML_BINDING_TABLE_DESC btd = {};
    btd.Dispatchable = initializer.Get();
    btd.CPUDescriptorHandle = dev.desc_heap->GetCPUDescriptorHandleForHeapStart();
    btd.GPUDescriptorHandle = dev.desc_heap->GetGPUDescriptorHandleForHeapStart();
    btd.SizeInDescriptors = init_desc_count;

    ComPtr<IDMLBindingTable> binding_table;
    hr = dev.dml_device->CreateBindingTable(&btd, IID_PPV_ARGS(&binding_table));
    if (FAILED(hr)) {
        // Initialization binding failed, but operator may still work
        result.compiled = compiled;
        return result;
    }

    if (init_temp) {
        DML_BUFFER_BINDING temp_bind = {};
        temp_bind.Buffer = init_temp.Get();
        temp_bind.SizeInBytes = init_bp.TemporaryResourceSize;
        DML_BINDING_DESC temp_desc = { DML_BINDING_TYPE_BUFFER, &temp_bind };
        binding_table->BindTemporaryResource(&temp_desc);
    }

    // Bind persistent resource during initialization (DML writes init data into it)
    if (result.persistent_resource) {
        DML_BUFFER_BINDING persist_bind = {};
        persist_bind.Buffer = result.persistent_resource.Get();
        persist_bind.SizeInBytes = exec_bp.PersistentResourceSize;
        DML_BINDING_DESC persist_desc = { DML_BINDING_TYPE_BUFFER, &persist_bind };
        binding_table->BindOutputs(1, &persist_desc);
    }

    // Record initialization dispatch
    dev.reset_cmd_list();
    ID3D12DescriptorHeap * heaps[] = { dev.desc_heap.Get() };
    dev.cmd_list->SetDescriptorHeaps(1, heaps);
    dev.dml_cmd_recorder->RecordDispatch(dev.cmd_list.Get(), initializer.Get(), binding_table.Get());
    dev.execute_and_wait();

    // Check device health after initialization
    HRESULT init_hr = dev.d3d_device->GetDeviceRemovedReason();
    if (init_hr != S_OK) {
        GGML_LOG_ERROR("DirectML: device removed during op initialization: 0x%08x\n", (unsigned)init_hr);
        return result; // compiled is null, signals failure
    }

    result.compiled = compiled;
    return result;
}

// Get a compiled operator from cache or compile + cache it
static dml_cached_op * dml_get_or_compile_op(ggml_dml_device_info & dev,
                                               const dml_op_cache_key & key,
                                               const DML_OPERATOR_DESC & op_desc) {
    auto it = dev.op_cache.find(key);
    if (it != dev.op_cache.end()) {
        return &it->second;
    }

    auto entry = dml_compile_op(dev, op_desc);
    if (!entry.compiled) return nullptr;

    auto [ins, _] = dev.op_cache.emplace(key, std::move(entry));
    return &ins->second;
}

// ---------------------------------------------------------------------------
// Per-op DML dispatch implementations
// ---------------------------------------------------------------------------

// Generic element-wise binary op (ADD, MUL)
static bool dml_op_elementwise_binary(ggml_dml_device_info & dev, struct ggml_tensor * node,
                                       DML_OPERATOR_TYPE op_type) {
    auto dst_info = dml_make_tensor_info(node);
    auto a_info   = dml_broadcast_tensor_info(node->src[0], dst_info.sizes);
    auto b_info   = dml_broadcast_tensor_info(node->src[1], dst_info.sizes);

    // Build DML tensor descriptions
    DML_BUFFER_TENSOR_DESC a_buf = {};
    a_buf.DataType = a_info.data_type;
    a_buf.DimensionCount = 4;
    a_buf.Sizes = a_info.sizes;
    a_buf.Strides = a_info.strides;
    a_buf.TotalTensorSizeInBytes = a_info.total_size;

    DML_BUFFER_TENSOR_DESC b_buf = {};
    b_buf.DataType = b_info.data_type;
    b_buf.DimensionCount = 4;
    b_buf.Sizes = b_info.sizes;
    b_buf.Strides = b_info.strides;
    b_buf.TotalTensorSizeInBytes = b_info.total_size;

    DML_BUFFER_TENSOR_DESC out_buf = {};
    out_buf.DataType = dst_info.data_type;
    out_buf.DimensionCount = 4;
    out_buf.Sizes = dst_info.sizes;
    out_buf.TotalTensorSizeInBytes = dst_info.total_size;

    DML_TENSOR_DESC a_desc = { DML_TENSOR_TYPE_BUFFER, &a_buf };
    DML_TENSOR_DESC b_desc = { DML_TENSOR_TYPE_BUFFER, &b_buf };
    DML_TENSOR_DESC out_desc = { DML_TENSOR_TYPE_BUFFER, &out_buf };

    DML_OPERATOR_DESC op_desc = {};
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_d = {};
    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_d = {};

    if (op_type == DML_OPERATOR_ELEMENT_WISE_ADD) {
        add_d.ATensor = &a_desc;
        add_d.BTensor = &b_desc;
        add_d.OutputTensor = &out_desc;
        op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
        op_desc.Desc = &add_d;
    } else {
        mul_d.ATensor = &a_desc;
        mul_d.BTensor = &b_desc;
        mul_d.OutputTensor = &out_desc;
        op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
        op_desc.Desc = &mul_d;
    }

    auto compiled = dml_get_or_compile_op(dev, dml_make_cache_key(node), op_desc);
    if (!compiled) return false;

    // Set up bindings
    DML_BUFFER_BINDING a_bind = { a_info.resource, a_info.buffer_offset, a_info.total_size };
    DML_BUFFER_BINDING b_bind = { b_info.resource, b_info.buffer_offset, b_info.total_size };
    DML_BUFFER_BINDING out_bind = { dst_info.resource, dst_info.buffer_offset, dst_info.total_size };

    std::vector<DML_BINDING_DESC> inputs = {
        { DML_BINDING_TYPE_BUFFER, &a_bind },
        { DML_BINDING_TYPE_BUFFER, &b_bind },
    };
    DML_BINDING_DESC output = { DML_BINDING_TYPE_BUFFER, &out_bind };
    return dml_execute_op(dev, compiled->compiled.Get(), inputs, output,
                           compiled->temp_resource.Get(), compiled->persistent_resource.Get());
}

static bool dml_op_add(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    return dml_op_elementwise_binary(dev, node, DML_OPERATOR_ELEMENT_WISE_ADD);
}

static bool dml_op_scale(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    // SCALE: multiply tensor by a scalar stored in op_params
    // Reuse element-wise multiply with a broadcast scalar
    float scale;
    memcpy(&scale, node->op_params, sizeof(float));

    auto dst_info = dml_make_tensor_info(node);
    auto src_info = dml_make_tensor_info(node->src[0]);

    // For SCALE, we use DML_ELEMENT_WISE_IDENTITY with ScaleBias
    DML_BUFFER_TENSOR_DESC src_buf = {};
    src_buf.DataType = src_info.data_type;
    src_buf.DimensionCount = 4;
    src_buf.Sizes = src_info.sizes;
    src_buf.TotalTensorSizeInBytes = src_info.total_size;

    DML_BUFFER_TENSOR_DESC out_buf = {};
    out_buf.DataType = dst_info.data_type;
    out_buf.DimensionCount = 4;
    out_buf.Sizes = dst_info.sizes;
    out_buf.TotalTensorSizeInBytes = dst_info.total_size;

    DML_TENSOR_DESC src_desc = { DML_TENSOR_TYPE_BUFFER, &src_buf };
    DML_TENSOR_DESC out_desc = { DML_TENSOR_TYPE_BUFFER, &out_buf };

    DML_SCALE_BIAS scale_bias = { scale, 0.0f };

    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_d = {};
    identity_d.InputTensor = &src_desc;
    identity_d.OutputTensor = &out_desc;
    identity_d.ScaleBias = &scale_bias;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identity_d };

    auto compiled = dml_get_or_compile_op(dev, dml_make_cache_key(node), op_desc);
    if (!compiled) return false;

    DML_BUFFER_BINDING src_bind = { src_info.resource, src_info.buffer_offset, src_info.total_size };
    DML_BUFFER_BINDING out_bind = { dst_info.resource, dst_info.buffer_offset, dst_info.total_size };

    std::vector<DML_BINDING_DESC> inputs = {
        { DML_BINDING_TYPE_BUFFER, &src_bind },
    };
    DML_BINDING_DESC output = { DML_BINDING_TYPE_BUFFER, &out_bind };
    return dml_execute_op(dev, compiled->compiled.Get(), inputs, output,
                           compiled->temp_resource.Get(), compiled->persistent_resource.Get());
}

static bool dml_op_soft_max(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    auto dst_info = dml_make_tensor_info(node);
    auto src_info = dml_make_tensor_info(node->src[0]);

    DML_BUFFER_TENSOR_DESC src_buf = {};
    src_buf.DataType = src_info.data_type;
    src_buf.DimensionCount = 4;
    src_buf.Sizes = src_info.sizes;
    src_buf.TotalTensorSizeInBytes = src_info.total_size;

    DML_BUFFER_TENSOR_DESC out_buf = {};
    out_buf.DataType = dst_info.data_type;
    out_buf.DimensionCount = 4;
    out_buf.Sizes = dst_info.sizes;
    out_buf.TotalTensorSizeInBytes = dst_info.total_size;

    DML_TENSOR_DESC src_desc = { DML_TENSOR_TYPE_BUFFER, &src_buf };
    DML_TENSOR_DESC out_desc = { DML_TENSOR_TYPE_BUFFER, &out_buf };

    // Softmax on innermost dimension (DML sizes[3] = GGML ne[0])
    UINT axis = 3;
    DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC sm_d = {};
    sm_d.InputTensor = &src_desc;
    sm_d.OutputTensor = &out_desc;
    sm_d.AxisCount = 1;
    sm_d.Axes = &axis;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ACTIVATION_SOFTMAX1, &sm_d };

    auto compiled = dml_get_or_compile_op(dev, dml_make_cache_key(node), op_desc);
    if (!compiled) return false;

    DML_BUFFER_BINDING src_bind = { src_info.resource, src_info.buffer_offset, src_info.total_size };
    DML_BUFFER_BINDING out_bind = { dst_info.resource, dst_info.buffer_offset, dst_info.total_size };

    std::vector<DML_BINDING_DESC> inputs = {
        { DML_BINDING_TYPE_BUFFER, &src_bind },
    };
    DML_BINDING_DESC output = { DML_BINDING_TYPE_BUFFER, &out_bind };
    return dml_execute_op(dev, compiled->compiled.Get(), inputs, output,
                           compiled->temp_resource.Get(), compiled->persistent_resource.Get());
}

// DML native GEMM for MUL_MAT — NPU-compatible (no HLSL compute shaders).
// GGML MUL_MAT(a, b) = a^T * b
//   a  (src[0]) shape: [K, M, batch2, batch3]  (weights)
//   b  (src[1]) shape: [K, N, batch2, batch3]  (input)
//   dst shape:         [M, N, batch2, batch3]
//
// DML GEMM: Output = alpha * op(A) * op(B)
// We set ATensor=[batch3, batch2, K, M], TransA=TRANSPOSE → op(A) = M×K
//           BTensor=[batch3, batch2, K, N], TransB=NONE    → op(B) = K×N
//           Output =[batch3, batch2, M, N]                         = M×N
// Helper: run a DML CAST operator (type conversion) from input buffer to output buffer.
// Caller must provide pre-allocated input/output D3D12 resources of the correct sizes.
static bool dml_run_cast(ggml_dml_device_info & dev,
                          DML_TENSOR_DATA_TYPE from_type, DML_TENSOR_DATA_TYPE to_type,
                          const UINT * sizes, UINT dim_count,
                          ID3D12Resource * in_res, UINT64 in_offset, UINT64 in_bytes,
                          ID3D12Resource * out_res, UINT64 out_offset, UINT64 out_bytes) {
    DML_BUFFER_TENSOR_DESC in_buf = {};
    in_buf.DataType = from_type;
    in_buf.DimensionCount = dim_count;
    in_buf.Sizes = sizes;
    in_buf.TotalTensorSizeInBytes = in_bytes;

    DML_BUFFER_TENSOR_DESC out_buf = {};
    out_buf.DataType = to_type;
    out_buf.DimensionCount = dim_count;
    out_buf.Sizes = sizes;
    out_buf.TotalTensorSizeInBytes = out_bytes;

    DML_TENSOR_DESC in_desc = { DML_TENSOR_TYPE_BUFFER, &in_buf };
    DML_TENSOR_DESC out_desc = { DML_TENSOR_TYPE_BUFFER, &out_buf };

    DML_CAST_OPERATOR_DESC cast_d = {};
    cast_d.InputTensor = &in_desc;
    cast_d.OutputTensor = &out_desc;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_CAST, &cast_d };

    // Check device health before CAST
    HRESULT dev_hr = dev.d3d_device->GetDeviceRemovedReason();
    if (dev_hr != S_OK) {
        GGML_LOG_ERROR("dml_run_cast: device already removed before cast: 0x%08x\n", (unsigned)dev_hr);
        return false;
    }

    auto result = dml_compile_op(dev, op_desc);
    if (!result.compiled) {
        dev_hr = dev.d3d_device->GetDeviceRemovedReason();
        GGML_LOG_ERROR("dml_run_cast: compile failed, device reason: 0x%08x, "
                       "from=%d to=%d elements=%u\n",
                       (unsigned)dev_hr, (int)from_type, (int)to_type, sizes[dim_count - 1]);
        return false;
    }

    DML_BUFFER_BINDING in_bind = { in_res, in_offset, in_bytes };
    DML_BUFFER_BINDING out_bind = { out_res, out_offset, out_bytes };
    std::vector<DML_BINDING_DESC> inputs = {
        { DML_BINDING_TYPE_BUFFER, &in_bind },
    };
    DML_BINDING_DESC output = { DML_BINDING_TYPE_BUFFER, &out_bind };
    return dml_execute_op(dev, result.compiled.Get(), inputs, output,
                          result.temp_resource.Get(), result.persistent_resource.Get());
}

static bool dml_op_mul_mat_gemm(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    const struct ggml_tensor * a = node->src[0];
    const struct ggml_tensor * b = node->src[1];

    const UINT K = (UINT)a->ne[0];
    const UINT M = (UINT)a->ne[1];
    const UINT N = (UINT)b->ne[1];
    const UINT batch2 = (UINT)node->ne[2];
    const UINT batch3 = (UINT)node->ne[3];

    DML_TENSOR_DATA_TYPE a_type = ggml_type_to_dml(a->type);
    DML_TENSOR_DATA_TYPE b_type = ggml_type_to_dml(b->type);
    DML_TENSOR_DATA_TYPE out_type = ggml_type_to_dml(node->type);

    // DML GEMM requires all tensors to be the same type. Promote to F32.
    DML_TENSOR_DATA_TYPE gemm_type = out_type;
    size_t gemm_elem = dml_type_size(gemm_type);

    // Compute buffer sizes
    UINT64 a_nelements = (UINT64)ggml_nelements(a);
    UINT64 b_nelements = (UINT64)ggml_nelements(b);
    UINT64 out_nelements = (UINT64)batch3 * batch2 * M * N;

    // Cast GGML's a (weights, may be F16) to gemm_type (F32) if needed
    ComPtr<ID3D12Resource> a_cast_buf;
    ID3D12Resource * a_gemm_res;
    UINT64 a_gemm_offset;
    UINT64 a_gemm_bytes = a_nelements * gemm_elem;

    if (a_type != gemm_type) {
        HRESULT hr = dml_create_buffer(dev.d3d_device.Get(), (size_t)a_gemm_bytes,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, a_cast_buf);
        if (FAILED(hr)) {
            GGML_LOG_ERROR("dml_op_mul_mat_gemm: failed to create A cast buffer: 0x%08x\n", (unsigned)hr);
            return false;
        }
        ID3D12Resource * a_src_res = dml_get_tensor_resource(a);
        UINT64 a_src_bytes = a_nelements * dml_type_size(a_type);
        UINT a_flat[4] = { 1, 1, 1, (UINT)a_nelements };
        if (!dml_run_cast(dev, a_type, gemm_type, a_flat, 4,
                          a_src_res, dml_tensor_offset(a), a_src_bytes,
                          a_cast_buf.Get(), 0, a_gemm_bytes)) {
            GGML_LOG_ERROR("dml_op_mul_mat_gemm: A cast failed\n");
            return false;
        }
        a_gemm_res = a_cast_buf.Get();
        a_gemm_offset = 0;
    } else {
        a_gemm_res = dml_get_tensor_resource(a);
        a_gemm_offset = dml_tensor_offset(a);
        a_gemm_bytes = a_nelements * dml_type_size(a_type);
    }

    // b (activations) should already be gemm_type (F32)
    ID3D12Resource * b_gemm_res = dml_get_tensor_resource(b);
    UINT64 b_gemm_offset = dml_tensor_offset(b);
    UINT64 b_gemm_bytes = b_nelements * gemm_elem;

    ID3D12Resource * out_gemm_res = dml_get_tensor_resource(node);
    UINT64 out_gemm_offset = dml_tensor_offset(node);
    UINT64 out_gemm_bytes = out_nelements * gemm_elem;

    // GGML: dst[m,n] = sum_k(a[k,m] * b[k,n]), a=[K,M,...], b=[K,N,...], dst=[M,N,...]
    // GGML stores K (ne[0]) contiguous. DML packed = last dimension contiguous.
    //
    // Declare DML sizes matching physical layout (no strides on inner 2 dims):
    //   DML_A = GGML's b: physical [N,K] packed, TransA=NONE → N×K
    //   DML_B = GGML's a: physical [M,K] packed, TransB=TRANSPOSE → (M×K)^T = K×M
    //   Output = N×K × K×M = [N,M] packed → M fastest → matches GGML column-major

    // DML ATensor = GGML's b: [batch3, batch2, N, K] packed
    UINT dml_a_sizes[4] = { batch3, batch2, N, K };
    UINT dml_a_strides[4] = { 0, 0, 0, 0 }; // Will be set only if broadcast needed
    bool dml_a_needs_strides = false;

    UINT b_batch2 = (UINT)b->ne[2];
    UINT b_batch3 = (UINT)b->ne[3];
    if (b_batch2 < batch2 || b_batch3 < batch3) {
        dml_a_strides[3] = 1;
        dml_a_strides[2] = K;
        dml_a_strides[1] = (b_batch2 < batch2) ? 0 : K * N;
        dml_a_strides[0] = (b_batch3 < batch3) ? 0 : b_batch2 * K * N;
        dml_a_needs_strides = true;
    }

    DML_BUFFER_TENSOR_DESC dml_a_buf = {};
    dml_a_buf.DataType = gemm_type;
    dml_a_buf.DimensionCount = 4;
    dml_a_buf.Sizes = dml_a_sizes;
    dml_a_buf.Strides = dml_a_needs_strides ? dml_a_strides : nullptr;
    dml_a_buf.TotalTensorSizeInBytes = b_gemm_bytes;

    // DML BTensor = GGML's a: [batch3, batch2, M, K] packed
    UINT dml_b_sizes[4] = { batch3, batch2, M, K };
    UINT dml_b_strides[4] = { 0, 0, 0, 0 };
    bool dml_b_needs_strides = false;

    UINT a_batch2 = (UINT)a->ne[2];
    UINT a_batch3 = (UINT)a->ne[3];
    if (a_batch2 < batch2 || a_batch3 < batch3) {
        dml_b_strides[3] = 1;
        dml_b_strides[2] = K;
        dml_b_strides[1] = (a_batch2 < batch2) ? 0 : K * M;
        dml_b_strides[0] = (a_batch3 < batch3) ? 0 : a_batch2 * K * M;
        dml_b_needs_strides = true;
    }

    DML_BUFFER_TENSOR_DESC dml_b_buf = {};
    dml_b_buf.DataType = gemm_type;
    dml_b_buf.DimensionCount = 4;
    dml_b_buf.Sizes = dml_b_sizes;
    dml_b_buf.Strides = dml_b_needs_strides ? dml_b_strides : nullptr;
    dml_b_buf.TotalTensorSizeInBytes = a_gemm_bytes;

    // Output: [batch3, batch2, N, M] packed → M fastest → matches GGML
    UINT out_sizes[4] = { batch3, batch2, N, M };
    DML_BUFFER_TENSOR_DESC out_gemm_buf = {};
    out_gemm_buf.DataType = gemm_type;
    out_gemm_buf.DimensionCount = 4;
    out_gemm_buf.Sizes = out_sizes;
    out_gemm_buf.TotalTensorSizeInBytes = out_gemm_bytes;

    DML_TENSOR_DESC a_desc = { DML_TENSOR_TYPE_BUFFER, &dml_a_buf };
    DML_TENSOR_DESC b_desc = { DML_TENSOR_TYPE_BUFFER, &dml_b_buf };
    DML_TENSOR_DESC out_desc = { DML_TENSOR_TYPE_BUFFER, &out_gemm_buf };

    DML_GEMM_OPERATOR_DESC gemm_d = {};
    gemm_d.ATensor = &a_desc;   // GGML's b (activations, packed [N,K])
    gemm_d.BTensor = &b_desc;   // GGML's a (weights, packed [M,K])
    gemm_d.CTensor = nullptr;
    gemm_d.OutputTensor = &out_desc;
    gemm_d.TransA = DML_MATRIX_TRANSFORM_NONE;       // N×K as-is
    gemm_d.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;   // (M×K)^T = K×M
    gemm_d.Alpha = 1.0f;
    gemm_d.Beta  = 0.0f;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_GEMM, &gemm_d };

    auto compiled = dml_get_or_compile_op(dev, dml_make_cache_key(node), op_desc);
    if (!compiled) return false;

    // Debug: log GEMM binding properties and buffer sizes
    {
        auto bp = compiled->compiled->GetBindingProperties();
        GGML_LOG_DEBUG("dml_op_mul_mat_gemm: K=%u M=%u N=%u batch=[%u,%u] "
                       "desc=%u temp=%llu persist=%llu\n",
                       K, M, N, batch2, batch3,
                       bp.RequiredDescriptorCount,
                       (unsigned long long)bp.TemporaryResourceSize,
                       (unsigned long long)bp.PersistentResourceSize);
        GGML_LOG_DEBUG("  A: res=%p off=%llu sz=%llu | B: res=%p off=%llu sz=%llu | "
                       "Out: res=%p off=%llu sz=%llu\n",
                       a_gemm_res, (unsigned long long)a_gemm_offset,
                       (unsigned long long)a_gemm_bytes,
                       b_gemm_res, (unsigned long long)b_gemm_offset,
                       (unsigned long long)b_gemm_bytes,
                       out_gemm_res, (unsigned long long)out_gemm_offset,
                       (unsigned long long)out_gemm_bytes);
    }

    // DML ATensor = GGML's b (activations), DML BTensor = GGML's a (weights)
    DML_BUFFER_BINDING dml_a_bind = { b_gemm_res, b_gemm_offset, b_gemm_bytes };
    DML_BUFFER_BINDING dml_b_bind = { a_gemm_res, a_gemm_offset, a_gemm_bytes };
    DML_BUFFER_BINDING out_bind = { out_gemm_res, out_gemm_offset, out_gemm_bytes };

    // GEMM has 3 input slots: ATensor, BTensor, CTensor (optional bias).
    // CTensor is nullptr in our operator desc, but BindInputs requires a slot for it.
    std::vector<DML_BINDING_DESC> inputs = {
        { DML_BINDING_TYPE_BUFFER, &dml_a_bind },
        { DML_BINDING_TYPE_BUFFER, &dml_b_bind },
        { DML_BINDING_TYPE_NONE,   nullptr },       // CTensor omitted
    };
    DML_BINDING_DESC output = { DML_BINDING_TYPE_BUFFER, &out_bind };
    return dml_execute_op(dev, compiled->compiled.Get(), inputs, output,
                          compiled->temp_resource.Get(), compiled->persistent_resource.Get());
}

static bool dml_op_rms_norm(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    GGML_UNUSED(dev);
    GGML_UNUSED(node);
    return false; // Handled by dml_cs_rms_norm instead
}

// ---------------------------------------------------------------------------
// Custom HLSL compute shader dispatch
// ---------------------------------------------------------------------------

static bool dml_dispatch_compute(ggml_dml_device_info & dev,
                                   ID3D12PipelineState * pso,
                                   const uint32_t * constants, uint32_t num_constants,
                                   ID3D12Resource * const * uavs, uint32_t num_uavs,
                                   uint32_t groups_x, uint32_t groups_y = 1,
                                   uint32_t groups_z = 1) {
    dev.reset_cmd_list();
    dev.cmd_list->SetComputeRootSignature(dev.compute_root_sig.Get());
    dev.cmd_list->SetPipelineState(pso);
    dev.cmd_list->SetComputeRoot32BitConstants(0, num_constants, constants, 0);
    // Always bind all 3 UAV root descriptors (D3D12 requires all root params set)
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t idx = (i < num_uavs) ? i : 0; // unused slots bind uavs[0] as placeholder
        dev.cmd_list->SetComputeRootUnorderedAccessView(1 + i,
            uavs[idx]->GetGPUVirtualAddress());
    }
    dev.cmd_list->Dispatch(groups_x, groups_y, groups_z);

    // UAV barrier on all resources
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    dev.cmd_list->ResourceBarrier(1, &barrier);

    dev.execute_and_wait();
    return true;
}

// Element-wise ADD using HLSL compute shader (with broadcasting support)
static bool dml_cs_add(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src0 = node->src[0];
    auto * src1 = node->src[1];
    uint32_t n_dst = (uint32_t)ggml_nelements(node);
    uint32_t n_src1 = (uint32_t)ggml_nelements(src1);
    uint32_t ne00 = (uint32_t)node->ne[0];

    GGML_ASSERT(dml_tensor_offset(src0) < (1ull << 32) && "ADD src0 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(src1) < (1ull << 32) && "ADD src1 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "ADD dst offset exceeds 4 GB");

    uint32_t src0_off = (uint32_t)dml_tensor_offset(src0);
    uint32_t src1_off = (uint32_t)dml_tensor_offset(src1);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    uint32_t constants[] = { n_dst, n_src1, ne00, src0_off, src1_off, dst_off };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src0),
                                 dml_get_tensor_resource(src1),
                                 dml_get_tensor_resource(node) };
    return dml_dispatch_compute(dev, dev.pso_add.Get(), constants, 6, uavs, 3,
                                 (n_dst + 255) / 256);
}

// Element-wise MUL using HLSL compute shader (with broadcasting support)
static bool dml_cs_mul(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src0 = node->src[0];
    auto * src1 = node->src[1];

    uint32_t n_dst = (uint32_t)ggml_nelements(node);
    uint32_t n_src1 = (uint32_t)ggml_nelements(src1);
    uint32_t ne00 = (uint32_t)node->ne[0];

    GGML_ASSERT(dml_tensor_offset(src0) < (1ull << 32) && "MUL src0 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(src1) < (1ull << 32) && "MUL src1 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "MUL dst offset exceeds 4 GB");

    uint32_t src0_off = (uint32_t)dml_tensor_offset(src0);
    uint32_t src1_off = (uint32_t)dml_tensor_offset(src1);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    uint32_t constants[] = { n_dst, n_src1, ne00, src0_off, src1_off, dst_off };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src0),
                                 dml_get_tensor_resource(src1),
                                 dml_get_tensor_resource(node) };
    return dml_dispatch_compute(dev, dev.pso_mul.Get(), constants, 6, uavs, 3,
                                 (n_dst + 255) / 256);
}

// SCALE using HLSL compute shader
static bool dml_cs_scale(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src = node->src[0];
    uint32_t n = (uint32_t)ggml_nelements(src);
    float scale;
    memcpy(&scale, node->op_params, sizeof(float));
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(uint32_t));

    GGML_ASSERT(dml_tensor_offset(src)  < (1ull << 32) && "SCALE src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "SCALE dst offset exceeds 4 GB");

    uint32_t src_off = (uint32_t)dml_tensor_offset(src);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    uint32_t constants[] = { n, scale_bits, src_off, dst_off };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src),
                                 dml_get_tensor_resource(node) };
    return dml_dispatch_compute(dev, dev.pso_scale.Get(), constants, 4, uavs, 2,
                                 (n + 255) / 256);
}

// SOFT_MAX using HLSL compute shader — one workgroup per row
static bool dml_cs_soft_max(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src = node->src[0];
    auto * mask = node->src[1]; // optional attention mask

    float scale;
    memcpy(&scale, (float *)node->op_params + 0, sizeof(float));
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(uint32_t));

    uint32_t ne00 = (uint32_t)node->ne[0];
    uint32_t nrows = (uint32_t)(ggml_nelements(node) / ne00);
    uint32_t has_mask = (mask != nullptr) ? 1u : 0u;

    GGML_ASSERT(dml_tensor_offset(src)  < (1ull << 32) && "SOFT_MAX src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "SOFT_MAX dst offset exceeds 4 GB");
    if (mask) {
        GGML_ASSERT(dml_tensor_offset(mask) < (1ull << 32) && "SOFT_MAX mask offset exceeds 4 GB");
    }

    uint32_t src_off = (uint32_t)dml_tensor_offset(src);
    uint32_t mask_off = has_mask ? (uint32_t)dml_tensor_offset(mask) : 0u;
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);
    uint32_t mask_nb1 = has_mask ? (uint32_t)mask->nb[1] : 0u;
    uint32_t mask_nrows = has_mask ? (uint32_t)(ggml_nelements(mask) / mask->ne[0]) : 1u;

    uint32_t constants[] = { ne00, nrows, scale_bits, has_mask, src_off, mask_off, dst_off, mask_nb1, mask_nrows };

    if (has_mask) {
        ID3D12Resource * uavs[] = { dml_get_tensor_resource(src),
                                     dml_get_tensor_resource(mask),
                                     dml_get_tensor_resource(node) };
        return dml_dispatch_compute(dev, dev.pso_soft_max.Get(), constants, 9, uavs, 3,
                                     nrows);
    } else {
        // No mask — bind src as u1 placeholder (won't be accessed)
        ID3D12Resource * uavs[] = { dml_get_tensor_resource(src),
                                     dml_get_tensor_resource(src),
                                     dml_get_tensor_resource(node) };
        return dml_dispatch_compute(dev, dev.pso_soft_max.Get(), constants, 9, uavs, 3,
                                     nrows);
    }
}

// SILU: x / (1 + exp(-x)), element-wise
static bool dml_cs_silu(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src = node->src[0];
    uint32_t n = (uint32_t)ggml_nelements(src);

    GGML_ASSERT(dml_tensor_offset(src)  < (1ull << 32) && "SILU src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "SILU dst offset exceeds 4 GB");

    uint32_t src_off = (uint32_t)dml_tensor_offset(src);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    uint32_t constants[] = { n, src_off, dst_off };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src),
                                 dml_get_tensor_resource(node) };
    return dml_dispatch_compute(dev, dev.pso_silu.Get(), constants, 3, uavs, 2,
                                 (n + 255) / 256);
}

// RMS_NORM: x / sqrt(mean(x^2) + eps), per-row normalization
static bool dml_cs_rms_norm(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src = node->src[0];
    uint32_t ne00 = (uint32_t)src->ne[0];
    uint32_t nrows = (uint32_t)(ggml_nelements(src) / ne00);
    float eps;
    memcpy(&eps, node->op_params, sizeof(float));
    uint32_t eps_bits;
    memcpy(&eps_bits, &eps, sizeof(uint32_t));

    GGML_ASSERT(dml_tensor_offset(src)  < (1ull << 32) && "RMS_NORM src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "RMS_NORM dst offset exceeds 4 GB");

    uint32_t src_off = (uint32_t)dml_tensor_offset(src);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    uint32_t constants[] = { ne00, nrows, eps_bits, src_off, dst_off };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src),
                                 dml_get_tensor_resource(node) };
    // One thread group per row (256 threads per group)
    return dml_dispatch_compute(dev, dev.pso_rms_norm.Get(), constants, 5, uavs, 2,
                                 nrows);
}

// DIAG_MASK_INF: causal attention mask (set future positions to -inf)
static bool dml_cs_diag_mask_inf(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src = node->src[0];
    int32_t n_past;
    memcpy(&n_past, node->op_params, sizeof(int32_t));

    uint32_t ne00 = (uint32_t)src->ne[0];
    uint32_t ne01 = (uint32_t)src->ne[1];
    uint32_t n_total = (uint32_t)ggml_nelements(src);

    GGML_ASSERT(dml_tensor_offset(src)  < (1ull << 32) && "DIAG_MASK_INF src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "DIAG_MASK_INF dst offset exceeds 4 GB");

    uint32_t src_off = (uint32_t)dml_tensor_offset(src);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    uint32_t constants[] = { ne00, ne01, (uint32_t)n_past, n_total, src_off, dst_off };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src),
                                 dml_get_tensor_resource(node) };
    return dml_dispatch_compute(dev, dev.pso_diag_mask_inf.Get(), constants, 6, uavs, 2,
                                 (n_total + 255) / 256);
}

// SET_ROWS: scatter-write F32 rows into dst at indexed positions (F16 or F32 dst)
// CPU-side implementation: reads src data from GPU, computes scatter on CPU, writes rows back to GPU.
// TODO: optimize with a GPU compute shader once the shader is validated end-to-end.
static bool dml_cs_set_rows(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0]; // F32 source data
    const struct ggml_tensor * src1 = node->src[1]; // row indices (I32 or I64)

    auto * src0_buf_ctx = (ggml_dml_buffer_context *)src0->buffer->context;
    auto * src1_buf_ctx = (ggml_dml_buffer_context *)src1->buffer->context;
    auto * dst_buf_ctx  = (ggml_dml_buffer_context *)node->buffer->context;

    uint32_t ne00 = (uint32_t)src0->ne[0];
    uint32_t ne01 = (uint32_t)src0->ne[1];
    uint32_t ne02 = (uint32_t)src0->ne[2];
    uint32_t ne03 = (uint32_t)src0->ne[3];
    uint32_t ne11 = (uint32_t)src1->ne[1];
    uint32_t ne12 = (uint32_t)src1->ne[2];

    size_t src0_bytes = ggml_nbytes(src0);
    size_t src1_bytes = ggml_nbytes(src1);

    std::vector<uint8_t> src0_data(src0_bytes);
    std::vector<uint8_t> src1_data(src1_bytes);

    auto gpu_read = [&](ggml_dml_buffer_context * buf_ctx, const struct ggml_tensor * t,
                        uint8_t * out, size_t bytes) {
        uint64_t off = dml_tensor_offset(t);
        ID3D12Resource * staging = dev.ensure_readback_staging(bytes);
        if (!staging) return;
        dev.reset_cmd_list();
        dev.cmd_list->CopyBufferRegion(staging, 0, buf_ctx->gpu_resource.Get(), off, bytes);
        dev.execute_and_wait();
        void * mapped = nullptr;
        D3D12_RANGE rr = { 0, (SIZE_T)bytes };
        if (SUCCEEDED(staging->Map(0, &rr, &mapped))) {
            memcpy(out, mapped, bytes);
            D3D12_RANGE wr = {0, 0};
            staging->Unmap(0, &wr);
        }
    };

    gpu_read(src0_buf_ctx, src0, src0_data.data(), src0_bytes);
    gpu_read(src1_buf_ctx, src1, src1_data.data(), src1_bytes);

    size_t row_bytes = ne00 * (node->type == GGML_TYPE_F16 ? 2 : 4);
    std::vector<uint8_t> row_buf(row_bytes);

    for (int64_t i03 = 0; i03 < (int64_t)ne03; ++i03) {
        for (int64_t i02 = 0; i02 < (int64_t)ne02; ++i02) {
            for (int64_t i01 = 0; i01 < (int64_t)ne01; ++i01) {
                int64_t i12 = i03 % (int64_t)ne12;
                int64_t i11 = i02 % (int64_t)ne11;
                int64_t i10 = i01;

                int64_t row_idx;
                size_t idx_byte_off = (size_t)(i10 * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]);
                if (src1->type == GGML_TYPE_I64) {
                    memcpy(&row_idx, src1_data.data() + idx_byte_off, sizeof(int64_t));
                } else {
                    int32_t idx32;
                    memcpy(&idx32, src1_data.data() + idx_byte_off, sizeof(int32_t));
                    row_idx = idx32;
                }

                const float * src_row = (const float *)(src0_data.data() +
                    (size_t)(i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]));

                if (node->type == GGML_TYPE_F16) {
                    uint16_t * dst_f16 = (uint16_t *)row_buf.data();
                    for (int64_t i = 0; i < (int64_t)ne00; i++) {
                        dst_f16[i] = ggml_fp32_to_fp16(src_row[i]);
                    }
                } else {
                    memcpy(row_buf.data(), src_row, row_bytes);
                }

                uint64_t dst_row_off = dml_tensor_offset(node) +
                    (uint64_t)(row_idx * node->nb[1] + i02 * node->nb[2] + i03 * node->nb[3]);
                ID3D12Resource * staging = dev.ensure_upload_staging(row_bytes);
                if (staging) {
                    void * mapped = nullptr;
                    D3D12_RANGE rr = {0, 0};
                    if (SUCCEEDED(staging->Map(0, &rr, &mapped))) {
                        memcpy(mapped, row_buf.data(), row_bytes);
                        D3D12_RANGE wr = { 0, (SIZE_T)row_bytes };
                        staging->Unmap(0, &wr);
                        dev.reset_cmd_list();
                        dev.cmd_list->CopyBufferRegion(dst_buf_ctx->gpu_resource.Get(), dst_row_off,
                                                        staging, 0, row_bytes);
                        dev.execute_and_wait();
                    }
                }
            }
        }
    }

    return true;
}

// CONT: strided-to-contiguous copy
static bool dml_cs_cont(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    auto * src0 = node->src[0];
    uint32_t n_total = (uint32_t)ggml_nelements(node);

    GGML_ASSERT(dml_tensor_offset(src0) < (1ull << 32) && "CONT src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "CONT dst offset exceeds 4 GB");

    uint64_t src_off = dml_tensor_offset(src0);
    uint64_t dst_off = dml_tensor_offset(node);

    // Fast path: if src is already contiguous, just do a buffer copy
    if (ggml_is_contiguous(src0)) {
        size_t nbytes = ggml_nbytes(node);
        dev.reset_cmd_list();
        dev.cmd_list->CopyBufferRegion(
            dml_get_tensor_resource(node), dst_off,
            dml_get_tensor_resource(src0), src_off,
            nbytes);
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = nullptr;
        dev.cmd_list->ResourceBarrier(1, &barrier);
        dev.execute_and_wait();
        return true;
    }

    // Non-contiguous path: read src data row-by-row from GPU, reorder on CPU, write back
    // This is correct but slow — will be replaced by compute shader once validated
    ID3D12Resource * src_res = dml_get_tensor_resource(src0);
    ID3D12Resource * dst_res = dml_get_tensor_resource(node);

    int64_t ne0 = node->ne[0];
    int64_t ne1 = node->ne[1];
    int64_t ne2 = node->ne[2];
    int64_t ne3 = node->ne[3];
    size_t nb00 = src0->nb[0];
    size_t nb01 = src0->nb[1];
    size_t nb02 = src0->nb[2];
    size_t nb03 = src0->nb[3];

    // Calculate source data extent for readback
    size_t src_extent = 0;
    for (int d = 0; d < 4; d++) {
        src_extent = std::max(src_extent,
            (size_t)((src0->ne[d] - 1) * src0->nb[d] + src0->nb[0]));
    }
    // More accurate: find max byte address
    size_t max_src_byte = (ne0-1)*nb00 + (ne1-1)*nb01 + (ne2-1)*nb02 + (ne3-1)*nb03 + nb00;

    // Read source from GPU
    ID3D12Resource * staging_read = dev.ensure_readback_staging(max_src_byte);
    if (!staging_read) return false;
    dev.reset_cmd_list();
    dev.cmd_list->CopyBufferRegion(staging_read, 0, src_res, src_off, max_src_byte);
    dev.execute_and_wait();

    const uint8_t * src_mapped = nullptr;
    D3D12_RANGE rr = { 0, (SIZE_T)max_src_byte };
    if (FAILED(staging_read->Map(0, &rr, (void**)&src_mapped))) return false;

    // Reorder on CPU
    size_t dst_bytes = n_total * sizeof(float);
    std::vector<float> dst_data(n_total);
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    size_t byte_off = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
                    float val;
                    memcpy(&val, src_mapped + byte_off, sizeof(float));
                    dst_data[idx++] = val;
                }
            }
        }
    }

    D3D12_RANGE wr_none = { 0, 0 };
    staging_read->Unmap(0, &wr_none);

    // Write contiguous result to GPU
    ID3D12Resource * staging_write = dev.ensure_upload_staging(dst_bytes);
    if (!staging_write) return false;
    void * write_mapped = nullptr;
    D3D12_RANGE rr2 = { 0, 0 };
    if (FAILED(staging_write->Map(0, &rr2, &write_mapped))) return false;
    memcpy(write_mapped, dst_data.data(), dst_bytes);
    D3D12_RANGE wr = { 0, (SIZE_T)dst_bytes };
    staging_write->Unmap(0, &wr);

    dev.reset_cmd_list();
    dev.cmd_list->CopyBufferRegion(dst_res, dst_off, staging_write, 0, dst_bytes);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    dev.cmd_list->ResourceBarrier(1, &barrier);
    dev.execute_and_wait();
    return true;
}

// ROPE: Rotary Position Embedding
static bool dml_cs_rope(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src0 = node->src[0]; // input tensor
    auto * src1 = node->src[1]; // position indices (I32)

    uint32_t ne0 = (uint32_t)node->ne[0];
    uint32_t ne1 = (uint32_t)node->ne[1];
    uint32_t ne2 = (uint32_t)node->ne[2];
    uint32_t ne3 = (uint32_t)node->ne[3];

    // Extract op_params
    int32_t n_dims;
    int32_t mode;
    float freq_base, freq_scale;
    memcpy(&n_dims,     (int32_t *)node->op_params + 1, sizeof(int32_t));
    memcpy(&mode,       (int32_t *)node->op_params + 2, sizeof(int32_t));
    memcpy(&freq_base,  (int32_t *)node->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *)node->op_params + 6, sizeof(float));

    if (n_dims == 0) n_dims = (int32_t)ne0;

    GGML_ASSERT(dml_tensor_offset(src0) < (1ull << 32) && "ROPE src offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(src1) < (1ull << 32) && "ROPE pos offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "ROPE dst offset exceeds 4 GB");

    uint32_t src_off = (uint32_t)dml_tensor_offset(src0);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);
    uint32_t pos_off = (uint32_t)dml_tensor_offset(src1);

    uint32_t freq_base_bits, freq_scale_bits;
    memcpy(&freq_base_bits, &freq_base, sizeof(uint32_t));
    memcpy(&freq_scale_bits, &freq_scale, sizeof(uint32_t));

    // Source strides in bytes
    uint32_t nb00 = (uint32_t)src0->nb[0];
    uint32_t nb01 = (uint32_t)src0->nb[1];
    uint32_t nb02 = (uint32_t)src0->nb[2];
    uint32_t nb03 = (uint32_t)src0->nb[3];

    // Dispatch rotary pairs
    uint32_t n_pairs = ((uint32_t)n_dims / 2) * ne1 * ne2 * ne3;
    uint32_t constants[] = { ne0, ne1, ne2, (uint32_t)n_dims, (uint32_t)mode,
                              src_off, dst_off, pos_off, freq_base_bits, freq_scale_bits,
                              ne3, nb00, nb01, nb02, nb03 };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src0),
                                 dml_get_tensor_resource(src1),
                                 dml_get_tensor_resource(node) };
    bool ok = dml_dispatch_compute(dev, dev.pso_rope.Get(), constants, 15, uavs, 3,
                                    (n_pairs + 255) / 256);
    if (!ok) return false;

    // Copy passthrough elements (i0 >= n_dims) if any
    if ((uint32_t)n_dims < ne0) {
        uint32_t pass_width = ne0 - (uint32_t)n_dims;
        uint32_t n_pass = pass_width * ne1 * ne2 * ne3;
        uint32_t pass_constants[] = { ne0, ne1, ne2, (uint32_t)n_dims,
                                       src_off, dst_off, ne3, nb00, nb01, nb02, nb03, n_pass };
        ID3D12Resource * pass_uavs[] = { dml_get_tensor_resource(src0),
                                          dml_get_tensor_resource(node) };
        ok = dml_dispatch_compute(dev, dev.pso_rope_passthrough.Get(), pass_constants, 12, pass_uavs, 2,
                                   (n_pass + 255) / 256);
    }
    return ok;
}

// GLU/SWIGLU
static bool dml_cs_swiglu(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src0 = node->src[0];
    auto * src1 = node->src[1]; // NULL for single-tensor mode

    int32_t glu_op, swapped;
    memcpy(&glu_op, (int32_t *)node->op_params + 0, sizeof(int32_t));
    memcpy(&swapped, (int32_t *)node->op_params + 1, sizeof(int32_t));

    uint32_t ne00 = (uint32_t)src0->ne[0];
    uint32_t nc = (uint32_t)node->ne[0]; // output width (ne00/2 for single, ne00 for split)
    uint32_t nrows = (uint32_t)(ggml_nelements(node) / nc);

    GGML_ASSERT(dml_tensor_offset(src0) < (1ull << 32) && "GLU src0 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "GLU dst offset exceeds 4 GB");

    uint32_t src0_off = (uint32_t)dml_tensor_offset(src0);
    uint32_t src1_off = src1 ? (uint32_t)dml_tensor_offset(src1) : 0;
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);
    uint32_t has_src1 = src1 ? 1 : 0;

    if (src1) {
        GGML_ASSERT(dml_tensor_offset(src1) < (1ull << 32) && "GLU src1 offset exceeds 4 GB");
    }

    uint32_t total = nc * nrows;
    uint32_t constants[] = { nc, nrows, (uint32_t)swapped, has_src1, ne00, src0_off, src1_off, dst_off };

    if (src1) {
        ID3D12Resource * uavs[] = { dml_get_tensor_resource(src0),
                                     dml_get_tensor_resource(src1),
                                     dml_get_tensor_resource(node) };
        return dml_dispatch_compute(dev, dev.pso_swiglu.Get(), constants, 8, uavs, 3,
                                     (total + 255) / 256);
    } else {
        ID3D12Resource * uavs[] = { dml_get_tensor_resource(src0),
                                     dml_get_tensor_resource(src0), // dummy for u1
                                     dml_get_tensor_resource(node) };
        return dml_dispatch_compute(dev, dev.pso_swiglu.Get(), constants, 8, uavs, 3,
                                     (total + 255) / 256);
    }
}

// GET_ROWS: gather rows from embedding table using integer indices
static bool dml_cs_get_rows(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    if (!dev.init_compute_shaders()) return false;

    auto * src0 = node->src[0]; // embedding table (F32 or F16 on device)
    auto * src1 = node->src[1]; // indices (I32)

    uint32_t ne00 = (uint32_t)src0->ne[0];
    uint32_t ne01 = (uint32_t)node->ne[1]; // rows to gather
    uint32_t ne02 = (uint32_t)node->ne[2]; // batch
    uint32_t n_total = ne00 * ne01 * ne02;

    GGML_ASSERT(dml_tensor_offset(src0) < (1ull << 32) && "GET_ROWS src0 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(src1) < (1ull << 32) && "GET_ROWS src1 offset exceeds 4 GB");
    GGML_ASSERT(dml_tensor_offset(node) < (1ull << 32) && "GET_ROWS dst offset exceeds 4 GB");

    uint32_t src0_off = (uint32_t)dml_tensor_offset(src0);
    uint32_t src1_off = (uint32_t)dml_tensor_offset(src1);
    uint32_t dst_off = (uint32_t)dml_tensor_offset(node);

    // On device, quantized tensors are stored as F16. Check the device type.
    // src0->type tells GGML type, but on device it's been expanded to F16 or F32.
    // Use nb[1] from the device perspective — for F16 it's ne00*2, for F32 it's ne00*4.
    uint32_t nb01 = (uint32_t)src0->nb[1];
    uint32_t src0_is_f16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) ? 1 : 0;

    uint32_t constants[] = { ne00, ne01, ne02, src0_off, src1_off, dst_off, nb01, n_total, src0_is_f16 };
    ID3D12Resource * uavs[] = { dml_get_tensor_resource(src0),
                                 dml_get_tensor_resource(src1),
                                 dml_get_tensor_resource(node) };
    return dml_dispatch_compute(dev, dev.pso_get_rows.Get(), constants, 9, uavs, 3,
                                 (n_total + 255) / 256);
}

static bool dml_op_mul_mat(ggml_dml_device_info & dev, struct ggml_tensor * node) {
    // GGML MUL_MAT(a, b) = a^T * b
    // a (src[0]) shape: [K, M, batch2, batch3]  (weights)
    // b (src[1]) shape: [K, N, batch2, batch3]  (input)
    // output shape:     [M, N, batch2, batch3]

    if (!dev.init_compute_shaders()) return false;

    const struct ggml_tensor * a = node->src[0];
    const struct ggml_tensor * b = node->src[1];

    const int64_t K = a->ne[0];
    const int64_t M = a->ne[1];
    const int64_t N = b->ne[1];
    const int64_t batch2 = node->ne[2];
    const int64_t batch3 = node->ne[3];

    auto a_info = dml_make_tensor_info(a);
    auto b_info = dml_make_tensor_info(b);
    auto dst_info = dml_make_tensor_info(node);

    DML_TENSOR_DATA_TYPE a_dt = a_info.data_type;

    // Step 5A: Assert offsets fit in 32-bit HLSL addressing
    GGML_ASSERT(a_info.buffer_offset < (1ull << 32) && "MUL_MAT src0 offset exceeds 4 GB");
    GGML_ASSERT(b_info.buffer_offset < (1ull << 32) && "MUL_MAT src1 offset exceeds 4 GB");
    GGML_ASSERT(dst_info.buffer_offset < (1ull << 32) && "MUL_MAT dst offset exceeds 4 GB");

    uint32_t batch_count = (uint32_t)(batch2 * batch3);
    uint32_t a_batch_count = (uint32_t)(a->ne[2] * a->ne[3]);
    uint32_t constants[] = {
        (uint32_t)M, (uint32_t)N, (uint32_t)K, batch_count, a_batch_count,
        (uint32_t)a_info.buffer_offset,
        (uint32_t)b_info.buffer_offset,
        (uint32_t)dst_info.buffer_offset,
    };

    ID3D12Resource * uavs[] = {
        a_info.resource,
        b_info.resource,
        dst_info.resource,
    };

    // Dispatch: one thread per output element, 256 threads per group
    uint32_t total_elements = (uint32_t)(M * N) * batch_count;
    uint32_t groups_x = (total_elements + 255) / 256;

    // Select shader based on A tensor type
    ID3D12PipelineState * pso = nullptr;
    if (a_dt == DML_TENSOR_DATA_TYPE_FLOAT16) {
        pso = dev.pso_mul_mat_f16_f32.Get();
    } else {
        pso = dev.pso_mul_mat_f32_f32.Get();
    }

    return dml_dispatch_compute(dev, pso, constants, 8, uavs, 3, groups_x);
}

// ---------------------------------------------------------------------------
// Graph compute — dispatches all nodes in the compute graph
// ---------------------------------------------------------------------------

static enum ggml_status ggml_backend_dml_graph_compute(ggml_backend_t backend,
                                                        struct ggml_cgraph * cgraph,
                                                        int batch_size) {
    auto * bctx = (ggml_dml_backend_context *)backend->context;
    auto & dev = dml_ctx().devices[bctx->device_index];

    // Check device health before computing
    HRESULT dev_hr = dev.d3d_device->GetDeviceRemovedReason();
    if (dev_hr != S_OK) {
        GGML_LOG_ERROR("graph_compute: device already removed at start: 0x%08x\n", (unsigned)dev_hr);
        return GGML_STATUS_FAILED;
    }

    // Test DML device health with a trivial op (only in debug mode)
    if (!dev.has_compute_shaders) {
        // Minimal IDENTITY op test
        UINT test_sizes[4] = {1, 1, 1, 1};
        DML_BUFFER_TENSOR_DESC test_buf = {};
        test_buf.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        test_buf.DimensionCount = 4;
        test_buf.Sizes = test_sizes;
        test_buf.TotalTensorSizeInBytes = 4;
        DML_TENSOR_DESC test_desc = { DML_TENSOR_TYPE_BUFFER, &test_buf };
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC test_id = {};
        test_id.InputTensor = &test_desc;
        test_id.OutputTensor = &test_desc;
        DML_OPERATOR_DESC test_op = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &test_id };
        ComPtr<IDMLOperator> test_dml_op;
        HRESULT test_hr = dev.dml_device->CreateOperator(&test_op, IID_PPV_ARGS(&test_dml_op));
        if (FAILED(test_hr)) {
            GGML_LOG_ERROR("graph_compute: DML CreateOperator test failed: 0x%08x, "
                           "dml_device=%p, d3d_device=%p\n",
                           (unsigned)test_hr, dev.dml_device.Get(), dev.d3d_device.Get());
            // Check D3D12 device again
            dev_hr = dev.d3d_device->GetDeviceRemovedReason();
            GGML_LOG_ERROR("graph_compute: D3D12 device reason: 0x%08x\n", (unsigned)dev_hr);
            return GGML_STATUS_FAILED;
        }
        GGML_LOG_DEBUG("graph_compute: DML device health check passed\n");
    }

    // Issue a global UAV barrier and fence synchronization to ensure all prior
    // GPU work (CopyBufferRegion from set_tensor uploads) is fully committed and
    // visible to subsequent compute shader reads. This is critical on ARM64 GPUs
    // where the copy engine and compute engine may have separate cache hierarchies.
    {
        dev.reset_cmd_list();
        D3D12_RESOURCE_BARRIER uav_barrier = {};
        uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav_barrier.UAV.pResource = nullptr;
        dev.cmd_list->ResourceBarrier(1, &uav_barrier);
        dev.execute_and_wait();
    }

    dev_hr = dev.d3d_device->GetDeviceRemovedReason();
    if (dev_hr != S_OK) {
        GGML_LOG_ERROR("graph_compute: device removed AFTER UAV barrier: 0x%08x\n", (unsigned)dev_hr);
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // Skip nodes that don't require computation
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        bool ok = false;
        switch (node->op) {
            // Ops with both DML native and HLSL paths:
            // DML native path uses compiled DML operators (NPU-compatible, UINT64 offsets)
            // HLSL path uses custom compute shaders (GPU-only, 32-bit offsets)
            case GGML_OP_ADD:
                ok = dev.has_compute_shaders ? dml_cs_add(dev, node)
                    : dml_op_elementwise_binary(dev, node, DML_OPERATOR_ELEMENT_WISE_ADD);
                break;
            case GGML_OP_MUL:
                ok = dev.has_compute_shaders ? dml_cs_mul(dev, node)
                    : dml_op_elementwise_binary(dev, node, DML_OPERATOR_ELEMENT_WISE_MULTIPLY);
                break;
            case GGML_OP_SCALE:
                ok = dev.has_compute_shaders ? dml_cs_scale(dev, node) : dml_op_scale(dev, node);
                break;
            case GGML_OP_SOFT_MAX:
                ok = dev.has_compute_shaders ? dml_cs_soft_max(dev, node) : dml_op_soft_max(dev, node);
                break;

            // Ops currently HLSL-only (NPU devices fall back to CPU via supports_op):
            case GGML_OP_RMS_NORM: ok = dml_cs_rms_norm(dev, node); break;
            case GGML_OP_MUL_MAT:
                ok = dev.has_compute_shaders ? dml_op_mul_mat(dev, node) : dml_op_mul_mat_gemm(dev, node);
                break;
            case GGML_OP_SET_ROWS:  ok = dml_cs_set_rows(dev, node); break;
            case GGML_OP_DIAG_MASK_INF: ok = dml_cs_diag_mask_inf(dev, node); break;
            case GGML_OP_CONT:      ok = dml_cs_cont(dev, node);     break;
            case GGML_OP_ROPE:      ok = dml_cs_rope(dev, node);     break;
            case GGML_OP_GLU:       ok = dml_cs_swiglu(dev, node);   break;
            case GGML_OP_GET_ROWS:  ok = dml_cs_get_rows(dev, node); break;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SILU: ok = dml_cs_silu(dev, node); break;
                    default:
                        GGML_LOG_ERROR("DirectML: unsupported unary op %s\n",
                                        ggml_unary_op_name(ggml_get_unary_op(node)));
                        return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("DirectML: unsupported op %s\n", ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }

        if (!ok) {
            GGML_LOG_ERROR("DirectML: op %s dispatch failed\n", ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }

        // In NPU mode, check DML device health after each op to catch silent failures
        if (!dev.has_compute_shaders) {
            HRESULT op_hr = dev.d3d_device->GetDeviceRemovedReason();
            if (op_hr != S_OK) {
                GGML_LOG_ERROR("DirectML: D3D12 device removed after op %s[%d]: 0x%08x\n",
                               ggml_op_name(node->op), i, (unsigned)op_hr);
                return GGML_STATUS_FAILED;
            }
            // Test DML device by creating a trivial operator
            DML_BUFFER_TENSOR_DESC chk_buf_desc = {};
            chk_buf_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
            UINT chk_sizes[] = {1,1,1,1};
            UINT chk_strides[] = {1,1,1,1};
            chk_buf_desc.Sizes = chk_sizes;
            chk_buf_desc.Strides = chk_strides;
            chk_buf_desc.DimensionCount = 4;
            chk_buf_desc.TotalTensorSizeInBytes = 4;
            DML_TENSOR_DESC chk_td = { DML_TENSOR_TYPE_BUFFER, &chk_buf_desc };
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC chk_id = {};
            chk_id.InputTensor = &chk_td;
            chk_id.OutputTensor = &chk_td;
            DML_OPERATOR_DESC chk_op = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &chk_id };
            ComPtr<IDMLOperator> chk_dml_op;
            HRESULT chk_hr = dev.dml_device->CreateOperator(&chk_op, IID_PPV_ARGS(&chk_dml_op));
            if (FAILED(chk_hr)) {
                GGML_LOG_ERROR("DirectML: DML device corrupted after op %s[%d]: 0x%08x\n",
                               ggml_op_name(node->op), i, (unsigned)chk_hr);
                return GGML_STATUS_FAILED;
            }
        }
    }

    // Final fence wait
    if (dev.fence->GetCompletedValue() < dev.fence_value) {
        dev.fence->SetEventOnCompletion(dev.fence_value, dev.fence_event);
        WaitForSingleObject(dev.fence_event, INFINITE);
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(batch_size);
}

static struct ggml_backend_i ggml_backend_dml_iface = {
    /* .get_name             = */ ggml_backend_dml_get_name,
    /* .free                 = */ ggml_backend_dml_free,
    /* .set_tensor_async     = */ nullptr,
    /* .get_tensor_async     = */ nullptr,
    /* .cpy_tensor_async     = */ nullptr,
    /* .synchronize          = */ ggml_backend_dml_synchronize,
    /* .graph_plan_create    = */ nullptr,
    /* .graph_plan_free      = */ nullptr,
    /* .graph_plan_update    = */ nullptr,
    /* .graph_plan_compute   = */ nullptr,
    /* .graph_compute        = */ ggml_backend_dml_graph_compute,
    /* .event_record         = */ nullptr,
    /* .event_wait           = */ nullptr,
    /* .graph_optimize        = */ nullptr,
    /* .graph_reserve        = */ nullptr,
    /* .buffer_size          = */ nullptr,
    /* .reset                = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend device
// ---------------------------------------------------------------------------

struct ggml_dml_device_context_wrapper {
    int device_index;
};

static const char * ggml_backend_dml_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    return dml_ctx().devices[ctx->device_index].name.c_str();
}

static const char * ggml_backend_dml_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    return dml_ctx().devices[ctx->device_index].description.c_str();
}

static void ggml_backend_dml_device_get_memory(ggml_backend_dev_t dev,
                                                size_t * free, size_t * total) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    auto & info = dml_ctx().devices[ctx->device_index];

    // Refresh memory from DXGI/PDH using LUID string
    size_t pdh_free = 0, pdh_total = 0;
    if (ggml_dxgi_pdh_get_device_memory(info.luid_str.c_str(),
            &pdh_free, &pdh_total, info.integrated) == 0 && pdh_total > 0) {
        *free = pdh_free;
        *total = pdh_total;
    } else {
        *free = info.free_memory;
        *total = info.total_memory;
    }
}

static enum ggml_backend_dev_type ggml_backend_dml_device_get_type(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    auto & info = dml_ctx().devices[ctx->device_index];
    if (info.is_npu) {
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }
    if (info.integrated) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_dml_device_get_props(ggml_backend_dev_t dev,
                                               struct ggml_backend_dev_props * props) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    auto & info = dml_ctx().devices[ctx->device_index];

    props->name = info.name.c_str();
    props->description = info.description.c_str();
    props->type = ggml_backend_dml_device_get_type(dev);

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dml_device_get_memory(dev, &free_mem, &total_mem);
    props->memory_free = free_mem;
    props->memory_total = total_mem;

    props->id = nullptr;
    props->device_id = info.pci_id.empty() ? nullptr : info.pci_id.c_str();
    props->caps = {
        /* .async              = */ false,
        /* .host_buffer        = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events             = */ false,
    };
    props->driver_major = 0;
    props->driver_minor = 0;
    props->compute_major = 0;
    props->compute_minor = 0;
    props->integrated = info.integrated ? 1 : 0;
    props->library = GGML_DML_NAME;
}

static ggml_backend_t ggml_backend_dml_device_init_backend(ggml_backend_dev_t dev,
                                                            const char * params) {
    auto * dev_ctx = (ggml_dml_device_context_wrapper *)dev->context;
    auto * bctx = new ggml_dml_backend_context{dev_ctx->device_index};

    static ggml_guid s_guid = {0xd1, 0x4e, 0xc7, 0x11, 0x3a, 0x2b, 0x4c, 0x5d,
                               0x9e, 0x6f, 0x7a, 0x8b, 0x9c, 0xad, 0xbe, 0xcf};

    auto * backend = new ggml_backend();
    backend->guid    = &s_guid;
    backend->iface   = ggml_backend_dml_iface;
    backend->device  = dev;
    backend->context = bctx;
    return backend;
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_dml_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    int idx = ctx->device_index;
    if (ggml_dml_buft_instances[idx].iface.get_name == nullptr) {
        ggml_dml_buft_contexts[idx].device_index = idx;
        ggml_dml_buft_instances[idx].iface   = ggml_backend_dml_buft_iface;
        ggml_dml_buft_instances[idx].device  = dev;
        ggml_dml_buft_instances[idx].context = &ggml_dml_buft_contexts[idx];
        ggml_dml_buft_instances[idx].no_alloc = false;
    }
    return &ggml_dml_buft_instances[idx];
}

static bool ggml_backend_dml_device_supports_op(ggml_backend_dev_t dev,
                                                 const struct ggml_tensor * op) {
    auto * ctx = (ggml_dml_device_context_wrapper *)dev->context;
    auto & info = dml_ctx().devices[ctx->device_index];
    bool has_cs = info.has_compute_shaders;

    switch (op->op) {
        // Metadata ops — always supported (zero-compute)
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        // Element-wise ops — have both DML native and HLSL paths
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32);

        case GGML_OP_SOFT_MAX:
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            if (has_cs) return true; // HLSL path handles scale+mask
            // DML native softmax: only support when no mask and scale == 1.0
            if (op->src[1]) return false;  // mask present → CPU fallback
            {
                float scale;
                memcpy(&scale, op->op_params, sizeof(float));
                if (scale != 1.0f) return false;  // non-unit scale → CPU fallback
            }
            return true;

        // Matrix multiplication — HLSL (GPU) or DML GEMM (NPU)
        case GGML_OP_MUL_MAT: {
            if (!ggml_is_contiguous(op->src[0]) ||
                !ggml_is_contiguous(op->src[1]) ||
                !ggml_is_contiguous(op)) return false;
            if (op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_F16)
                return false;
            if (has_cs) {
                return op->src[0]->view_src == NULL &&
                       op->src[1]->view_src == NULL &&
                       op->view_src == NULL &&
                       dml_tensor_offset(op->src[0]) < (1ull << 32) &&
                       dml_tensor_offset(op->src[1]) < (1ull << 32) &&
                       dml_tensor_offset(op)         < (1ull << 32);
            }
            return true; // DML GEMM: UINT64 offsets, no 4GB limit
        }

        // --- HLSL-only ops: require compute shader support (GPU) ---
        // NPU devices fall back to CPU for these ops.

        case GGML_OP_RMS_NORM:
            return has_cs && (op->src[0]->type == GGML_TYPE_F32);

        case GGML_OP_DIAG_MASK_INF:
            return has_cs && (op->src[0]->type == GGML_TYPE_F32);

        case GGML_OP_UNARY:
            if (!has_cs) return false;
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU: return (op->src[0]->type == GGML_TYPE_F32);
                default: return false;
            }

        case GGML_OP_SET_ROWS:
            // SET_ROWS uses CPU-side readback+scatter+upload (no HLSL needed).
            // Required on all devices for KV cache writes.
            return (op->src[0]->type == GGML_TYPE_F32) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);

        case GGML_OP_CONT:
            return false; // Disabled: graph partition issue

        case GGML_OP_ROPE: {
            if (!has_cs) return false;
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            int32_t mode;
            memcpy(&mode, (int32_t *)op->op_params + 2, sizeof(int32_t));
            if (mode != 0 && mode != 2) return false;
            if (op->src[2] != NULL) return false;
            return true;
        }

        case GGML_OP_GLU: {
            if (!has_cs) return false;
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            int32_t glu_op;
            memcpy(&glu_op, (int32_t *)op->op_params + 0, sizeof(int32_t));
            return (glu_op == GGML_GLU_OP_SWIGLU);
        }

        case GGML_OP_GET_ROWS:
            return has_cs &&
                   (op->type == GGML_TYPE_F32) &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    ggml_is_quantized(op->src[0]->type)) &&
                   (op->src[1]->type == GGML_TYPE_I32);

        default:
            return false;
    }
}

static bool ggml_backend_dml_device_supports_buft(ggml_backend_dev_t dev,
                                                    ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_dml_buft_get_name;
    GGML_UNUSED(dev);
}

static bool ggml_backend_dml_device_offload_op(ggml_backend_dev_t dev,
                                                const struct ggml_tensor * op) {
    // Offload heavy compute ops even if weights are in a different buffer type
    const int min_batch_size = 32;
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1] >= min_batch_size;
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static struct ggml_backend_device_i ggml_backend_dml_device_iface = {
    /* .get_name             = */ ggml_backend_dml_device_get_name,
    /* .get_description      = */ ggml_backend_dml_device_get_description,
    /* .get_memory           = */ ggml_backend_dml_device_get_memory,
    /* .get_type             = */ ggml_backend_dml_device_get_type,
    /* .get_props            = */ ggml_backend_dml_device_get_props,
    /* .init_backend         = */ ggml_backend_dml_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_dml_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_dml_device_supports_op,
    /* .supports_buft        = */ ggml_backend_dml_device_supports_buft,
    /* .offload_op           = */ ggml_backend_dml_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
    /* .reset                = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend registration
// ---------------------------------------------------------------------------

static ggml_backend_device ggml_dml_device_instances[GGML_DML_MAX_DEVICES] = {};
static ggml_dml_device_context_wrapper ggml_dml_device_wrappers[GGML_DML_MAX_DEVICES] = {};

static const char * ggml_backend_dml_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_DML_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_dml_reg_get_device_count(ggml_backend_reg_t reg) {
    return dml_ctx().devices.size();
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_dml_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < dml_ctx().devices.size());
    if (ggml_dml_device_instances[index].iface.get_name == nullptr) {
        ggml_dml_device_wrappers[index].device_index = (int)index;
        ggml_dml_device_instances[index].iface   = ggml_backend_dml_device_iface;
        ggml_dml_device_instances[index].reg     = reg;
        ggml_dml_device_instances[index].context = &ggml_dml_device_wrappers[index];
    }
    return &ggml_dml_device_instances[index];
}

static struct ggml_backend_reg_i ggml_backend_dml_reg_iface = {
    /* .get_name         = */ ggml_backend_dml_reg_get_name,
    /* .get_device_count = */ ggml_backend_dml_reg_get_device_count,
    /* .get_device       = */ ggml_backend_dml_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg ggml_dml_reg_instance = {};

ggml_backend_reg_t ggml_backend_dml_reg(void) {
    if (ggml_dml_reg_instance.iface.get_name == nullptr) {
        ggml_dml_reg_instance.api_version = GGML_BACKEND_API_VERSION;
        ggml_dml_reg_instance.iface       = ggml_backend_dml_reg_iface;
        ggml_dml_reg_instance.context     = nullptr;
    }
    return &ggml_dml_reg_instance;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ggml_backend_t ggml_backend_dml_init(size_t dev_num) {
    if (dev_num >= dml_ctx().devices.size()) return nullptr;
    ggml_backend_dev_t dev = ggml_backend_dml_reg_get_device(ggml_backend_dml_reg(), dev_num);
    return ggml_backend_dml_device_init_backend(dev, nullptr);
}

bool ggml_backend_is_dml(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_dml_get_name;
}

int ggml_backend_dml_get_device_count(void) {
    return (int)dml_ctx().devices.size();
}

void ggml_backend_dml_get_device_description(int device, char * description, size_t description_size) {
    if (device < 0 || device >= (int)dml_ctx().devices.size()) return;
    strncpy(description, dml_ctx().devices[device].description.c_str(), description_size - 1);
    description[description_size - 1] = '\0';
}

void ggml_backend_dml_get_device_memory(int device, size_t * free, size_t * total) {
    if (device < 0 || device >= (int)dml_ctx().devices.size()) {
        *free = 0;
        *total = 0;
        return;
    }
    *free = dml_ctx().devices[device].free_memory;
    *total = dml_ctx().devices[device].total_memory;
}

ggml_backend_buffer_type_t ggml_backend_dml_buffer_type(size_t dev_num) {
    ggml_backend_dev_t dev = ggml_backend_dml_reg_get_device(ggml_backend_dml_reg(), dev_num);
    return ggml_backend_dml_device_get_buffer_type(dev);
}

// Dynamic loading support
GGML_BACKEND_DL_IMPL(ggml_backend_dml_reg)

#else // !_WIN32

// Stub implementation for non-Windows platforms

ggml_backend_reg_t ggml_backend_dml_reg(void) {
    return nullptr;
}

ggml_backend_t ggml_backend_dml_init(size_t dev_num) {
    GGML_UNUSED(dev_num);
    return nullptr;
}

bool ggml_backend_is_dml(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return false;
}

int ggml_backend_dml_get_device_count(void) {
    return 0;
}

void ggml_backend_dml_get_device_description(int device, char * description, size_t description_size) {
    GGML_UNUSED(device);
    GGML_UNUSED(description);
    GGML_UNUSED(description_size);
}

void ggml_backend_dml_get_device_memory(int device, size_t * free, size_t * total) {
    GGML_UNUSED(device);
    *free = 0;
    *total = 0;
}

ggml_backend_buffer_type_t ggml_backend_dml_buffer_type(size_t dev_num) {
    GGML_UNUSED(dev_num);
    return nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_dml_reg)

#endif // _WIN32
