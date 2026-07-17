package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

// The mapped MoE expert projection fast paths are shared by multiple MoE model
// implementations. They are most useful for routed expert projections with
// many small per-expert matmuls, where MLX's generic gathered matmul path can
// leave avoidable routing and launch overhead on Metal.
// Keep the public model-facing API narrow: exported FastMoE helpers perform
// dtype dispatch and keep fallback decisions inside this package.
// TODO(cuda): add CUDA equivalents for these custom Metal kernels before
// enabling the same fast path on CUDA-backed MLX; non-Metal backends must keep
// using the generic fallback.

import (
	"sync"
	"unsafe"
)

var (
	moeExpertMapMetalKernelOnce sync.Once
	moeExpertMapMetalKernel     C.mlx_fast_metal_kernel
	moeExpertMapMetalDisabled   bool

	moeExpertBlockMapMetalKernelOnce sync.Once
	moeExpertBlockMapMetalKernel     C.mlx_fast_metal_kernel
	moeExpertBlockMapMetalDisabled   bool

	nvfp4MoEMappedMetalKernelOnce sync.Once
	nvfp4MoEMappedMetalKernel     C.mlx_fast_metal_kernel
	nvfp4MoEMappedMetalDisabled   bool

	nvfp4MoEBlockMappedMetalKernelOnce sync.Once
	nvfp4MoEBlockMappedMetalKernel     C.mlx_fast_metal_kernel
	nvfp4MoEBlockMappedMetalDisabled   bool

	mxfp8MoEMappedMetalKernelOnce sync.Once
	mxfp8MoEMappedMetalKernel     C.mlx_fast_metal_kernel
	mxfp8MoEMappedMetalDisabled   bool

	mxfp8MoEBlockMappedMetalKernelOnce sync.Once
	mxfp8MoEBlockMappedMetalKernel     C.mlx_fast_metal_kernel
	mxfp8MoEBlockMappedMetalDisabled   bool

	moeGatherMMBlockMappedMetalKernelOnce sync.Once
	moeGatherMMBlockMappedMetalKernel     C.mlx_fast_metal_kernel
	moeGatherMMBlockMappedMetalDisabled   bool
)

const (
	moeExpertBlockSize          = 32
	maxMoEExpertBlockMapExperts = 1024
)

// Mapped MoE kernels use Metal 4 tensor ops when available, with a
// simdgroup_matrix fallback for older OS releases. Keeping both paths in one
// source string preserves the same Go-side kernel API across supported macOS
// versions.
const moeMappedMetalKernelHeader = `
#include <metal_stdlib>
#include <metal_simdgroup_matrix>

#if __has_include(<metal_tensor>)
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define OLLAMA_MLX_HAS_METAL_TENSOR 1
#else
#define OLLAMA_MLX_HAS_METAL_TENSOR 0
#endif

using namespace metal;

#if OLLAMA_MLX_HAS_METAL_TENSOR
using namespace mpp::tensor_ops;
#endif

#define MOE_MAPPED_FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

inline half moe_mapped_nvfp4_value(uint nibble) {
  half v = as_type<half>(ushort((nibble & 7) << 9));
  v *= 16384.0h;
  return (nibble & 8) ? -v : v;
}

inline half moe_mapped_nvfp4_scale(uint8_t sb) {
  half scale = as_type<half>(ushort(uint16_t(sb & 127) << 7));
  scale *= 256.0h;
  return (sb & 128) ? -scale : scale;
}

inline half moe_mapped_mxfp8_value(uint8_t b) {
  return moe_mapped_nvfp4_scale(b);
}

inline float moe_mapped_mxfp8_scale(uint8_t sb) {
  uint32_t out = (sb == 0) ? 0x400000u : (uint32_t(sb) << 23);
  return as_type<float>(out);
}
`

const moeExpertMapMetalKernelSource = `
constexpr int Threads = 128;

auto expert = threadgroup_position_in_grid.x;
if (expert >= E) {
  return;
}
auto tid = thread_index_in_threadgroup;
constexpr int Total = T * TopK;

threadgroup int local_counts[Threads];

uint n = 0;
for (int flat = int(tid); flat < Total; flat += Threads) {
  if (uint(indices[flat]) == expert) {
    n++;
  }
}
local_counts[tid] = int(n);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint offset = 1; offset < Threads; offset <<= 1) {
  int add = 0;
  if (tid >= offset) {
    add = local_counts[tid - offset];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  local_counts[tid] += add;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

int out = local_counts[tid] - int(n);
if (tid == Threads - 1) {
  counts[expert] = local_counts[tid];
}

for (int flat = int(tid); flat < Total; flat += Threads) {
  if (uint(indices[flat]) == expert) {
    ids[size_t(expert) * T + out] = flat;
    out++;
  }
}
`

const moeExpertBlockMapMetalKernelSource = `
constexpr int Threads = 256;
constexpr int NR1 = 32;
constexpr int Total = T * TopK;

auto tid = thread_index_in_threadgroup;

threadgroup atomic_int local_counts[E];
threadgroup atomic_int write_counts[E];
threadgroup int block_bases[E];

for (int expert = int(tid); expert < E; expert += Threads) {
  atomic_store_explicit(&local_counts[expert], 0, memory_order_relaxed);
  atomic_store_explicit(&write_counts[expert], 0, memory_order_relaxed);
  block_bases[expert] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int flat = int(tid); flat < Total; flat += Threads) {
  int expert = int(indices[flat]);
  if (expert >= 0 && expert < E) {
    atomic_fetch_add_explicit(&local_counts[expert], 1, memory_order_relaxed);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0) {
  int next_block = 0;
  for (int expert = 0; expert < E; expert++) {
    int count = atomic_load_explicit(&local_counts[expert], memory_order_relaxed);
    atomic_store_explicit(counts + expert, count, memory_order_relaxed);
    block_bases[expert] = next_block;

    int blocks = (count + NR1 - 1) / NR1;
    for (int block = 0; block < blocks; block++) {
      int dst = next_block + block;
      if (dst < MaxBlocks) {
        atomic_store_explicit(block_experts + dst, expert, memory_order_relaxed);
        atomic_store_explicit(block_offsets + dst, block * NR1, memory_order_relaxed);
      }
    }
    next_block += blocks;
  }
  atomic_store_explicit(block_count, next_block, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int flat = int(tid); flat < Total; flat += Threads) {
  int expert = int(indices[flat]);
  if (expert >= 0 && expert < E) {
    int out = atomic_fetch_add_explicit(&write_counts[expert], 1, memory_order_relaxed);
    int dst = block_bases[expert] + out / NR1;
    if (dst < MaxBlocks) {
      atomic_store_explicit(block_ids + size_t(dst) * NR1 + size_t(out % NR1), flat, memory_order_relaxed);
    }
  }
}
`

const moeGatherMMBlockMappedMetalKernelSource = `
#if OLLAMA_MLX_HAS_METAL_TENSOR
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int block = int(threadgroup_position_in_grid.x);
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (block >= int(block_count[0]) || r0 >= N) {
  return;
}

const int expert = int(block_experts[block]);
const int r1 = int(block_offsets[block]);
if (expert >= E) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup bfloat sa[NR0 * NK];
threadgroup bfloat sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

auto tA = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = WeightTransposed == 0
    ? (size_t(expert) * K + size_t(k0)) * N + size_t(col)
    : (size_t(expert) * N + size_t(col)) * K + size_t(k0);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);
    const size_t w_offset = WeightTransposed == 0 ? size_t(i) * N : size_t(i);
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = bfloat(static_cast<float>(w[w_base + w_offset]));
  }

  const size_t sb_base = NK * row1 + 8 * il1;
  if (row1 < nr1) {
    const int id = block_ids[size_t(block) * NR1 + size_t(row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      *(sb + sb_base + i) = bfloat(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      *(sb + sb_base + i) = bfloat(0.0f);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  const int id = block_ids[size_t(block) * NR1 + size_t(j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#else
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int block = int(threadgroup_position_in_grid.x);
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (block >= int(block_count[0]) || r0 >= N) {
  return;
}

const int expert = int(block_experts[block]);
const int r1 = int(block_offsets[block]);
if (expert >= E) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup float sx[NR1 * NK];
threadgroup float sw[NR0 * NK];
threadgroup float temp[NR1 * NR0];

for (int i = int(tiitg); i < NR1 * NR0; i += 128) {
  temp[i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = WeightTransposed == 0
    ? (size_t(expert) * K + size_t(k0)) * N + size_t(col)
    : (size_t(expert) * N + size_t(col)) * K + size_t(k0);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short local_k = short(16 * il0 + i);
    const size_t w_offset = WeightTransposed == 0 ? size_t(i) * N : size_t(i);
    sw[int(local_k) * NR0 + int(lr0)] = static_cast<float>(w[w_base + w_offset]);
  }

  const size_t sb_base = int(row1) * NK + int(iy);
  if (row1 < nr1) {
    const int id = block_ids[size_t(block) * NR1 + size_t(row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      sx[sb_base + i] = xv;
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      sx[sb_base + i] = 0.0f;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short tile = sgitg; tile < 32; tile += 4) {
    const short tile_r1 = short((tile / 8) * 8);
    const short tile_r0 = short((tile % 8) * 8);

    simdgroup_float8x8 mx;
    simdgroup_float8x8 mw;
    simdgroup_float8x8 acc;

    simdgroup_load(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
    MOE_MAPPED_FOR_UNROLL (short k = 0; k < NK; k += 8) {
      simdgroup_load(mx, sx + int(tile_r1) * NK + int(k), NK);
      simdgroup_load(mw, sw + int(k) * NR0 + int(tile_r0), NR0);
      simdgroup_multiply_accumulate(acc, mx, mw, acc);
    }
    simdgroup_store(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (short j = sgitg; j < nr1; j += 4) {
  const int id = block_ids[size_t(block) * NR1 + size_t(j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#endif
`

const nvfp4MoEMappedMetalKernelSource = `
#if OLLAMA_MLX_HAS_METAL_TENSOR
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int expert = int(threadgroup_position_in_grid.z);
const int r1 = int(threadgroup_position_in_grid.x) * NR1;
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (expert >= E || r0 >= N) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup bfloat sa[NR0 * NK];
threadgroup bfloat sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K / 2;
const size_t k_groups = K / 16;

auto tA = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0 / 2);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(k0 / 16);
  const half scale = moe_mapped_nvfp4_scale(scales[s_base]);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);

    const uint8_t packed = wb[w_base + size_t(i / 2)];
    const uint nibble = ((i & 1) == 0) ? uint(packed & 0x0f) : uint((packed >> 4) & 0x0f);
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = bfloat(scale * moe_mapped_nvfp4_value(nibble));
  }

  const size_t sb_base = NK * row1 + 8 * il1;
  if (row1 < nr1) {
    const int id = ids[size_t(expert) * T + size_t(r1 + row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      *(sb + sb_base + i) = bfloat(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      *(sb + sb_base + i) = bfloat(0.0f);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  const int id = ids[size_t(expert) * T + size_t(r1 + j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#else
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int expert = int(threadgroup_position_in_grid.z);
const int r1 = int(threadgroup_position_in_grid.x) * NR1;
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (expert >= E || r0 >= N) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup half sx[NR1 * NK];
threadgroup half sw[NR0 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K / 2;
const size_t k_groups = K / 16;

for (int i = int(tiitg); i < NR1 * NR0; i += 128) {
  temp[i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0 / 2);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(k0 / 16);
  const half scale = moe_mapped_nvfp4_scale(scales[s_base]);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short local_k = short(16 * il0 + i);
    const uint8_t packed = wb[w_base + size_t(i / 2)];
    const uint nibble = ((i & 1) == 0) ? uint(packed & 0x0f) : uint((packed >> 4) & 0x0f);
    sw[int(local_k) * NR0 + int(lr0)] = scale * moe_mapped_nvfp4_value(nibble);
  }

  const size_t sb_base = int(row1) * NK + int(iy);
  if (row1 < nr1) {
    const int id = ids[size_t(expert) * T + size_t(r1 + row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      sx[sb_base + i] = half(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      sx[sb_base + i] = half(0.0h);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short tile = sgitg; tile < 32; tile += 4) {
    const short tile_r1 = short((tile / 8) * 8);
    const short tile_r0 = short((tile % 8) * 8);

    simdgroup_half8x8 mx;
    simdgroup_half8x8 mw;
    simdgroup_float8x8 acc;

    simdgroup_load(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
    MOE_MAPPED_FOR_UNROLL (short k = 0; k < NK; k += 8) {
      simdgroup_load(mx, sx + int(tile_r1) * NK + int(k), NK);
      simdgroup_load(mw, sw + int(k) * NR0 + int(tile_r0), NR0);
      simdgroup_multiply_accumulate(acc, mx, mw, acc);
    }
    simdgroup_store(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (short j = sgitg; j < nr1; j += 4) {
  const int id = ids[size_t(expert) * T + size_t(r1 + j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#endif
`

const nvfp4MoEBlockMappedMetalKernelSource = `
#if OLLAMA_MLX_HAS_METAL_TENSOR
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int block = int(threadgroup_position_in_grid.x);
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (block >= int(block_count[0]) || r0 >= N) {
  return;
}

const int expert = int(block_experts[block]);
const int r1 = int(block_offsets[block]);
if (expert >= E) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup bfloat sa[NR0 * NK];
threadgroup bfloat sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K / 2;
const size_t k_groups = K / 16;

auto tA = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0 / 2);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(k0 / 16);
  const half scale = moe_mapped_nvfp4_scale(scales[s_base]);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);

    const uint8_t packed = wb[w_base + size_t(i / 2)];
    const uint nibble = ((i & 1) == 0) ? uint(packed & 0x0f) : uint((packed >> 4) & 0x0f);
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = bfloat(scale * moe_mapped_nvfp4_value(nibble));
  }

  const size_t sb_base = NK * row1 + 8 * il1;
  if (row1 < nr1) {
    const int id = block_ids[size_t(block) * NR1 + size_t(row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      *(sb + sb_base + i) = bfloat(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      *(sb + sb_base + i) = bfloat(0.0f);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  const int id = block_ids[size_t(block) * NR1 + size_t(j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#else
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int block = int(threadgroup_position_in_grid.x);
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (block >= int(block_count[0]) || r0 >= N) {
  return;
}

const int expert = int(block_experts[block]);
const int r1 = int(block_offsets[block]);
if (expert >= E) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup half sx[NR1 * NK];
threadgroup half sw[NR0 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K / 2;
const size_t k_groups = K / 16;

for (int i = int(tiitg); i < NR1 * NR0; i += 128) {
  temp[i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0 / 2);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(k0 / 16);
  const half scale = moe_mapped_nvfp4_scale(scales[s_base]);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short local_k = short(16 * il0 + i);
    const uint8_t packed = wb[w_base + size_t(i / 2)];
    const uint nibble = ((i & 1) == 0) ? uint(packed & 0x0f) : uint((packed >> 4) & 0x0f);
    sw[int(local_k) * NR0 + int(lr0)] = scale * moe_mapped_nvfp4_value(nibble);
  }

  const size_t sb_base = int(row1) * NK + int(iy);
  if (row1 < nr1) {
    const int id = block_ids[size_t(block) * NR1 + size_t(row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      sx[sb_base + i] = half(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      sx[sb_base + i] = half(0.0h);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short tile = sgitg; tile < 32; tile += 4) {
    const short tile_r1 = short((tile / 8) * 8);
    const short tile_r0 = short((tile % 8) * 8);

    simdgroup_half8x8 mx;
    simdgroup_half8x8 mw;
    simdgroup_float8x8 acc;

    simdgroup_load(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
    MOE_MAPPED_FOR_UNROLL (short k = 0; k < NK; k += 8) {
      simdgroup_load(mx, sx + int(tile_r1) * NK + int(k), NK);
      simdgroup_load(mw, sw + int(k) * NR0 + int(tile_r0), NR0);
      simdgroup_multiply_accumulate(acc, mx, mw, acc);
    }
    simdgroup_store(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (short j = sgitg; j < nr1; j += 4) {
  const int id = block_ids[size_t(block) * NR1 + size_t(j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#endif
`

const mxfp8MoEMappedMetalKernelSource = `
#if OLLAMA_MLX_HAS_METAL_TENSOR
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int expert = int(threadgroup_position_in_grid.z);
const int r1 = int(threadgroup_position_in_grid.x) * NR1;
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (expert >= E || r0 >= N) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup half sa[NR0 * NK];
threadgroup half sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K;
const size_t k_groups = K / 32;

auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(loop_k / 32);
  const float scale = moe_mapped_mxfp8_scale(scales[s_base]);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);

    const uint8_t packed = wb[w_base + size_t(i)];
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = half(scale * float(moe_mapped_mxfp8_value(packed)));
  }

  const size_t sb_base = NK * row1 + 8 * il1;
  if (row1 < nr1) {
    const int id = ids[size_t(expert) * T + size_t(r1 + row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      *(sb + sb_base + i) = half(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      *(sb + sb_base + i) = 0.0h;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  const int id = ids[size_t(expert) * T + size_t(r1 + j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#else
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int expert = int(threadgroup_position_in_grid.z);
const int r1 = int(threadgroup_position_in_grid.x) * NR1;
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (expert >= E || r0 >= N) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup half sx[NR1 * NK];
threadgroup half sw[NR0 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K;
const size_t k_groups = K / 32;

for (int i = int(tiitg); i < NR1 * NR0; i += 128) {
  temp[i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(loop_k / 32);
  const float scale = moe_mapped_mxfp8_scale(scales[s_base]);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short local_k = short(16 * il0 + i);
    const uint8_t packed = wb[w_base + size_t(i)];
    sw[int(local_k) * NR0 + int(lr0)] = half(scale * float(moe_mapped_mxfp8_value(packed)));
  }

  const size_t sb_base = int(row1) * NK + int(iy);
  if (row1 < nr1) {
    const int id = ids[size_t(expert) * T + size_t(r1 + row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      sx[sb_base + i] = half(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      sx[sb_base + i] = 0.0h;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short tile = sgitg; tile < 32; tile += 4) {
    const short tile_r1 = short((tile / 8) * 8);
    const short tile_r0 = short((tile % 8) * 8);

    simdgroup_half8x8 mx;
    simdgroup_half8x8 mw;
    simdgroup_float8x8 acc;

    simdgroup_load(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
    MOE_MAPPED_FOR_UNROLL (short k = 0; k < NK; k += 8) {
      simdgroup_load(mx, sx + int(tile_r1) * NK + int(k), NK);
      simdgroup_load(mw, sw + int(k) * NR0 + int(tile_r0), NR0);
      simdgroup_multiply_accumulate(acc, mx, mw, acc);
    }
    simdgroup_store(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (short j = sgitg; j < nr1; j += 4) {
  const int id = ids[size_t(expert) * T + size_t(r1 + j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#endif
`

const mxfp8MoEBlockMappedMetalKernelSource = `
#if OLLAMA_MLX_HAS_METAL_TENSOR
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int block = int(threadgroup_position_in_grid.x);
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (block >= int(block_count[0]) || r0 >= N) {
  return;
}

const int expert = int(block_experts[block]);
const int r1 = int(block_offsets[block]);
if (expert >= E) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup half sa[NR0 * NK];
threadgroup half sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K;
const size_t k_groups = K / 32;

auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(loop_k / 32);
  const float scale = moe_mapped_mxfp8_scale(scales[s_base]);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);

    const uint8_t packed = wb[w_base + size_t(i)];
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = half(scale * float(moe_mapped_mxfp8_value(packed)));
  }

  const size_t sb_base = NK * row1 + 8 * il1;
  if (row1 < nr1) {
    const int id = block_ids[size_t(block) * NR1 + size_t(row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      *(sb + sb_base + i) = half(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      *(sb + sb_base + i) = 0.0h;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  const int id = block_ids[size_t(block) * NR1 + size_t(j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#else
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int block = int(threadgroup_position_in_grid.x);
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (block >= int(block_count[0]) || r0 >= N) {
  return;
}

const int expert = int(block_experts[block]);
const int r1 = int(block_offsets[block]);
if (expert >= E) {
  return;
}

const int count = int(counts[expert]);
if (r1 >= count) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((count - r1 < NR1) ? (count - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short il0 = short(tiitg % NL0);
const short row1 = short(tiitg / NL1);
const short il1 = short(tiitg % NL1);
const short iy = short(8 * il1);

threadgroup half sx[NR1 * NK];
threadgroup half sw[NR0 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K;
const size_t k_groups = K / 32;

for (int i = int(tiitg); i < NR1 * NR0; i += 128) {
  temp[i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = (size_t(expert) * N + col) * k_bytes + size_t(k0);
  const size_t s_base = (size_t(expert) * N + col) * k_groups + size_t(loop_k / 32);
  const float scale = moe_mapped_mxfp8_scale(scales[s_base]);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short local_k = short(16 * il0 + i);
    const uint8_t packed = wb[w_base + size_t(i)];
    sw[int(local_k) * NR0 + int(lr0)] = half(scale * float(moe_mapped_mxfp8_value(packed)));
  }

  const size_t sb_base = int(row1) * NK + int(iy);
  if (row1 < nr1) {
    const int id = block_ids[size_t(block) * NR1 + size_t(row1)];
    const int token = id / TopK;
    const int slot = id - token * TopK;
    const int inputSlot = InputSlots == 1 ? 0 : slot;
    const size_t x_base = (size_t(token) * InputSlots + size_t(inputSlot)) * K + size_t(loop_k + iy);

    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      float xv = static_cast<float>(x[x_base + i]);
      if (ApplyReluSquared != 0) {
        xv = (xv > 0.0f) ? (xv * xv) : 0.0f;
      }
      sx[sb_base + i] = half(xv);
    }
  } else {
    MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
      sx[sb_base + i] = 0.0h;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short tile = sgitg; tile < 32; tile += 4) {
    const short tile_r1 = short((tile / 8) * 8);
    const short tile_r0 = short((tile % 8) * 8);

    simdgroup_half8x8 mx;
    simdgroup_half8x8 mw;
    simdgroup_float8x8 acc;

    simdgroup_load(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
    MOE_MAPPED_FOR_UNROLL (short k = 0; k < NK; k += 8) {
      simdgroup_load(mx, sx + int(tile_r1) * NK + int(k), NK);
      simdgroup_load(mw, sw + int(k) * NR0 + int(tile_r0), NR0);
      simdgroup_multiply_accumulate(acc, mx, mw, acc);
    }
    simdgroup_store(acc, temp + int(tile_r1) * NR0 + int(tile_r0), NR0);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (short j = sgitg; j < nr1; j += 4) {
  const int id = block_ids[size_t(block) * NR1 + size_t(j)];
  const int token = id / TopK;
  const int slot = id - token * TopK;
  device OutT* dst = y + (size_t(token) * TopK + size_t(slot)) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
#endif
`

func initMoEExpertMapMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"indices"})
	if !ok {
		moeExpertMapMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"counts", "ids"})
	if !ok {
		moeExpertMapMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("moe_expert_map")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeExpertMapMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	moeExpertMapMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func initMoEExpertBlockMapMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"indices"})
	if !ok {
		moeExpertBlockMapMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"block_count", "counts", "block_experts", "block_offsets", "block_ids"})
	if !ok {
		moeExpertBlockMapMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("moe_expert_block_map")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeExpertBlockMapMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	moeExpertBlockMapMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(true),
	)
}

func initMoEMappedMetalKernel(target *C.mlx_fast_metal_kernel, disabled *bool, name, source string) {
	inputs, freeInputs, ok := cStringVector([]string{"x", "w", "scales", "counts", "ids"})
	if !ok {
		*disabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		*disabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeMappedMetalKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	*target = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func initMoEBlockMappedMetalKernel(target *C.mlx_fast_metal_kernel, disabled *bool, name, source string) {
	inputs, freeInputs, ok := cStringVector([]string{"x", "w", "scales", "counts", "block_count", "block_experts", "block_offsets", "block_ids"})
	if !ok {
		*disabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		*disabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeMappedMetalKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	*target = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func initMoEGatherMMBlockMappedMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"x", "w", "counts", "block_count", "block_experts", "block_offsets", "block_ids"})
	if !ok {
		moeGatherMMBlockMappedMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		moeGatherMMBlockMappedMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("moe_gather_mm_block_mapped")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeGatherMMBlockMappedMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeMappedMetalKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	moeGatherMMBlockMappedMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func metalTensorOpsAvailable() bool {
	// Dense quantized linear kernels currently only have a Metal 4 tensor-op
	// implementation. The mapped MoE kernels carry their older-OS fallback in
	// the Metal source itself and do not use this guard.
	return macOSMajorVersion() >= 26
}

func initNVFP4MoEMappedMetalKernel() {
	initMoEMappedMetalKernel(&nvfp4MoEMappedMetalKernel, &nvfp4MoEMappedMetalDisabled, "moe_gather_qmm_mapped_fp4", nvfp4MoEMappedMetalKernelSource)
}

func initNVFP4MoEBlockMappedMetalKernel() {
	initMoEBlockMappedMetalKernel(&nvfp4MoEBlockMappedMetalKernel, &nvfp4MoEBlockMappedMetalDisabled, "moe_gather_qmm_block_mapped_fp4", nvfp4MoEBlockMappedMetalKernelSource)
}

func initMXFP8MoEMappedMetalKernel() {
	initMoEMappedMetalKernel(&mxfp8MoEMappedMetalKernel, &mxfp8MoEMappedMetalDisabled, "moe_gather_qmm_mapped_fp8", mxfp8MoEMappedMetalKernelSource)
}

func initMXFP8MoEBlockMappedMetalKernel() {
	initMoEBlockMappedMetalKernel(&mxfp8MoEBlockMappedMetalKernel, &mxfp8MoEBlockMappedMetalDisabled, "moe_gather_qmm_block_mapped_fp8", mxfp8MoEBlockMappedMetalKernelSource)
}

func addMoEExpertMapTemplateArgs(cfg C.mlx_fast_metal_kernel_config, tokens, topK, experts int) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "T", value: tokens},
		{name: "TopK", value: topK},
		{name: "E", value: experts},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			return false
		}
	}
	return true
}

func addMoEExpertBlockMapTemplateArgs(cfg C.mlx_fast_metal_kernel_config, tokens, topK, experts, maxBlocks int) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "T", value: tokens},
		{name: "TopK", value: topK},
		{name: "E", value: experts},
		{name: "MaxBlocks", value: maxBlocks},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			return false
		}
	}
	return true
}

func addMoEMappedTemplateArgs(cfg C.mlx_fast_metal_kernel_config, tokens, topK, inputSlots, experts, N, K int, outDType DType, applyReluSquared, weightTransposed bool) bool {
	reluSquaredValue := 0
	if applyReluSquared {
		reluSquaredValue = 1
	}
	weightTransposedValue := 0
	if weightTransposed {
		weightTransposedValue = 1
	}
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "T", value: tokens},
		{name: "TopK", value: topK},
		{name: "InputSlots", value: inputSlots},
		{name: "E", value: experts},
		{name: "N", value: N},
		{name: "K", value: K},
		{name: "ApplyReluSquared", value: reluSquaredValue},
		{name: "WeightTransposed", value: weightTransposedValue},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			return false
		}
	}

	cn := C.CString("OutT")
	rc := C.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, cn, C.mlx_dtype(outDType))
	C.free(unsafe.Pointer(cn))
	return rc == 0
}

func moeExpertMapValidate(indices *Array, experts int) (tokens, topK int, ok bool) {
	if indices == nil || experts <= 0 {
		return 0, 0, false
	}
	if indices.DType() != DTypeInt32 && indices.DType() != DTypeUint32 {
		return 0, 0, false
	}
	dims := indices.Dims()
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return 0, 0, false
	}
	return dims[0], dims[1], true
}

func maxMoEExpertBlocks(tokens, topK, experts int) int {
	total := tokens * topK
	nonEmpty := min(experts, total)
	return nonEmpty + (total-nonEmpty)/moeExpertBlockSize
}

// fastMoEExpertMap builds per-expert token maps from unsorted top-k expert
// indices. The ids output is shaped [experts, tokens] and stores
// token*topK+slot for the first counts[expert] entries.
func fastMoEExpertMap(indices *Array, experts int) (counts, ids *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, nil, false
	}
	if moeExpertMapMetalDisabled {
		return nil, nil, false
	}
	tokens, topK, ok := moeExpertMapValidate(indices, experts)
	if !ok {
		return nil, nil, false
	}

	moeExpertMapMetalKernelOnce.Do(initMoEExpertMapMetalKernel)
	if moeExpertMapMetalDisabled {
		return nil, nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEExpertMapTemplateArgs(cfg, tokens, topK, experts) {
		return nil, nil, false
	}

	countShape := []C.int{C.int(experts)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(countShape), C.size_t(len(countShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, false
	}
	idsShape := []C.int{C.int(experts), C.int(tokens)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(idsShape), C.size_t(len(idsShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(experts*128), 1, 1) != 0 {
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, nil, false
	}

	inputs := []C.mlx_array{indices.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeExpertMapMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 2 {
		return nil, nil, false
	}

	counts = New("MOE_EXPERT_COUNTS")
	C.mlx_vector_array_get(&counts.ctx, outVec, 0)
	ids = New("MOE_EXPERT_IDS")
	C.mlx_vector_array_get(&ids.ctx, outVec, 1)
	return counts, ids, true
}

// fastMoEExpertBlockMap builds per-expert counts plus a compact list of
// non-empty expert row blocks and their token ids for block-mapped MoE
// kernels to dispatch over.
func fastMoEExpertBlockMap(indices *Array, experts int) (counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, nil, nil, nil, nil, false
	}
	if moeExpertBlockMapMetalDisabled {
		return nil, nil, nil, nil, nil, false
	}
	tokens, topK, ok := moeExpertMapValidate(indices, experts)
	if !ok {
		return nil, nil, nil, nil, nil, false
	}
	// The block-map kernel allocates threadgroup arrays indexed by expert.
	// Larger expert sets fall back to the mapped expert path instead of
	// risking a Metal compile failure on devices with smaller threadgroup
	// memory limits.
	if tokens*topK > 32768 || experts > maxMoEExpertBlockMapExperts {
		return nil, nil, nil, nil, nil, false
	}
	maxBlocks := maxMoEExpertBlocks(tokens, topK, experts)

	moeExpertBlockMapMetalKernelOnce.Do(initMoEExpertBlockMapMetalKernel)
	if moeExpertBlockMapMetalDisabled {
		return nil, nil, nil, nil, nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEExpertBlockMapTemplateArgs(cfg, tokens, topK, experts, maxBlocks) {
		return nil, nil, nil, nil, nil, false
	}

	blockCountShape := []C.int{1}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(blockCountShape), C.size_t(len(blockCountShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	countShape := []C.int{C.int(experts)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(countShape), C.size_t(len(countShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	blockShape := []C.int{C.int(maxBlocks)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(blockShape), C.size_t(len(blockShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(blockShape), C.size_t(len(blockShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	blockIDsShape := []C.int{C.int(maxBlocks * moeExpertBlockSize)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(blockIDsShape), C.size_t(len(blockIDsShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_init_value(cfg, 0) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, 256, 1, 1) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1) != 0 {
		return nil, nil, nil, nil, nil, false
	}

	inputs := []C.mlx_array{indices.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeExpertBlockMapMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, nil, nil, nil, nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 5 {
		return nil, nil, nil, nil, nil, false
	}

	blockCount = New("MOE_EXPERT_BLOCK_COUNT")
	C.mlx_vector_array_get(&blockCount.ctx, outVec, 0)
	counts = New("MOE_EXPERT_BLOCK_COUNTS")
	C.mlx_vector_array_get(&counts.ctx, outVec, 1)
	blockExperts = New("MOE_EXPERT_BLOCK_EXPERTS")
	C.mlx_vector_array_get(&blockExperts.ctx, outVec, 2)
	blockOffsets = New("MOE_EXPERT_BLOCK_OFFSETS")
	C.mlx_vector_array_get(&blockOffsets.ctx, outVec, 3)
	blockIDs = New("MOE_EXPERT_BLOCK_IDS")
	C.mlx_vector_array_get(&blockIDs.ctx, outVec, 4)
	return counts, blockCount, blockExperts, blockOffsets, blockIDs, true
}

func nvfp4MoEMappedValidate(x, w, scales, counts, ids *Array, topK int) (tokens, inputSlots, experts, N, K int, ok bool) {
	tokens, inputSlots, experts, N, K, ok = nvfp4MoEValidateBase(x, w, scales, counts, topK)
	if !ok || ids == nil || ids.DType() != DTypeInt32 {
		return 0, 0, 0, 0, 0, false
	}
	id := ids.Dims()
	if len(id) != 2 || id[0] != experts || id[1] != tokens {
		return 0, 0, 0, 0, 0, false
	}
	return tokens, inputSlots, experts, N, K, true
}

func fpMoEValidateBase(x, w, scales, counts *Array, topK, weightDiv, scaleDiv int) (tokens, inputSlots, experts, N, K int, ok bool) {
	if x == nil || w == nil || scales == nil || counts == nil || topK <= 0 {
		return 0, 0, 0, 0, 0, false
	}
	xd := x.Dims()
	wd := w.Dims()
	sd := scales.Dims()
	cd := counts.Dims()
	if len(xd) != 3 || len(wd) != 3 || len(sd) != 3 || len(cd) != 1 {
		return 0, 0, 0, 0, 0, false
	}
	if x.DType() != DTypeFloat32 && x.DType() != DTypeFloat16 && x.DType() != DTypeBFloat16 {
		return 0, 0, 0, 0, 0, false
	}
	if w.DType() != DTypeUint32 || scales.DType() != DTypeUint8 || counts.DType() != DTypeInt32 {
		return 0, 0, 0, 0, 0, false
	}
	tokens, inputSlots, K = xd[0], xd[1], xd[2]
	experts, N = wd[0], wd[1]
	if tokens <= 0 || inputSlots <= 0 || topK <= 0 || N <= 0 || K <= 0 || K%32 != 0 {
		return 0, 0, 0, 0, 0, false
	}
	if inputSlots != 1 && inputSlots != topK {
		return 0, 0, 0, 0, 0, false
	}
	if wd[2] != K/weightDiv || sd[0] != experts || sd[1] != N || sd[2] != K/scaleDiv || cd[0] != experts {
		return 0, 0, 0, 0, 0, false
	}
	return tokens, inputSlots, experts, N, K, true
}

func nvfp4MoEValidateBase(x, w, scales, counts *Array, topK int) (tokens, inputSlots, experts, N, K int, ok bool) {
	return fpMoEValidateBase(x, w, scales, counts, topK, 8, 16)
}

func mxfp8MoEMappedValidate(x, w, scales, counts, ids *Array, topK int) (tokens, inputSlots, experts, N, K int, ok bool) {
	tokens, inputSlots, experts, N, K, ok = mxfp8MoEValidateBase(x, w, scales, counts, topK)
	if !ok || ids == nil || ids.DType() != DTypeInt32 {
		return 0, 0, 0, 0, 0, false
	}
	id := ids.Dims()
	if len(id) != 2 || id[0] != experts || id[1] != tokens {
		return 0, 0, 0, 0, 0, false
	}
	return tokens, inputSlots, experts, N, K, true
}

func mxfp8MoEValidateBase(x, w, scales, counts *Array, topK int) (tokens, inputSlots, experts, N, K int, ok bool) {
	return fpMoEValidateBase(x, w, scales, counts, topK, 4, 32)
}

func moeBlockMapValidate(blockCount, blockExperts, blockOffsets, blockIDs *Array) (maxBlocks int, ok bool) {
	if blockCount == nil || blockExperts == nil || blockOffsets == nil || blockIDs == nil {
		return 0, false
	}
	if blockCount.DType() != DTypeInt32 || blockExperts.DType() != DTypeInt32 || blockOffsets.DType() != DTypeInt32 || blockIDs.DType() != DTypeInt32 {
		return 0, false
	}
	cd := blockCount.Dims()
	ed := blockExperts.Dims()
	od := blockOffsets.Dims()
	bd := blockIDs.Dims()
	if len(cd) != 1 || cd[0] != 1 || len(ed) != 1 || len(od) != 1 || len(bd) != 1 ||
		ed[0] <= 0 || od[0] != ed[0] || bd[0] != ed[0]*moeExpertBlockSize {
		return 0, false
	}
	return ed[0], true
}

func nvfp4MoEBlockMappedValidate(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int) (tokens, inputSlots, experts, N, K, maxBlocks int, ok bool) {
	tokens, inputSlots, experts, N, K, ok = nvfp4MoEValidateBase(x, w, scales, counts, topK)
	if !ok {
		return 0, 0, 0, 0, 0, 0, false
	}
	maxBlocks, ok = moeBlockMapValidate(blockCount, blockExperts, blockOffsets, blockIDs)
	if !ok {
		return 0, 0, 0, 0, 0, 0, false
	}
	return tokens, inputSlots, experts, N, K, maxBlocks, true
}

func mxfp8MoEBlockMappedValidate(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int) (tokens, inputSlots, experts, N, K, maxBlocks int, ok bool) {
	tokens, inputSlots, experts, N, K, ok = mxfp8MoEValidateBase(x, w, scales, counts, topK)
	if !ok {
		return 0, 0, 0, 0, 0, 0, false
	}
	maxBlocks, ok = moeBlockMapValidate(blockCount, blockExperts, blockOffsets, blockIDs)
	if !ok {
		return 0, 0, 0, 0, 0, 0, false
	}
	return tokens, inputSlots, experts, N, K, maxBlocks, true
}

func moeGatherMMSupportedDType(dtype DType) bool {
	return dtype == DTypeBFloat16 && MetalIsAvailable()
}

func moeGatherMMBlockMappedValidate(x, w, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int) (tokens, inputSlots, experts, N, K, maxBlocks int, weightTransposed, ok bool) {
	if x == nil || w == nil || counts == nil || topK <= 0 {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	xd := x.Dims()
	wd := w.Dims()
	cd := counts.Dims()
	if len(xd) != 3 || len(wd) != 3 || len(cd) != 1 {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	if !moeGatherMMSupportedDType(x.DType()) || !moeGatherMMSupportedDType(w.DType()) || counts.DType() != DTypeInt32 {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	tokens, inputSlots, K = xd[0], xd[1], xd[2]
	experts = wd[0]
	switch {
	case wd[1] == K:
		N = wd[2]
	case wd[2] == K:
		N = wd[1]
		weightTransposed = true
	default:
		return 0, 0, 0, 0, 0, 0, false, false
	}
	if tokens <= 0 || inputSlots <= 0 || experts <= 0 || N <= 0 || K <= 0 || K%32 != 0 {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	if inputSlots != 1 && inputSlots != topK {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	if cd[0] != experts {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	maxBlocks, ok = moeBlockMapValidate(blockCount, blockExperts, blockOffsets, blockIDs)
	if !ok {
		return 0, 0, 0, 0, 0, 0, false, false
	}
	return tokens, inputSlots, experts, N, K, maxBlocks, weightTransposed, true
}

// SupportsMoEGatherQMMBlockMapped reports whether FastMoEGatherQMMBlockMapped
// can dispatch the quantized expert projection described by the quantization
// parameters.
func SupportsMoEGatherQMMBlockMapped(groupSize, bits int, mode string) bool {
	return (groupSize == 16 && bits == 4 && mode == "nvfp4") ||
		(groupSize == 32 && bits == 8 && mode == "mxfp8")
}

// MoEGatherQMMMap stores the routed-token map reused by multiple expert
// projections in one MoE layer.
type MoEGatherQMMMap struct {
	indices *Array
	experts int

	counts *Array
	ids    *Array

	blockCount   *Array
	blockExperts *Array
	blockOffsets *Array
	blockIDs     *Array
}

// MoEExpertMap is the routed-token map shared by mapped MoE kernels.
type MoEExpertMap = MoEGatherQMMMap

// NewMoEGatherQMMMap builds the routed-token map used by
// FastMoEGatherQMMBlockMapped. It prefers the compact block map and falls back
// to the older expert map when the shape cannot use the block-map kernel.
func NewMoEGatherQMMMap(indices *Array, experts int) (*MoEGatherQMMMap, bool) {
	counts, blockCount, blockExperts, blockOffsets, blockIDs, ok := fastMoEExpertBlockMap(indices, experts)
	if ok {
		return &MoEGatherQMMMap{
			indices:      indices,
			experts:      experts,
			counts:       counts,
			blockCount:   blockCount,
			blockExperts: blockExperts,
			blockOffsets: blockOffsets,
			blockIDs:     blockIDs,
		}, true
	}

	counts, ids, ok := fastMoEExpertMap(indices, experts)
	if !ok {
		return nil, false
	}
	return &MoEGatherQMMMap{
		indices: indices,
		experts: experts,
		counts:  counts,
		ids:     ids,
	}, true
}

// NewMoEExpertMap builds the routed-token map used by mapped MoE kernels.
func NewMoEExpertMap(indices *Array, experts int) (*MoEExpertMap, bool) {
	return NewMoEGatherQMMMap(indices, experts)
}

func (m *MoEGatherQMMMap) expertMap() (counts, ids *Array, ok bool) {
	if m == nil {
		return nil, nil, false
	}
	if m.ids != nil {
		return m.counts, m.ids, true
	}
	counts, ids, ok = fastMoEExpertMap(m.indices, m.experts)
	if !ok {
		return nil, nil, false
	}
	m.counts = counts
	m.ids = ids
	return counts, ids, true
}

func applyMoEGatherMappedKernel(kernel C.mlx_fast_metal_kernel, name string, outDType DType, inputs []C.mlx_array, tokens, topK, inputSlots, experts, N, K, gridX, gridY, gridZ int, applyReluSquared, weightTransposed bool) (y *Array, ok bool) {
	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEMappedTemplateArgs(cfg, tokens, topK, inputSlots, experts, N, K, outDType, applyReluSquared, weightTransposed) {
		return nil, false
	}

	yShape := []C.int{C.int(tokens), C.int(topK), C.int(N)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(yShape), C.size_t(len(yShape)), C.mlx_dtype(outDType)) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), C.int(gridY), C.int(gridZ)) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, false
	}

	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, kernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New(name)
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}

// FastMoEGatherQMMBlockMapped computes x @ W[expert].T in original token/slot
// order for supported quantized MoE expert weights. It uses the compact block
// map when available, then falls back to the non-block expert map if the current
// Metal toolchain cannot use the block-mapped kernel.
func FastMoEGatherQMMBlockMapped(x, w, scales *Array, expertMap *MoEGatherQMMMap, topK, groupSize, bits int, mode string) (y *Array, ok bool) {
	return fastMoEGatherQMMBlockMapped(x, w, scales, expertMap, topK, groupSize, bits, mode, false)
}

// FastMoEGatherQMMBlockMappedReLUSquared is like
// FastMoEGatherQMMBlockMapped, but applies relu(x)^2 inside the projection.
// This matches Nemotron-H's routed down-projection shape without exposing the
// dtype-specific mapped kernels.
func FastMoEGatherQMMBlockMappedReLUSquared(x, w, scales *Array, expertMap *MoEGatherQMMMap, topK, groupSize, bits int, mode string) (y *Array, ok bool) {
	return fastMoEGatherQMMBlockMapped(x, w, scales, expertMap, topK, groupSize, bits, mode, true)
}

// SupportsMoEGatherMMBlockMapped reports whether FastMoEGatherMMBlockMapped can
// dispatch dense expert weights with the given dtype. Metal 4 uses BF16 tensor
// ops; older Metal keeps BF16 precision by loading operands as float for
// simdgroup-matrix accumulation instead of routing through FP16 math.
func SupportsMoEGatherMMBlockMapped(dtype DType) bool {
	return moeGatherMMSupportedDType(dtype)
}

// FastMoEGatherMMBlockMapped computes x @ W[expert] in original token/slot
// order for dense MoE expert weights stored as [experts, in, out]. It only uses
// the compact block map; callers should fall back to GatherMM when it returns
// ok=false.
func FastMoEGatherMMBlockMapped(x, w *Array, expertMap *MoEExpertMap, topK int) (y *Array, ok bool) {
	return fastMoEGatherMMBlockMapped(x, w, expertMap, topK, false)
}

// FastMoEGatherMMBlockMappedReLUSquared is like FastMoEGatherMMBlockMapped, but
// applies relu(x)^2 inside the projection.
func FastMoEGatherMMBlockMappedReLUSquared(x, w *Array, expertMap *MoEExpertMap, topK int) (y *Array, ok bool) {
	return fastMoEGatherMMBlockMapped(x, w, expertMap, topK, true)
}

func fastMoEGatherMMBlockMapped(x, w *Array, expertMap *MoEExpertMap, topK int, applyReluSquared bool) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if x == nil || w == nil || expertMap == nil {
		return nil, false
	}
	if !SupportsMoEGatherMMBlockMapped(x.DType()) {
		return nil, false
	}
	if expertMap.blockCount == nil || expertMap.blockExperts == nil || expertMap.blockOffsets == nil || expertMap.blockIDs == nil {
		return nil, false
	}

	tokens, inputSlots, experts, N, K, maxBlocks, weightTransposed, ok := moeGatherMMBlockMappedValidate(x, w, expertMap.counts, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs, topK)
	if !ok {
		return nil, false
	}

	if moeGatherMMBlockMappedMetalDisabled {
		return nil, false
	}
	moeGatherMMBlockMappedMetalKernelOnce.Do(initMoEGatherMMBlockMappedMetalKernel)
	if moeGatherMMBlockMappedMetalDisabled {
		return nil, false
	}

	gridX := maxBlocks * 128
	gridY := (N + 63) / 64
	inputs := []C.mlx_array{x.ctx, w.ctx, expertMap.counts.ctx, expertMap.blockCount.ctx, expertMap.blockExperts.ctx, expertMap.blockOffsets.ctx, expertMap.blockIDs.ctx}
	out, ok := applyMoEGatherMappedKernel(moeGatherMMBlockMappedMetalKernel, "MOE_GATHER_MM_BLOCK_MAPPED", x.DType(), inputs, tokens, topK, inputSlots, experts, N, K, gridX, gridY, 1, applyReluSquared, weightTransposed)
	if !ok {
		return nil, false
	}
	return out, true
}

func fastMoEGatherQMMBlockMapped(x, w, scales *Array, expertMap *MoEGatherQMMMap, topK, groupSize, bits int, mode string, applyReluSquared bool) (y *Array, ok bool) {
	if !SupportsMoEGatherQMMBlockMapped(groupSize, bits, mode) {
		return nil, false
	}
	if w == nil || !w.Valid() || w.NumDims() != 3 || expertMap == nil {
		return nil, false
	}
	if expertMap.blockCount != nil && expertMap.blockExperts != nil && expertMap.blockOffsets != nil && expertMap.blockIDs != nil {
		switch mode {
		case "nvfp4":
			if out, ok := fastNVFP4MoEGatherQMMBlockMapped(x, w, scales, expertMap.counts, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs, topK, applyReluSquared); ok {
				return out, true
			}
		case "mxfp8":
			if out, ok := fastMXFP8MoEGatherQMMBlockMapped(x, w, scales, expertMap.counts, expertMap.blockCount, expertMap.blockExperts, expertMap.blockOffsets, expertMap.blockIDs, topK, applyReluSquared); ok {
				return out, true
			}
		}
	}

	counts, ids, ok := expertMap.expertMap()
	if !ok {
		return nil, false
	}
	switch mode {
	case "nvfp4":
		return fastNVFP4MoEGatherQMMMapped(x, w, scales, counts, ids, topK, applyReluSquared)
	case "mxfp8":
		return fastMXFP8MoEGatherQMMMapped(x, w, scales, counts, ids, topK, applyReluSquared)
	default:
		return nil, false
	}
}

// fastNVFP4MoEGatherQMMMapped computes x @ W[expert].T for ids produced by
// fastMoEExpertMap. It returns [tokens, topK, N] in original token/slot order.
func fastNVFP4MoEGatherQMMMapped(x, w, scales, counts, ids *Array, topK int, applyReluSquared bool) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	tokens, inputSlots, experts, N, K, ok := nvfp4MoEMappedValidate(x, w, scales, counts, ids, topK)
	if !ok {
		return nil, false
	}

	if nvfp4MoEMappedMetalDisabled {
		return nil, false
	}
	nvfp4MoEMappedMetalKernelOnce.Do(initNVFP4MoEMappedMetalKernel)
	if nvfp4MoEMappedMetalDisabled {
		return nil, false
	}

	gridX := ((tokens + 31) / 32) * 128
	gridY := (N + 63) / 64
	inputs := []C.mlx_array{x.ctx, w.ctx, scales.ctx, counts.ctx, ids.ctx}
	return applyMoEGatherMappedKernel(nvfp4MoEMappedMetalKernel, "MOE_GATHER_QMM_MAPPED", x.DType(), inputs, tokens, topK, inputSlots, experts, N, K, gridX, gridY, experts, applyReluSquared, false)
}

// fastNVFP4MoEGatherQMMBlockMapped is like fastNVFP4MoEGatherQMMMapped, but
// dispatches only the populated expert row blocks produced by
// fastMoEExpertBlockMap.
func fastNVFP4MoEGatherQMMBlockMapped(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int, applyReluSquared bool) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	tokens, inputSlots, experts, N, K, maxBlocks, ok := nvfp4MoEBlockMappedValidate(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs, topK)
	if !ok {
		return nil, false
	}

	if nvfp4MoEBlockMappedMetalDisabled {
		return nil, false
	}
	nvfp4MoEBlockMappedMetalKernelOnce.Do(initNVFP4MoEBlockMappedMetalKernel)
	if nvfp4MoEBlockMappedMetalDisabled {
		return nil, false
	}

	gridX := maxBlocks * 128
	gridY := (N + 63) / 64
	inputs := []C.mlx_array{x.ctx, w.ctx, scales.ctx, counts.ctx, blockCount.ctx, blockExperts.ctx, blockOffsets.ctx, blockIDs.ctx}
	return applyMoEGatherMappedKernel(nvfp4MoEBlockMappedMetalKernel, "MOE_GATHER_QMM_BLOCK_MAPPED", x.DType(), inputs, tokens, topK, inputSlots, experts, N, K, gridX, gridY, 1, applyReluSquared, false)
}

// fastMXFP8MoEGatherQMMMapped computes x @ W[expert].T for ids produced by
// fastMoEExpertMap. It returns [tokens, topK, N] in original token/slot order.
func fastMXFP8MoEGatherQMMMapped(x, w, scales, counts, ids *Array, topK int, applyReluSquared bool) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	tokens, inputSlots, experts, N, K, ok := mxfp8MoEMappedValidate(x, w, scales, counts, ids, topK)
	if !ok {
		return nil, false
	}

	if mxfp8MoEMappedMetalDisabled {
		return nil, false
	}
	mxfp8MoEMappedMetalKernelOnce.Do(initMXFP8MoEMappedMetalKernel)
	if mxfp8MoEMappedMetalDisabled {
		return nil, false
	}

	gridX := ((tokens + 31) / 32) * 128
	gridY := (N + 63) / 64
	inputs := []C.mlx_array{x.ctx, w.ctx, scales.ctx, counts.ctx, ids.ctx}
	return applyMoEGatherMappedKernel(mxfp8MoEMappedMetalKernel, "MOE_GATHER_QMM_MAPPED", x.DType(), inputs, tokens, topK, inputSlots, experts, N, K, gridX, gridY, experts, applyReluSquared, false)
}

// fastMXFP8MoEGatherQMMBlockMapped is like fastMXFP8MoEGatherQMMMapped, but
// dispatches only the populated expert row blocks produced by
// fastMoEExpertBlockMap.
func fastMXFP8MoEGatherQMMBlockMapped(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs *Array, topK int, applyReluSquared bool) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	tokens, inputSlots, experts, N, K, maxBlocks, ok := mxfp8MoEBlockMappedValidate(x, w, scales, counts, blockCount, blockExperts, blockOffsets, blockIDs, topK)
	if !ok {
		return nil, false
	}

	if mxfp8MoEBlockMappedMetalDisabled {
		return nil, false
	}
	mxfp8MoEBlockMappedMetalKernelOnce.Do(initMXFP8MoEBlockMappedMetalKernel)
	if mxfp8MoEBlockMappedMetalDisabled {
		return nil, false
	}

	gridX := maxBlocks * 128
	gridY := (N + 63) / 64
	inputs := []C.mlx_array{x.ctx, w.ctx, scales.ctx, counts.ctx, blockCount.ctx, blockExperts.ctx, blockOffsets.ctx, blockIDs.ctx}
	return applyMoEGatherMappedKernel(mxfp8MoEBlockMappedMetalKernel, "MOE_GATHER_QMM_BLOCK_MAPPED", x.DType(), inputs, tokens, topK, inputSlots, experts, N, K, gridX, gridY, 1, applyReluSquared, false)
}
