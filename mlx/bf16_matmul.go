package mlx

import "math"

// MatmulBF16Weight computes x [M,K] times weight [N,K] transposed. Short projections
// are bandwidth bound, so weights stay in BF16 storage while the result accumulates in FP32.
// It returns nil outside the qualified Metal path for up to two eight-row tiles.
func MatmulBF16Weight(x, weight *Array) *Array {
	if x == nil || weight == nil || x.NumDims() != 2 || weight.NumDims() != 2 ||
		(x.DType() != DTypeFloat32 && x.DType() != DTypeBFloat16) || weight.DType() != DTypeBFloat16 {
		return nil
	}
	rows, width, columns := x.Dim(0), x.Dim(1), weight.Dim(0)
	// At these dimensions native FP32 matmul may split K across workgroups,
	// changing the accumulation order of a compact single-pass projection.
	if width >= 128 && columns <= width {
		return nil
	}
	if rows < 1 || rows > 16 || width < 1 || width != weight.Dim(1) || columns < 1 || !SupportsBF16WeightMatmul() {
		return nil
	}
	if usesBF16TensorMatmul() {
		return matmulBF16WeightTensor(x, weight, rows, width, columns)
	}

	const tile = 8
	paddedRows := (rows + tile - 1) / tile * tile
	paddedWidth := (width + tile - 1) / tile * tile
	paddedColumns := (columns + tile - 1) / tile * tile
	if paddedColumns > math.MaxInt32/4 || paddedWidth > math.MaxInt32 {
		return nil
	}
	x = x.AsType(DTypeFloat32)
	if rows != paddedRows || width != paddedWidth {
		x = PadConstant(x, []int{0, 1}, []int{0, 0}, []int{paddedRows - rows, paddedWidth - width})
	}
	if weight.Size() < tile {
		// MLX passes tiny arguments in Metal's constant address space, which
		// SIMD matrix loads cannot use. Padding at most seven rows gives it
		// a device buffer without changing the logical matrix.
		weight = PadConstant(weight, []int{0}, []int{0}, []int{tile - columns})
	}
	result, ok := bf16WeightMatmul.applyMetal(gpuLaunch{
		inputs:  []*Array{x, weight},
		ints:    []gpuIntArg{{"Width", width}, {"Columns", columns}},
		outputs: []gpuOutputSpec{{"matmul", []int32{int32(paddedRows), int32(paddedColumns)}, DTypeFloat32}},
		grid:    [3]int{paddedColumns / tile * 32, paddedRows / tile, 1}, threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	if rows == paddedRows && columns == paddedColumns {
		return result[0]
	}
	return result[0].Slice(Slice(0, rows), Slice(0, columns))
}

func matmulBF16WeightTensor(x, weight *Array, rows, width, columns int) *Array {
	const tileRows, tileColumns, kAlignment = 16, 32, 16
	// Contracting projections can use a split-K reduction in MLX. The
	// vocabulary projection expands the hidden dimension and has one accumulator.
	if columns < width || columns > math.MaxInt32-tileColumns || width > math.MaxInt32-kAlignment {
		return nil
	}
	paddedColumns := (columns + tileColumns - 1) / tileColumns * tileColumns
	paddedWidth := (width + kAlignment - 1) / kAlignment * kAlignment
	if rows != tileRows || width != paddedWidth {
		x = PadConstant(x, []int{0, 1}, []int{0, 0}, []int{tileRows - rows, paddedWidth - width})
	}
	if columns != paddedColumns || width != paddedWidth {
		weight = PadConstant(weight, []int{0, 1}, []int{0, 0}, []int{paddedColumns - columns, paddedWidth - width})
	}
	result, ok := bf16TensorMatmul.applyMetal(gpuLaunch{
		inputs:  []*Array{x, weight},
		dtypes:  []gpuDTypeArg{{"Hidden", x.DType()}},
		ints:    []gpuIntArg{{"Width", paddedWidth}, {"Columns", paddedColumns}},
		outputs: []gpuOutputSpec{{"matmul", []int32{tileRows, int32(paddedColumns)}, DTypeFloat32}},
		grid:    [3]int{paddedColumns / tileColumns * 32, 1, 1}, threadGroup: [3]int{32, 1, 1},
	})
	if !ok {
		return nil
	}
	return result[0].Slice(Slice(0, rows), Slice(0, columns))
}

// On Metal 4, stock FP32 projections can use tensor operations with TF32 inputs.
// A mixed-type tile preserves their rounding while reading each BF16 weight once.
var bf16TensorMatmul = &gpuKernel{
	name:    "bf16_tensor_matmul",
	inputs:  []string{"hidden", "weight"},
	outputs: []string{"scores"},
	metal: gpuSource{
		header: `
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
`,
		source: `
constexpr int rows = 16, channels = 32;
const uint channel = threadgroup_position_in_grid.x * channels;
using dims = dextents<int, 2>;
using strides = array<int, 2>;
tensor<device Hidden, dims, tensor_inline> input(
    (device Hidden*)hidden, dims(Width, rows), strides{1, Width});
tensor<device bfloat, dims, tensor_inline> weights(
    (device bfloat*)weight + size_t(channel) * Width, dims(Width, channels), strides{1, Width});
tensor<device float, dims, tensor_inline> output(
    scores + channel, dims(channels, rows), strides{1, Columns});
constexpr mpp::tensor_ops::matmul2d_descriptor shape(rows, channels, int(dynamic_extent), false, true, true);
mpp::tensor_ops::matmul2d<shape, execution_simdgroup> product;
product.run(input, weights, output);
`,
	},
}

// Stock FP32 matmul reads four bytes per weight. Widening BF16 tiles in registers
// halves the weight traffic while preserving FP32 multiplication and accumulation.
var bf16WeightMatmul = &gpuKernel{
	name:    "bf16_weight_matmul",
	inputs:  []string{"x", "weight"},
	outputs: []string{"result"},
	metal: gpuSource{
		header: `
#include <metal_simdgroup_matrix>
using namespace metal;
`,
		source: `
constexpr uint xStride = (Width + 7u) / 8u * 8u;
constexpr uint outStride = (Columns + 7u) / 8u * 8u;
uint row = 8 * thread_position_in_grid.y;
uint col = 8 * (thread_position_in_grid.x / 32);
simdgroup_float8x8 sum = make_filled_simdgroup_matrix<float, 8, 8>(0);
threadgroup bfloat edge[4][64];
for (uint k = 0; k < Width; k += 8) {
    simdgroup_float8x8 lhs;
    simdgroup_load(lhs, x + size_t(row) * xStride + k, xStride);
    simdgroup_bfloat8x8 rhs;
    if (Width % 8 == 0 && (Columns % 8 == 0 || col + 8 <= Columns)) {
        simdgroup_load(rhs, weight + size_t(col) * Width + k, Width, ulong2(0), true);
    } else {
        // A partial or unaligned weight tile is zero-filled in its own SIMD group.
        threadgroup bfloat* tile = edge[simdgroup_index_in_threadgroup];
        for (uint i = thread_index_in_simdgroup; i < 64; i += 32) {
            uint n = col + i / 8, d = k + i % 8;
            tile[i] = n < Columns && d < Width ? weight[size_t(n) * Width + d] : bfloat(0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(rhs, tile, 8, ulong2(0), true);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_float8x8 expanded;
    expanded.thread_elements()[0] = float(rhs.thread_elements()[0]);
    expanded.thread_elements()[1] = float(rhs.thread_elements()[1]);
    simdgroup_multiply_accumulate(sum, lhs, expanded, sum);
}
simdgroup_store(sum, result + size_t(row) * outStride + col, outStride);
`,
	},
}
