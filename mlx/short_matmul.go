package mlx

import "math"

// ShortMatmul multiplies a few rows x [rows, K] by weight [N, K] transposed,
// reading the weight once. Each row accumulates in FP32 in the same order
// whatever the number of rows, so a row's result does not depend on the rows
// computed with it. The result has the promoted dtype of x and weight.
// It returns nil when the Metal kernel cannot run.
func ShortMatmul(x, weight *Array) *Array {
	if !MetalIsAvailable() || x == nil || weight == nil || x.NumDims() != 2 || weight.NumDims() != 2 {
		return nil
	}
	rows, width, columns := x.Dim(0), x.Dim(1), weight.Dim(0)
	if rows < 1 || rows > 8 || width%8 != 0 || weight.Dim(1) != width || columns < 1 ||
		uint64(weight.Size()) > math.MaxUint32 {
		return nil
	}
	floats := func(a *Array) bool { return a.DType() == DTypeFloat32 || a.DType() == DTypeBFloat16 }
	if !floats(x) || !floats(weight) {
		return nil
	}
	out := DTypeBFloat16
	if x.DType() == DTypeFloat32 || weight.DType() == DTypeFloat32 {
		out = DTypeFloat32
	}
	// Vocabulary-sized products stage the rows once per threadgroup for four
	// columns per SIMD group. Narrow ones spread over more threadgroups.
	staged, perSimdgroup, simdgroups := 0, 1, 8
	if columns >= 32768 {
		staged, perSimdgroup = 1, 4
	}
	if columns < 1024 {
		simdgroups = 2
	}
	span := simdgroups * perSimdgroup
	result, ok := shortMatmul.applyMetal(gpuLaunch{
		inputs:      []*Array{x, weight},
		dtypes:      []gpuDTypeArg{{"OutT", out}},
		ints:        []gpuIntArg{{"Rows", rows}, {"Width", width}, {"Columns", perSimdgroup}, {"Staged", staged}},
		outputs:     []gpuOutputSpec{{"short_matmul", []int32{int32(rows), int32(columns)}, out}},
		grid:        [3]int{(columns + span - 1) / span * 32 * simdgroups, 1, 1},
		threadGroup: [3]int{32 * simdgroups, 1, 1},
	})
	if !ok {
		return nil
	}
	return result[0]
}

var shortMatmul = &gpuKernel{
	name:    "short_matmul",
	inputs:  []string{"x", "weight"},
	outputs: []string{"y"},
	metal: gpuSource{header: `
inline void load8(const device bfloat16_t* p, thread float* out) {
    uint4 v = *((const device uint4*)p);
    out[0] = as_type<float>(v.x << 16); out[1] = as_type<float>(v.x & 0xffff0000u);
    out[2] = as_type<float>(v.y << 16); out[3] = as_type<float>(v.y & 0xffff0000u);
    out[4] = as_type<float>(v.z << 16); out[5] = as_type<float>(v.z & 0xffff0000u);
    out[6] = as_type<float>(v.w << 16); out[7] = as_type<float>(v.w & 0xffff0000u);
}
inline void load8(const device float* p, thread float* out) {
    float4 a = *((const device float4*)p), b = *((const device float4*)(p + 4));
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}
`, source: `
// Each lane reduces 8 consecutive values of every 256 in the same order
// whether or not the rows are staged, for any row count.
const uint columns = weight_shape[0];
const uint lane = thread_index_in_simdgroup, threads = threads_per_threadgroup.x;
const uint first = (threadgroup_position_in_grid.x * (threads / 32) + simdgroup_index_in_threadgroup) * Columns;
threadgroup float staged[Staged ? Rows * 256 : 1];
float acc[Columns][Rows] = {};
for (uint k0 = 0; k0 < Width; k0 += 256) {
    if (Staged) {
        for (uint i = thread_index_in_threadgroup; i < Rows * 256; i += threads) {
            uint k = k0 + i % 256;
            staged[i] = k < Width ? float(x[(i / 256) * Width + k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint k = k0 + lane * 8;
    for (uint c = 0; c < Columns && k < Width; ++c) {
        if (first + c >= columns) break;
        float wv[8];
        load8(weight + size_t(first + c) * Width + k, wv);
        for (uint m = 0; m < Rows; ++m) {
            float xv[8];
            if (Staged) {
                for (uint j = 0; j < 8; ++j) xv[j] = staged[m * 256 + lane * 8 + j];
            } else {
                load8(x + m * Width + k, xv);
            }
            for (uint j = 0; j < 8; ++j) acc[c][m] = fma(xv[j], wv[j], acc[c][m]);
        }
    }
    if (Staged) threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint c = 0; c < Columns; ++c) {
    for (uint m = 0; m < Rows; ++m) {
        float sum = simd_sum(acc[c][m]);
        if (lane == 0 && first + c < columns) y[m * columns + first + c] = OutT(sum);
    }
}
`},
}
