package mlx

import "math"

// mxfp8Metal reconstructs MXFP8 values: blocks of 32 share a power-of-two
// scale at least the block maximum over 448, and each value rounds to E4M3FN.
const mxfp8Metal = nvfp4RoundingMetal + `
inline float mxfp8_value(float x, float block_max) {
    float raw = block_max * (1.0f / 448.0f);
    uint bits = as_type<uint>(raw);
    uint exponent = min((bits >> 23) + ((bits & 0x7fffffu) != 0u ? 1u : 0u), 254u);
    if (!(raw > 0x1p-127f)) exponent = 0u;
    float scale = max(as_type<float>(exponent << 23), 0x1p-127f);
    float inverse = exponent == 0u ? 0x1p127f : 1.0f / scale;
    float scaled = clamp(x * inverse, -448.0f, 448.0f);
    float q = round_e4m3(abs(scaled));
    return (scaled < 0.0f ? -q : q) * scale;
}
`

// MXFP8Quantize rounds rows of x [..., width] to MXFP8 activations with 32
// values per block and returns their reconstruction in dtype. It returns nil
// when the Metal kernel cannot run.
func MXFP8Quantize(x *Array, dtype DType) *Array {
	if !MetalIsAvailable() || x == nil || x.NumDims() == 0 || x.Dim(-1)%32 != 0 || x.Size() == 0 ||
		uint64(x.Size()) > math.MaxInt32 || (dtype != DTypeFloat32 && dtype != DTypeBFloat16) {
		return nil
	}
	shape := make([]int32, x.NumDims())
	for i := range shape {
		shape[i] = int32(x.Dim(i))
	}
	out, ok := mxfp8Quantize.applyMetal(gpuLaunch{
		inputs:      []*Array{x},
		dtypes:      []gpuDTypeArg{{"OutT", dtype}},
		outputs:     []gpuOutputSpec{{"mxfp8_values", shape, dtype}},
		grid:        [3]int{x.Size(), 1, 1},
		threadGroup: [3]int{256, 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

var mxfp8Quantize = &gpuKernel{
	name:    "mxfp8_quantize",
	inputs:  []string{"x"},
	outputs: []string{"values"},
	metal: gpuSource{header: mxfp8Metal, source: `
#pragma clang fp contract(off) reassociate(off)
uint i = thread_position_in_grid.x;
float v = float(x[i]);
float block_max = simd_max(abs(v));
values[i] = OutT(mxfp8_value(v, block_max));
`},
}
