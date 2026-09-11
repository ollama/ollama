package mlx

import "fmt"

// B and T arrive as runtime scalars rather than template arguments so
// windows of any length share one compiled pipeline; only the channel
// geometry specializes the kernel.
const depthwiseConvSiLUMetalSource = `
auto elem = thread_position_in_grid.x;
int B = dims[0];
int T = dims[1];
uint total = uint(B) * uint(T) * uint(C);
if (elem >= total) {
  return;
}

int c = int(elem % uint(C));
int t = int(elem / uint(C)) % T;
int b = int(elem) / (C * T);
auto in_base = (b * (T + K - 1) + t) * C + c;

float acc = 0.0f;
for (int i = 0; i < K; ++i) {
  acc += static_cast<float>(x[in_base + i * C]) * static_cast<float>(w[c * K + i]);
}

// The graph path is Add(conv(x, w), bias), so it rounds twice. Folding the
// bias into the float accumulator would round once and drift by an ULP.
InT conv_out = static_cast<InT>(acc);
conv_out = static_cast<InT>(static_cast<float>(conv_out) + DEPTHWISE_CONV_BIAS(c));
InT sigmoid = stable_sigmoid(conv_out);
out[elem] = static_cast<InT>(conv_out * sigmoid);
`

const depthwiseConvSiLUMetalHeader = `
template <typename T>
T stable_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
`

var (
	depthwiseConvSiLU = &gpuKernel{
		name:    "depthwise_conv_silu",
		inputs:  []string{"x", "w", "dims"},
		outputs: []string{"out"},
		metal: gpuSource{
			source: depthwiseConvSiLUMetalSource,
			header: depthwiseConvSiLUMetalHeader + "#define DEPTHWISE_CONV_BIAS(c) 0.0f\n",
		},
		fallback: func(launch gpuLaunch) []*Array {
			return []*Array{depthwiseConvSiLUGraph(launch.inputs[0], launch.inputs[1], nil)}
		},
	}
	depthwiseConvSiLUBias = &gpuKernel{
		name:    "depthwise_conv_silu_bias",
		inputs:  []string{"x", "w", "bias", "dims"},
		outputs: []string{"out"},
		metal: gpuSource{
			source: depthwiseConvSiLUMetalSource,
			header: depthwiseConvSiLUMetalHeader + "#define DEPTHWISE_CONV_BIAS(c) static_cast<float>(bias[c])\n",
		},
		fallback: func(launch gpuLaunch) []*Array {
			in := launch.inputs
			return []*Array{depthwiseConvSiLUGraph(in[0], in[1], in[2])}
		},
	}
)

func depthwiseConvSiLUGraph(x, w, bias *Array) *Array {
	Cdim, K := int32(w.Dim(0)), int32(w.Dim(1))
	return SiLU(Conv1d(x, Reshape(w, Cdim, K, 1), bias, 1, 0, 1, Cdim))
}

// DepthwiseConvSiLU computes SiLU of a valid depthwise conv: x
// [B, T+K-1, C] and w [C, K] give [B, T, C], each output reading the K
// trailing input rows starting at its own index. bias, when non-nil, is [C].
// Inputs that fit the fused kernel's contract run there; anything else runs
// the same computation as graph ops, bit for bit.
func DepthwiseConvSiLU(x, w, bias *Array, outLen int) *Array {
	if x == nil || w == nil || x.NumDims() != 3 || w.NumDims() != 2 {
		panic("mlx.DepthwiseConvSiLU: need x [B, T+K-1, C] and w [C, K]")
	}
	B, Cdim, K := x.Dim(0), x.Dim(2), w.Dim(1)
	if w.Dim(0) != Cdim || K <= 0 || x.Dim(1) != outLen+K-1 {
		panic(fmt.Sprintf("mlx.DepthwiseConvSiLU: shapes x %v, w %v do not fit outLen %d", x.Dims(), w.Dims(), outLen))
	}
	if bias != nil && (bias.NumDims() != 1 || bias.Dim(0) != Cdim) {
		panic(fmt.Sprintf("mlx.DepthwiseConvSiLU: bias %v does not match %d channels", bias.Dims(), Cdim))
	}
	if x.DType() != w.DType() || (x.DType() != DTypeBFloat16 && x.DType() != DTypeFloat32) ||
		(bias != nil && bias.DType() != x.DType()) {
		return depthwiseConvSiLUGraph(x, w, bias)
	}

	kernel := depthwiseConvSiLU
	inputs := []*Array{x, w}
	if bias != nil {
		kernel = depthwiseConvSiLUBias
		inputs = append(inputs, bias)
	}
	inputs = append(inputs, NewArrayInt32([]int32{int32(B), int32(outLen)}, []int32{2}))

	total := B * outLen * Cdim
	outs := kernel.run(gpuLaunch{
		dtypes: []gpuDTypeArg{{"InT", x.DType()}},
		ints:   []gpuIntArg{{"C", Cdim}, {"K", K}},
		outputs: []gpuOutputSpec{
			{"DEPTHWISE_CONV_SILU", []int32{int32(B), int32(outLen), int32(Cdim)}, x.DType()},
		},
		grid:        [3]int{(total + 255) / 256 * 256, 1, 1},
		threadGroup: [3]int{256, 1, 1},
		inputs:      inputs,
	})
	return outs[0]
}
