package mlx

const (
	connectionMixProjected = iota
	connectionMixVariance
	connectionMixScales
	connectionMixBase
	connectionMixEpsilons
)

const (
	connectionMixCopies = iota
	connectionMixIterations
)

// One thread per token keeps every Sinkhorn iteration of its small matrix in
// registers, where the graph takes two reductions per iteration.
var connectionMix = &gpuKernel{
	name:    "connection_mix",
	inputs:  []string{connectionMixProjected: "projected", connectionMixVariance: "variance", connectionMixScales: "scales", connectionMixBase: "base", connectionMixEpsilons: "eps"},
	outputs: []string{"pre", "post", "residual"},
	metal: gpuSource{header: `
// The graph ops' forms and rounding, so the kernel tracks the graph.
#pragma METAL fp contract(off)
inline float sigmoid(float x) {
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return x < 0.0f ? y : 1.0f - y;
}
`, source: `
uint n = thread_position_in_grid.x;
constexpr int W = (C + 2) * C;
const device float* logits = projected + n * W;
float norm = metal::precise::rsqrt(variance[n] + eps[0]);
float mix_eps = eps[1];
for (int i = 0; i < C; ++i) {
  pre[n * C + i] = sigmoid(logits[i] * norm * scales[0] + base[i]) + mix_eps;
  post[n * C + i] = sigmoid(logits[C + i] * norm * scales[1] + base[C + i]) * 2.0f;
}
float m[C * C];
for (int r = 0; r < C; ++r) {
  float top = -INFINITY;
  for (int c = 0; c < C; ++c) {
    int k = 2 * C + r * C + c;
    m[r * C + c] = logits[k] * norm * scales[2] + base[k];
    top = metal::max(top, m[r * C + c]);
  }
  float total = 0.0f;
  for (int c = 0; c < C; ++c) {
    m[r * C + c] = metal::fast::exp(m[r * C + c] - top);
    total += m[r * C + c];
  }
  float inv_total = 1.0f / total;
  for (int c = 0; c < C; ++c) {
    m[r * C + c] = m[r * C + c] * inv_total + mix_eps;
  }
}
for (int it = 0; it < ITERS; ++it) {
  if (it > 0) {
    for (int r = 0; r < C; ++r) {
      float sum = 0.0f;
      for (int c = 0; c < C; ++c) sum += m[r * C + c];
      sum += mix_eps;
      for (int c = 0; c < C; ++c) m[r * C + c] /= sum;
    }
  }
  for (int c = 0; c < C; ++c) {
    float sum = 0.0f;
    for (int r = 0; r < C; ++r) sum += m[r * C + c];
    sum += mix_eps;
    for (int r = 0; r < C; ++r) m[r * C + c] /= sum;
  }
}
for (int k = 0; k < C * C; ++k) {
  residual[n * C * C + k] = m[k];
}
`},
	// Match the graph's CUDA rounding across each Sinkhorn iteration.
	cuda: gpuSource{source: `
int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
if (n >= variance_shape[0] * variance_shape[1]) {
  return;
}
constexpr int W = (C + 2) * C;
const float* logits = projected + n * W;
float norm = rsqrtf(__fadd_rn(variance[n], eps[0]));
float mix_eps = eps[1];
auto sigmoid = [](float x) {
  float y = 1.0f / (1.0f + expf(fabsf(x)));
  return x < 0.0f ? y : 1.0f - y;
};
for (int i = 0; i < C; ++i) {
  float x = __fadd_rn(__fmul_rn(__fmul_rn(logits[i], norm), scales[0]), base[i]);
  pre[n * C + i] = __fadd_rn(sigmoid(x), mix_eps);
  float y = __fadd_rn(__fmul_rn(__fmul_rn(logits[C + i], norm), scales[1]), base[C + i]);
  post[n * C + i] = __fmul_rn(sigmoid(y), 2.0f);
}
float m[C * C];
for (int r = 0; r < C; ++r) {
  float top = -INFINITY;
  for (int c = 0; c < C; ++c) {
    int k = 2 * C + r * C + c;
    m[r * C + c] = __fadd_rn(__fmul_rn(__fmul_rn(logits[k], norm), scales[2]), base[k]);
    top = fmaxf(top, m[r * C + c]);
  }
  float total = 0.0f;
  for (int c = 0; c < C; ++c) {
    total = __fadd_rn(total, __expf(m[r * C + c] - top));
  }
  float inv_total = 1.0f / total;
  for (int c = 0; c < C; ++c) {
    m[r * C + c] = __fadd_rn(__fmul_rn(__expf(m[r * C + c] - top), inv_total), mix_eps);
  }
}
for (int it = 0; it < ITERS; ++it) {
  if (it > 0) {
    for (int r = 0; r < C; ++r) {
      float sum = 0.0f;
      for (int c = 0; c < C; ++c) sum = __fadd_rn(sum, m[r * C + c]);
      sum = __fadd_rn(sum, mix_eps);
      for (int c = 0; c < C; ++c) m[r * C + c] = __fdiv_rn(m[r * C + c], sum);
    }
  }
  for (int c = 0; c < C; ++c) {
    float sum = 0.0f;
    for (int r = 0; r < C; ++r) sum = __fadd_rn(sum, m[r * C + c]);
    sum = __fadd_rn(sum, mix_eps);
    for (int r = 0; r < C; ++r) m[r * C + c] = __fdiv_rn(m[r * C + c], sum);
  }
}
for (int k = 0; k < C * C; ++k) {
  residual[n * C * C + k] = m[k];
}
`},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		pre, post, residual := connectionMixGraph(in[connectionMixProjected], in[connectionMixVariance], in[connectionMixScales], in[connectionMixBase], in[connectionMixEpsilons],
			launch.ints[connectionMixCopies].value, launch.ints[connectionMixIterations].value)
		return []*Array{pre, post, residual}
	},
}

func connectionMixGraph(projected, variance, scales, base, eps *Array, copies, iterations int) (pre, post, residual *Array) {
	normEps, mixEps := eps.Slice(Slice(0, 1)), eps.Slice(Slice(1, 2))
	logits := Mul(projected, RSqrt(Add(variance, normEps)))
	part := func(scale, start, end int) *Array {
		return Add(Mul(logits.Slice(Slice(), Slice(), Slice(start, end)), scales.Slice(Slice(scale, scale+1))), base.Slice(Slice(start, end)))
	}
	pre = Add(Sigmoid(part(0, 0, copies)), mixEps)
	post = MulScalar(Sigmoid(part(1, copies, 2*copies)), 2)
	matrix := Reshape(part(2, 2*copies, (copies+2)*copies), int32(logits.Dim(0)), int32(logits.Dim(1)), int32(copies), int32(copies))
	return pre, post, sinkhornGraph(iterations)(matrix, mixEps)
}

// sinkhornGraphs caches the compiled Sinkhorn normalization per iteration
// count. MLX runs on one thread, so the map needs no lock.
var sinkhornGraphs = map[int]func(*Array, *Array) *Array{}

func sinkhornGraph(iterations int) func(*Array, *Array) *Array {
	if fn := sinkhornGraphs[iterations]; fn != nil {
		return fn
	}
	fn := Compile2("sinkhorn", func(matrix, eps *Array) *Array {
		matrix = Add(SoftmaxAxis(matrix, -1, true), eps)
		matrix = Div(matrix, Add(Sum(matrix, -2, true), eps))
		for range iterations - 1 {
			matrix = Div(matrix, Add(Sum(matrix, -1, true), eps))
			matrix = Div(matrix, Add(Sum(matrix, -2, true), eps))
		}
		return matrix
	}, Shapeless())
	sinkhornGraphs[iterations] = fn
	return fn
}

// ConnectionMix splits a hyper-connection's mix logits. projected is
// [B, L, (C+2)*C] and variance [B, L, 1], both FP32; the logits are
// projected·rsqrt(variance+normEps), scaled by scales[0..2] per part and
// offset by base. It returns the next branch's input weights
// sigmoid(·)+mixEps [B, L, C], this branch's output weights 2·sigmoid(·)
// [B, L, C], and the residual mix [B, L, C, C]: a row softmax plus mixEps,
// then iterations rounds of Sinkhorn normalization (columns first; every
// round after the first starts with rows), each dividing by sum+mixEps.
func ConnectionMix(projected, variance, scales, base *Array, copies, iterations int, normEps, mixEps float32) (pre, post, residual *Array) {
	width := (copies + 2) * copies
	if projected.NumDims() != 3 || projected.Dim(2) != width || variance.Size() != projected.Dim(0)*projected.Dim(1) || base.Size() != width || scales.Size() != 3 || iterations < 1 {
		panic("mlx: ConnectionMix requires projected [B, L, (C+2)*C], variance [B, L, 1], scales [3], base [(C+2)*C], iterations >= 1")
	}
	b, l := int32(projected.Dim(0)), int32(projected.Dim(1))
	n := projected.Dim(0) * projected.Dim(1)
	eps := FromValues([]float32{normEps, mixEps}, 2)
	out := connectionMix.run(gpuLaunch{
		ints: []gpuIntArg{connectionMixCopies: {"C", copies}, connectionMixIterations: {"ITERS", iterations}},
		outputs: []gpuOutputSpec{
			{"CONNECTION_PRE", []int32{b, l, int32(copies)}, DTypeFloat32},
			{"CONNECTION_POST", []int32{b, l, int32(copies)}, DTypeFloat32},
			{"CONNECTION_RESIDUAL", []int32{b, l, int32(copies), int32(copies)}, DTypeFloat32},
		},
		grid:        [3]int{n, 1, 1},
		threadGroup: [3]int{min(n, 256), 1, 1},
		inputs: []*Array{
			connectionMixProjected: projected.AsType(DTypeFloat32),
			connectionMixVariance:  variance.AsType(DTypeFloat32),
			connectionMixScales:    scales.AsType(DTypeFloat32),
			connectionMixBase:      base.AsType(DTypeFloat32),
			connectionMixEpsilons:  eps,
		},
	})
	return out[0], out[1], out[2]
}
