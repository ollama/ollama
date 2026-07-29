package mlx

var gatedDelta = &gpuKernel{
	name:    "gated_delta_step",
	inputs:  []string{"q", "k", "v", "g", "beta", "state_in", "T"},
	outputs: []string{"y", "state_out"},
	metal:   gpuSource{source: gatedDeltaMetalKernelSource},
	cuda:    gpuSource{source: gatedDeltaCUDAKernelSource},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		y, state := gatedDeltaFallback(in[0], in[1], in[2], in[3], in[4], in[5])
		return []*Array{y, state}
	},
}

const gatedDeltaMetalKernelSource = `
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

// q, k: [B, T, Hk, Dk]
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

// v, y: [B, T, Hv, Dv]
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

// state_in, state_out: [B, Hv, Dv, Dk]
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// g: [B, T, Hv]
auto g_ = g + b_idx * T * Hv;
auto beta_ = beta + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[hv_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);

  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[dv_idx] = static_cast<InT>(out);
  }

  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv;
  beta_ += Hv;
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<StT>(state[i]);
}
`

const gatedDeltaCUDAKernelSource = `
auto tid_x = threadIdx.x;
auto tid_y = threadIdx.y;
auto grid_y = blockIdx.y * blockDim.y + tid_y;
auto grid_z = blockIdx.z;

int T_val = static_cast<int>(*T);

auto n = grid_z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

// q, k: [B, T, Hk, Dk]
auto q_ = q + b_idx * T_val * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T_val * Hk * Dk + hk_idx * Dk;

// v, y: [B, T, Hv, Dv]
auto dv_idx = grid_y;
auto v_ = v + b_idx * T_val * Hv * Dv + hv_idx * Dv;
y += b_idx * T_val * Hv * Dv + hv_idx * Dv;

auto dk_idx = tid_x;

// state_in, state_out: [B, Hv, Dv, Dk]
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// g: [B, T, Hv]
auto g_ = g + b_idx * T_val * Hv;
auto beta_ = beta + b_idx * T_val * Hv;

for (int t = 0; t < T_val; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * static_cast<float>(g_[hv_idx]);
    kv_mem += state[i] * static_cast<float>(k_[s_idx]);
  }
  // Warp reduction (full warp, 32 threads in x)
  for (int offset = 16; offset > 0; offset >>= 1)
    kv_mem += __shfl_down_sync(0xffffffff, kv_mem, offset);
  kv_mem = __shfl_sync(0xffffffff, kv_mem, 0);

  auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
    out += state[i] * static_cast<float>(q_[s_idx]);
  }
  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1)
    out += __shfl_down_sync(0xffffffff, out, offset);
  if (tid_x == 0) {
    y[dv_idx] = static_cast<InT>(out);
  }

  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv;
  beta_ += Hv;
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<StT>(state[i]);
}
`

// gatedDeltaDims are the batch and head geometry of one scan, recovered
// from the input shapes.
type gatedDeltaDims struct {
	B, T, Hk, Dk, Hv, Dv int
}

// resolveGatedDeltaDims validates the inputs against the GPU kernels'
// contract and recovers the launch geometry. ok=false routes to the graph
// fallback: shapes that disagree, Dk not a multiple of the 32-lane simd
// width, or mixed input dtypes.
func resolveGatedDeltaDims(q, k, v, g, beta, state *Array) (gatedDeltaDims, bool) {
	var dims gatedDeltaDims
	if q == nil || k == nil || v == nil || g == nil || beta == nil || state == nil {
		return dims, false
	}
	qd, kd, vd, gd, bd, sd := q.Dims(), k.Dims(), v.Dims(), g.Dims(), beta.Dims(), state.Dims()
	if len(qd) != 4 || len(kd) != 4 || len(vd) != 4 || len(gd) != 3 || len(bd) != 3 || len(sd) != 4 {
		return dims, false
	}
	dims.B, dims.T, dims.Hk, dims.Dk = qd[0], qd[1], qd[2], qd[3]
	if dims.T <= 0 || dims.Hk <= 0 || dims.Dk <= 0 || dims.Dk%32 != 0 {
		return dims, false
	}
	if kd[0] != dims.B || kd[1] != dims.T || kd[2] != dims.Hk || kd[3] != dims.Dk {
		return dims, false
	}
	dims.Hv, dims.Dv = vd[2], vd[3]
	if vd[0] != dims.B || vd[1] != dims.T || dims.Hv <= 0 || dims.Dv <= 0 || dims.Hv%dims.Hk != 0 {
		return dims, false
	}
	if gd[0] != dims.B || gd[1] != dims.T || gd[2] != dims.Hv {
		return dims, false
	}
	if bd[0] != dims.B || bd[1] != dims.T || bd[2] != dims.Hv {
		return dims, false
	}
	if sd[0] != dims.B || sd[1] != dims.Hv || sd[2] != dims.Dv || sd[3] != dims.Dk {
		return dims, false
	}
	if k.DType() != q.DType() || v.DType() != q.DType() || g.DType() != q.DType() || beta.DType() != q.DType() {
		return dims, false
	}
	return dims, true
}

func repeatHeadsForGatedDelta(x *Array, repeatFactor int) *Array {
	if repeatFactor <= 1 {
		return x
	}
	shape := x.Dims()
	x = ExpandDims(x, 3)
	x = Tile(x, []int32{1, 1, 1, int32(repeatFactor), 1})
	return Reshape(x, int32(shape[0]), int32(shape[1]), int32(shape[2]*repeatFactor), int32(shape[3]))
}

func gatedDeltaFallback(q, k, v, g, beta, state *Array) (y, nextState *Array) {
	if q == nil || k == nil || v == nil || g == nil || beta == nil || state == nil {
		return nil, nil
	}

	qd := q.Dims()
	kd := k.Dims()
	vd := v.Dims()
	gd := g.Dims()
	bd := beta.Dims()
	sd := state.Dims()
	if len(qd) != 4 || len(kd) != 4 || len(vd) != 4 || len(gd) != 3 || len(bd) != 3 || len(sd) != 4 {
		return nil, nil
	}

	B, T, Hk, Dk := int32(qd[0]), int32(qd[1]), int32(qd[2]), int32(qd[3])
	Hv, Dv := int32(vd[2]), int32(vd[3])
	if T <= 0 || Hk <= 0 || Dk <= 0 || Hv <= 0 || Dv <= 0 || Hv%Hk != 0 {
		return nil, nil
	}
	if kd[0] != int(B) || kd[1] != int(T) || kd[2] != int(Hk) || kd[3] != int(Dk) {
		return nil, nil
	}
	if vd[0] != int(B) || vd[1] != int(T) {
		return nil, nil
	}
	if gd[0] != int(B) || gd[1] != int(T) || gd[2] != int(Hv) {
		return nil, nil
	}
	if bd[0] != int(B) || bd[1] != int(T) || bd[2] != int(Hv) {
		return nil, nil
	}
	if sd[0] != int(B) || sd[1] != int(Hv) || sd[2] != int(Dv) || sd[3] != int(Dk) {
		return nil, nil
	}

	repeatFactor := int(Hv / Hk)
	q = repeatHeadsForGatedDelta(q, repeatFactor)
	k = repeatHeadsForGatedDelta(k, repeatFactor)

	nextState = state
	if T == 1 {
		qt := Squeeze(q, 1)
		kt := Squeeze(k, 1)
		vt := Squeeze(v, 1)
		gt := Squeeze(g, 1)
		bt := Squeeze(beta, 1)

		nextState = Mul(nextState, ExpandDims(ExpandDims(gt, -1), -1))
		kvMem := Sum(Mul(nextState, ExpandDims(kt, 2)), -1, false)
		delta := Mul(Sub(vt, kvMem), ExpandDims(bt, -1))
		nextState = Add(nextState, Mul(ExpandDims(kt, 2), ExpandDims(delta, -1)))
		yt := Sum(Mul(nextState, ExpandDims(qt, 2)), -1, false)
		return ExpandDims(yt, 1), nextState
	}

	outs := make([]*Array, 0, T)
	for t := range T {
		qt := Squeeze(SliceStartStop(q, []int32{0, t, 0, 0}, []int32{B, t + 1, Hv, Dk}), 1)
		kt := Squeeze(SliceStartStop(k, []int32{0, t, 0, 0}, []int32{B, t + 1, Hv, Dk}), 1)
		vt := Squeeze(SliceStartStop(v, []int32{0, t, 0, 0}, []int32{B, t + 1, Hv, Dv}), 1)
		gt := Squeeze(SliceStartStop(g, []int32{0, t, 0}, []int32{B, t + 1, Hv}), 1)
		bt := Squeeze(SliceStartStop(beta, []int32{0, t, 0}, []int32{B, t + 1, Hv}), 1)

		nextState = Mul(nextState, ExpandDims(ExpandDims(gt, -1), -1))
		kvMem := Sum(Mul(nextState, ExpandDims(kt, 2)), -1, false)
		delta := Mul(Sub(vt, kvMem), ExpandDims(bt, -1))
		nextState = Add(nextState, Mul(ExpandDims(kt, 2), ExpandDims(delta, -1)))
		yt := Sum(Mul(nextState, ExpandDims(qt, 2)), -1, false)
		outs = append(outs, ExpandDims(yt, 1))
	}
	return Concatenate(outs, 1), nextState
}

// FastGatedDelta runs the recurrent update operation.
//
// When mask is non-nil, it must be a [B, T] bool tensor identifying real
// (true) vs. padded (false) positions in q/k/v/g/beta. Padded positions
// are substituted with neutral values (q=k=v=beta=0, g=1) so each padded
// kernel iteration is a no-op — state passes through unchanged and the
// final state equals the state after the last real token of each row.
//
// Inputs that fit the GPU kernels' contract run there (CUDA or Metal, with
// the graph implementation covering boxes where neither can run); anything
// else runs the graph implementation directly.
func FastGatedDelta(q, k, v, g, beta, state, mask *Array) (y, nextState *Array) {
	// TODO: handle this more efficiently with a masked kernel (MLX-LM has one).
	if mask != nil {
		B := int32(mask.Dim(0))
		T := int32(mask.Dim(1))
		m4 := Reshape(mask, B, T, 1, 1)
		m3 := Reshape(mask, B, T, 1)
		zeroQ := FromValue(float32(0)).AsType(q.DType())
		zeroK := FromValue(float32(0)).AsType(k.DType())
		zeroV := FromValue(float32(0)).AsType(v.DType())
		zeroBeta := FromValue(float32(0)).AsType(beta.DType())
		oneG := FromValue(float32(1)).AsType(g.DType())
		q = Where(m4, q, zeroQ)
		k = Where(m4, k, zeroK)
		v = Where(m4, v, zeroV)
		beta = Where(m3, beta, zeroBeta)
		g = Where(m3, g, oneG)
	}

	if dims, ok := resolveGatedDeltaDims(q, k, v, g, beta, state); ok {
		outs := gatedDelta.run(gpuLaunch{
			dtypes: []gpuDTypeArg{{"InT", q.DType()}, {"StT", state.DType()}},
			ints:   []gpuIntArg{{"Dk", dims.Dk}, {"Dv", dims.Dv}, {"Hk", dims.Hk}, {"Hv", dims.Hv}},
			outputs: []gpuOutputSpec{
				{"GATED_DELTA_Y", []int32{int32(dims.B), int32(dims.T), int32(dims.Hv), int32(dims.Dv)}, q.DType()},
				{"GATED_DELTA_STATE", []int32{int32(dims.B), int32(dims.Hv), int32(dims.Dv), int32(dims.Dk)}, state.DType()},
			},
			grid:        [3]int{32, dims.Dv, dims.B * dims.Hv},
			threadGroup: [3]int{32, min(dims.Dv, 4), 1},
			inputs:      []*Array{q, k, v, g, beta, state, FromValue(dims.T)},
		})
		return outs[0], outs[1]
	}

	y, nextState = gatedDeltaFallback(q, k, v, g, beta, state)
	if y == nil || nextState == nil {
		panic("mlx.FastGatedDelta: fallback failed (invalid inputs or unsupported shapes)")
	}
	return y, nextState
}
