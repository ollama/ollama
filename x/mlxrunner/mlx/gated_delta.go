package mlx

import "math"

var gatedDeltaRecurrenceKernel = &gpuKernel{
	name:    "gated_delta_recurrence",
	inputs:  []string{"q", "k", "v", "g", "beta", "state_in", "T"},
	outputs: []string{"y", "state_out"},
	metal:   gpuSource{source: gatedDeltaRecurrenceMetalSource},
	cuda:    gpuSource{source: gatedDeltaRecurrenceCUDASource},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		y, state := gatedDeltaRecurrenceGraph(in[0], in[1], in[2], in[3], in[4], in[5])
		return []*Array{y, state}
	},
}

const gatedDeltaRecurrenceMetalSource = `
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

const gatedDeltaRecurrenceCUDASource = `
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

// gatedDeltaRecurrenceDims are the batch and head geometry of one scan,
// recovered from the input shapes.
type gatedDeltaRecurrenceDims struct {
	B, T, Hk, Dk, Hv, Dv int
}

// resolveGatedDeltaRecurrenceDims validates the inputs against the GPU
// kernels' contract and recovers the launch geometry. ok=false routes to
// the graph implementation: shapes that disagree, Dk not a multiple of the
// 32-lane simd width, or mixed input dtypes.
func resolveGatedDeltaRecurrenceDims(q, k, v, g, beta, state *Array) (gatedDeltaRecurrenceDims, bool) {
	var dims gatedDeltaRecurrenceDims
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

func gatedDeltaRecurrenceGraph(q, k, v, g, beta, state *Array) (y, nextState *Array) {
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

// gatedDeltaRecurrence runs the scan. Inputs that fit the GPU kernels'
// contract run there (CUDA or Metal, with the graph implementation covering
// boxes where neither can run); anything else runs the graph implementation
// directly.
func gatedDeltaRecurrence(q, k, v, g, beta, state *Array) (y, nextState *Array) {
	if dims, ok := resolveGatedDeltaRecurrenceDims(q, k, v, g, beta, state); ok {
		outs := gatedDeltaRecurrenceKernel.run(gpuLaunch{
			dtypes: []gpuDTypeArg{{"InT", q.DType()}, {"StT", state.DType()}},
			ints:   []gpuIntArg{{"Dk", dims.Dk}, {"Dv", dims.Dv}, {"Hk", dims.Hk}, {"Hv", dims.Hv}},
			outputs: []gpuOutputSpec{
				{"GATED_DELTA_RECURRENCE_Y", []int32{int32(dims.B), int32(dims.T), int32(dims.Hv), int32(dims.Dv)}, q.DType()},
				{"GATED_DELTA_RECURRENCE_STATE", []int32{int32(dims.B), int32(dims.Hv), int32(dims.Dv), int32(dims.Dk)}, state.DType()},
			},
			grid:        [3]int{32, dims.Dv, dims.B * dims.Hv},
			threadGroup: [3]int{32, min(dims.Dv, 4), 1},
			inputs:      []*Array{q, k, v, g, beta, state, FromValue(dims.T)},
		})
		return outs[0], outs[1]
	}

	y, nextState = gatedDeltaRecurrenceGraph(q, k, v, g, beta, state)
	if y == nil || nextState == nil {
		panic("mlx: gated-delta recurrence: invalid inputs or unsupported shapes")
	}
	return y, nextState
}

// gatedDeltaMaxTokens caps the fused scan length at the current token plus
// a ten-token draft — several times the depth the EV controller selects in
// practice. SeqT is a template argument, so every accepted length compiles
// its own pipeline variant; the cap bounds that set, and longer windows run
// the same step as graph ops.
const gatedDeltaMaxTokens = 11

var (
	gatedDelta = &gpuKernel{
		name:    "gated_delta",
		inputs:  []string{"packed", "ba", "dt_bias", "a_exp", "qk_scale", "state_in"},
		outputs: []string{"y", "state_out"},
		metal: gpuSource{
			source: gatedDeltaMetalSource,
			header: gatedDeltaMetalHeader + "#define GDN_STORE_INTERIOR(index, value)\n",
		},
		fallback: func(launch gpuLaunch) []*Array {
			in := launch.inputs
			y, end, _ := gatedDeltaGraph(in[0], in[1], in[2], in[3], in[5], false)
			return []*Array{y, end}
		},
	}
	gatedDeltaStates = &gpuKernel{
		name:    "gated_delta_states",
		inputs:  []string{"packed", "ba", "dt_bias", "a_exp", "qk_scale", "state_in"},
		outputs: []string{"y", "state_out", "state_seq"},
		metal: gpuSource{
			source: gatedDeltaMetalSource,
			header: gatedDeltaMetalHeader + "#define GDN_STORE_INTERIOR(index, value) state_seq[index] = value\n",
		},
		fallback: func(launch gpuLaunch) []*Array {
			in := launch.inputs
			y, end, interior := gatedDeltaGraph(in[0], in[1], in[2], in[3], in[5], true)
			for i, s := range interior {
				interior[i] = ExpandDims(s, 0)
			}
			return []*Array{y, end, Concatenate(interior, 0)}
		},
	}
)

// gatedDeltaGraph is the graph implementation of the fused gated-delta step
// over the kernels' contract domain: q/k RMS norm and scaling, the decay
// gate, and the (per-token, for captureAll) scan. Geometry is recovered
// from the packed and state shapes, and the q/k scales are recomputed from
// Dk with the same bits the kernel input carries.
func gatedDeltaGraph(packed, ba, dtBias, aExp, state *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	B, T := int32(packed.Dim(0)), int32(packed.Dim(1))
	sd := state.Dims()
	Hv, Dv, Dk := int32(sd[1]), int32(sd[2]), int32(sd[3])
	keyDim := (int32(packed.Dim(2)) - Hv*Dv) / 2
	Hk := keyDim / Dk

	q := SliceStartStop(packed, []int32{0, 0, 0}, []int32{B, T, keyDim})
	k := SliceStartStop(packed, []int32{0, 0, keyDim}, []int32{B, T, 2 * keyDim})
	v := SliceStartStop(packed, []int32{0, 0, 2 * keyDim}, []int32{B, T, 2*keyDim + Hv*Dv})
	q = Reshape(q, B, T, Hk, Dk)
	k = Reshape(k, B, T, Hk, Dk)
	v = Reshape(v, B, T, Hv, Dv)
	invScale := gatedDeltaInvScale(int(Dk))
	q = MulScalar(RMSNormFn(q, nil, 1e-6), invScale*invScale)
	k = MulScalar(RMSNormFn(k, nil, 1e-6), invScale)

	beta := SliceStartStop(ba, []int32{0, 0, 0}, []int32{B, T, Hv})
	alpha := SliceStartStop(ba, []int32{0, 0, Hv}, []int32{B, T, 2 * Hv})
	decay := Softplus(Add(alpha, dtBias))
	decay = Mul(decay, aExp)
	decay = Exp(MulScalar(decay, -1)).AsType(alpha.DType())
	betaGate := Sigmoid(beta)

	if !captureAll || T == 1 {
		y, nextState = gatedDeltaRecurrence(q, k, v, decay, betaGate, state)
		return y, nextState, nil
	}

	sliceT := func(a *Array, t int32) *Array {
		dims := a.Dims()
		start := make([]int32, len(dims))
		stop := make([]int32, len(dims))
		for d := range dims {
			stop[d] = int32(dims[d])
		}
		start[1], stop[1] = t, t+1
		return SliceStartStop(a, start, stop)
	}
	outs := make([]*Array, T)
	for t := range T {
		outs[t], state = gatedDeltaRecurrence(sliceT(q, t), sliceT(k, t), sliceT(v, t), sliceT(decay, t), sliceT(betaGate, t), state)
		if t+1 < T {
			interior = append(interior, state)
		}
	}
	return Concatenate(outs, 1), state, interior
}

// The fused scan consumes the activated causal-conv output rows [q | k | v]
// plus the packed [beta | alpha] projection and performs q/k RMS norm and
// scaling, the decay gate, and the gated-delta recurrence in one launch.
const gatedDeltaMetalSource = `
constexpr int SIMDGroups = 4;
constexpr int ValuesPerSIMD = DvTile / SIMDGroups;
constexpr int StatePerLane = Dk / 32;
constexpr int QDim = Hk * Dk;

auto lane = thread_position_in_threadgroup.x;
auto simd_idx = thread_position_in_threadgroup.y;
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
auto value_tile = threadgroup_position_in_grid.y;

threadgroup InT q_cache[Dk];
threadgroup InT k_cache[Dk];
threadgroup InT decay_cache[1];
threadgroup InT beta_cache[1];

float state[ValuesPerSIMD][StatePerLane];
for (int value_iter = 0; value_iter < ValuesPerSIMD; ++value_iter) {
  auto dv_idx = value_tile * DvTile + simd_idx + value_iter * SIMDGroups;
  auto state_offset = (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < StatePerLane; ++i) {
    auto d = StatePerLane * lane + i;
    state[value_iter][i] = state_in[state_offset + d];
  }
}

for (int t = 0; t < SeqT; ++t) {
  if (simd_idx == 0) {
    auto token = packed + (b_idx * SeqT + t) * PackedDim;
    float q_values[StatePerLane];
    float k_values[StatePerLane];
    float q_squares = 0.0f;
    float k_squares = 0.0f;

    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      float q_value = static_cast<float>(token[hk_idx * Dk + d]);
      float k_value = static_cast<float>(token[QDim + hk_idx * Dk + d]);
      q_values[i] = q_value;
      k_values[i] = k_value;
      q_squares += q_value * q_value;
      k_squares += k_value * k_value;
    }

    q_squares = simd_sum(q_squares);
    k_squares = simd_sum(k_squares);
    float q_inv_rms = metal::precise::rsqrt(q_squares / float(Dk) + 1.0e-6f);
    float k_inv_rms = metal::precise::rsqrt(k_squares / float(Dk) + 1.0e-6f);
    InT q_scale = static_cast<InT>(qk_scale[0]);
    InT k_scale = static_cast<InT>(qk_scale[1]);
    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      InT q_norm = static_cast<InT>(q_values[i] * q_inv_rms);
      InT k_norm = static_cast<InT>(k_values[i] * k_inv_rms);
      q_cache[d] = static_cast<InT>(q_norm * q_scale);
      k_cache[d] = static_cast<InT>(k_norm * k_scale);
    }

    if (lane == 0) {
      auto ba_row = (b_idx * SeqT + t) * 2 * Hv;
      InT gate_input = static_cast<InT>(ba[ba_row + Hv + hv_idx] + dt_bias[hv_idx]);
      InT softplus = gdn_logaddexp(gate_input, static_cast<InT>(0));
      float decay = metal::precise::exp(-static_cast<float>(softplus) * a_exp[hv_idx]);
      decay_cache[0] = static_cast<InT>(decay);
      beta_cache[0] = gdn_sigmoid(ba[ba_row + hv_idx]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int value_iter = 0; value_iter < ValuesPerSIMD; ++value_iter) {
    auto dv_idx = value_tile * DvTile + simd_idx + value_iter * SIMDGroups;
    float projection = 0.0f;
    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      state[value_iter][i] *= static_cast<float>(decay_cache[0]);
      projection += state[value_iter][i] * static_cast<float>(k_cache[d]);
    }
    projection = simd_sum(projection);

    auto token = packed + (b_idx * SeqT + t) * PackedDim;
    float v_value = static_cast<float>(token[2 * QDim + hv_idx * Dv + dv_idx]);
    float delta = (v_value - projection) * static_cast<float>(beta_cache[0]);

    float out = 0.0f;
    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      state[value_iter][i] += static_cast<float>(k_cache[d]) * delta;
      out += state[value_iter][i] * static_cast<float>(q_cache[d]);
    }
    out = simd_sum(out);
    if (lane == 0) {
      y[((b_idx * SeqT + t) * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
    }

    if (t + 1 < SeqT) {
      auto seq_offset = ((t * threads_per_grid.z + n) * Dv + dv_idx) * Dk;
      for (int i = 0; i < StatePerLane; ++i) {
        auto d = StatePerLane * lane + i;
        GDN_STORE_INTERIOR(seq_offset + d, state[value_iter][i]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int value_iter = 0; value_iter < ValuesPerSIMD; ++value_iter) {
  auto dv_idx = value_tile * DvTile + simd_idx + value_iter * SIMDGroups;
  auto state_offset = (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < StatePerLane; ++i) {
    auto d = StatePerLane * lane + i;
    state_out[state_offset + d] = state[value_iter][i];
  }
}
`

const gatedDeltaMetalHeader = `
template <typename T>
T gdn_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

template <typename T>
T gdn_logaddexp(T x, T y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<T>::quiet_NaN();
  }
  constexpr T inf = metal::numeric_limits<T>::infinity();
  T maxval = metal::max(x, y);
  T minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
      ? maxval
      : (maxval + log1p(metal::exp(minval - maxval)));
}
`

// gatedDeltaInvScale is the q/k norm scale for key dimension dk; the
// kernel's qk_scale input and the graph implementation share these bits.
func gatedDeltaInvScale(dk int) float32 {
	return float32(1.0 / math.Sqrt(float64(dk)))
}

// gatedDeltaDims are the batch and head geometry the kernel is instantiated
// for, recovered from the input shapes.
type gatedDeltaDims struct {
	B, Hk, Dk, Hv, Dv, PackedDim, T int
}

// GatedDelta runs the whole gated-delta step — q/k norms, decay gate, and
// the scan — in one launch over the activated causal-conv output. packed is
// [B, T, 2*Hk*Dk + Hv*Dv] with rows packed [q | k | v], ba is [B, T, 2*Hv]
// packed [beta | alpha] rows, and state is [B, Hv, Dv, Dk]. captureAll
// additionally emits every interior per-token state. Inputs that fit the
// one-launch kernels' contract run there; anything else runs the same step
// as graph ops.
//
// When mask is non-nil, it must be a [B, T] bool tensor identifying real
// (true) vs. padded (false) positions. Padded rows are neutralized before
// the kernels' own preprocessing: zeroed conv rows make the q/k/v norms
// emit zeros, and -inf beta/alpha rows yield beta 0 and decay 1, so each
// padded position is an identity step with zero output, exactly as on the
// recurrence path.
func GatedDelta(packed, ba, dtBias, aExp, state, mask *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	if mask != nil {
		mask3 := Reshape(mask, int32(mask.Dim(0)), int32(mask.Dim(1)), 1)
		zero := FromValue(float32(0)).AsType(packed.DType())
		packed = Where(mask3, packed, zero)
		negInf := FromValue(float32(math.Inf(-1))).AsType(ba.DType())
		ba = Where(mask3, ba, negInf)
	}

	dims, ok := resolveGatedDeltaDims(packed, ba, dtBias, aExp, state)
	if !ok {
		return gatedDeltaGraph(packed, ba, dtBias, aExp, state, captureAll)
	}

	inv := gatedDeltaInvScale(dims.Dk)
	qkScale := FromValues([]float32{inv * inv, inv}, 2)
	dvTile := 32
	if dims.Dv%dvTile != 0 {
		// resolveGatedDeltaDims guarantees Dv%16 == 0.
		dvTile = 16
	}
	useAllStates := captureAll && dims.T > 1
	kernel := gatedDelta
	outputs := []gpuOutputSpec{
		{"GATED_DELTA_Y", []int32{int32(dims.B), int32(dims.T), int32(dims.Hv), int32(dims.Dv)}, packed.DType()},
		{"GATED_DELTA_STATE", []int32{int32(dims.B), int32(dims.Hv), int32(dims.Dv), int32(dims.Dk)}, DTypeFloat32},
	}
	if useAllStates {
		kernel = gatedDeltaStates
		outputs = append(outputs, gpuOutputSpec{
			"GATED_DELTA_STATE_SEQ", []int32{int32(dims.T - 1), int32(dims.B), int32(dims.Hv), int32(dims.Dv), int32(dims.Dk)}, DTypeFloat32,
		})
	}

	outs := kernel.run(gpuLaunch{
		dtypes: []gpuDTypeArg{{"InT", packed.DType()}},
		ints: []gpuIntArg{
			{"Hk", dims.Hk},
			{"Dk", dims.Dk},
			{"Hv", dims.Hv},
			{"Dv", dims.Dv},
			{"PackedDim", dims.PackedDim},
			{"SeqT", dims.T},
			{"DvTile", dvTile},
		},
		outputs:     outputs,
		grid:        [3]int{32, (dims.Dv / dvTile) * 4, dims.B * dims.Hv},
		threadGroup: [3]int{32, 4, 1},
		inputs:      []*Array{packed, ba, dtBias, aExp, qkScale, state},
	})
	if useAllStates {
		interior = sliceGatedDeltaStates(outs[2], dims)
	}
	return outs[0], outs[1], interior
}

func resolveGatedDeltaDims(packed, ba, dtBias, aExp, state *Array) (gatedDeltaDims, bool) {
	var dims gatedDeltaDims
	if packed == nil || ba == nil || dtBias == nil || aExp == nil || state == nil {
		return dims, false
	}
	sd := state.Dims()
	if len(sd) != 4 || sd[0] < 1 {
		return dims, false
	}
	dims.B, dims.Hv, dims.Dv, dims.Dk = sd[0], sd[1], sd[2], sd[3]

	pd := packed.Dims()
	if len(pd) != 3 || pd[0] != dims.B || pd[1] < 1 || pd[1] > gatedDeltaMaxTokens {
		return dims, false
	}
	dims.T, dims.PackedDim = pd[1], pd[2]

	keyRows := dims.PackedDim - dims.Hv*dims.Dv
	if keyRows <= 0 || dims.Dk <= 0 || keyRows%(2*dims.Dk) != 0 {
		return dims, false
	}
	dims.Hk = keyRows / (2 * dims.Dk)
	if dims.Hk <= 0 || dims.Hv%dims.Hk != 0 || dims.Dk%32 != 0 || dims.Dv%16 != 0 {
		return dims, false
	}

	if !exactShape(ba, dims.B, dims.T, 2*dims.Hv) ||
		!exactShape(dtBias, dims.Hv) ||
		!exactShape(aExp, dims.Hv) {
		return dims, false
	}
	if packed.DType() != DTypeBFloat16 || ba.DType() != DTypeBFloat16 ||
		dtBias.DType() != DTypeBFloat16 || aExp.DType() != DTypeFloat32 ||
		state.DType() != DTypeFloat32 {
		return dims, false
	}
	return dims, true
}

func sliceGatedDeltaStates(stateSeq *Array, dims gatedDeltaDims) []*Array {
	interior := make([]*Array, dims.T-1)
	for t := range interior {
		s := SliceStartStop(stateSeq,
			[]int32{int32(t), 0, 0, 0, 0},
			[]int32{int32(t) + 1, int32(dims.B), int32(dims.Hv), int32(dims.Dv), int32(dims.Dk)})
		interior[t] = Reshape(s, int32(dims.B), int32(dims.Hv), int32(dims.Dv), int32(dims.Dk))
	}
	return interior
}

func exactShape(value *Array, shape ...int) bool {
	dims := value.Dims()
	if len(dims) != len(shape) {
		return false
	}
	for i := range shape {
		if dims[i] != shape[i] {
			return false
		}
	}
	return true
}
