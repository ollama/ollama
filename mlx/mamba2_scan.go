package mlx

import "math"

// The states variant records the state after every token but the last, which is
// what MTP verify needs to roll back to any position.
var (
	mamba2Scan = &gpuKernel{
		name:    "mamba2_scan",
		inputs:  []string{"hidden", "b_state", "c_state", "dt", "state_in", "a", "d", "dt_bias", "T"},
		outputs: []string{"y", "state_out"},
		metal: gpuSource{
			source: mamba2ScanMetalSource,
			header: "#define MAMBA2_STORE_INTERIOR(index, value)\n",
		},
		fallback: func(launch gpuLaunch) []*Array {
			in := launch.inputs
			y, end, _ := mamba2ScanGraph(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], false)
			return []*Array{y, end}
		},
	}
	mamba2ScanStates = &gpuKernel{
		name:    "mamba2_scan_states",
		inputs:  []string{"hidden", "b_state", "c_state", "dt", "state_in", "a", "d", "dt_bias", "T"},
		outputs: []string{"y", "state_out", "state_seq"},
		metal: gpuSource{
			source: mamba2ScanMetalSource,
			header: "#define MAMBA2_STORE_INTERIOR(index, value) state_seq[index] = value\n",
		},
		fallback: func(launch gpuLaunch) []*Array {
			in := launch.inputs
			y, end, interior := mamba2ScanGraph(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], true)
			for i, s := range interior {
				interior[i] = ExpandDims(s, 0)
			}
			return []*Array{y, end, Concatenate(interior, 0)}
		},
	}
)

const mamba2ScanMetalSource = `
auto lane = thread_position_in_threadgroup.x;
auto d_idx = thread_position_in_grid.y;
auto bh_idx = thread_position_in_grid.z;
auto b_idx = bh_idx / H;
auto h_idx = bh_idx % H;
// B/C groups repeat across contiguous head blocks.
auto g_idx = h_idx / (H / G);
constexpr int n_per_t = S / 32;
constexpr int state_count = B * H * D * S;

auto state_offset = ((b_idx * H + h_idx) * D + d_idx) * S;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * lane + i;
  state[i] = static_cast<float>(state_in[state_offset + s_idx]);
}

for (int t = 0; t < T; ++t) {
  auto bth = (b_idx * T + t) * H + h_idx;
  float dt_raw = static_cast<float>(dt[bth]) + static_cast<float>(dt_bias[h_idx]);
  float dt_val = log(1.0f + exp(dt_raw));
  float decay = exp(dt_val * static_cast<float>(a[h_idx]));
  float x_val = static_cast<float>(hidden[bth * D + d_idx]);

  float out = 0.0f;
  auto bs_base = ((b_idx * T + t) * G + g_idx) * S;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * lane + i;
    float b_val = static_cast<float>(b_state[bs_base + s_idx]);
    float c_val = static_cast<float>(c_state[bs_base + s_idx]);
    state[i] = state[i] * decay + x_val * (dt_val * b_val);
    out += state[i] * c_val;
  }

  if (t + 1 < T) {
    auto seq_offset = t * state_count + state_offset;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * lane + i;
      MAMBA2_STORE_INTERIOR(seq_offset + s_idx, state[i]);
    }
  }

  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[bth * D + d_idx] = out + x_val * static_cast<float>(d[h_idx]);
  }
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * lane + i;
  state_out[state_offset + s_idx] = state[i];
}
`

type mamba2ScanDims struct {
	B, T, H, G, D, S int
}

func resolveMamba2ScanDims(hidden, bState, cState, dt, state, a, d, dtBias *Array) (mamba2ScanDims, bool) {
	var dims mamba2ScanDims
	if hidden == nil || bState == nil || cState == nil || dt == nil || state == nil || a == nil || d == nil || dtBias == nil {
		return dims, false
	}
	if hidden.DType() != DTypeFloat32 || bState.DType() != DTypeFloat32 || cState.DType() != DTypeFloat32 || dt.DType() != DTypeFloat32 ||
		state.DType() != DTypeFloat32 || a.DType() != DTypeFloat32 || d.DType() != DTypeFloat32 || dtBias.DType() != DTypeFloat32 {
		return dims, false
	}

	hd, bd, cd := hidden.Dims(), bState.Dims(), cState.Dims()
	dd, sd := dt.Dims(), state.Dims()
	ad, wd, td := a.Dims(), d.Dims(), dtBias.Dims()
	if len(hd) != 4 || len(bd) != 4 || len(cd) != 4 || len(dd) != 3 || len(sd) != 4 || len(ad) != 1 || len(wd) != 1 || len(td) != 1 {
		return dims, false
	}

	dims.B, dims.T, dims.H, dims.D = hd[0], hd[1], hd[2], hd[3]
	dims.G, dims.S = bd[2], bd[3]
	// S must be a multiple of the Metal simdgroup width (32): the kernel
	// partitions the S state slots across 32 lanes (n_per_t = S/32) and
	// reduces with simd_sum, so a non-multiple would drop tail slots.
	if dims.B <= 0 || dims.T <= 0 || dims.H <= 0 || dims.G <= 0 || dims.D <= 0 || dims.S <= 0 || dims.H%dims.G != 0 || dims.S%32 != 0 {
		return dims, false
	}
	if bd[0] != dims.B || bd[1] != dims.T || cd[0] != dims.B || cd[1] != dims.T || cd[2] != dims.G || cd[3] != dims.S {
		return dims, false
	}
	if dd[0] != dims.B || dd[1] != dims.T || dd[2] != dims.H {
		return dims, false
	}
	if sd[0] != dims.B || sd[1] != dims.H || sd[2] != dims.D || sd[3] != dims.S {
		return dims, false
	}
	if ad[0] != dims.H || wd[0] != dims.H || td[0] != dims.H {
		return dims, false
	}
	return dims, true
}

func repeatMambaGroups(x *Array, repeats int32) *Array {
	if repeats <= 1 {
		return x
	}
	// Mamba2 maps each B/C group to a contiguous block of heads.
	dims := x.Dims()
	x = ExpandDims(x, 3)
	x = Tile(x, []int32{1, 1, 1, repeats, 1})
	return Reshape(x, int32(dims[0]), int32(dims[1]), int32(dims[2])*repeats, int32(dims[3]))
}

func sliceMambaTime(x *Array, t int32) *Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[1], stop[1] = t, t+1
	return Squeeze(SliceStartStop(x, start, stop), 1)
}

func mamba2ScanGraph(hidden, bState, cState, dt, state, a, d, dtBias *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	B := int32(hidden.Dim(0))
	T := int32(hidden.Dim(1))
	H := int32(hidden.Dim(2))
	G := int32(bState.Dim(2))

	if G > 0 && H%G == 0 {
		bState = repeatMambaGroups(bState, H/G)
		cState = repeatMambaGroups(cState, H/G)
	}

	a4 := Reshape(a, 1, H, 1, 1)
	d3 := Reshape(d, 1, H, 1)
	bias2 := Reshape(dtBias, 1, H)

	outs := make([]*Array, 0, T)
	for t := range T {
		xt := sliceMambaTime(hidden, t)
		bt := sliceMambaTime(bState, t)
		ct := sliceMambaTime(cState, t)

		dtt := Add(sliceMambaTime(dt, t), bias2)
		dtt = Log(AddScalar(Exp(dtt), 1))
		dA := Exp(Mul(Reshape(dtt, B, H, 1, 1), a4))
		dB := Mul(Reshape(dtt, B, H, 1), bt)

		state = Add(Mul(state, dA), Mul(ExpandDims(xt, -1), ExpandDims(dB, 2)))
		yt := Sum(Mul(state, ExpandDims(ct, 2)), 3, false)
		outs = append(outs, Add(yt, Mul(xt, d3)))

		if captureAll && t+1 < T {
			interior = append(interior, state)
		}
	}
	return Stack(outs, 1), state, interior
}

// Mamba2Scan runs the Mamba2 recurrent scan. Inputs must be float32 with
// shapes hidden [B, T, H, D], bState/cState [B, T, G, S] where H%G == 0,
// dt [B, T, H], state [B, H, D, S], and a/d/dtBias [H]. captureAll also
// returns the state after every token but the last.
//
// mask, when non-nil, is a [B, T] bool marking real (true) vs padded
// positions; padded positions are identity steps.
func Mamba2Scan(hidden, bState, cState, dt, state, a, d, dtBias, mask *Array, captureAll bool) (y, nextState *Array, interior []*Array) {
	if mask != nil {
		B, T := int32(mask.Dim(0)), int32(mask.Dim(1))
		zero := FromValue(float32(0)).AsType(hidden.DType())
		mask4 := Reshape(mask, B, T, 1, 1)
		hidden = Where(mask4, hidden, zero)
		cState = Where(mask4, cState, zero)
		negInf := FromValue(float32(math.Inf(-1))).AsType(dt.DType())
		dt = Where(Reshape(mask, B, T, 1), dt, negInf)
	}

	dims, ok := resolveMamba2ScanDims(hidden, bState, cState, dt, state, a, d, dtBias)
	if !ok {
		return mamba2ScanGraph(hidden, bState, cState, dt, state, a, d, dtBias, captureAll)
	}

	kernel := mamba2Scan
	outputs := []gpuOutputSpec{
		{"MAMBA2_SCAN_Y", []int32{int32(dims.B), int32(dims.T), int32(dims.H), int32(dims.D)}, DTypeFloat32},
		{"MAMBA2_SCAN_STATE", []int32{int32(dims.B), int32(dims.H), int32(dims.D), int32(dims.S)}, DTypeFloat32},
	}
	useAllStates := captureAll && dims.T > 1
	if useAllStates {
		kernel = mamba2ScanStates
		outputs = append(outputs, gpuOutputSpec{
			"MAMBA2_SCAN_STATE_SEQ",
			[]int32{int32(dims.T - 1), int32(dims.B), int32(dims.H), int32(dims.D), int32(dims.S)},
			DTypeFloat32,
		})
	}

	outs := kernel.run(gpuLaunch{
		ints: []gpuIntArg{
			{"B", dims.B},
			{"H", dims.H},
			{"G", dims.G},
			{"D", dims.D},
			{"S", dims.S},
		},
		outputs:     outputs,
		grid:        [3]int{32, dims.D, dims.B * dims.H},
		threadGroup: [3]int{32, min(dims.D, 4), 1},
		inputs:      []*Array{hidden, bState, cState, dt, state, a, d, dtBias, FromValue(dims.T)},
	})
	if useAllStates {
		interior = sliceMamba2ScanStates(outs[2], dims)
	}
	return outs[0], outs[1], interior
}

func sliceMamba2ScanStates(stateSeq *Array, dims mamba2ScanDims) []*Array {
	interior := make([]*Array, dims.T-1)
	for t := range interior {
		s := SliceStartStop(stateSeq,
			[]int32{int32(t), 0, 0, 0, 0},
			[]int32{int32(t) + 1, int32(dims.B), int32(dims.H), int32(dims.D), int32(dims.S)})
		interior[t] = Reshape(s, int32(dims.B), int32(dims.H), int32(dims.D), int32(dims.S))
	}
	return interior
}
