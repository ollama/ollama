package mlx

var (
	mamba2Scan = &gpuKernel{
		name:    "mamba2_scan",
		inputs:  []string{"hidden", "b_state", "c_state", "dt", "state_in", "a", "d", "dt_bias", "T"},
		outputs: []string{"y", "state_out"},
		metal:   gpuSource{source: mamba2ScanMetalKernelSource},
	}
	mamba2ScanSnapshot = &gpuKernel{
		name:    "mamba2_scan_snapshot",
		inputs:  []string{"hidden", "b_state", "c_state", "dt", "state_in", "a", "d", "dt_bias", "T", "split"},
		outputs: []string{"y", "states_out"},
		metal:   gpuSource{source: mamba2ScanSnapshotMetalKernelSource},
	}
)

const mamba2ScanMetalKernelSource = `
auto lane = thread_position_in_threadgroup.x;
auto d_idx = thread_position_in_grid.y;
auto bh_idx = thread_position_in_grid.z;
auto b_idx = bh_idx / H;
auto h_idx = bh_idx % H;
// B/C groups repeat across contiguous head blocks.
auto g_idx = h_idx / (H / G);
constexpr int n_per_t = S / 32;

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

const mamba2ScanSnapshotMetalKernelSource = `
auto lane = thread_position_in_threadgroup.x;
auto d_idx = thread_position_in_grid.y;
auto bh_idx = thread_position_in_grid.z;
auto b_idx = bh_idx / H;
auto h_idx = bh_idx % H;
// B/C groups repeat across contiguous head blocks.
auto g_idx = h_idx / (H / G);
constexpr int n_per_t = S / 32;

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

  if (t + 1 == split) {
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * lane + i;
      states_out[state_offset + s_idx] = state[i];
    }
  }

  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[bth * D + d_idx] = out + x_val * static_cast<float>(d[h_idx]);
  }
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * lane + i;
  constexpr int state_count = B * H * D * S;
  states_out[state_count + state_offset + s_idx] = state[i];
}
`

func mamba2ScanValidate(hidden, bState, cState, dt, state, a, d, dtBias *Array) (B, T, H, G, D, S int, ok bool) {
	if hidden == nil || bState == nil || cState == nil || dt == nil || state == nil || a == nil || d == nil || dtBias == nil {
		return 0, 0, 0, 0, 0, 0, false
	}
	if hidden.DType() != DTypeFloat32 || bState.DType() != DTypeFloat32 || cState.DType() != DTypeFloat32 || dt.DType() != DTypeFloat32 ||
		state.DType() != DTypeFloat32 || a.DType() != DTypeFloat32 || d.DType() != DTypeFloat32 || dtBias.DType() != DTypeFloat32 {
		return 0, 0, 0, 0, 0, 0, false
	}

	hd := hidden.Dims()
	bd := bState.Dims()
	cd := cState.Dims()
	dd := dt.Dims()
	sd := state.Dims()
	ad := a.Dims()
	wd := d.Dims()
	td := dtBias.Dims()
	if len(hd) != 4 || len(bd) != 4 || len(cd) != 4 || len(dd) != 3 || len(sd) != 4 || len(ad) != 1 || len(wd) != 1 || len(td) != 1 {
		return 0, 0, 0, 0, 0, 0, false
	}

	B, T, H, D = hd[0], hd[1], hd[2], hd[3]
	G = bd[2]
	S = bd[3]
	// S must be a multiple of the Metal simdgroup width (32): the kernel
	// partitions the S state slots across 32 lanes (n_per_t = S/32) and
	// reduces with simd_sum, so a non-multiple would drop tail slots.
	if B <= 0 || T <= 0 || H <= 0 || G <= 0 || D <= 0 || S <= 0 || H%G != 0 || S%32 != 0 {
		return 0, 0, 0, 0, 0, 0, false
	}
	if bd[0] != B || bd[1] != T || cd[0] != B || cd[1] != T || cd[2] != G || cd[3] != S {
		return 0, 0, 0, 0, 0, 0, false
	}
	if dd[0] != B || dd[1] != T || dd[2] != H {
		return 0, 0, 0, 0, 0, 0, false
	}
	if sd[0] != B || sd[1] != H || sd[2] != D || sd[3] != S {
		return 0, 0, 0, 0, 0, 0, false
	}
	if ad[0] != H || wd[0] != H || td[0] != H {
		return 0, 0, 0, 0, 0, 0, false
	}

	return B, T, H, G, D, S, true
}

func mamba2ScanTemplateArgs(B, H, G, D, S int) []gpuIntArg {
	return []gpuIntArg{
		{"B", B},
		{"H", H},
		{"G", G},
		{"D", D},
		{"S", S},
	}
}

func mamba2ScanMetalKernelApply(hidden, bState, cState, dt, state, a, d, dtBias *Array) (y, nextState *Array, ok bool) {
	B, T, H, G, D, S, ok := mamba2ScanValidate(hidden, bState, cState, dt, state, a, d, dtBias)
	if !ok {
		return nil, nil, false
	}

	outs, ok := mamba2Scan.applyMetal(gpuLaunch{
		ints: mamba2ScanTemplateArgs(B, H, G, D, S),
		outputs: []gpuOutputSpec{
			{"MAMBA2_SCAN_Y", []int32{int32(B), int32(T), int32(H), int32(D)}, DTypeFloat32},
			{"MAMBA2_SCAN_STATE", []int32{int32(B), int32(H), int32(D), int32(S)}, DTypeFloat32},
		},
		grid:        [3]int{32, D, B * H},
		threadGroup: [3]int{32, min(D, 4), 1},
		inputs:      []*Array{hidden, bState, cState, dt, state, a, d, dtBias, FromValue(T)},
	})
	if !ok {
		return nil, nil, false
	}
	return outs[0], outs[1], true
}

func mamba2ScanSnapshotMetalKernelApply(hidden, bState, cState, dt, state, a, d, dtBias *Array, split int) (y, nextState, snapshotState *Array, ok bool) {
	B, T, H, G, D, S, ok := mamba2ScanValidate(hidden, bState, cState, dt, state, a, d, dtBias)
	if !ok || split <= 0 || split >= T {
		return nil, nil, nil, false
	}

	outs, ok := mamba2ScanSnapshot.applyMetal(gpuLaunch{
		ints: mamba2ScanTemplateArgs(B, H, G, D, S),
		outputs: []gpuOutputSpec{
			{"MAMBA2_SCAN_SNAPSHOT_Y", []int32{int32(B), int32(T), int32(H), int32(D)}, DTypeFloat32},
			{"MAMBA2_SCAN_SNAPSHOT_STATES", []int32{2, int32(B), int32(H), int32(D), int32(S)}, DTypeFloat32},
		},
		grid:        [3]int{32, D, B * H},
		threadGroup: [3]int{32, min(D, 4), 1},
		inputs:      []*Array{hidden, bState, cState, dt, state, a, d, dtBias, FromValue(T), FromValue(split)},
	})
	if !ok {
		return nil, nil, nil, false
	}

	y = outs[0]
	states := outs[1]
	snapshotState = Squeeze(
		SliceStartStop(states, []int32{0, 0, 0, 0, 0}, []int32{1, int32(B), int32(H), int32(D), int32(S)}),
		0,
	)
	nextState = Squeeze(
		SliceStartStop(states, []int32{1, 0, 0, 0, 0}, []int32{2, int32(B), int32(H), int32(D), int32(S)}),
		0,
	)
	return y, nextState, snapshotState, true
}

// FastMamba2Scan runs the Mamba2 recurrent scan as a fused custom kernel when
// the backend and shapes are supported. It returns ok=false for unsupported
// backends or shapes so callers can use a backend-neutral fallback. Inputs must
// be float32 with shapes:
// hidden [B,T,H,D], bState/cState [B,T,G,S] where H%G==0, dt [B,T,H],
// state [B,H,D,S], and a/d/dtBias [H].
func FastMamba2Scan(hidden, bState, cState, dt, state, a, d, dtBias *Array) (y, nextState *Array, ok bool) {
	if y, nextState, ok := mamba2ScanMetalKernelApply(hidden, bState, cState, dt, state, a, d, dtBias); ok {
		return y, nextState, true
	}
	return nil, nil, false
}

// FastMamba2ScanWithSnapshot runs one fused scan and also returns the recurrent
// state after split tokens. It returns ok=false for unsupported backends or
// shapes so callers can use a backend-neutral fallback. The helper preserves
// the RecurrentCache snapshot contract without forcing the model to launch
// separate prefix/suffix scans.
func FastMamba2ScanWithSnapshot(hidden, bState, cState, dt, state, a, d, dtBias *Array, split int) (y, nextState, snapshotState *Array, ok bool) {
	if y, nextState, snapshotState, ok := mamba2ScanSnapshotMetalKernelApply(hidden, bState, cState, dt, state, a, d, dtBias, split); ok {
		return y, nextState, snapshotState, true
	}
	return nil, nil, nil, false
}
