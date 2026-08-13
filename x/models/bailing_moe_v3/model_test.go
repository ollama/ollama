package bailing_moe_v3

import (
	"encoding/json"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

func requireMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}
}

func validConfigData(t *testing.T, overrides map[string]any) []byte {
	t.Helper()
	cfg := map[string]any{
		"model_type":                            "bailing_hybrid",
		"hidden_size":                           1536,
		"num_hidden_layers":                     24,
		"num_attention_heads":                   16,
		"head_dim":                              128,
		"layer_group_size":                      4,
		"no_kda_lora":                           true,
		"qk_head_dim":                           192,
		"qk_nope_head_dim":                      128,
		"qk_rope_head_dim":                      64,
		"rope_interleave":                       true,
		"gated_attention_proj_granularity_type": "head_wise",
		"num_experts":                           128,
		"num_experts_per_tok":                   8,
		"n_group":                               8,
		"topk_group":                            4,
	}
	for key, value := range overrides {
		if value == nil {
			delete(cfg, key)
		} else {
			cfg[key] = value
		}
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestParseConfigAndLayerSchedule(t *testing.T) {
	cfg, err := parseConfig(validConfigData(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ShortConvKernelSize != 4 || cfg.RMSNormEps != 1e-6 || cfg.RoutedScalingFactor != 1 {
		t.Fatalf("defaults not applied: %+v", cfg)
	}

	var got []int
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		if isMLALayer(&cfg, i) {
			got = append(got, int(i))
		}
	}
	want := []int{3, 7, 11, 15, 19, 23}
	if len(got) != len(want) {
		t.Fatalf("MLA layers = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MLA layers = %v, want %v", got, want)
		}
	}
}

func TestParseConfigRejectsInvalidRouting(t *testing.T) {
	tests := []struct {
		name      string
		overrides map[string]any
		want      string
	}{
		{name: "missing n_group", overrides: map[string]any{"n_group": nil}, want: "invalid n_group: 0"},
		{name: "uneven groups", overrides: map[string]any{"n_group": 7}, want: "must be divisible"},
		{name: "one expert per group", overrides: map[string]any{"n_group": 128}, want: "at least 2"},
		{name: "missing topk_group", overrides: map[string]any{"topk_group": nil}, want: "invalid topk_group: 0"},
		{name: "too many groups", overrides: map[string]any{"topk_group": 9}, want: "must not exceed n_group"},
		{name: "missing experts per token", overrides: map[string]any{"num_experts_per_tok": nil}, want: "invalid num_experts_per_tok: 0"},
		{name: "too many experts per token", overrides: map[string]any{"num_experts_per_tok": 65}, want: "exceeds the 64 candidates"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseConfig(validConfigData(t, tt.overrides))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseConfig() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestParseConfigRejectsRopeScaling(t *testing.T) {
	if _, err := parseConfig(validConfigData(t, map[string]any{"rope_scaling": json.RawMessage("null")})); err != nil {
		t.Fatalf("parseConfig() rejected null rope_scaling: %v", err)
	}

	_, err := parseConfig(validConfigData(t, map[string]any{
		"rope_scaling": map[string]any{"rope_type": "yarn", "factor": 4},
	}))
	if err == nil || !strings.Contains(err.Error(), "rope_scaling is not supported") {
		t.Fatalf("parseConfig() error = %v, want unsupported rope_scaling", err)
	}
}

func TestSupportsGatherQMM(t *testing.T) {
	tests := []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "affine", bits: 8, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "mxfp4", bits: 8, want: false},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "", bits: 4, want: false},
		{mode: "", bits: 8, want: false},
	}
	for _, tt := range tests {
		if got := supportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("supportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

func TestLoadStackedProjectionQuantizationBoundary(t *testing.T) {
	requireMLX(t)
	const base = "model.layers.0.mlp.experts.gate_proj"
	key := base + ".weight"
	weight := mlx.FromValues([]uint32{0}, 1, 1, 1)
	scale := mlx.FromValues([]uint8{127}, 1, 1, 1)
	affineScale := mlx.FromValues([]float32{0.25}, 1, 1, 1)
	affineBias := mlx.FromValues([]float32{-2}, 1, 1, 1)

	tests := []struct {
		quantType string
		scale     *mlx.Array
		bias      *mlx.Array
		wantMode  string
		wantBits  int
		wantErr   bool
	}{
		{quantType: "mxfp8", scale: scale, wantMode: "mxfp8", wantBits: 8},
		{quantType: "mxfp4", scale: scale, wantMode: "mxfp4", wantBits: 4},
		{quantType: "int4", scale: affineScale, bias: affineBias, wantMode: "affine", wantBits: 4},
		{quantType: "int8", scale: affineScale, bias: affineBias, wantMode: "affine", wantBits: 8},
	}
	for _, tt := range tests {
		t.Run(tt.quantType, func(t *testing.T) {
			cfg := &Config{TensorQuant: map[string]*model.TensorQuantInfo{
				key: {QuantType: tt.quantType, GroupSize: 32},
			}}
			tensors := map[string]*mlx.Array{key: weight, key + "_scale": tt.scale}
			if tt.bias != nil {
				tensors[key+"_qbias"] = tt.bias
			}
			got, err := loadStackedProjection(tensors, cfg, base)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "unsupported quantization") {
					t.Fatalf("loadStackedProjection() error = %v, want unsupported quantization", err)
				}
				return
			}
			if err != nil || got == nil || got.Scales != tt.scale || got.Biases != tt.bias || got.Mode != tt.wantMode || got.Bits != tt.wantBits {
				t.Fatalf("loadStackedProjection() = %+v, %v; want mode=%q bits=%d GatherQMM weights", got, err, tt.wantMode, tt.wantBits)
			}
		})
	}

	cfg := &Config{TensorQuant: map[string]*model.TensorQuantInfo{
		key: {QuantType: "mxfp8", GroupSize: 32},
	}}
	if _, err := loadStackedProjection(map[string]*mlx.Array{key: weight}, cfg, base); err == nil || !strings.Contains(err.Error(), "missing its scale tensor") {
		t.Fatalf("loadStackedProjection() error = %v, want missing scale rejection", err)
	}

	cfg = &Config{TensorQuant: map[string]*model.TensorQuantInfo{
		key: {QuantType: "int4", GroupSize: 32},
	}}
	if _, err := loadStackedProjection(map[string]*mlx.Array{
		key: weight, key + "_scale": affineScale,
	}, cfg, base); err == nil || !strings.Contains(err.Error(), "missing its qbias tensor") {
		t.Fatalf("loadStackedProjection() error = %v, want missing qbias rejection", err)
	}
}

func TestGatherQMMCompressedInt4Values(t *testing.T) {
	requireMLX(t)

	pack := func(values []int8) []uint32 {
		if len(values)%8 != 0 {
			t.Fatalf("pack input length %d is not divisible by 8", len(values))
		}
		words := make([]uint32, len(values)/8)
		for i, value := range values {
			if value < -8 || value > 7 {
				t.Fatalf("INT4 value %d out of range", value)
			}
			words[i/8] |= uint32(uint8(value+8)) << (4 * (i % 8))
		}
		return words
	}

	expert0 := make([]int8, 32)
	expert1 := make([]int8, 32)
	for i := range expert0 {
		expert0[i] = int8(i%16 - 8)
		expert1[i] = 1
	}
	packed := append(pack(expert0), pack(expert1)...)
	weight := mlx.FromValues(packed, 2, 1, 4)
	scales := mlx.FromValues([]float32{0.5, 2}, 2, 1, 1)
	biases := mlx.FromValues([]float32{-4, -16}, 2, 1, 1)
	xValues := make([]float32, 2*32)
	for i := range xValues {
		xValues[i] = 1
	}
	x := mlx.FromValues(xValues, 2, 1, 1, 32)
	rhsIndices := mlx.FromValues([]int32{0, 1}, 2, 1)

	got := mlx.GatherQMM(x, weight, scales, biases, nil, rhsIndices, true, 32, 4, "affine", false)
	got = got.AsType(mlx.DTypeFloat32)
	mlx.Eval(got)
	values := got.Floats()
	want := []float32{-8, 64}
	if len(values) != len(want) {
		t.Fatalf("GatherQMM output shape %v values=%v, want two values", got.Dims(), values)
	}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Errorf("GatherQMM output[%d] = %g, want %g", i, values[i], want[i])
		}
	}
}

func TestKDASegmentedCacheMatchesSingleScan(t *testing.T) {
	requireMLX(t)
	const B, T, H, D = 1, 5, 2, 3
	values := func(n int, offset, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = offset + float32((i*7)%19-9)*scale
		}
		return v
	}
	shape := []int{B, T, H, D}
	q := mlx.FromValues(values(B*T*H*D, 0.1, 0.03), shape...)
	k := mlx.FromValues(values(B*T*H*D, -0.2, 0.02), shape...)
	v := mlx.FromValues(values(B*T*H*D, 0.05, 0.04), shape...)
	a := mlx.FromValues(values(B*T*H*D, -0.1, 0.01), shape...)
	beta := mlx.FromValues(values(B*T*H, 0.3, 0.05), B, T, H)
	aExp := mlx.FromValues([]float32{0.8, 1.1}, H)
	dtBias := mlx.FromValues(values(H*D, 0.2, 0.02), H*D)
	initial := mlx.Zeros(mlx.DTypeFloat32, B, H, D, D)

	wantOut, wantState := kdaScan(q, k, v, a, beta, aExp, dtBias, initial, true, -5)
	gotOut, states := runKDASegments(q, k, v, a, beta, aExp, dtBias, initial, []int{2, 4}, true, -5)
	gotState := states[len(states)-1]
	wantOut = wantOut.AsType(mlx.DTypeFloat32)
	gotOut = gotOut.AsType(mlx.DTypeFloat32)
	mlx.Eval(wantOut, gotOut, wantState, gotState)

	assertClose := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
		}
		for i := range want {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
				t.Fatalf("%s[%d] = %g, want %g (diff %g)", label, i, got[i], want[i], diff)
			}
		}
	}
	assertClose("output", gotOut.Floats(), wantOut.Floats())
	assertClose("state", gotState.Floats(), wantState.Floats())
}

func TestGroupedRouterBiasOnlyAffectsSelection(t *testing.T) {
	requireMLX(t)
	gateWeight := mlx.FromValues([]float32{0, .2, .4, .6, .8, 1, 1.2, 1.4}, 8, 1)
	bias := mlx.FromValues([]float32{10, 0, 0, 0, 0, 0, 0, 0}, 8)
	router := &Router{Gate: nn.NewLinear(gateWeight, nil), ExpertBias: bias}
	cfg := &Config{
		HiddenSize: 1, NumExperts: 8, NGroup: 4, TopKGroup: 2,
		NumExpertsPerTok: 2, NormTopKProb: true, RoutedScalingFactor: 2.5,
	}
	indices, weights := router.Forward(mlx.FromValues([]float32{1}, 1, 1, 1), cfg)
	mlx.Eval(indices, weights)
	ids := indices.Ints()
	gotIDs := append([]int(nil), ids...)
	sort.Ints(gotIDs)
	if gotIDs[0] != 0 || gotIDs[1] != 7 {
		t.Fatalf("selected experts = %v, want [0 7]", gotIDs)
	}

	weightByID := map[int]float32{}
	for i, id := range ids {
		weightByID[id] = weights.Floats()[i]
	}
	s0 := float32(1 / (1 + math.Exp(0)))
	s7 := float32(1 / (1 + math.Exp(-1.4)))
	denom := s0 + s7
	want0, want7 := 2.5*s0/denom, 2.5*s7/denom
	if math.Abs(float64(weightByID[0]-want0)) > 1e-6 || math.Abs(float64(weightByID[7]-want7)) > 1e-6 {
		t.Fatalf("weights = %v, want expert 0=%g expert 7=%g", weightByID, want0, want7)
	}
}

func TestInterleavedToHalf(t *testing.T) {
	requireMLX(t)
	x := mlx.FromValues([]float32{0, 1, 2, 3, 4, 5, 6, 7}, 1, 1, 1, 8)
	got := interleavedToHalf(x)
	mlx.Eval(got)
	want := []float32{0, 2, 4, 6, 1, 3, 5, 7}
	for i, value := range got.Floats() {
		if value != want[i] {
			t.Fatalf("interleavedToHalf[%d] = %g, want %g", i, value, want[i])
		}
	}
}

func TestFuseGateUpDropsOriginals(t *testing.T) {
	requireMLX(t)
	newLinear := func(dtype mlx.DType) *nn.Linear {
		w := mlx.Zeros(dtype, 4, 8)
		mlx.Eval(w)
		return &nn.Linear{Weight: w}
	}

	m := &DenseMLP{
		GateProj: newLinear(mlx.DTypeBFloat16),
		UpProj:   newLinear(mlx.DTypeBFloat16),
		DownProj: newLinear(mlx.DTypeBFloat16),
	}
	m.fuseGateUp()
	if m.GateUpProj == nil {
		t.Fatal("expected uniform projections to fuse")
	}
	if m.GateProj != nil || m.UpProj != nil {
		t.Fatal("expected fused originals to be dropped so they are not pinned twice")
	}

	// Mixed representations do not fuse; the originals must survive for the
	// per-projection fallback path.
	mixed := &DenseMLP{
		GateProj: newLinear(mlx.DTypeBFloat16),
		UpProj:   newLinear(mlx.DTypeFloat32),
		DownProj: newLinear(mlx.DTypeBFloat16),
	}
	mixed.fuseGateUp()
	if mixed.GateUpProj != nil {
		t.Fatal("expected mixed-dtype projections not to fuse")
	}
	if mixed.GateProj == nil || mixed.UpProj == nil {
		t.Fatal("unfused originals must be retained")
	}
}

// TestAbsorbedMLAMatchesExpanded proves the absorbed attention path computes
// the same output as the expanded reference: caching the 576-dim latent and
// folding kv_b_proj into the query/output paths is a pure reassociation,
// (q W_k^T) . c == q . (W_k c), so scores and outputs must match to f32
// tolerance.
func TestAbsorbedMLAMatchesExpanded(t *testing.T) {
	requireMLX(t)
	const (
		B, L, H  = 1, 7, 4
		nopeDim  = 16
		ropeDim  = 8
		vDim     = 12
		loraRank = 24
		headDim  = nopeDim + vDim
	)
	values := func(n int, offset, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = offset + float32((i*13)%23-11)*scale
		}
		return v
	}

	wKVB := mlx.FromValues(values(H*headDim*loraRank, 0.01, 0.015), H*headDim, loraRank)
	qNope := mlx.FromValues(values(B*H*L*nopeDim, 0.05, 0.02), B, H, L, nopeDim)
	qPE := mlx.FromValues(values(B*H*L*ropeDim, -0.02, 0.03), B, H, L, ropeDim)
	latent := mlx.FromValues(values(B*L*loraRank, 0.03, 0.02), B, L, loraRank)
	kPE := mlx.FromValues(values(B*L*ropeDim, 0.02, 0.025), B, 1, L, ropeDim)
	scale := float32(1 / math.Sqrt(float64(nopeDim+ropeDim)))

	// Expanded reference: kv_b_proj -> per-head kNope/values, full SDPA.
	kvExpanded := mlx.Reshape(latent, B*L, loraRank).Matmul(wKVB.Transpose(1, 0))
	kvExpanded = mlx.Transpose(mlx.Reshape(kvExpanded, B, L, H, headDim), 0, 2, 1, 3)
	kNope := mlx.SliceStartStop(kvExpanded, []int32{0, 0, 0, 0}, []int32{B, H, L, nopeDim})
	vals := mlx.SliceStartStop(kvExpanded, []int32{0, 0, 0, nopeDim}, []int32{B, H, L, headDim})
	keysRef := mlx.Concatenate([]*mlx.Array{kNope, mlx.Tile(kPE, []int32{1, H, 1, 1})}, 3)
	queriesRef := mlx.Concatenate([]*mlx.Array{qNope, qPE}, 3)
	scoresRef := mlx.MulScalar(queriesRef.Matmul(keysRef.Transpose(0, 1, 3, 2)), scale)
	probsRef := mlx.SoftmaxAxis(scoresRef, -1, true)
	outRef := probsRef.Matmul(vals)

	// Absorbed path: EmbedQ/UnembedOut factors, MQA over the shared latent.
	w3 := mlx.Reshape(wKVB, H, headDim, loraRank)
	wk := mlx.SliceStartStop(w3, []int32{0, 0, 0}, []int32{H, nopeDim, loraRank})
	wv := mlx.SliceStartStop(w3, []int32{0, nopeDim, 0}, []int32{H, headDim, loraRank})
	embedQ := nn.NewMultiLinear(mlx.Contiguous(mlx.Transpose(wk, 0, 2, 1), false))
	unembedOut := nn.NewMultiLinear(mlx.Contiguous(wv, false))

	qLatent := embedQ.Forward(qNope)
	keysAbs := mlx.Concatenate([]*mlx.Array{mlx.Reshape(latent, B, 1, L, loraRank), kPE}, 3)
	queriesAbs := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)
	scoresAbs := mlx.MulScalar(queriesAbs.Matmul(keysAbs.Transpose(0, 1, 3, 2)), scale)
	probsAbs := mlx.SoftmaxAxis(scoresAbs, -1, true)
	latentOut := probsAbs.Matmul(mlx.Reshape(latent, B, 1, L, loraRank))
	outAbs := unembedOut.Forward(latentOut)

	refF := outRef.AsType(mlx.DTypeFloat32)
	absF := outAbs.AsType(mlx.DTypeFloat32)
	mlx.Eval(refF, absF)
	r, a := refF.Floats(), absF.Floats()
	if len(r) != len(a) || len(r) != B*H*L*vDim {
		t.Fatalf("output sizes: ref %d abs %d want %d", len(r), len(a), B*H*L*vDim)
	}
	for i := range r {
		if diff := math.Abs(float64(r[i] - a[i])); diff > 1e-4 { // CUDA f32 matmul rounding (TF32) needs looser tolerance than Metal
			t.Fatalf("output[%d] = %g (absorbed) vs %g (expanded), diff %g", i, a[i], r[i], diff)
		}
	}
}

// TestKDAChunkedMatchesStepwise drives kdaScan with a full-chunk-plus-tail
// sequence and checks it against the exact per-token reference (the compiled
// single-step graph chained token by token).
func TestKDAChunkedMatchesStepwise(t *testing.T) {
	requireMLX(t)
	const (
		B, T, H, D = 1, 71, 4, 32
		lowerBound = float32(-8)
	)
	values := func(n int, offset, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = offset + float32((i*29)%37-18)*scale
		}
		return v
	}
	q := mlx.FromValues(values(B*T*H*D, 0.03, 0.021), B, T, H, D)
	k := mlx.FromValues(values(B*T*H*D, -0.01, 0.017), B, T, H, D)
	v := mlx.FromValues(values(B*T*H*D, 0.05, 0.013), B, T, H, D)
	a := mlx.FromValues(values(B*T*H*D, -0.4, 0.05), B, T, H, D)
	beta := mlx.FromValues(values(B*T*H, 0.2, 0.11), B, T, H)
	aExp := mlx.FromValues(values(H, 0.9, 0.05), H)
	dtBias := mlx.FromValues(values(H*D, 0.1, 0.03), H, D)
	state0 := mlx.FromValues(values(H*D*D, 0.0, 0.002), B, H, D, D)

	for _, safeGate := range []bool{false, true} {
		got, gotState := kdaScan(q, k, v, a, beta, aExp, dtBias, state0, safeGate, lowerBound)

		refState := state0
		refOuts := make([]*mlx.Array, 0, T)
		for i := int32(0); i < T; i++ {
			r := timeRange{start: i, end: i + 1}
			var y *mlx.Array
			y, refState = kdaScan(
				sliceTime(q, r), sliceTime(k, r), sliceTime(v, r),
				sliceTime(a, r), sliceTime(beta, r), aExp, dtBias, refState,
				safeGate, lowerBound,
			)
			refOuts = append(refOuts, y)
		}
		ref := mlx.Concatenate(refOuts, 1)

		gf := got.AsType(mlx.DTypeFloat32)
		rf := ref.AsType(mlx.DTypeFloat32)
		sf := gotState.AsType(mlx.DTypeFloat32)
		rsf := refState.AsType(mlx.DTypeFloat32)
		mlx.Eval(gf, rf, sf, rsf)

		check := func(name string, av, bv []float32) {
			if len(av) != len(bv) {
				t.Fatalf("%s size %d vs %d", name, len(av), len(bv))
			}
			var maxd float64
			var maxi int
			var sum float64
			for i := range av {
				diff := math.Abs(float64(av[i] - bv[i]))
				sum += diff
				if diff > maxd {
					maxd, maxi = diff, i
				}
			}
			t.Logf("safeGate=%v %s: max=%g at %d (%g vs %g) mean=%g",
				safeGate, name, maxd, maxi, av[maxi], bv[maxi], sum/float64(len(av)))
			if maxd > 5e-2 { // CUDA TF32 matmuls dominate the tolerance; exactness proven with MLX_ENABLE_TF32=0 (max 1e-5)
				t.Fatalf("safeGate=%v %s diverges: max diff %g", safeGate, name, maxd)
			}
		}
		check("y", gf.Floats(), rf.Floats())
		check("state", sf.Floats(), rsf.Floats())
	}
}
