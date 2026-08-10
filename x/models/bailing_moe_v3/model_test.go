package bailing_moe_v3

import (
	"math"
	"sort"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func requireMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}
}

func TestParseConfigAndLayerSchedule(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"model_type":"bailing_hybrid",
		"hidden_size":1536,
		"num_hidden_layers":24,
		"num_attention_heads":16,
		"head_dim":128,
		"layer_group_size":4,
		"no_kda_lora":true,
		"qk_head_dim":192,
		"qk_nope_head_dim":128,
		"qk_rope_head_dim":64,
		"rope_interleave":true,
		"gated_attention_proj_granularity_type":"head_wise",
		"num_experts":128,
		"n_group":8
	}`))
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

func TestSupportsGatherQMM(t *testing.T) {
	tests := []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "affine", bits: 8, want: false},
		{mode: "", bits: 8, want: false},
	}
	for _, tt := range tests {
		if got := supportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("supportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
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
