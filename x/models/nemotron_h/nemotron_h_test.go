package nemotron_h

import (
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestParseConfigNestedWrapper(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["NemotronH_Nano_VL_V2"],
		"model_type": "NemotronH_Nano_VL_V2",
		"llm_config": {
			"model_type": "nemotron_h",
			"hidden_size": 2688,
			"num_hidden_layers": 4,
			"hybrid_override_pattern": "M*E-",
			"num_attention_heads": 32,
			"num_key_value_heads": 2,
			"head_dim": 84,
			"mamba_num_heads": 64,
			"mamba_head_dim": 64,
			"conv_kernel": 4,
			"ssm_state_size": 128,
			"n_groups": 8,
			"n_routed_experts": 128,
			"num_experts_per_tok": 6,
			"routed_scaling_factor": 2.5,
			"layer_norm_epsilon": 0.00001
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig returned error: %v", err)
	}

	if got, want := cfg.ModelType, "nemotron_h"; got != want {
		t.Fatalf("ModelType = %q, want %q", got, want)
	}
	if got, want := string(cfg.LayerTypes), "M*E-"; got != want {
		t.Fatalf("LayerTypes = %q, want %q", got, want)
	}
	if got, want := cfg.NGroups, int32(8); got != want {
		t.Fatalf("NGroups = %d, want %d", got, want)
	}
	if got, want := cfg.RoutedScalingFactor, float32(2.5); got != want {
		t.Fatalf("RoutedScalingFactor = %v, want %v", got, want)
	}
}

func TestParseConfigRejectsBadPattern(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"hidden_size": 4,
		"num_hidden_layers": 2,
		"hybrid_override_pattern": "M",
		"num_attention_heads": 2
	}`))
	if err == nil || !strings.Contains(err.Error(), "hybrid_override_pattern length") {
		t.Fatalf("parseConfig error = %v, want hybrid pattern length error", err)
	}
}

func TestParseConfigRejectsUndividedExpertGroups(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"hybrid_override_pattern": "E",
		"num_attention_heads": 2,
		"n_routed_experts": 10,
		"n_group": 3,
		"num_experts_per_tok": 2
	}`))
	if err == nil || !strings.Contains(err.Error(), "must be divisible by n_group") {
		t.Fatalf("parseConfig error = %v, want expert-group divisibility error", err)
	}
}

func TestNewCachesLayout(t *testing.T) {
	m := &Model{
		Config: &Config{
			ConvKernel:    4,
			MambaNumHeads: 2,
			MambaHeadDim:  2,
			SSMStateSize:  3,
			NGroups:       1,
		},
		Layers: []*Layer{
			{Type: 'M'},
			{Type: '*'},
			{Type: 'E'},
			{Type: '-'},
		},
	}

	caches := m.NewCaches()
	if got, want := len(caches), 4; got != want {
		t.Fatalf("len(NewCaches()) = %d, want %d", got, want)
	}
	if _, ok := caches[0].(*cache.RecurrentCache); !ok {
		t.Fatalf("caches[0] = %T, want *cache.RecurrentCache", caches[0])
	}
	if _, ok := caches[1].(*cache.KVCache); !ok {
		t.Fatalf("caches[1] = %T, want *cache.KVCache", caches[1])
	}
	if caches[2] != nil || caches[3] != nil {
		t.Fatalf("MLP-only caches = %T/%T, want nil/nil", caches[2], caches[3])
	}
}

func TestSupportsGatherQMM(t *testing.T) {
	for _, tt := range []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "unknown", bits: 4, want: false},
	} {
		if got := supportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("supportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

func TestApplyExpertWeightGlobalScale(t *testing.T) {
	mlxtest.Setup(t)

	weight := mlx.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}, 2, 2, 2)
	scale := mlx.FromValues([]float32{2, 3}, 2)

	got := applyExpertWeightGlobalScale(weight, scale)
	mlx.Eval(got)

	assertAllClose(t, "scaled expert weight", got.Floats(), []float32{
		2, 4,
		6, 8,
		15, 18,
		21, 24,
	}, 1e-5)
}

func TestMamba2LoopScanPaddedRowDoesNotAdvanceState(t *testing.T) {
	mlxtest.Setup(t)

	const (
		B = int32(2)
		L = int32(3)
		H = int32(2)
		D = int32(2)
		S = int32(32)
	)
	cfg := &Config{
		MambaNumHeads: H,
		MambaHeadDim:  D,
		NGroups:       H,
		SSMStateSize:  S,
	}

	hidden := testValues(0.1, int(B), int(L), int(H), int(D))
	bState := testValues(0.2, int(B), int(L), int(H), int(S))
	cState := testValues(0.3, int(B), int(L), int(H), int(S))
	dt := testValues(-0.4, int(B), int(L), int(H))
	state := testValues(0.5, int(B), int(H), int(D), int(S))
	a := mlx.MulScalar(ones(mlx.DTypeFloat32, 1, int(H), 1, 1), -0.25)
	d := mlx.MulScalar(ones(mlx.DTypeFloat32, 1, int(H), 1), 0.1)
	dtBias := mlx.Zeros(mlx.DTypeFloat32, int(B), int(H))

	fullBatch := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, int(B), int(L)),
		SeqOffsets:   []int32{0, 0},
		SeqQueryLens: []int32{L, 1},
	}
	mask := nn.PaddingMask(fullBatch, L)
	if mask == nil {
		t.Fatal("expected padding mask")
	}

	gotY, gotState, _ := mamba2LoopScan(hidden, bState, cState, dt, state, a, d, dtBias, mask, B, L, cfg, nil)

	refY0, refState0, _ := mamba2LoopScan(
		sliceBatch(hidden, 0, 1, L),
		sliceBatch(bState, 0, 1, L),
		sliceBatch(cState, 0, 1, L),
		sliceBatch(dt, 0, 1, L),
		sliceStateBatch(state, 0),
		a,
		d,
		mlx.Zeros(mlx.DTypeFloat32, 1, int(H)),
		nil,
		1,
		L,
		cfg,
		nil,
	)
	refY1, refState1, _ := mamba2LoopScan(
		sliceBatch(hidden, 1, 2, 1),
		sliceBatch(bState, 1, 2, 1),
		sliceBatch(cState, 1, 2, 1),
		sliceBatch(dt, 1, 2, 1),
		sliceStateBatch(state, 1),
		a,
		d,
		mlx.Zeros(mlx.DTypeFloat32, 1, int(H)),
		nil,
		1,
		1,
		cfg,
		nil,
	)

	gotY0 := sliceBatch(gotY, 0, 1, L)
	gotY1Real := sliceBatch(gotY, 1, 2, 1)
	gotY1Pad := mlx.SliceStartStop(gotY, []int32{1, 1, 0, 0}, []int32{2, L, H, D})
	gotState0 := sliceStateBatch(gotState, 0)
	gotState1 := sliceStateBatch(gotState, 1)

	mlx.Eval(gotY0, gotY1Real, gotY1Pad, gotState0, gotState1, refY0, refY1, refState0, refState1)
	assertAllClose(t, "row0 output", gotY0.Floats(), refY0.Floats(), 1e-5)
	assertAllClose(t, "row0 state", gotState0.Floats(), refState0.Floats(), 1e-5)
	assertAllClose(t, "row1 real output", gotY1Real.Floats(), refY1.Floats(), 1e-5)
	assertAllClose(t, "row1 state", gotState1.Floats(), refState1.Floats(), 1e-5)
	assertAllClose(t, "row1 padded output", gotY1Pad.Floats(), make([]float32, gotY1Pad.Size()), 1e-5)
}

func TestMambaConvSharedHelperMatchesExplicitPath(t *testing.T) {
	mlxtest.Setup(t)

	const (
		B = int32(2)
		L = int32(3)
		C = int32(4)
		K = int32(3)
	)
	convTail := int(K - 1)
	xBC := testValues(0.1, int(B), int(L), int(C))
	convState := testValues(0.2, int(B), convTail, int(C))
	weight := testValues(-0.3, int(C), int(K))
	bias := testValues(0.4, int(C))
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, int(B), int(L)),
		SeqOffsets:   []int32{0, 0},
		SeqQueryLens: []int32{L, 1},
	}

	conv := nn.NewConv1d(mlx.ExpandDims(weight, 2), nil, 1, 0, 1, C)
	got, gotStates := nn.CausalConv1D(b, xBC, conv, convTail,
		nn.WithRecurrentState(convState, nil),
		nn.WithSnapshotSplits([]int{1, 2}),
	)
	got = mlx.SiLU(mlx.Add(got, bias))

	mask := nn.PaddingMask(b, L)
	zero := mlx.FromValue(float32(0))
	got = mlx.Where(mlx.ExpandDims(mask, 2), got, zero.AsType(got.DType()))

	maskedXBC := mlx.Where(mlx.ExpandDims(mask, 2), xBC, zero.AsType(xBC.DType()))
	convInput := mlx.Concatenate([]*mlx.Array{convState, maskedXBC}, 1)
	want := mlx.SiLU(mlx.Add(conv.Forward(convInput), bias))
	want = mlx.Where(mlx.ExpandDims(mask, 2), want, zero.AsType(want.DType()))
	wantStates := []*mlx.Array{
		nn.CausalConvStateAt(convInput, b.SeqQueryLens, convTail, 1),
		nn.CausalConvStateAt(convInput, b.SeqQueryLens, convTail, 2),
		nn.CausalConvStateAt(convInput, b.SeqQueryLens, convTail, L),
	}
	for i, st := range wantStates {
		wantStates[i] = mlx.Contiguous(st, false)
	}

	eval := append([]*mlx.Array{got, want}, gotStates...)
	eval = append(eval, wantStates...)
	mlx.Eval(eval...)
	assertAllClose(t, "conv output", got.Floats(), want.Floats(), 1e-5)
	if len(gotStates) != len(wantStates) {
		t.Fatalf("len(conv states) = %d, want %d", len(gotStates), len(wantStates))
	}
	for i := range gotStates {
		assertAllClose(t, "conv state", gotStates[i].Floats(), wantStates[i].Floats(), 1e-5)
	}
}

func ones(dtype mlx.DType, shape ...int) *mlx.Array {
	return mlx.AddScalar(mlx.Zeros(dtype, shape...), 1)
}

func testValues(seed float32, shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = seed + 0.001*float32(i)
	}
	return mlx.FromValues(vals, shape...)
}

func sliceBatch(x *mlx.Array, start, stop int32, length int32) *mlx.Array {
	dims := x.Dims()
	starts := make([]int32, len(dims))
	stops := make([]int32, len(dims))
	for i, d := range dims {
		stops[i] = int32(d)
	}
	starts[0] = start
	stops[0] = stop
	stops[1] = length
	return mlx.SliceStartStop(x, starts, stops)
}

func sliceStateBatch(x *mlx.Array, row int32) *mlx.Array {
	return mlx.SliceStartStop(x, []int32{row, 0, 0, 0}, []int32{row + 1, int32(x.Dim(1)), int32(x.Dim(2)), int32(x.Dim(3))})
}

func assertAllClose(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}
