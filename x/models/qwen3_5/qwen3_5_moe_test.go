package qwen3_5

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// MLX streams are bound to the thread that created them.
func useMLXTestThread(t *testing.T) {
	t.Helper()

	runtime.LockOSThread()
	initialized := false
	t.Cleanup(func() {
		if initialized {
			mlx.Sweep()
			mlx.ClearCache()
		}
		runtime.UnlockOSThread()
	})

	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	initialized = true
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}
}

func moeTestConfig() *Config {
	return &Config{
		NumExperts:       8,
		NumExpertsPerTok: 2,
		HiddenSize:       128,
		QuantGroupSize:   32,
		QuantBits:        8,
		QuantMode:        "affine",
	}
}

func patternValues(n int, seed float64) []float32 {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(math.Sin(seed + float64(i)*0.37))
	}
	return vals
}

func expertSlice(a *mlx.Array, e int32) *mlx.Array {
	dims := a.Dims()
	return mlx.Squeeze(mlx.SliceStartStop(a, []int32{e, 0, 0}, []int32{e + 1, int32(dims[1]), int32(dims[2])}), 0)
}

type moeTestWeights struct {
	gateQ, gateS, gateB *mlx.Array
	upQ, upS, upB       *mlx.Array
	downQ, downS, downB *mlx.Array
}

func makeMoETestWeights(cfg *Config) moeTestWeights {
	E, I, H := int(cfg.NumExperts), 64, int(cfg.HiddenSize)
	gate := mlx.FromValues(patternValues(E*I*H, 1), E, I, H)
	up := mlx.FromValues(patternValues(E*I*H, 2), E, I, H)
	down := mlx.FromValues(patternValues(E*H*I, 3), E, H, I)
	var w moeTestWeights
	gs, bits, mode := cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode
	w.gateQ, w.gateS, w.gateB = mlx.Quantize(gate, gs, bits, mode)
	w.upQ, w.upS, w.upB = mlx.Quantize(up, gs, bits, mode)
	w.downQ, w.downS, w.downB = mlx.Quantize(down, gs, bits, mode)
	return w
}

func (w moeTestWeights) dequantized(cfg *Config) (gate, up, down *mlx.Array) {
	gs, bits, mode := cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode
	return mlx.Dequantize(w.gateQ, w.gateS, w.gateB, gs, bits, mode),
		mlx.Dequantize(w.upQ, w.upS, w.upB, gs, bits, mode),
		mlx.Dequantize(w.downQ, w.downS, w.downB, gs, bits, mode)
}

func moeForward(t *testing.T, tensors map[string]*mlx.Array, cfg *Config, L int32) (*SwitchMLP, []float32) {
	t.Helper()
	mlp, err := loadSwitchMLP(tensors, cfg, true, "model.layers.0")
	if err != nil {
		t.Fatalf("loadSwitchMLP: %v", err)
	}

	x := mlx.FromValues(patternValues(int(L*cfg.HiddenSize), 9), 1, int(L), int(cfg.HiddenSize))
	idx := make([]uint32, L*cfg.NumExpertsPerTok)
	for i := range idx {
		idx[i] = uint32((i*7 + 3) % int(cfg.NumExperts))
	}
	indices := mlx.FromValues(idx, 1, int(L), int(cfg.NumExpertsPerTok))

	out := mlp.Forward(x, indices, cfg)
	mlx.Eval(out)
	return mlp, out.Floats()
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
			m = d
		}
	}
	return m
}

// Every supported checkpoint layout must produce the same routed-expert
// output once normalized to the packed gate_up representation.
func TestSwitchMLPSourceLayoutEquivalence(t *testing.T) {
	useMLXTestThread(t)

	cfg := moeTestConfig()
	w := makeMoETestWeights(cfg)
	prefix := "model.layers.0"

	packed := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.experts.gate_up_proj.weight":       mlx.Concatenate([]*mlx.Array{w.gateQ, w.upQ}, 1),
			prefix + ".mlp.experts.gate_up_proj.weight_scale": mlx.Concatenate([]*mlx.Array{w.gateS, w.upS}, 1),
			prefix + ".mlp.experts.gate_up_proj.weight_qbias": mlx.Concatenate([]*mlx.Array{w.gateB, w.upB}, 1),
			prefix + ".mlp.experts.down_proj.weight":          w.downQ,
			prefix + ".mlp.experts.down_proj.weight_scale":    w.downS,
			prefix + ".mlp.experts.down_proj.weight_qbias":    w.downB,
		}
	}
	separate := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.switch_mlp.gate_proj.weight":       w.gateQ,
			prefix + ".mlp.switch_mlp.gate_proj.weight_scale": w.gateS,
			prefix + ".mlp.switch_mlp.gate_proj.weight_qbias": w.gateB,
			prefix + ".mlp.switch_mlp.up_proj.weight":         w.upQ,
			prefix + ".mlp.switch_mlp.up_proj.weight_scale":   w.upS,
			prefix + ".mlp.switch_mlp.up_proj.weight_qbias":   w.upB,
			prefix + ".mlp.switch_mlp.down_proj.weight":       w.downQ,
			prefix + ".mlp.switch_mlp.down_proj.weight_scale": w.downS,
			prefix + ".mlp.switch_mlp.down_proj.weight_qbias": w.downB,
		}
	}
	perExpert := func() map[string]*mlx.Array {
		tensors := map[string]*mlx.Array{}
		for e := range cfg.NumExperts {
			for proj, parts := range map[string][3]*mlx.Array{
				"gate_proj": {w.gateQ, w.gateS, w.gateB},
				"up_proj":   {w.upQ, w.upS, w.upB},
				"down_proj": {w.downQ, w.downS, w.downB},
			} {
				base := fmt.Sprintf("%s.mlp.experts.%d.%s.weight", prefix, e, proj)
				tensors[base] = expertSlice(parts[0], e)
				tensors[base+"_scale"] = expertSlice(parts[1], e)
				tensors[base+"_qbias"] = expertSlice(parts[2], e)
			}
		}
		return tensors
	}

	for _, L := range []int32{2, 64} { // below and above the sorted-gather threshold
		mlp, want := moeForward(t, packed(), cfg, L)
		if mlp.GateUpWeightQ == nil || mlp.DownWeightQ == nil {
			t.Fatalf("L=%d: packed layout did not take the quantized path", L)
		}
		for name, tensors := range map[string]func() map[string]*mlx.Array{"separate": separate, "per_expert": perExpert} {
			mlp, got := moeForward(t, tensors(), cfg, L)
			if mlp.GateUpWeightQ == nil || mlp.DownWeightQ == nil {
				t.Fatalf("L=%d %s: layout did not take the quantized path", L, name)
			}
			if d := maxAbsDiff(want, got); d != 0 {
				t.Errorf("L=%d %s: output differs from packed layout, max abs diff %g", L, name, d)
			}
		}
	}
}

// Tensor-level scales must be applied to gather outputs on the quantized
// path and folded into the weights on every fp path.
func TestSwitchMLPGlobalScales(t *testing.T) {
	useMLXTestThread(t)

	cfg := moeTestConfig()
	w := makeMoETestWeights(cfg)
	deqGate, deqUp, deqDown := w.dequantized(cfg)
	prefix := "model.layers.0"

	// Sorted gathers run on reduced-precision NAX kernels.
	const tolerance = 0.1

	const gsGate, gsUp, gsDown = 0.5, 0.25, 2.0
	separate := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.switch_mlp.gate_proj.weight":              w.gateQ,
			prefix + ".mlp.switch_mlp.gate_proj.weight_scale":        w.gateS,
			prefix + ".mlp.switch_mlp.gate_proj.weight_qbias":        w.gateB,
			prefix + ".mlp.switch_mlp.gate_proj.weight.global_scale": mlx.FromValues([]float32{gsGate}, 1),
			prefix + ".mlp.switch_mlp.up_proj.weight":                w.upQ,
			prefix + ".mlp.switch_mlp.up_proj.weight_scale":          w.upS,
			prefix + ".mlp.switch_mlp.up_proj.weight_qbias":          w.upB,
			prefix + ".mlp.switch_mlp.up_proj.weight.global_scale":   mlx.FromValues([]float32{gsUp}, 1),
			prefix + ".mlp.switch_mlp.down_proj.weight":              w.downQ,
			prefix + ".mlp.switch_mlp.down_proj.weight_scale":        w.downS,
			prefix + ".mlp.switch_mlp.down_proj.weight_qbias":        w.downB,
			prefix + ".mlp.switch_mlp.down_proj.weight.global_scale": mlx.FromValues([]float32{gsDown}, 1),
		}
	}

	// Packed checkpoints carry one scale covering both gate_up halves.
	packed := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.experts.gate_up_proj.weight":              mlx.Concatenate([]*mlx.Array{w.gateQ, w.upQ}, 1),
			prefix + ".mlp.experts.gate_up_proj.weight_scale":        mlx.Concatenate([]*mlx.Array{w.gateS, w.upS}, 1),
			prefix + ".mlp.experts.gate_up_proj.weight_qbias":        mlx.Concatenate([]*mlx.Array{w.gateB, w.upB}, 1),
			prefix + ".mlp.experts.gate_up_proj.weight.global_scale": mlx.FromValues([]float32{gsGate}, 1),
			prefix + ".mlp.experts.down_proj.weight":                 w.downQ,
			prefix + ".mlp.experts.down_proj.weight_scale":           w.downS,
			prefix + ".mlp.experts.down_proj.weight_qbias":           w.downB,
			prefix + ".mlp.experts.down_proj.weight.global_scale":    mlx.FromValues([]float32{gsDown}, 1),
		}
	}

	// References: the same weights dequantized with the scales pre-folded.
	foldedSeparate := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.switch_mlp.gate_proj.weight": mlx.MulScalar(deqGate, gsGate),
			prefix + ".mlp.switch_mlp.up_proj.weight":   mlx.MulScalar(deqUp, gsUp),
			prefix + ".mlp.switch_mlp.down_proj.weight": mlx.MulScalar(deqDown, gsDown),
		}
	}
	foldedPacked := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.switch_mlp.gate_proj.weight": mlx.MulScalar(deqGate, gsGate),
			prefix + ".mlp.switch_mlp.up_proj.weight":   mlx.MulScalar(deqUp, gsGate),
			prefix + ".mlp.switch_mlp.down_proj.weight": mlx.MulScalar(deqDown, gsDown),
		}
	}

	// Quantized gate/up with fp down: gate_up stays on the gather kernel
	// with its scales; only down runs fp.
	mixed := func() map[string]*mlx.Array {
		tensors := separate()
		delete(tensors, prefix+".mlp.switch_mlp.gate_proj.weight.global_scale")
		delete(tensors, prefix+".mlp.switch_mlp.up_proj.weight.global_scale")
		delete(tensors, prefix+".mlp.switch_mlp.down_proj.weight_scale")
		delete(tensors, prefix+".mlp.switch_mlp.down_proj.weight_qbias")
		delete(tensors, prefix+".mlp.switch_mlp.down_proj.weight.global_scale")
		tensors[prefix+".mlp.switch_mlp.down_proj.weight"] = mlx.MulScalar(deqDown, gsDown)
		return tensors
	}
	foldedMixed := func() map[string]*mlx.Array {
		return map[string]*mlx.Array{
			prefix + ".mlp.switch_mlp.gate_proj.weight": deqGate,
			prefix + ".mlp.switch_mlp.up_proj.weight":   deqUp,
			prefix + ".mlp.switch_mlp.down_proj.weight": mlx.MulScalar(deqDown, gsDown),
		}
	}

	for _, L := range []int32{2, 64} {
		mlpF, want := moeForward(t, foldedSeparate(), cfg, L)
		if mlpF.GateUpWeightQ != nil || mlpF.GateUpGlobalScale != nil || mlpF.DownGlobalScale != nil {
			t.Fatalf("L=%d: fp layout must fold tensor-level scales into the weights", L)
		}

		mlpQ, got := moeForward(t, separate(), cfg, L)
		if mlpQ.GateUpWeightQ != nil || mlpQ.GateUpGlobalScale != nil {
			t.Fatalf("L=%d: differing gate/up scales must decline quantized fusion", L)
		}
		if mlpQ.DownWeightQ == nil || mlpQ.DownGlobalScale == nil {
			t.Fatalf("L=%d: down must keep its tensor-level scale for the gather path", L)
		}
		if d := maxAbsDiff(want, got); d > tolerance {
			t.Errorf("L=%d: quantized output diverges from folded fp reference, max abs diff %g", L, d)
		}

		mlpM, gotMixed := moeForward(t, mixed(), cfg, L)
		if mlpM.GateUpWeightQ == nil || mlpM.GateUpGlobalScale != nil || mlpM.DownWeightQ != nil || mlpM.DownGlobalScale != nil {
			t.Fatalf("L=%d: mixed layout must keep gate_up quantized and run down fp", L)
		}
		_, wantMixed := moeForward(t, foldedMixed(), cfg, L)
		if d := maxAbsDiff(wantMixed, gotMixed); d > tolerance {
			t.Errorf("L=%d: mixed layout diverges from fp reference, max abs diff %g", L, d)
		}

		mlpP, gotPacked := moeForward(t, packed(), cfg, L)
		if mlpP.GateUpWeightQ == nil || mlpP.GateUpGlobalScale == nil || mlpP.DownGlobalScale == nil {
			t.Fatalf("L=%d: packed layout must keep tensor-level scales for the gather path", L)
		}
		_, wantPacked := moeForward(t, foldedPacked(), cfg, L)
		if d := maxAbsDiff(wantPacked, gotPacked); d > tolerance {
			t.Errorf("L=%d: packed output diverges from folded fp reference, max abs diff %g", L, d)
		}
	}
}
