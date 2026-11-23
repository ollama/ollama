package olala

import (
	"fmt"
	"maps"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

const releasedConfig = `{
	"architectures": ["OlalaForCausalLM"],
	"model_type": "olala",
	"vocab_size": 151936,
	"hidden_size": 1536,
	"num_hidden_layers": 36,
	"layers_config": "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM",
	"num_attention_heads": 48,
	"num_signal_heads_diff": 36,
	"head_dim": 128,
	"mamba_d_state": 128,
	"mamba_headdim": 64,
	"mamba_mimo_dim": 4,
	"max_position_embeddings": 100000
}`

func TestParseConfig(t *testing.T) {
	cfg, err := parseConfig([]byte(releasedConfig))
	if err != nil {
		t.Fatal(err)
	}
	got := []int32{cfg.mH, cfg.mP, cfg.mN, cfg.mR, cfg.mA, cfg.nStatic, cfg.nSig, cfg.nNoise, cfg.snr}
	want := []int32{48, 64, 128, 4, 32, 6288, 36, 12, 3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("derived dims = %v, want %v", got, want)
		}
	}
	if cfg.attnScale != 1.0/128 || cfg.gateBias != 1.15 {
		t.Fatalf("attnScale = %v, gateBias = %v", cfg.attnScale, cfg.gateBias)
	}
	if n := newModelFromConfig(cfg, nil).MaxContextLength(); n != 32768 {
		t.Fatalf("MaxContextLength = %d, want the 32768 attention window", n)
	}
}

func TestParseConfigRejectsUnknownLayers(t *testing.T) {
	for _, layers := range []string{"MMX", "MV"} {
		_, err := parseConfig([]byte(fmt.Sprintf(`{"num_hidden_layers": 3, "layers_config": %q}`, layers)))
		if err == nil {
			t.Fatalf("layers_config %q: expected an error", layers)
		}
	}
}

func TestNewCachesLayout(t *testing.T) {
	cfg, err := parseConfig([]byte(releasedConfig))
	if err != nil {
		t.Fatal(err)
	}
	m := newModelFromConfig(cfg, nil)
	caches := m.NewCaches()
	if len(caches) != 36+7 {
		t.Fatalf("got %d caches, want one per layer plus one per attention layer", len(caches))
	}
	for i, l := range m.Layers {
		switch l.typ {
		case 'M':
			if _, ok := caches[i].(*cache.RecurrentCache); !ok {
				t.Fatalf("layer %d: got %T, want *cache.RecurrentCache", i, caches[i])
			}
		case 'V':
			if _, ok := caches[i].(*cache.KVCache); !ok {
				t.Fatalf("layer %d: got %T, want *cache.KVCache", i, caches[i])
			}
			if _, ok := caches[m.shiftSlot[i]].(*cache.RecurrentCache); !ok {
				t.Fatalf("layer %d token shift: got %T", i, caches[m.shiftSlot[i]])
			}
		}
	}
}

// A small Olala with the released head sizes (the fused kernels require
// N = 128, A = 32, Dh = 128 and Hq = 4 * Hk) and random weights.
const tinyConfig = `{
	"vocab_size": 64,
	"hidden_size": 256,
	"num_hidden_layers": 3,
	"layers_config": "MVM",
	"num_attention_heads": 8,
	"num_signal_heads_diff": 6,
	"head_dim": 128,
	"tpa_rank": 4,
	"mamba_d_state": 128,
	"mamba_headdim": 64,
	"mamba_mimo_dim": 4,
	"moe_num_routed_experts": 32,
	"moe_num_active_experts": 4,
	"moe_routed_input_dim": 128,
	"max_position_embeddings": 4096
}`

type tinyWeights struct {
	rng     *rand.Rand
	tensors map[string]*mlx.Array
	quant   map[string]*model.TensorQuantInfo
	// bits quantizes the large matrices (0: bf16); expertBits the experts.
	bits, expertBits int
}

func (w *tinyWeights) values(n int, mean, std float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(mean + std*w.rng.NormFloat64())
	}
	return out
}

func (w *tinyWeights) put(name string, mean, std float64, dtype mlx.DType, shape ...int) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	w.tensors[name] = mlx.FromValues(w.values(n, mean, std), shape...).AsType(dtype)
}

// linear stores a (rows, cols) matrix, quantized when bits > 0.
func (w *tinyWeights) linear(name string, bits int, shape ...int) {
	std := 1 / math.Sqrt(float64(shape[len(shape)-1]))
	if bits == 0 {
		w.put(name+".weight", 0, std, mlx.DTypeBFloat16, shape...)
		return
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	full := mlx.FromValues(w.values(n, 0, std), shape...).AsType(mlx.DTypeBFloat16)
	q, scales, biases := mlx.Quantize(full, 64, bits, "affine")
	w.tensors[name+".weight"] = q
	w.tensors[name+".weight_scale"] = scales
	w.tensors[name+".weight_qbias"] = biases
	// per-tensor metadata, as an import records it
	w.quant[name+".weight"] = &model.TensorQuantInfo{QuantType: fmt.Sprintf("INT%d", bits), GroupSize: 64}
}

// newTinyShell returns the tiny model before its weights are loaded.
func newTinyShell(t *mlxtest.T) *Model {
	cfg, err := parseConfig([]byte(tinyConfig))
	if err != nil {
		t.Fatal(err)
	}
	cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = 64, 8, "affine"
	return newModelFromConfig(cfg, nil)
}

func newTinyModel(t *mlxtest.T, bits, expertBits int) *Model {
	m := newTinyShell(t)
	tensors, quant := tinyTensors(m, bits, expertBits)
	m.cfg.TensorQuant = quant
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatal(err)
	}
	return m
}

// tinyTensors returns random checkpoint tensors for m, named as the runner
// hands them to LoadWeights, and their quantization metadata.
func tinyTensors(m *Model, bits, expertBits int) (map[string]*mlx.Array, map[string]*model.TensorQuantInfo) {
	cfg := *m.cfg
	w := &tinyWeights{rng: rand.New(rand.NewSource(1)), tensors: map[string]*mlx.Array{}, quant: map[string]*model.TensorQuantInfo{}, bits: bits, expertBits: expertBits}
	D := int(cfg.HiddenSize)
	f32 := mlx.DTypeFloat32
	w.put("model.embedding.weight", 0, 1, mlx.DTypeBFloat16, int(cfg.VocabSize), D)
	w.linear("lm_head", 0, int(cfg.VocabSize), D)
	H, P, N, R, A := int(cfg.mH), int(cfg.mP), int(cfg.mN), int(cfg.mR), int(cfg.mA)
	Hq, Hk, Dh, Rk := int(cfg.NumAttentionHeads), int(cfg.nNoise), int(cfg.HeadDim), int(cfg.TPARank)
	E, latent, hidden, shared := int(cfg.NumRoutedExperts), int(cfg.RoutedInputDim), 128, 256
	for i, l := range m.Layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		// the runner's names for the "<x>.scale" / "<x>.bias" geodesic pairs
		w.put(p+"geodesic_mixer_scale", 1, 0.1, f32, 1)
		w.put(p+"geodesic_mixer_qbias", 0, 0.05, f32, 1)
		w.put(p+"geodesic_mlp_scale", 1, 0.1, f32, 1)
		w.put(p+"geodesic_mlp_qbias", 0, 0.05, f32, 1)
		if l.typ == 'M' {
			w.linear(p+"mixer.in_proj", bits, int(cfg.nStatic)+2*R*N+A, D)
			w.put(p+"mixer.B_bias", 1, 0.1, f32, H, R, N)
			w.put(p+"mixer.C_bias", 1, 0.1, f32, H, R, N)
			w.put(p+"mixer.B_norm.norm.weight", 0, 0.1, f32, N)
			w.put(p+"mixer.C_norm.norm.weight", 0, 0.1, f32, N)
			w.put(p+"mixer.in_proj_mimo_x", 1, 0.2, f32, H, R, P)
			w.put(p+"mixer.in_proj_mimo_z", 1, 0.2, f32, H, R, P)
			w.put(p+"mixer.out_proj_mimo", 1, 0.2, f32, H, R, P)
			w.put(p+"mixer.dt_bias", -2, 0.5, f32, H)
			w.put(p+"mixer.D", 1, 0.1, f32, H)
			w.linear(p+"mixer_proj", bits, D, H*P)
		} else {
			w.linear(p+"mixer.in_proj", bits, Hq*Dh+2*Rk*Dh+int(cfg.nSig)*Dh, D)
			w.put(p+"mixer.in_proj_small.weight", 0, 1/math.Sqrt(float64(D)), f32, 2*Hk*Rk+2*Hk+int(cfg.nNoise), D)
			w.put(p+"mixer.q_norm.norm.weight", 0, 0.1, f32, Dh)
			w.put(p+"mixer.k_norm.norm.weight", 0, 0.1, f32, Dh)
			w.put(p+"mixer.softmax_scaler", 1, 0.1, f32, Hq)
			w.linear(p+"mixer_proj", bits, D, int(cfg.nSig)*Dh)
		}
		w.put(p+"mlp.moe_gate.weight", 0, 1/math.Sqrt(float64(D)), f32, E, D)
		w.put(p+"mlp.expert_bias", 0, 0.1, f32, E)
		w.linear(p+"mlp.in_cat", bits, latent+shared, D)
		w.linear(p+"mlp.out_cat", bits, D, latent+shared)
		w.linear(p+"mlp.experts.experts", expertBits, E, hidden, latent)
		w.linear(p+"mlp.experts.output_experts", expertBits, E, latent, hidden)
	}
	return w.tensors, w.quant
}

func tinyTokens(n int) []int32 {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32((i*7 + 3) % 64)
	}
	return ids
}

// forward runs ids through the model at offset off and returns the hidden
// state of the last position as floats.
func forward(m *Model, caches []cache.Cache, ids []int32, off int) []float32 {
	L := len(ids)
	h, _ := m.Forward(&batch.Batch{
		InputIDs:     mlx.FromValues(ids, 1, L),
		SeqOffsets:   []int32{int32(off)},
		SeqQueryLens: []int32{int32(L)},
	}, cache.Layers(caches))
	last := mlx.Reshape(mlx.SliceStartStop(h, []int32{0, int32(L - 1), 0}, []int32{1, int32(L), int32(h.Dim(2))}), -1)
	state := []*mlx.Array{last}
	for _, c := range caches {
		state = append(state, c.State()...)
	}
	mlx.Eval(state...)
	return last.Floats()
}

func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num / den)
}

// prefillAndDecode prefills prompt in one forward, then decodes next one
// token at a time, and returns the last hidden state after each step.
func prefillAndDecode(m *Model, prompt, next []int32) [][]float32 {
	caches := m.NewCaches()
	out := [][]float32{forward(m, caches, prompt, 0)}
	for i, id := range next {
		out = append(out, forward(m, caches, []int32{id}, len(prompt)+i))
	}
	return out
}

func assertClose(t *mlxtest.T, what string, got, want []float32, tol float64) {
	if e := relErr(got, want); e > tol {
		t.Fatalf("%s: relative error %.5f > %g", what, e, tol)
	}
}

func TestLoadWeightsReportsMissingTensor(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		for _, name := range []string{
			"model.layers.1.mixer.softmax_scaler",
			"model.layers.0.mixer.B_bias",
			"model.layers.2.mlp.experts.output_experts.weight",
			"model.embedding.weight",
		} {
			m := newTinyShell(t)
			tensors, quant := tinyTensors(m, 0, 0)
			delete(tensors, name)
			m.cfg.TensorQuant = quant
			err := m.LoadWeights(tensors)
			if err == nil {
				t.Fatalf("missing %s: LoadWeights succeeded", name)
			}
			if !strings.Contains(err.Error(), name) {
				t.Fatalf("missing %s: error %q does not name it", name, err)
			}
		}
	})
}

// The runner loads weights in a scope that keeps only the arrays mlx.Collect
// finds on the model; everything the forward pass reads must be among them.
func TestWeightLifetime(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		prompt, next := tinyTokens(20), []int32{4, 9}
		for _, bits := range []int{0, 8} {
			want := prefillAndDecode(newTinyModel(t, bits, bits), prompt, next)

			var m *Model
			mlx.ScopedArrays(func() []*mlx.Array {
				m = newTinyShell(t)
				tensors, quant := tinyTensors(m, bits, bits)
				mlx.Eval(slices.Collect(maps.Values(tensors))...)
				m.cfg.TensorQuant = quant
				if err := m.LoadWeights(tensors); err != nil {
					t.Fatal(err)
				}
				return mlx.Collect(m)
			})
			got := prefillAndDecode(m, prompt, next)
			for i := range want {
				assertClose(t, fmt.Sprintf("bits=%d step %d", bits, i), got[i], want[i], 1e-5)
			}
		}
	})
}

// The runner prefills long prompts in several forwards.
func TestCacheContinuation(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := newTinyModel(t, 0, 0)
		ids := tinyTokens(40)
		whole := m.NewCaches()
		want := forward(m, whole, ids, 0)

		split := m.NewCaches()
		var got []float32
		for _, r := range [][2]int{{0, 23}, {23, 39}, {39, 40}} {
			got = forward(m, split, ids[r[0]:r[1]], r[0])
		}
		assertClose(t, "split prefill", got, want, 1e-2)

		// and both caches decode the same next token
		assertClose(t, "decode after split prefill",
			forward(m, split, []int32{7}, 40), forward(m, whole, []int32{7}, 40), 1e-2)
	})
}

// Prefix caching captures the state at interior offsets of a forward and
// restores it later.
func TestSnapshotRestore(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := newTinyModel(t, 0, 0)
		ids := tinyTokens(40)
		const at = 24
		want := forward(m, m.NewCaches(), ids, 0)

		caches := m.NewCaches()
		for _, c := range caches {
			c.PrepareSnapshots([]int{at})
		}
		got := forward(m, caches, ids, 0)
		assertClose(t, "forward with a snapshot split", got, want, 1e-2)

		snaps := make([]cache.Snapshot, len(caches))
		for i, c := range caches {
			taken := c.TakeSnapshots()
			if len(taken) != 1 || taken[0] == nil {
				t.Fatalf("cache %d (%T): no snapshot at %d", i, c, at)
			}
			snaps[i] = taken[0]
		}
		for i, c := range caches {
			if !c.Restore(snaps[i], at) {
				t.Fatalf("cache %d (%T): restore to %d failed", i, c, at)
			}
		}
		got = forward(m, caches, ids[at:], at)
		assertClose(t, "replay after restore", got, want, 1e-2)
	})
}

// Several requests decode together with one token per row.
func TestBatchedDecode(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := newTinyModel(t, 0, 0)
		prompts := [][]int32{tinyTokens(30), tinyTokens(20)[5:]}
		next := []int32{11, 3}

		var want [][]float32
		rows := make([][]cache.Cache, len(prompts))
		for i, p := range prompts {
			want = append(want, prefillAndDecode(m, p, next[i:i+1])[1])
			rows[i] = m.NewCaches()
			forward(m, rows[i], p, 0)
		}

		layers, err := cache.Batch(rows)
		if err != nil {
			t.Fatal(err)
		}
		h, _ := m.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues(next, 2, 1),
			SeqOffsets:   []int32{int32(len(prompts[0])), int32(len(prompts[1]))},
			SeqQueryLens: []int32{1, 1},
		}, layers)
		mlx.Eval(h)
		D := h.Dim(2)
		all := h.Floats()
		for i := range prompts {
			assertClose(t, fmt.Sprintf("row %d", i), all[i*D:(i+1)*D], want[i], 1e-2)
		}
	})
}

// Long prompts take paths short ones do not: Mamba-3 pieces of m3MaxPiece
// tokens, sorted expert dispatch from 64 tokens, and query chunks when the
// attention scores exceed attnScoreBudget.
func TestLongPrefillMatchesShortPieces(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		saved := attnScoreBudget
		defer func() { attnScoreBudget = saved }()
		// about 50 queries per chunk at 300 keys: (4 bytes * 8 heads * 300 keys) * 50
		attnScoreBudget = 4 * 8 * 300 * 50

		m := newTinyModel(t, 0, 0)
		ids := tinyTokens(300)
		if len(ids) <= m3MaxPiece {
			t.Fatalf("prompt must exceed m3MaxPiece (%d)", m3MaxPiece)
		}
		long := m.NewCaches()
		want := forward(m, long, ids, 0)

		short := m.NewCaches()
		var got []float32
		for p := 0; p < len(ids); p += 50 {
			got = forward(m, short, ids[p:min(len(ids), p+50)], p)
		}
		assertClose(t, "300-token prefill", want, got, 1e-2)
		assertClose(t, "decode after the long prefill",
			forward(m, long, []int32{9}, len(ids)), forward(m, short, []int32{9}, len(ids)), 1e-2)
	})
}

// The chunked Mamba-3 prefill and the attention prefill must match feeding
// the tokens one at a time.
func TestChunkedPrefillMatchesStepByStep(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := newTinyModel(t, 0, 0)
		ids := tinyTokens(45) // two Mamba chunks, the second one partial

		got := forward(m, m.NewCaches(), ids, 0)

		caches := m.NewCaches()
		var want []float32
		for i, id := range ids {
			want = forward(m, caches, []int32{id}, i)
		}
		e := relErr(got, want)
		if e > 1e-2 {
			t.Fatalf("chunked prefill differs from step by step: relative error %.4f", e)
		}
		t.Logf("relative error %.5f", e)
	})
}
