package phonon

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func tinySpeechModel() *Model {
	linear := func(out, in int) *nn.Linear {
		values := make([]float32, out*in)
		for i := range values {
			values[i] = float32(i%7-3) / 32
		}
		return nn.NewLinear(mlx.FromValues(values, out, in), nil)
	}
	norm := func() *nn.LayerNorm {
		return &nn.LayerNorm{Weight: mlx.FromValues([]float32{1, 1, 1, 1}, 4), Eps: 1e-5}
	}
	m := &Model{
		cfg: &config{Temperature: .3, EOSThreshold: 100},
		BOS: mlx.FromValues([]float32{.25, -.5}, 1, 1, 2), Conditions: mlx.Zeros(mlx.DTypeFloat32, 1, 1, 4),
		InputLinear: linear(4, 2), OutNorm: norm(), EOS: linear(1, 4),
		EmbMean: mlx.Zeros(mlx.DTypeFloat32, 2), EmbStd: mlx.FromValues([]float32{1, 1}, 2),
		Flow: &Transformer{NumHeads: 2, MaxPeriod: 10000, GELU: "erf", Layers: []*Layer{{Norm1: norm(), Norm2: norm(), InProj: linear(12, 4), OutProj: linear(4, 4), Linear1: linear(8, 4), Linear2: linear(4, 8)}}},
		Head: &FlowHead{CondEmbed: linear(4, 4), InputProj: linear(4, 2), Final: linear(2, 4), Modulation: linear(20, 4), Time: mlx.Zeros(mlx.DTypeFloat32, 4), Blocks: []*FlowBlock{{Norm: norm(), MLP0: linear(8, 4), MLP2: linear(4, 8)}}},
		Mimi: &Mimi{OutputProj: mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2), Upsample: &ConvTr{Weight: mlx.FromValues([]float32{.5, .5, .25, .25}, 2, 2), Stride: 1, Depthwise: true}, Transformer: &Transformer{NumHeads: 1, MaxPeriod: 10000}, Context: 4},
	}
	m.sample = mlx.Compile("phonon_sample", m.sampleFrame)
	return m
}

func TestSpeechSampleCompiledMatchesEager(t *testing.T) {
	// Keep the package's float32 comparisons independent of TF32 setup order.
	t.Setenv("MLX_ENABLE_TF32", "0")
	for _, rows := range []int{1, 2} {
		mlxtest.RunSubtest(t, fmt.Sprintf("rows=%d", rows), func(t *mlxtest.T) {
			m := tinySpeechModel()
			hiddenValues := make([]float32, rows*4)
			noiseValues := make([]float32, rows*2)
			for i := range hiddenValues {
				hiddenValues[i] = float32(i%7-3) / 16
			}
			for i := range noiseValues {
				noiseValues[i] = float32(i%5-2) / 8
			}
			hidden := mlx.FromValues(hiddenValues, rows, 1, 4)
			noise := mlx.FromValues(noiseValues, rows, 2)
			got, want := m.sample(hidden, noise), m.sampleFrame(hidden, noise)
			for i := range want {
				actual, expected := got[i].AsType(mlx.DTypeFloat32), want[i].AsType(mlx.DTypeFloat32)
				mlx.Eval(actual, expected)
				a, b := actual.Floats(), expected.Floats()
				if len(a) == 0 || len(a) != len(b) {
					t.Fatalf("output %d value count = %d, want %d", i, len(a), len(b))
				}
				for j := range b {
					if delta := math.Abs(float64(a[j] - b[j])); math.IsNaN(delta) || delta > 1e-5 {
						t.Fatalf("output %d element %d = %g, want %g", i, j, a[j], b[j])
					}
				}
			}
		})
	}
}

func BenchmarkSpeechSampleEager(b *testing.B) {
	benchmarkSpeechSample(b, false)
}

func BenchmarkSpeechSampleCompiled(b *testing.B) {
	benchmarkSpeechSample(b, true)
}

func benchmarkSpeechSample(b *testing.B, compiled bool) {
	thread := mlxtest.Worker(b)
	if err := thread.Do(b.Context(), func() error {
		defer b.StopTimer()
		mlx.Scoped(func() {
			m := tinySpeechModel()
			hidden := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 2, 1, 4), .25)
			noise := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 2, 2), .125)
			sample := m.sampleFrame
			if compiled {
				sample = m.sample
			}
			mlx.Scoped(func() { mlx.Eval(sample(hidden, noise)...) })
			b.ResetTimer()
			for range b.N {
				mlx.Scoped(func() { mlx.Eval(sample(hidden, noise)...) })
			}
		})
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}

func TestSpeechFrameBatchMatchesIndependent(t *testing.T) {
	// Compare float32 arithmetic. TF32 rounds batched GEMMs differently
	// from the independent GEMVs; production precision is unchanged.
	t.Setenv("MLX_ENABLE_TF32", "0")
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := tinySpeechModel()
		var owners, refs [2][]cache.Cache
		for i := range owners {
			owners[i], refs[i] = []cache.Cache{cache.NewKVCache()}, []cache.Cache{cache.NewKVCache()}
			defer owners[i][0].Free()
			defer refs[i][0].Free()
			prompt := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, i+2, 4), float32(i+1)/4)
			mlx.Eval(m.Flow.forward(prompt, 0, owners[i]), m.Flow.forward(prompt, 0, refs[i]))
		}
		for step, order := range [][]int{{0, 1}, {1, 0}, {1}, {0, 1}} {
			var inputs, noises []*mlx.Array
			var caches [][]cache.Cache
			var positions []int32
			var want [2][]float32
			for _, id := range order {
				prev := mlx.FromValues([]float32{float32(id+1) / 4, -float32(step+1) / 4}, 1, 1, 2)
				noise := mlx.FromValues([]float32{float32(step+2) / 8, float32(id+2) / 8}, 1, 2)
				inputs, noises, caches = append(inputs, prev), append(noises, noise), append(caches, owners[id])
				positions = append(positions, int32(owners[id][0].Offset()))
				out, err := m.speechFrame(prev, []int32{int32(refs[id][0].Offset())}, [][]cache.Cache{refs[id]}, noise)
				if err != nil {
					t.Fatal(err)
				}
				for j := range out {
					want[j] = append(want[j], out[j].Floats()...)
				}
			}
			out, err := m.speechFrame(mlx.Concatenate(inputs, 0), positions, caches, mlx.Concatenate(noises, 0))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(out[0].Dims(), []int{len(order), 1, 2}) || out[1].Dim(0) != len(order) {
				t.Fatal("speech computation lost batch dimension")
			}
			for j, a := range out {
				actual := a.Floats()
				if len(actual) == 0 || len(actual) != len(want[j]) {
					t.Fatalf("step %d output %d value count = %d, want %d", step, j, len(actual), len(want[j]))
				}
				for k, got := range actual {
					if delta := math.Abs(float64(got - want[j][k])); math.IsNaN(delta) || delta > 1e-5 {
						t.Fatalf("step %d output %d[%d]=%g, want %g", step, j, k, got, want[j][k])
					}
				}
			}
			for _, id := range order {
				if owners[id][0].Offset() != refs[id][0].Offset() {
					t.Fatal("history moved between requests")
				}
			}
		}
	})
}

func TestSpeechBatchPreservesFrameGroups(t *testing.T) {
	// Compare float32 arithmetic. TF32 rounds batched GEMMs differently
	// from the independent GEMVs; production precision is unchanged.
	t.Setenv("MLX_ENABLE_TF32", "0")
	for _, eos := range []bool{false, true} {
		mlxtest.RunSubtest(t, map[bool]string{false: "budget", true: "eos"}[eos], func(t *mlxtest.T) {
			m := tinySpeechModel()
			if eos {
				m.cfg.EOSThreshold = -100
			}
			newState := func(id int) *speechState {
				s := &speechState{caches: []cache.Cache{cache.NewKVCache()}, held: mlx.NewScope(), mimi: m.Mimi.newState(), rng: rand.New(rand.NewPCG(uint64(id+1), 42)), noise: make([]float32, 2), prev: m.BOS.Clone(), stop: 12, framesAfterEOS: 2}
				s.held.Attach(s.prev)
				// Start one stream mid-group so the cohort has different frame counts.
				s.generated = id * 3
				if s.generated > 0 {
					mlx.Eval(m.Flow.forward(mlx.Zeros(mlx.DTypeFloat32, 1, s.generated, 4), 0, s.caches))
				}
				return s
			}
			a, b, ar, br := newState(0), newState(1), newState(0), newState(1)
			defer a.Close()
			defer b.Close()
			defer ar.Close()
			defer br.Close()
			finished := false
			for round := range 4 {
				got, err := m.DecodeSpeech([]model.SpeechState{a, b})
				if err != nil {
					t.Fatal(err)
				}
				for i, ref := range []*speechState{ar, br} {
					want, err := m.DecodeSpeech([]model.SpeechState{ref})
					if err != nil {
						t.Fatal(err)
					}
					if got[i].Done != want[0].Done || got[i].Err != nil || len(got[i].Samples) != len(want[0].Samples) {
						t.Fatalf("round %d row %d completion or sample length differs", round, i)
					}
					for j, value := range got[i].Samples {
						if delta := math.Abs(float64(value - want[0].Samples[j])); math.IsNaN(delta) || delta > 1e-5 {
							t.Fatalf("round %d row %d sample %d differs", round, i, j)
						}
					}
				}
				if round == 0 && (len(got[0].Samples) != 2 || len(got[1].Samples) != map[bool]int{false: 16, true: 6}[eos]) {
					t.Fatalf("first-frame/codec grouping changed: %d %d", len(got[0].Samples), len(got[1].Samples))
				}
				if got[0].Done && got[1].Done {
					finished = true
					break
				}
			}
			if !finished {
				t.Fatal("speech streams did not finish")
			}
		})
	}
}
