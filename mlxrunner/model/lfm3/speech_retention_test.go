package lfm3

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Exercise the real speech forward and attention cache without model downloads.
// Request boundaries deliberately have no enclosing array scope: production
// speakChunk closes its held output/cache scopes, not the runner's root scope.
func TestSpeechForwardReleasesRequestCache(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		library, err := mlx.LoadedLibraryPath()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("native_library=%s mlx_version=%s gpu=%t", library, mlx.Version(), mlx.GPUIsAvailable())
		mlx.Scoped(func() {
			const width = 64
			const eps = 1e-5
			norm := func() *nn.RMSNorm {
				values := make([]float32, width)
				for i := range values {
					values[i] = 1
				}
				return nn.NewRMSNorm(mlx.FromValues(values, width), eps)
			}
			linear := func(rows int) nn.LinearLayer {
				return nn.NewLinear(mlx.Zeros(mlx.DTypeFloat32, rows, width), nil)
			}
			m := &SpeechModel{VLModel: &VLModel{Model: &Model{
				Config: &Config{
					HiddenSize: width, NumAttentionHeads: 1, NumKeyValueHeads: 1,
					HeadDim: width, NormEps: eps, Scale: 0.125,
					FullRope: ropeConfig{RopeTheta: 10000, PartialRotaryFactor: 1},
				},
				EmbeddingNorm: norm(),
				Layers: []*Layer{{
					LayerType: "full_attention", OperatorNorm: norm(), FFNNorm: norm(),
					Attention: &Attention{
						QProj: linear(width), KProj: linear(width), VProj: linear(width),
						OProj: linear(width), GateProj: linear(1), QNorm: norm(), KNorm: norm(),
					},
					MLP: &MLP{W1: linear(width), W2: linear(width), W3: linear(width)},
				}},
			}}}
			mlx.Eval(mlx.Collect(m)...)

			request := func() {
				caches := m.NewCaches()
				defer func() {
					for _, c := range caches {
						c.Free()
					}
				}()
				held := mlx.NewScope()
				defer held.Close()
				for pos := range 6 {
					hidden, err := m.forwardSpeech(caches, pos, func() (*mlx.Array, *mlx.Array, error) {
						return mlx.Zeros(mlx.DTypeInt32, 1, 1), mlx.Zeros(mlx.DTypeFloat32, 1, 1, width), nil
					})
					if err != nil {
						t.Fatal(err)
					}
					held.Attach(hidden)
					values := hidden.Floats() // Finish async work while the output is still owned.
					if len(values) != width {
						t.Fatalf("hidden width = %d, want %d", len(values), width)
					}
					for _, v := range values {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Fatalf("invalid hidden value %g", v)
						}
					}
					held.Discard(hidden)
				}
			}
			settle := func() uint64 {
				mlx.DefaultStream().Synchronize()
				mlx.ClearCache()
				return uint64(mlx.ActiveMemory())
			}
			request() // Warm kernels before recording the baseline.
			baseline := settle()
			for i := range 8 {
				request()
				active := settle()
				t.Logf("request=%d baseline_bytes=%d active_bytes=%d", i+1, baseline, active)
				// One fresh pair of 256 x 64 float32 KV buffers already costs
				// 128 KiB. Allow 32 KiB for warmed backend accounting variation.
				if active > baseline+32*1024 {
					t.Fatalf("speech request retained allocations: baseline=%d active=%d delta=%d bytes", baseline, active, active-baseline)
				}
			}
		})
	})
}
