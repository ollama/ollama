package inkling2

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestTargetProjectionUsesConsistentRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const hidden, output = 2304, 128
			rng := rand.New(rand.NewPCG(448, 1536))
			values := func(n int) []float32 {
				data := make([]float32, n)
				for i := range data {
					data[i] = rng.Float32() - 0.5
				}
				return data
			}
			input := mlx.FromValues(values(2*shortTargetRows*hidden), 2*shortTargetRows, hidden).AsType(mlx.DTypeBFloat16)
			weight := mlx.FromValues(values(output*hidden), output, hidden).AsType(mlx.DTypeBFloat16)
			bias := mlx.FromValues(values(output), output).AsType(mlx.DTypeBFloat16)
			linear := nn.NewLinear(weight, bias)

			var want []float32
			for i := range 2 {
				row := input.Slice(mlx.Slice(i*shortTargetRows, (i+1)*shortTargetRows), mlx.Slice())
				want = append(want, linear.Forward(row).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()...)
			}
			batched := projectTarget(linear, input.Reshape(2, shortTargetRows, hidden))
			if got := batched.AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats(); !slices.Equal(got, want) {
				t.Fatal("joining two verification blocks changed their projection")
			}
			for tokens := 1; tokens <= shortTargetRows; tokens++ {
				ids := input.Slice(mlx.Slice(0, tokens), mlx.Slice()).Reshape(1, tokens, hidden)
				got := projectTarget(linear, ids).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
				if !slices.Equal(got, want[:tokens*output]) {
					t.Errorf("projection at width %d differs from width %d", tokens, shortTargetRows)
				}
			}

			prefill := input.Reshape(1, 2*shortTargetRows, hidden)
			got := projectTarget(linear, prefill).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
			ordinary := linear.Forward(prefill).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
			if !slices.Equal(got, ordinary) {
				t.Error("larger prefill changed from the ordinary projection path")
			}
		})
	})
}

// Short forwards unembed in launches of shortTargetRows flattened rows that
// apply the softcap themselves, so requests sharing a launch get the logits
// each would get alone. Metal's precise tanh and MLX's differ by an FP32
// rounding or two of the capped values.
func TestShortVocabularyLogitsTileRequests(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("the BF16 short head loads on Metal only")
		}
		mlx.Scoped(func() {
			const width, block, copies, logitCap = 2304, 1000, 33, 30
			rng := rand.New(rand.NewPCG(53, 59))
			// A vocabulary-sized head takes ShortMatmul's staged rows. Scaled
			// copies of one random block keep its columns distinct without a
			// large host buffer.
			gains := mlx.AddScalar(mlx.MulScalar(mlx.Arange(0, copies, 1, mlx.DTypeFloat32), 1.0/32), 1).AsType(mlx.DTypeBFloat16)
			head := mlx.Mul(draftTestValues(rng, 0.02, 1, block, width), gains.Reshape(copies, 1, 1)).Reshape(copies*block, width)
			mlx.Eval(head)
			m := &targetModel{config: textConfig{LogitCap: logitCap}, Output: nn.NewLinear(head, nil), ShortOutput: nn.NewLinear(head, nil)}
			for _, sh := range [][2]int{{1, 1}, {3, 3}, {2, shortTargetRows}} {
				mlx.Scoped(func() {
					rows, tokens := sh[0], sh[1]
					name := fmt.Sprintf("%d requests of %d rows", rows, tokens)
					hidden := draftTestValues(rng, 16, rows, tokens, width)
					fused := m.shortVocabularyLogits(hidden)
					if fused == nil {
						t.Fatalf("%s: the capped short head declined", name)
					}
					got := m.Unembed(hidden)
					requireIdentical(t, name, got, fused)
					for i := range rows {
						alone := m.Unembed(hidden.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()))
						requireIdentical(t, fmt.Sprintf("%s: request %d alone", name, i), alone, got.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()))
					}
					requireClose(t, name+" against the capped projection", got, softcap(m.shortVocabularyProjection(hidden), logitCap), 1e-5)
				})
			}
			if m.shortVocabularyLogits(draftTestValues(rng, 1, 1, shortTargetRows+1, width)) != nil {
				t.Fatal("a long forward took the short head")
			}
		})
	})
}
