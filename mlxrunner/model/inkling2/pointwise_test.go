package inkling2

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestMetalAttentionRounding(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		mlx.EnableCompile()
		random := rand.New(rand.NewPCG(131, 49))
		values := func(shape ...int) *mlx.Array {
			n := 1
			for _, size := range shape {
				n *= size
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(random.NormFloat64())
			}
			return mlx.FromValues(data, shape...).AsType(mlx.DTypeBFloat16)
		}
		equal := func(name string, got, want *mlx.Array) {
			t.Helper()
			if got.DType() != want.DType() || !slices.Equal(got.Dims(), want.Dims()) {
				t.Fatalf("%s got %s %v, want %s %v", name, got.DType(), got.Dims(), want.DType(), want.Dims())
			}
			a := got.AsType(mlx.DTypeFloat32).Floats()
			b := want.AsType(mlx.DTypeFloat32).Floats()
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("%s value %d: got %g, want %g", name, i, a[i], b[i])
				}
			}
		}
		for _, tokens := range []int{1, 8, 129} {
			mlx.Scoped(func() {
				x := values(2, tokens, 8, 128)
				weight := values(8, 128)
				temperature := values(2, tokens, 1, 1).AsType(mlx.DTypeFloat32)
				for _, scales := range [][]*mlx.Array{{}, {weight}, {weight, temperature}} {
					inputs := append([]*mlx.Array{x, mlx.FromValue(float32(8))}, scales...)
					equal("L2 clip", compiledClipL2(inputs...)[0], clipL2Tensors(inputs...)[0])
				}
				gate := values(2, tokens, 8, 128)
				equal("attention gate", compiledAttentionGate(x, gate), gateAttention(x, gate, x.DType()))
			})
		}
	})
}
