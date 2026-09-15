package testutil

import (
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/tokenizer"
)

type perplexitySweepProbe struct {
	forwardScratch       *mlx.Array
	scratchLiveAtUnembed bool
}

func (*perplexitySweepProbe) LoadWeights(map[string]*mlx.Array) error { return nil }
func (*perplexitySweepProbe) NewCaches() []cache.Cache                { return nil }
func (*perplexitySweepProbe) Tokenizer() *tokenizer.Tokenizer         { return nil }
func (*perplexitySweepProbe) MaxContextLength() int                   { return 32 }

func (m *perplexitySweepProbe) Forward(b *batch.Batch, _ []cache.Cache) (*mlx.Array, *mlx.Array) {
	length := b.InputIDs.Dim(1)
	m.forwardScratch = mlx.Zeros(mlx.DTypeFloat32, 1, length, 4)
	hidden := mlx.AddScalar(m.forwardScratch, 1)
	return hidden, hidden
}

func (m *perplexitySweepProbe) Unembed(x *mlx.Array) *mlx.Array {
	m.scratchLiveAtUnembed = m.forwardScratch.Valid()
	return mlx.Zeros(mlx.DTypeFloat32, 1, x.Dim(1), 8)
}

func TestPerplexitySweepsForwardBeforeUnembed(t *testing.T) {
	mlxtest.Setup(t)
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}
	t.Cleanup(func() {
		mlx.Sweep()
		mlx.ClearCache()
	})

	tests := []struct {
		name  string
		score func(*perplexitySweepProbe) error
	}{
		{
			name: "harness",
			score: func(m *perplexitySweepProbe) error {
				_, _, _, err := scoreLastN(m, []int32{1, 2, 3, 4}, []int32{1, 2, 3, 4}, 4)
				return err
			},
		},
		{
			name: "window",
			score: func(m *perplexitySweepProbe) error {
				_, _, _, err := scoreSecondHalf(m, []int32{1, 2, 3, 4}, []int32{1, 2, 3, 4}, 2)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &perplexitySweepProbe{}
			if err := tt.score(m); err != nil {
				t.Fatal(err)
			}
			if m.scratchLiveAtUnembed {
				t.Fatal("forward scratch remained live when unembed started")
			}
			mlx.Sweep()
			mlx.ClearCache()
		})
	}
}
