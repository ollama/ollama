package testutil

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

type perplexityReleaseProbe struct {
	forwardScratch       *mlx.Array
	scratchLiveAtUnembed bool
}

func (*perplexityReleaseProbe) LoadWeights(map[string]*mlx.Array) error { return nil }
func (*perplexityReleaseProbe) NewCaches() []cache.Cache                { return nil }
func (*perplexityReleaseProbe) Tokenizer() *tokenizer.Tokenizer         { return nil }
func (*perplexityReleaseProbe) MaxContextLength() int                   { return 32 }

func (m *perplexityReleaseProbe) Forward(b *batch.Batch, _ []cache.Cache) (*mlx.Array, *mlx.Array) {
	length := b.InputIDs.Dim(1)
	m.forwardScratch = mlx.Zeros(mlx.DTypeFloat32, 1, length, 4)
	hidden := mlx.AddScalar(m.forwardScratch, 1)
	return hidden, hidden
}

func (m *perplexityReleaseProbe) Unembed(x *mlx.Array) *mlx.Array {
	m.scratchLiveAtUnembed = !released(m.forwardScratch)
	return mlx.Zeros(mlx.DTypeFloat32, 1, x.Dim(1), 8)
}

// released reports whether a's scope has already ended (its handle freed).
// mlx exposes no validity check, so it relies on Scope.Attach refusing an
// array whose scope ended; a still-live array is handed straight back to the
// current scope.
func released(a *mlx.Array) (freed bool) {
	s := mlx.NewScope()
	defer func() {
		if r := recover(); r != nil {
			if !strings.Contains(fmt.Sprint(r), "used after its scope ended") {
				panic(r)
			}
			freed = true
		}
	}()
	s.Attach(a)
	s.Detach(a)
	return false
}

func TestPerplexityReleasesForwardBeforeUnembed(t *testing.T) {
	mlxtest.SkipIfUnavailable(t)

	tests := []struct {
		name  string
		score func(*perplexityReleaseProbe) error
	}{
		{
			name: "harness",
			score: func(m *perplexityReleaseProbe) error {
				_, _, _, err := scoreLastN(m, []int32{1, 2, 3, 4}, []int32{1, 2, 3, 4}, 4)
				return err
			},
		},
		{
			name: "window",
			score: func(m *perplexityReleaseProbe) error {
				_, _, _, err := scoreSecondHalf(m, []int32{1, 2, 3, 4}, []int32{1, 2, 3, 4}, 2)
				return err
			},
		},
	}

	for _, tt := range tests {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			m := &perplexityReleaseProbe{}
			if err := tt.score(m); err != nil {
				t.Fatal(err)
			}
			if m.scratchLiveAtUnembed {
				t.Fatal("forward scratch remained live when unembed started")
			}
			mlx.ClearCache()
		})
	}
}

// TestLogSumExpMatchesNaive checks the stable logsumexp against the naive
// form on values small enough that the naive form does not overflow, and
// that it stays finite where the naive form would.
func TestLogSumExpMatchesNaive(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		x := mlx.FromValues([]float32{1, 2, 3, -1, 0, 0.5}, 1, 2, 3)
		got := hostFloat32(logSumExp(x, 2))
		want := hostFloat32(mlx.Log(mlx.Exp(x).SumAxis(2, true)))
		if len(got) != 2 || len(want) != 2 {
			t.Fatalf("logSumExp shape: got %d values, want 2", len(got))
		}
		for i := range got {
			if d := got[i] - want[i]; d > 1e-5 || d < -1e-5 {
				t.Errorf("logSumExp[%d] = %v, want %v", i, got[i], want[i])
			}
		}

		big := hostFloat32(logSumExp(mlx.FromValues([]float32{1000, 1000}, 1, 2), 1))
		if len(big) != 1 || big[0] < 1000.69 || big[0] > 1000.70 {
			t.Errorf("logSumExp([1000, 1000]) = %v, want ~1000.693", big)
		}
	})
}
