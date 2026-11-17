//go:build linux || windows

package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestCUDAUnsupportedRetention(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.CUDAIsAvailable() {
			t.Skip("CUDA device required")
		}
		count := mlx.GPUCount()
		if count == 1 {
			integrated, err := mlx.IntegratedGPU()
			if err != nil {
				t.Fatal(err)
			}
			if integrated {
				t.Skip("requires discrete or multiple CUDA devices")
			}
		}
		observer, err := newSnapshotRetention(4)
		if observer != nil {
			observer.Close()
			t.Fatal("adaptive history enabled for an unsupported CUDA configuration")
		}
		// Unsupported Windows budget reporting may return an error along with
		// the nil observer. Neither outcome enables adaptive retention.
		t.Logf("CUDA devices=%d, observer error=%v", count, err)
		s := &batchScheduler{}
		s.slots = []*batchSlot{{cache: &prefixCache{}}}
		s.r = &Runner{parallel: 4}
		s.trimSnapshots(true)
		if s.retention != nil || !s.snapshotCheck.IsZero() {
			t.Fatal("scheduler enabled adaptive history without an observer")
		}
		r := &Runner{parallel: 4}
		if got := r.snapshotLimit(); got != 2<<30 {
			t.Fatalf("history allowance per slot = %d, want 2 GiB", got)
		}
	})
}
