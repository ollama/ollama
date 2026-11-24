package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestParallelContextUsesExplicitNumCtx(t *testing.T) {
	const trained = 1 << 20
	for _, tc := range []struct {
		name string
		auto bool
		want int
	}{
		{"VRAM tier", true, 256 << 10},
		{"explicit", false, 4096},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := Client{softContextLength: 4096, numCtxAuto: tc.auto}
			if got := parallelContextLength(trained, c.parallelContextHint()); got != tc.want {
				t.Errorf("projected context = %d, want %d", got, tc.want)
			}
			if got := c.reportedContextLength(trained); got != 4096 {
				t.Errorf("reported context = %d, want 4096", got)
			}
		})
	}
}

func TestParallelContextLength(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		trained, requested, want int
	}{
		{"unknown model size", 0, 4096, 0},
		{"short model", 8192, 0, 8192},
		{"short requested window", 1 << 20, 4096, 4096},
		{"request exceeds model", 8192, 32768, 8192},
		{"large model default", 1 << 20, 0, 256 << 10},
		{"large requested window", 1 << 20, 512 << 10, 256 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := parallelContextLength(tc.trained, tc.requested); got != tc.want {
				t.Fatalf("projected context = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestDefaultParallelism(t *testing.T) {
	const gib = 1 << 30
	for _, tc := range []struct {
		name                        string
		fullContextBytes, available int64
		want                        int
	}{
		{"unknown", 0, 128 * gib, 1},
		{"pressure", 4 * gib, -gib, 1},
		{"tight", 4 * gib, 10 * gib, 1},
		{"two", 4 * gib, 12 * gib, 2},
		{"four", 4 * gib, 20 * gib, 4},
		{"eight", 4 * gib, 36 * gib, 8},
		{"long_context", 16 * gib, 36 * gib, 2},
		{"large_system", 4 * gib, 512 * gib, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := defaultParallelism(tc.fullContextBytes, tc.available); got != tc.want {
				t.Fatalf("got %d clients, want %d", got, tc.want)
			}
		})
	}
}

func TestParallelismPreservesExplicitSetting(t *testing.T) {
	r := Runner{parallel: 3}
	r.configureParallelism()
	if r.parallel != 3 {
		t.Fatal("explicit parallelism changed")
	}
}

func TestModelShapesBudget(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &scoringTestModel{}
		width, bytes := modelShapes(m, nil, 1024)
		// One float32 K/V value per token, two float32 recurrent values,
		// and eight hidden values retained while scoring.
		const target = 2*1024*2*4 + 3*2*4
		const hidden = 2 * 1024 * 8 * 4
		if width != 8 || bytes != target+hidden {
			t.Fatalf("width=%d bytes=%d, want 8 and %d", width, bytes, target+hidden)
		}
		width, bytes = modelShapes(m, nil, 0)
		if width != 8 || bytes != 0 {
			t.Fatalf("explicit capacity: width=%d bytes=%d, want 8 and 0", width, bytes)
		}
	})
}
