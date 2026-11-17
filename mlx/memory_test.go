package mlx

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestSetWiredLimitRejectsOversizeWithoutChangingLimit(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !GPUIsAvailable() {
			t.Skip("MLX GPU not available")
		}
		if err := checkWiredLimitRejectsOversize(); err != nil {
			t.Fatal(err)
		}
	})
}

func checkWiredLimitRejectsOversize() (err error) {
	maxRecommended, err := MaxRecommendedWorkingSetSize()
	if err != nil {
		return err
	}
	if maxRecommended == math.MaxInt {
		return errors.New("recommended working set cannot be exceeded by an int")
	}

	previous, err := SetWiredLimit(maxRecommended)
	if err != nil {
		return err
	}
	defer func() {
		if _, restoreErr := SetWiredLimit(previous); restoreErr != nil {
			err = errors.Join(err, fmt.Errorf("restore wired limit: %w", restoreErr))
		}
	}()

	if _, err := SetWiredLimit(maxRecommended + 1); err == nil {
		return errors.New("SetWiredLimit accepted a limit above the recommended working set")
	}

	current, err := SetWiredLimit(maxRecommended)
	if err != nil {
		return err
	}
	if current != maxRecommended {
		return fmt.Errorf("wired limit after rejected update = %d, want %d", current, maxRecommended)
	}

	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := Matmul(a, a)
	Eval(b)
	if got, want := b.Floats(), []float32{7, 10, 15, 22}; !slices.Equal(got, want) {
		return fmt.Errorf("evaluation after rejected update = %v, want %v", got, want)
	}
	return nil
}

func TestMeasurePeakMemoryPreservesReportingPeak(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !GPUIsAvailable() {
			t.Skip("GPU allocator counters required")
		}
		ResetPeakMemory()
		before := ActiveMemory()
		large := MeasurePeakMemory(func() {
			Scoped(func() { Eval(Zeros(DTypeFloat32, 4096)) })
		})
		DefaultStream().Synchronize()
		if large <= before {
			t.Fatal("fixture did not allocate")
		}
		small := MeasurePeakMemory(func() {
			Scoped(func() { Eval(Zeros(DTypeFloat32, 4)) })
		})
		if small >= large {
			t.Fatalf("small interval peak %d must be below %d", small, large)
		}
		if got := PeakMemory(); got != large {
			t.Fatalf("cumulative peak = %d, want %d", got, large)
		}
		DefaultStream().Synchronize()
		ResetPeakMemory()
		if got := PeakMemory(); got >= large {
			t.Fatalf("reset kept prior interval peak: %d", got)
		}
	})
}

func TestCacheLimitRejectsNegativeValues(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("negative limit reached the native unsigned API")
		}
	}()
	SetCacheLimit(-1)
}

func TestCacheLimitReleasesUnusedBuffers(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !GPUIsAvailable() {
			t.Skip("GPU allocator counters required")
		}
		previous := SetCacheLimit(0)
		defer SetCacheLimit(previous)
		ClearCache()
		Scoped(func() { Eval(Zeros(DTypeFloat32, 4096)) })
		DefaultStream().Synchronize()
		if got := CacheMemory(); got != 0 {
			t.Fatalf("disabled allocator pool retained %d bytes", got)
		}
	})
}
