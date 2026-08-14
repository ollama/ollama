package mlx

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestSetWiredLimitRejectsOversizeWithoutChangingLimit(t *testing.T) {
	skipIfNoMLX(t)
	if !GPUIsAvailable() {
		t.Skip("MLX GPU not available")
	}

	var testErr error
	withMLXThread(t, func() {
		testErr = checkWiredLimitRejectsOversize()
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
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
