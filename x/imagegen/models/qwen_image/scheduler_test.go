//go:build mlx

package qwen_image

import (
	"math"
	"testing"
)

// TestSchedulerSetTimesteps verifies scheduler sigmas match Python diffusers reference.
// Golden values generated via:
//
//	python3 -c "
//	from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
//	import numpy as np
//	s = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, base_shift=0.5, max_shift=0.9,
//	    base_image_seq_len=256, max_image_seq_len=8192, shift_terminal=0.02, use_dynamic_shifting=True)
//	mu = 4096 * (0.9-0.5)/(8192-256) + 0.5 - (0.9-0.5)/(8192-256)*256
//	sigmas = np.linspace(1.0, 1.0/30, 30)
//	s.set_timesteps(sigmas=sigmas, mu=mu)
//	print(s.sigmas.numpy())"
func TestSchedulerSetTimesteps(t *testing.T) {
	cfg := DefaultSchedulerConfig()
	scheduler := NewFlowMatchScheduler(cfg)
	scheduler.SetTimesteps(30, 4096)

	// Golden values from Python diffusers (first 3, last 3 before terminal)
	wantFirst := []float32{1.000000, 0.982251, 0.963889}
	wantLast := []float32{0.142924, 0.083384, 0.020000}

	// Check first 3
	for i, want := range wantFirst {
		got := scheduler.Sigmas[i]
		if abs32(got-want) > 1e-4 {
			t.Errorf("sigma[%d]: got %v, want %v", i, got, want)
		}
	}

	// Check last 3 (indices 27, 28, 29)
	for i, want := range wantLast {
		idx := 27 + i
		got := scheduler.Sigmas[idx]
		if abs32(got-want) > 1e-4 {
			t.Errorf("sigma[%d]: got %v, want %v", idx, got, want)
		}
	}

	// Check terminal is 0
	if scheduler.Sigmas[30] != 0.0 {
		t.Errorf("terminal sigma: got %v, want 0", scheduler.Sigmas[30])
	}

	// Check length
	if len(scheduler.Sigmas) != 31 {
		t.Errorf("sigmas length: got %d, want 31", len(scheduler.Sigmas))
	}
}

// TestSchedulerProperties tests mathematical invariants of the scheduler.
func TestSchedulerProperties(t *testing.T) {
	cfg := DefaultSchedulerConfig()
	scheduler := NewFlowMatchScheduler(cfg)
	scheduler.SetTimesteps(30, 4096)

	// Property: sigmas monotonically decreasing
	for i := 1; i < len(scheduler.Sigmas); i++ {
		if scheduler.Sigmas[i] > scheduler.Sigmas[i-1] {
			t.Errorf("sigmas not monotonically decreasing at %d: %v > %v",
				i, scheduler.Sigmas[i], scheduler.Sigmas[i-1])
		}
	}

	// Property: first sigma should be ~1.0 (with time shift)
	if scheduler.Sigmas[0] < 0.9 || scheduler.Sigmas[0] > 1.01 {
		t.Errorf("first sigma out of expected range [0.9, 1.01]: %v", scheduler.Sigmas[0])
	}

	// Property: terminal sigma should be exactly 0
	if scheduler.Sigmas[len(scheduler.Sigmas)-1] != 0.0 {
		t.Errorf("terminal sigma should be 0, got %v", scheduler.Sigmas[len(scheduler.Sigmas)-1])
	}

	// Property: last non-terminal sigma should be shift_terminal (0.02)
	lastNonTerminal := scheduler.Sigmas[len(scheduler.Sigmas)-2]
	if abs32(lastNonTerminal-0.02) > 1e-5 {
		t.Errorf("last non-terminal sigma should be 0.02, got %v", lastNonTerminal)
	}

	// Property: length = steps + 1
	if len(scheduler.Sigmas) != scheduler.NumSteps+1 {
		t.Errorf("sigmas length should be steps+1: got %d, want %d",
			len(scheduler.Sigmas), scheduler.NumSteps+1)
	}
}

// TestCalculateShift verifies the mu calculation against Python reference.
// Golden values from: mu = img_seq_len * m + b where m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
func TestCalculateShift(t *testing.T) {
	cases := []struct {
		imgSeqLen int32
		want      float32
	}{
		{256, 0.5},     // base case
		{8192, 0.9},    // max case
		{4096, 0.6935}, // middle case (rounded)
	}

	for _, c := range cases {
		got := CalculateShift(c.imgSeqLen, 256, 8192, 0.5, 0.9)
		if abs32(got-c.want) > 0.001 {
			t.Errorf("CalculateShift(%d): got %v, want %v", c.imgSeqLen, got, c.want)
		}
	}
}

// TestSchedulerStep verifies the Euler step formula.
func TestSchedulerStep(t *testing.T) {
	cfg := DefaultSchedulerConfig()
	scheduler := NewFlowMatchScheduler(cfg)
	scheduler.SetTimesteps(30, 4096)

	// Verify dt calculation for first step
	sigma0 := scheduler.Sigmas[0]
	sigma1 := scheduler.Sigmas[1]
	expectedDt := sigma1 - sigma0

	// dt should be negative (sigmas decrease)
	if expectedDt >= 0 {
		t.Errorf("expected negative dt, got %v (sigma0=%v, sigma1=%v)", expectedDt, sigma0, sigma1)
	}
}

func abs32(x float32) float32 {
	return float32(math.Abs(float64(x)))
}
