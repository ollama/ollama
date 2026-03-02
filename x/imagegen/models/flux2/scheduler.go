//go:build mlx

package flux2

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// SchedulerConfig holds Flow-Match scheduler configuration
type SchedulerConfig struct {
	NumTrainTimesteps  int32   `json:"num_train_timesteps"`  // 1000
	Shift              float32 `json:"shift"`                // 3.0 for Klein
	UseDynamicShifting bool    `json:"use_dynamic_shifting"` // true
	TimeShiftType      string  `json:"time_shift_type"`      // "exponential" or "linear"
}

// DefaultSchedulerConfig returns default config for Klein
func DefaultSchedulerConfig() *SchedulerConfig {
	return &SchedulerConfig{
		NumTrainTimesteps:  1000,
		Shift:              3.0, // Klein uses 3.0
		UseDynamicShifting: true,
		TimeShiftType:      "exponential",
	}
}

// FlowMatchScheduler implements the Flow-Match Euler discrete scheduler
type FlowMatchScheduler struct {
	Config    *SchedulerConfig
	Timesteps []float32 // Discretized timesteps (t from 1 to 0)
	Sigmas    []float32 // Noise levels at each timestep
	NumSteps  int       // Number of inference steps
}

// NewFlowMatchScheduler creates a new scheduler
func NewFlowMatchScheduler(cfg *SchedulerConfig) *FlowMatchScheduler {
	return &FlowMatchScheduler{
		Config: cfg,
	}
}

// SetTimesteps sets up the scheduler for the given number of inference steps
func (s *FlowMatchScheduler) SetTimesteps(numSteps int) {
	s.SetTimestepsWithMu(numSteps, 0)
}

// SetTimestepsWithMu sets up scheduler matching diffusers set_timesteps(sigmas=..., mu=...)
func (s *FlowMatchScheduler) SetTimestepsWithMu(numSteps int, mu float32) {
	s.NumSteps = numSteps

	// diffusers: sigmas = linspace(1, 1/num_steps, num_steps)
	// Then applies time shift, appends 0.0 at end
	s.Sigmas = make([]float32, numSteps+1)

	for i := 0; i < numSteps; i++ {
		// linspace(1, 1/num_steps, num_steps)
		var sigma float32
		if numSteps == 1 {
			sigma = 1.0
		} else {
			sigma = 1.0 - float32(i)/float32(numSteps-1)*(1.0-1.0/float32(numSteps))
		}

		// Apply time shift if using dynamic shifting
		if s.Config.UseDynamicShifting && mu != 0 {
			sigma = s.timeShift(mu, sigma)
		} else {
			// If not dynamic shifting, apply fixed shift scaling like diffusers
			shift := s.Config.Shift
			sigma = shift * sigma / (1 + (shift-1)*sigma)
		}
		s.Sigmas[i] = sigma
	}
	// Append terminal zero
	s.Sigmas[numSteps] = 0.0

	// Timesteps scaled to training range (matches diffusers: timesteps = sigmas * num_train_timesteps)
	s.Timesteps = make([]float32, numSteps+1)
	for i, v := range s.Sigmas {
		s.Timesteps[i] = v * float32(s.Config.NumTrainTimesteps)
	}
}

// timeShift applies the dynamic time shift
func (s *FlowMatchScheduler) timeShift(mu float32, t float32) float32 {
	if t <= 0 {
		return 0
	}
	if s.Config.TimeShiftType == "linear" {
		return mu / (mu + (1.0/t-1.0))
	}
	// Default: exponential
	expMu := float32(math.Exp(float64(mu)))
	return expMu / (expMu + (1.0/t - 1.0))
}

// Step performs one denoising step
func (s *FlowMatchScheduler) Step(modelOutput, sample *mlx.Array, timestepIdx int) *mlx.Array {
	sigma := s.Sigmas[timestepIdx]
	sigmaNext := s.Sigmas[timestepIdx+1]

	// Euler step: x_{t-dt} = x_t + (sigma_next - sigma) * v_t
	dt := sigmaNext - sigma

	// Upcast to float32 for precision (matches diffusers)
	sampleF32 := mlx.AsType(sample, mlx.DtypeFloat32)
	outputF32 := mlx.AsType(modelOutput, mlx.DtypeFloat32)

	scaledOutput := mlx.MulScalar(outputF32, dt)
	result := mlx.Add(sampleF32, scaledOutput)

	// Cast back to bfloat16
	return mlx.ToBFloat16(result)
}

// GetTimestep returns the timestep value at the given index
func (s *FlowMatchScheduler) GetTimestep(idx int) float32 {
	if idx < len(s.Timesteps) {
		return s.Timesteps[idx]
	}
	return 0.0
}

// InitNoise creates initial noise for sampling
func (s *FlowMatchScheduler) InitNoise(shape []int32, seed int64) *mlx.Array {
	return mlx.RandomNormalWithDtype(shape, uint64(seed), mlx.DtypeBFloat16)
}

// CalculateShift computes the mu shift value for dynamic scheduling
// Matches diffusers compute_empirical_mu function
func CalculateShift(imgSeqLen int32, numSteps int) float32 {
	a1, b1 := float32(8.73809524e-05), float32(1.89833333)
	a2, b2 := float32(0.00016927), float32(0.45666666)

	seqLen := float32(imgSeqLen)

	if imgSeqLen > 4300 {
		return a2*seqLen + b2
	}

	m200 := a2*seqLen + b2
	m10 := a1*seqLen + b1

	a := (m200 - m10) / 190.0
	b := m200 - 200.0*a
	return a*float32(numSteps) + b
}
