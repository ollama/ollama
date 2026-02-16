//go:build mlx

package zimage

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// FlowMatchSchedulerConfig holds scheduler configuration
type FlowMatchSchedulerConfig struct {
	NumTrainTimesteps  int32   `json:"num_train_timesteps"`  // 1000
	Shift              float32 `json:"shift"`                // 3.0
	UseDynamicShifting bool    `json:"use_dynamic_shifting"` // false
}

// DefaultFlowMatchSchedulerConfig returns default config
func DefaultFlowMatchSchedulerConfig() *FlowMatchSchedulerConfig {
	return &FlowMatchSchedulerConfig{
		NumTrainTimesteps:  1000,
		Shift:              3.0,
		UseDynamicShifting: true, // Z-Image-Turbo uses dynamic shifting
	}
}

// FlowMatchEulerScheduler implements the Flow Match Euler discrete scheduler
// This is used in Z-Image-Turbo for fast sampling
type FlowMatchEulerScheduler struct {
	Config    *FlowMatchSchedulerConfig
	Timesteps []float32 // Discretized timesteps
	Sigmas    []float32 // Noise levels at each timestep
	NumSteps  int       // Number of inference steps
}

// NewFlowMatchEulerScheduler creates a new scheduler
func NewFlowMatchEulerScheduler(cfg *FlowMatchSchedulerConfig) *FlowMatchEulerScheduler {
	return &FlowMatchEulerScheduler{
		Config: cfg,
	}
}

// SetTimesteps sets up the scheduler for the given number of inference steps
func (s *FlowMatchEulerScheduler) SetTimesteps(numSteps int) {
	s.SetTimestepsWithMu(numSteps, 0)
}

// SetTimestepsWithMu sets up the scheduler with dynamic mu shift
func (s *FlowMatchEulerScheduler) SetTimestepsWithMu(numSteps int, mu float32) {
	s.NumSteps = numSteps

	// Create evenly spaced timesteps from 1.0 to 0.0 (flow matching goes t=1 to t=0)
	// Match Python: np.linspace(1.0, 0.0, num_inference_steps + 1)
	s.Timesteps = make([]float32, numSteps+1)
	s.Sigmas = make([]float32, numSteps+1)

	for i := 0; i <= numSteps; i++ {
		t := 1.0 - float32(i)/float32(numSteps)

		// Apply time shift if using dynamic shifting
		if s.Config.UseDynamicShifting && mu != 0 {
			t = s.timeShift(mu, t)
		}

		s.Timesteps[i] = t
		s.Sigmas[i] = t
	}
}

// timeShift applies the dynamic time shift (match Python)
func (s *FlowMatchEulerScheduler) timeShift(mu float32, t float32) float32 {
	if t <= 0 {
		return 0
	}
	// exp(mu) / (exp(mu) + (1/t - 1))
	expMu := float32(math.Exp(float64(mu)))
	return expMu / (expMu + (1.0/t - 1.0))
}

// Step performs one denoising step
// modelOutput: predicted velocity/noise from the model
// timestepIdx: current timestep index
// sample: current noisy sample
// Returns: denoised sample for next step
func (s *FlowMatchEulerScheduler) Step(modelOutput, sample *mlx.Array, timestepIdx int) *mlx.Array {
	// Get current and next sigma
	sigma := s.Sigmas[timestepIdx]
	sigmaNext := s.Sigmas[timestepIdx+1]

	// Euler step: x_{t-dt} = x_t + (sigma_next - sigma) * v_t
	// where v_t is the velocity predicted by the model
	dt := sigmaNext - sigma // This is negative (going from noise to clean)

	// x_next = x + dt * velocity
	scaledOutput := mlx.MulScalar(modelOutput, dt)
	return mlx.Add(sample, scaledOutput)
}

// ScaleSample scales the sample for model input (identity for flow matching)
func (s *FlowMatchEulerScheduler) ScaleSample(sample *mlx.Array, timestepIdx int) *mlx.Array {
	// Flow matching doesn't need scaling
	return sample
}

// GetTimestep returns the timestep value at the given index
func (s *FlowMatchEulerScheduler) GetTimestep(idx int) float32 {
	if idx < len(s.Timesteps) {
		return s.Timesteps[idx]
	}
	return 0.0
}

// GetTimesteps returns all timesteps (implements Scheduler interface)
func (s *FlowMatchEulerScheduler) GetTimesteps() []float32 {
	return s.Timesteps
}

// AddNoise adds noise to clean samples for a given timestep
// Used for img2img or inpainting
func (s *FlowMatchEulerScheduler) AddNoise(cleanSample, noise *mlx.Array, timestepIdx int) *mlx.Array {
	// In flow matching: x_t = (1-t) * x_0 + t * noise
	t := s.Timesteps[timestepIdx]
	oneMinusT := 1.0 - t

	scaledClean := mlx.MulScalar(cleanSample, oneMinusT)
	scaledNoise := mlx.MulScalar(noise, t)

	return mlx.Add(scaledClean, scaledNoise)
}

// InitNoise creates initial noise for sampling (BFloat16 for GPU efficiency)
func (s *FlowMatchEulerScheduler) InitNoise(shape []int32, seed int64) *mlx.Array {
	return mlx.RandomNormalWithDtype(shape, uint64(seed), mlx.DtypeBFloat16)
}

// GetLatentShape returns the latent shape for a given image size
func GetLatentShape(batchSize, height, width, latentChannels int32, patchSize int32) []int32 {
	// Latent is 8x smaller than image (VAE downscale)
	latentH := height / 8
	latentW := width / 8

	return []int32{batchSize, latentChannels, latentH, latentW}
}
