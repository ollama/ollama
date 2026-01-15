//go:build mlx

package glm_image

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// FlowMatchSchedulerConfig holds scheduler configuration
type FlowMatchSchedulerConfig struct {
	NumTrainTimesteps  int32   `json:"num_train_timesteps"`   // 1000
	BaseShift          float32 `json:"base_shift"`            // 0.25
	MaxShift           float32 `json:"max_shift"`             // 0.75
	BaseImageSeqLen    int32   `json:"base_image_seq_len"`    // 256
	MaxImageSeqLen     int32   `json:"max_image_seq_len"`     // 4096
	UseDynamicShifting bool    `json:"use_dynamic_shifting"`  // true
	TimeShiftType      string  `json:"time_shift_type"`       // "linear"
}

// DefaultSchedulerConfig returns the default config for GLM-Image
func DefaultSchedulerConfig() *FlowMatchSchedulerConfig {
	return &FlowMatchSchedulerConfig{
		NumTrainTimesteps:  1000,
		BaseShift:          0.25,
		MaxShift:           0.75,
		BaseImageSeqLen:    256,
		MaxImageSeqLen:     4096,
		UseDynamicShifting: true,
		TimeShiftType:      "linear",
	}
}

// FlowMatchScheduler implements FlowMatchEulerDiscreteScheduler
type FlowMatchScheduler struct {
	Config    *FlowMatchSchedulerConfig
	Timesteps []float32 // Raw timesteps for transformer conditioning (unshifted)
	Sigmas    []float32 // Shifted sigmas for Euler step calculation
	NumSteps  int
}

// NewFlowMatchScheduler creates a new scheduler
func NewFlowMatchScheduler(cfg *FlowMatchSchedulerConfig) *FlowMatchScheduler {
	return &FlowMatchScheduler{Config: cfg}
}

// SetTimestepsWithDynamicShift sets timesteps with dynamic shifting based on image size
// Following diffusers: raw timesteps are used for conditioning, shifted sigmas for step calculation
func (s *FlowMatchScheduler) SetTimestepsWithDynamicShift(numSteps int, imgSeqLen int32) {
	s.NumSteps = numSteps

	// Calculate shift (mu) based on image sequence length
	mu := s.calculateShift(imgSeqLen)

	// Create timesteps: linspace from sigma_max_t to sigma_min_t
	// sigma_max = 1.0, sigma_min ~= 0.001 (near 0 but not exactly 0)
	// Then apply time shift and append terminal sigma=0
	s.Timesteps = make([]float32, numSteps)
	s.Sigmas = make([]float32, numSteps+1) // +1 for terminal sigma

	numTrainTimesteps := float32(s.Config.NumTrainTimesteps)

	// Create base sigmas: linspace from 1.0 to small value (matching diffusers)
	for i := 0; i < numSteps; i++ {
		// linspace from 1000 to ~20 (sigma_min * num_train_timesteps)
		tRaw := numTrainTimesteps - float32(i)*(numTrainTimesteps-1.0)/float32(numSteps-1)
		s.Timesteps[i] = tRaw

		// Convert to sigma [0, 1]
		sigma := tRaw / numTrainTimesteps

		// Apply time shift if enabled
		if s.Config.UseDynamicShifting && mu > 0 {
			sigma = s.applyShift(mu, sigma)
		}

		s.Sigmas[i] = sigma
	}

	// Append terminal sigma = 0 (the final clean image)
	s.Sigmas[numSteps] = 0
}

// calculateShift computes dynamic shift based on image sequence length
// Uses the sqrt-based formula from diffusers:
// m = (image_seq_len / base_seq_len) ** 0.5
// mu = m * max_shift + base_shift
func (s *FlowMatchScheduler) calculateShift(imgSeqLen int32) float32 {
	cfg := s.Config

	if !cfg.UseDynamicShifting {
		return 0
	}

	// Sqrt-based shift calculation (matches diffusers pipeline_glm_image.py)
	m := float32(math.Sqrt(float64(imgSeqLen) / float64(cfg.BaseImageSeqLen)))
	mu := m*cfg.MaxShift + cfg.BaseShift
	return mu
}

// applyShift applies time shift transformation
// mu: the computed shift value
// t: sigma value in [0, 1]
func (s *FlowMatchScheduler) applyShift(mu float32, t float32) float32 {
	if t <= 0 {
		return 0
	}
	if t >= 1 {
		return 1
	}

	// sigma=1.0 for both shift types
	sigma := float32(1.0)

	if s.Config.TimeShiftType == "linear" {
		// Linear: mu / (mu + (1/t - 1)^sigma)
		return mu / (mu + float32(math.Pow(float64(1.0/t-1.0), float64(sigma))))
	}

	// Exponential (default): exp(mu) / (exp(mu) + (1/t - 1)^sigma)
	expMu := float32(math.Exp(float64(mu)))
	return expMu / (expMu + float32(math.Pow(float64(1.0/t-1.0), float64(sigma))))
}

// Step performs one denoising step
func (s *FlowMatchScheduler) Step(modelOutput, sample *mlx.Array, stepIdx int) *mlx.Array {
	sigma := s.Sigmas[stepIdx]
	sigmaNext := s.Sigmas[stepIdx+1]

	// Euler step: x_{t-dt} = x_t + dt * v_t
	dt := sigmaNext - sigma // Negative (going from noise to clean)

	scaledOutput := mlx.MulScalar(modelOutput, dt)
	return mlx.Add(sample, scaledOutput)
}

// InitNoise creates initial noise
func (s *FlowMatchScheduler) InitNoise(shape []int32, seed int64) *mlx.Array {
	return mlx.RandomNormalWithDtype(shape, uint64(seed), mlx.DtypeBFloat16)
}

// AddNoise adds noise to clean samples for a given timestep (for img2img)
func (s *FlowMatchScheduler) AddNoise(cleanSample, noise *mlx.Array, timestepIdx int) *mlx.Array {
	// In flow matching: x_t = (1-sigma) * x_0 + sigma * noise
	// Use sigmas (shifted) for the interpolation
	sigma := s.Sigmas[timestepIdx]
	oneMinusSigma := 1.0 - sigma

	scaledClean := mlx.MulScalar(cleanSample, oneMinusSigma)
	scaledNoise := mlx.MulScalar(noise, sigma)

	return mlx.Add(scaledClean, scaledNoise)
}

// GetTimesteps returns all timesteps
func (s *FlowMatchScheduler) GetTimesteps() []float32 {
	return s.Timesteps
}
