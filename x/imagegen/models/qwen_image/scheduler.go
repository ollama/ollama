//go:build mlx

package qwen_image

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// SchedulerConfig holds FlowMatchEulerDiscreteScheduler configuration
type SchedulerConfig struct {
	NumTrainTimesteps int32   `json:"num_train_timesteps"` // 1000
	BaseShift         float32 `json:"base_shift"`          // 0.5
	MaxShift          float32 `json:"max_shift"`           // 0.9
	BaseImageSeqLen   int32   `json:"base_image_seq_len"`  // 256
	MaxImageSeqLen    int32   `json:"max_image_seq_len"`   // 8192
	ShiftTerminal     float32 `json:"shift_terminal"`      // 0.02
	UseDynamicShift   bool    `json:"use_dynamic_shifting"` // true
}

// DefaultSchedulerConfig returns config for FlowMatchEulerDiscreteScheduler
func DefaultSchedulerConfig() *SchedulerConfig {
	return &SchedulerConfig{
		NumTrainTimesteps: 1000,
		BaseShift:         0.5,
		MaxShift:          0.9, // Matches scheduler_config.json
		BaseImageSeqLen:   256,
		MaxImageSeqLen:    8192,
		ShiftTerminal:     0.02,
		UseDynamicShift:   true,
	}
}

// FlowMatchScheduler implements the Flow Match Euler discrete scheduler
type FlowMatchScheduler struct {
	Config    *SchedulerConfig
	Timesteps []float32
	Sigmas    []float32
	NumSteps  int
}

// NewFlowMatchScheduler creates a new scheduler
func NewFlowMatchScheduler(cfg *SchedulerConfig) *FlowMatchScheduler {
	return &FlowMatchScheduler{
		Config: cfg,
	}
}

// CalculateShift computes the dynamic shift based on image sequence length
// This matches Python's calculate_shift function
func CalculateShift(imageSeqLen int32, baseSeqLen int32, maxSeqLen int32, baseShift float32, maxShift float32) float32 {
	m := (maxShift - baseShift) / float32(maxSeqLen-baseSeqLen)
	b := baseShift - m*float32(baseSeqLen)
	mu := float32(imageSeqLen)*m + b
	return mu
}

// SetTimesteps sets up the scheduler for the given number of inference steps
// Matches Python diffusers FlowMatchEulerDiscreteScheduler behavior:
// 1. Create sigmas from sigma_max to sigma_min (linspace)
// 2. Apply time_shift with mu (if dynamic shifting)
// 3. Apply stretch_shift_to_terminal to make final value = shift_terminal
func (s *FlowMatchScheduler) SetTimesteps(numSteps int, imageSeqLen int32) {
	s.NumSteps = numSteps

	// Calculate mu for dynamic shifting
	var mu float32
	if s.Config.UseDynamicShift {
		mu = CalculateShift(
			imageSeqLen,
			s.Config.BaseImageSeqLen,
			s.Config.MaxImageSeqLen,
			s.Config.BaseShift,
			s.Config.MaxShift,
		)
	}

	// Step 1: Create sigmas from 1.0 to 1/num_steps
	// Python (pipeline_qwenimage.py:639):
	//   sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
	// This gives sigmas from 1.0 to 1/30 = 0.033 for 30 steps
	sigmas := make([]float32, numSteps)
	sigmaMax := float32(1.0)
	sigmaMin := 1.0 / float32(numSteps) // 1/30 = 0.033 for 30 steps
	if numSteps == 1 {
		sigmas[0] = sigmaMax
	} else {
		for i := 0; i < numSteps; i++ {
			sigmas[i] = sigmaMax + float32(i)*(sigmaMin-sigmaMax)/float32(numSteps-1)
		}
	}

	// Step 2: Apply time shift if using dynamic shifting
	if s.Config.UseDynamicShift && mu != 0 {
		for i := range sigmas {
			sigmas[i] = s.timeShift(mu, sigmas[i])
		}
	}

	// Step 3: Apply stretch_shift_to_terminal
	if s.Config.ShiftTerminal > 0 {
		sigmas = s.stretchShiftToTerminal(sigmas)
	}

	// Step 4: Append terminal sigma (0) and store
	// Note: Python's scheduler.timesteps are sigmas*1000, but the pipeline divides by 1000
	// before passing to transformer. We skip both steps and just use sigmas directly.
	s.Sigmas = make([]float32, numSteps+1)
	s.Timesteps = make([]float32, numSteps+1)
	for i := 0; i < numSteps; i++ {
		s.Sigmas[i] = sigmas[i]
		s.Timesteps[i] = sigmas[i]
	}
	s.Sigmas[numSteps] = 0.0
	s.Timesteps[numSteps] = 0.0
}

// stretchShiftToTerminal stretches and shifts the timestep schedule
// so the final value equals shift_terminal (matches Python behavior)
func (s *FlowMatchScheduler) stretchShiftToTerminal(sigmas []float32) []float32 {
	if len(sigmas) == 0 {
		return sigmas
	}

	// one_minus_z = 1 - t
	// scale_factor = one_minus_z[-1] / (1 - shift_terminal)
	// stretched_t = 1 - (one_minus_z / scale_factor)
	lastSigma := sigmas[len(sigmas)-1]
	scaleFactor := (1.0 - lastSigma) / (1.0 - s.Config.ShiftTerminal)

	// Handle edge case: if scaleFactor is 0 or near 0, skip stretch
	// This happens when lastSigma ≈ 1.0 (e.g., single step with timeshift)
	if scaleFactor < 1e-6 {
		return sigmas
	}

	result := make([]float32, len(sigmas))
	for i, t := range sigmas {
		oneMinusZ := 1.0 - t
		result[i] = 1.0 - (oneMinusZ / scaleFactor)
	}
	return result
}

// timeShift applies the dynamic time shift (exponential)
// exp(mu) / (exp(mu) + (1/t - 1))
func (s *FlowMatchScheduler) timeShift(mu float32, t float32) float32 {
	if t <= 0 {
		return 0
	}
	expMu := float32(math.Exp(float64(mu)))
	return expMu / (expMu + (1.0/t - 1.0))
}

// Step performs one denoising step
// modelOutput: predicted velocity from the transformer
// sample: current noisy sample
// timestepIdx: current timestep index
func (s *FlowMatchScheduler) Step(modelOutput, sample *mlx.Array, timestepIdx int) *mlx.Array {
	// Get current and next sigma
	sigma := s.Sigmas[timestepIdx]
	sigmaNext := s.Sigmas[timestepIdx+1]

	// Euler step: x_{t-dt} = x_t + (sigma_next - sigma) * v_t
	dt := sigmaNext - sigma

	// Upcast to float32 to avoid precision issues (matches Python diffusers)
	sampleF32 := mlx.AsType(sample, mlx.DtypeFloat32)
	modelOutputF32 := mlx.AsType(modelOutput, mlx.DtypeFloat32)

	scaledOutput := mlx.MulScalar(modelOutputF32, dt)
	result := mlx.Add(sampleF32, scaledOutput)

	// Cast back to original dtype
	return mlx.ToBFloat16(result)
}

// GetTimestep returns the timestep value at the given index
func (s *FlowMatchScheduler) GetTimestep(idx int) float32 {
	if idx < len(s.Timesteps) {
		return s.Timesteps[idx]
	}
	return 0.0
}

// InitNoise creates initial noise for sampling in unpacked format [B, C, T, H, W]
func (s *FlowMatchScheduler) InitNoise(shape []int32, seed int64) *mlx.Array {
	return mlx.RandomNormal(shape, uint64(seed))
}

// InitNoisePacked creates initial noise directly in packed format [B, L, C*4]
// This matches how Python diffusers generates noise - directly in packed space.
// Generating in unpacked format and then packing produces different spatial
// correlation structure, which affects model output quality.
func (s *FlowMatchScheduler) InitNoisePacked(batchSize, seqLen, channels int32, seed int64) *mlx.Array {
	shape := []int32{batchSize, seqLen, channels}
	return mlx.RandomNormal(shape, uint64(seed))
}

// GetLatentShape returns the latent shape for a given image size
// For qwen_image: VAE downscale is 8x (spatial), latent has 16 channels
func GetLatentShape(batchSize, height, width int32) []int32 {
	latentH := height / 8
	latentW := width / 8
	return []int32{batchSize, 16, 1, latentH, latentW} // [B, C, T, H, W]
}

// GetPatchedLatentShape returns the patchified latent shape
// After patchification: [B, L, C*patch_size^2] where L = H/2 * W/2
func GetPatchedLatentShape(batchSize, height, width, patchSize int32) []int32 {
	latentH := height / 8
	latentW := width / 8
	pH := latentH / patchSize
	pW := latentW / patchSize
	inChannels := int32(64) // 16 * patch_size^2
	return []int32{batchSize, pH * pW, inChannels}
}
