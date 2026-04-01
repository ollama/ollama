package kan

import (
	"fmt"
	"log/slog"
	"math"
	"sync"
)

// ShadowTrainer manages online training of KAN layers to match softmax output.
// It runs KAN forward passes in parallel with softmax and uses finite-difference
// gradient estimation to update the KAN coefficients.
type ShadowTrainer struct {
	cfg    Config
	layers map[string]*layerState
	mu     sync.RWMutex
}

type layerState struct {
	kan              *Layer
	stepCount        int
	emaLoss          float64
	convergenceCount int
	converged        bool
}

// NewShadowTrainer creates a new trainer with the given configuration.
func NewShadowTrainer(cfg Config) *ShadowTrainer {
	return &ShadowTrainer{
		cfg:    cfg,
		layers: make(map[string]*layerState),
	}
}

// LayerKey returns the map key for a given layer index.
func LayerKey(layerIdx int) string {
	return fmt.Sprintf("layer_%d", layerIdx)
}

// GetOrCreateLayer returns the KAN layer for the given key, creating it if needed.
func (s *ShadowTrainer) GetOrCreateLayer(key string) *Layer {
	s.mu.Lock()
	defer s.mu.Unlock()

	state, ok := s.layers[key]
	if !ok {
		state = &layerState{
			kan: NewLayer(s.cfg),
		}
		s.layers[key] = state
	}
	return state.kan
}

// IsConverged returns whether a specific layer's KAN has converged.
func (s *ShadowTrainer) IsConverged(key string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if state, ok := s.layers[key]; ok {
		return state.converged
	}
	return false
}

// IsFullyConverged returns whether all layers have converged.
func (s *ShadowTrainer) IsFullyConverged() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.layers) == 0 {
		return false
	}
	for _, state := range s.layers {
		if !state.converged {
			return false
		}
	}
	return true
}

// ShouldTrain returns whether this step should perform a training update,
// based on the TrainEveryN configuration.
func (s *ShadowTrainer) ShouldTrain(key string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	state, ok := s.layers[key]
	if !ok {
		return true // First step, always train
	}
	if state.converged {
		return false
	}
	return state.stepCount%s.cfg.TrainEveryN == 0
}

// TrainStep performs one training step for a layer's KAN.
//
// It computes the MSE loss between the softmax output and the KAN output,
// then uses finite-difference gradient estimation to update coefficients.
//
// Parameters:
//   - key: layer identifier (e.g., "layer_0")
//   - logits: pre-softmax attention logits (flat float32 slice)
//   - softmaxOut: the ground truth softmax output (flat float32 slice)
//   - seqK, seqQ: dimensions for row-wise normalization
//
// Returns the current loss value.
func (s *ShadowTrainer) TrainStep(key string, logits, softmaxOut []float32, seqK, seqQ int) float64 {
	s.mu.Lock()
	state, ok := s.layers[key]
	if !ok {
		state = &layerState{
			kan: NewLayer(s.cfg),
		}
		s.layers[key] = state
	}
	state.stepCount++
	s.mu.Unlock()

	if state.converged {
		return state.emaLoss
	}

	// Compute current KAN output and loss
	kanOut := state.kan.Forward(logits, seqK, seqQ)
	baseLoss := mse(softmaxOut, kanOut)

	// Finite-difference gradient estimation for each coefficient
	coeffs := state.kan.GetCoefficients()
	grads := make([]float32, len(coeffs))
	eps := s.cfg.GradEpsilon

	for i := range coeffs {
		// Perturb coefficient +epsilon
		perturbed := make([]float32, len(coeffs))
		copy(perturbed, coeffs)
		perturbed[i] += eps
		state.kan.UpdateCoefficients(perturbed)
		kanOutPlus := state.kan.Forward(logits, seqK, seqQ)
		lossPlus := mse(softmaxOut, kanOutPlus)

		// Single-sided finite difference: grad = (loss+ - loss) / eps
		grads[i] = float32((lossPlus - baseLoss) / float64(eps))
	}

	// SGD update
	newCoeffs := make([]float32, len(coeffs))
	for i := range coeffs {
		newCoeffs[i] = coeffs[i] - s.cfg.LearningRate*grads[i]
	}

	// Apply geometric mean normalization + redistribution
	state.kan.UpdateCoefficients(newCoeffs)

	// Update convergence tracking
	s.mu.Lock()
	defer s.mu.Unlock()

	// Compute loss after update
	finalOut := state.kan.Forward(logits, seqK, seqQ)
	finalLoss := mse(softmaxOut, finalOut)

	if state.emaLoss == 0 {
		state.emaLoss = finalLoss
	} else {
		state.emaLoss = 0.99*state.emaLoss + 0.01*finalLoss
	}

	if state.emaLoss < float64(s.cfg.ConvergenceThreshold) {
		state.convergenceCount++
		if state.convergenceCount >= s.cfg.ConvergenceWindow {
			state.converged = true
			slog.Info("KAN attention converged", "layer", key, "ema_loss", state.emaLoss, "steps", state.stepCount)
		}
	} else {
		state.convergenceCount = 0
	}

	if state.stepCount%100 == 0 {
		slog.Debug("KAN training progress", "layer", key, "step", state.stepCount, "ema_loss", state.emaLoss, "converged", state.converged)
	}

	return state.emaLoss
}

// GetLayerWeights returns the current KAN weights for a layer, or nil if not found.
func (s *ShadowTrainer) GetLayerWeights(key string) []float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if state, ok := s.layers[key]; ok {
		return state.kan.GetCoefficients()
	}
	return nil
}

// SetLayerWeights loads pre-trained weights for a layer and marks it converged.
func (s *ShadowTrainer) SetLayerWeights(key string, weights []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	state, ok := s.layers[key]
	if !ok {
		state = &layerState{
			kan: NewLayerFromWeights(s.cfg, weights),
		}
		s.layers[key] = state
	} else {
		state.kan.UpdateCoefficients(weights)
	}
	state.converged = true
}

// LayerKeys returns all registered layer keys.
func (s *ShadowTrainer) LayerKeys() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keys := make([]string, 0, len(s.layers))
	for k := range s.layers {
		keys = append(keys, k)
	}
	return keys
}

// Stats returns a human-readable summary of training progress.
func (s *ShadowTrainer) Stats() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()

	convergedCount := 0
	totalSteps := 0
	for _, state := range s.layers {
		if state.converged {
			convergedCount++
		}
		totalSteps += state.stepCount
	}

	return map[string]any{
		"total_layers":    len(s.layers),
		"converged":       convergedCount,
		"total_steps":     totalSteps,
		"fully_converged": convergedCount == len(s.layers) && len(s.layers) > 0,
	}
}

// mse computes mean squared error between two float32 slices.
func mse(expected, actual []float32) float64 {
	if len(expected) != len(actual) {
		n := len(expected)
		if len(actual) < n {
			n = len(actual)
		}
		if n == 0 {
			return math.MaxFloat64
		}
		sum := float64(0)
		for i := 0; i < n; i++ {
			d := float64(expected[i]) - float64(actual[i])
			sum += d * d
		}
		return sum / float64(n)
	}

	sum := float64(0)
	for i := range expected {
		d := float64(expected[i]) - float64(actual[i])
		sum += d * d
	}
	return sum / float64(len(expected))
}
