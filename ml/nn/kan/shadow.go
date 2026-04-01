package kan

import (
	"fmt"
	"log/slog"
	"math"
	"sync"
)

// adamState holds the per-parameter moment estimates for Adam optimizer.
type adamState struct {
	m []float64 // First moment (mean of gradients)
	v []float64 // Second moment (mean of squared gradients)
	t int       // Timestep (for bias correction)
}

func newAdamState(n int) *adamState {
	return &adamState{
		m: make([]float64, n),
		v: make([]float64, n),
		t: 0,
	}
}

// ShadowTrainer manages online training of KAN layers to match softmax output.
// It runs KAN forward passes in parallel with softmax and uses finite-difference
// gradient estimation with Adam optimizer to update the KAN coefficients.
type ShadowTrainer struct {
	cfg    Config
	layers map[string]*layerState
	mu     sync.RWMutex
}

type layerState struct {
	kan              *Layer
	adam             *adamState
	stepCount        int
	emaLoss          float64
	convergenceCount int
	converged        bool

	// Phase 2: self-evolution after graduation
	phase2Active      bool
	phase2Adam        *adamState
	graduationWeights []float32 // checkpoint at graduation for drift detection
	phase2Steps       int
	emaSharpness      float64 // EMA of attention sharpness (negative entropy)

	// graduationSlope is the raw B-spline slope at the moment of convergence.
	// Used as the reference point for computing effectiveScale.
	graduationSlope float64

	// effectiveScale is the RELATIVE slope change from graduation.
	// effectiveScale = currentSlope / graduationSlope
	// At convergence: 1.0 (KAN matches softmax, graph should too).
	// After Phase 2 sharpening: >1.0 (steeper attention).
	// Used in the GGML graph: softmax(effectiveScale * logits).
	effectiveScale float64

	// Dynamic head spawning: plateau detection
	bestLoss      float64 // best EMA loss seen so far
	plateauCount  int     // steps since best loss improved significantly
	lastHeadSpawn int     // step at which last head was spawned
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
			kan:  NewLayer(s.cfg),
			adam: newAdamState(s.cfg.NumBasis),
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

// TrainStep performs one training step for a layer's KAN using Adam optimizer.
//
// It computes the MSE loss between the softmax output and the KAN output,
// uses finite-difference gradient estimation, then applies Adam
// (adaptive moment estimation) for the parameter update.
//
// Adam maintains per-parameter running averages of gradients (first moment)
// and squared gradients (second moment), with bias correction. This gives:
//   - Adaptive per-parameter learning rates
//   - Momentum for faster convergence
//   - Robustness to noisy gradients from finite differences
//
// Parameters:
//   - key: layer identifier (e.g., "layer_0")
//   - logits: pre-softmax attention logits (flat float32 slice)
//   - softmaxOut: the ground truth softmax output (flat float32 slice)
//   - seqK, seqQ: dimensions for row-wise normalization
//
// Returns the current EMA loss value.
func (s *ShadowTrainer) TrainStep(key string, logits, softmaxOut []float32, seqK, seqQ int) float64 {
	s.mu.Lock()
	state, ok := s.layers[key]
	if !ok {
		state = &layerState{
			kan:  NewLayer(s.cfg),
			adam: newAdamState(s.cfg.NumBasis),
		}
		s.layers[key] = state
	}
	state.stepCount++
	s.mu.Unlock()

	if state.converged {
		return state.emaLoss
	}

	// Compute current KAN output and loss using a scratch layer
	// to avoid mutating the live KAN during gradient estimation
	coeffs := state.kan.GetCoefficients()
	scratch := NewLayerFromWeights(s.cfg, coeffs)
	kanOut := scratch.Forward(logits, seqK, seqQ)
	baseLoss := mse(softmaxOut, kanOut)

	// Finite-difference gradient estimation for each coefficient
	grads := make([]float64, len(coeffs))
	eps := s.cfg.GradEpsilon

	for i := range coeffs {
		// Perturb coefficient +epsilon on scratch layer
		perturbed := make([]float32, len(coeffs))
		copy(perturbed, coeffs)
		perturbed[i] += eps
		scratch.UpdateCoefficients(perturbed)
		kanOutPlus := scratch.Forward(logits, seqK, seqQ)
		lossPlus := mse(softmaxOut, kanOutPlus)

		// Single-sided finite difference: grad = (loss+ - loss) / eps
		g := (lossPlus - baseLoss) / float64(eps)
		if math.IsNaN(g) || math.IsInf(g, 0) {
			g = 0
		}
		grads[i] = g
	}

	// Adam optimizer update
	adam := state.adam
	adam.t++
	beta1 := s.cfg.AdamBeta1
	beta2 := s.cfg.AdamBeta2
	epsAdam := s.cfg.AdamEpsilon
	lr := float64(s.cfg.LearningRate)

	// Bias correction factors
	bc1 := 1.0 - math.Pow(beta1, float64(adam.t))
	bc2 := 1.0 - math.Pow(beta2, float64(adam.t))

	newCoeffs := make([]float32, len(coeffs))
	for i := range coeffs {
		// Update biased first moment estimate: m = β1*m + (1-β1)*g
		adam.m[i] = beta1*adam.m[i] + (1.0-beta1)*grads[i]
		// Update biased second raw moment estimate: v = β2*v + (1-β2)*g²
		adam.v[i] = beta2*adam.v[i] + (1.0-beta2)*grads[i]*grads[i]

		// Bias-corrected estimates
		mHat := adam.m[i] / bc1
		vHat := adam.v[i] / bc2

		// Parameter update: θ = θ - lr * m̂ / (√v̂ + ε)
		update := lr * mHat / (math.Sqrt(vHat) + epsAdam)
		if math.IsNaN(update) || math.IsInf(update, 0) {
			update = 0
		}
		newCoeffs[i] = coeffs[i] - float32(update)
	}

	// Apply geometric mean normalization + redistribution
	state.kan.UpdateCoefficients(newCoeffs)

	// Update convergence tracking
	s.mu.Lock()
	defer s.mu.Unlock()

	// Compute loss after update
	finalOut := state.kan.Forward(logits, seqK, seqQ)
	finalLoss := mse(softmaxOut, finalOut)

	if math.IsNaN(finalLoss) || math.IsInf(finalLoss, 0) {
		finalLoss = state.emaLoss // Skip this step if numerical issues
	}

	if state.emaLoss == 0 || math.IsNaN(state.emaLoss) {
		state.emaLoss = finalLoss
		state.bestLoss = finalLoss
	} else {
		state.emaLoss = 0.99*state.emaLoss + 0.01*finalLoss
	}

	// === Dynamic head spawning: plateau detection ===
	// If loss hasn't improved significantly in PlateauWindow steps,
	// spawn a new cooperative head to break through the plateau.
	if state.bestLoss > 0 && state.emaLoss < state.bestLoss*(1.0-s.cfg.PlateauImprovement) {
		// Significant improvement — reset plateau counter
		state.bestLoss = state.emaLoss
		state.plateauCount = 0
	} else {
		state.plateauCount++
	}

	if state.plateauCount >= s.cfg.PlateauWindow &&
		state.kan.NumHeads() < s.cfg.MaxHeads &&
		state.stepCount-state.lastHeadSpawn > s.cfg.PlateauWindow {
		// Spawn a new cooperative head
		numHeads := state.kan.AddHead(s.cfg.NumBasis)

		// Extend Adam state for the new head's parameters
		newSize := numHeads * s.cfg.NumBasis
		newAdam := newAdamState(newSize)
		copy(newAdam.m, state.adam.m)
		copy(newAdam.v, state.adam.v)
		newAdam.t = state.adam.t
		state.adam = newAdam

		state.plateauCount = 0
		state.bestLoss = state.emaLoss
		state.lastHeadSpawn = state.stepCount

		slog.Info("KAN spawned new cooperative head",
			"layer", key, "heads", numHeads,
			"ema_loss", state.emaLoss, "step", state.stepCount)
	}

	if state.emaLoss < float64(s.cfg.ConvergenceThreshold) {
		state.convergenceCount++
		if state.convergenceCount >= s.cfg.ConvergenceWindow {
			state.converged = true
			state.graduationSlope = rawSlope(state.kan)
			state.effectiveScale = 1.0 // At convergence, KAN ≈ softmax
			// Activate Phase 2 self-evolution if enabled
			if s.cfg.Phase2Enabled && !state.phase2Active {
				state.phase2Active = true
				state.graduationWeights = state.kan.GetCoefficients()
				state.phase2Adam = newAdamState(len(state.graduationWeights))
				slog.Info("KAN Phase 2 activated: self-evolution enabled", "layer", key,
					"graduation_slope", state.graduationSlope)
			}
			slog.Info("KAN attention converged", "layer", key, "ema_loss", state.emaLoss,
				"steps", state.stepCount)
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
// Sets the graduation slope from the loaded weights so Phase 2 evolution
// can compute relative effective scale correctly.
func (s *ShadowTrainer) SetLayerWeights(key string, weights []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	state, ok := s.layers[key]
	if !ok {
		state = &layerState{
			kan:  NewLayerFromWeights(s.cfg, weights),
			adam: newAdamState(len(weights)),
		}
		s.layers[key] = state
	} else {
		state.kan.UpdateCoefficients(weights)
	}
	state.converged = true
	state.graduationSlope = rawSlope(state.kan)
	state.effectiveScale = 1.0 // Loaded weights = graduation state

	// Enable Phase 2 if configured
	if s.cfg.Phase2Enabled && !state.phase2Active {
		state.phase2Active = true
		state.graduationWeights = state.kan.GetCoefficients()
		state.phase2Adam = newAdamState(len(state.graduationWeights))
	}
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

// GetEffectiveScale returns the cached effective scale for a converged layer.
// Returns 1.0 (identity/standard softmax) if the layer hasn't converged yet.
func (s *ShadowTrainer) GetEffectiveScale(key string) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if state, ok := s.layers[key]; ok && state.effectiveScale > 0 {
		return state.effectiveScale
	}
	return 1.0
}

// rawSlope estimates the linear slope of the KAN's B-spline transform
// by evaluating at symmetric sample points and fitting y = α*x (forced
// through origin). The symmetric points make the intercept term cancel,
// so this correctly extracts just the slope even when the KAN has an offset.
func rawSlope(layer *Layer) float64 {
	// Sample points spanning the typical logit range
	points := []float32{-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0}
	sumXY := 0.0
	sumXX := 0.0
	for _, x := range points {
		y := float64(layer.EvaluateRaw(x))
		sumXY += float64(x) * y
		sumXX += float64(x) * float64(x)
	}
	if sumXX == 0 {
		return 1.0
	}
	scale := sumXY / sumXX
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return 1.0
	}
	return scale
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

// Phase2Step performs one self-evolution step for a graduated KAN layer.
//
// Unlike Phase 1 (match softmax), Phase 2 optimizes a self-supervised objective:
// **attention sharpness** -- the negative entropy of the attention weights.
// Lower entropy = sharper, more confident attention = the model "knows what to
// look at." This encourages the KAN to learn crisper attention patterns than
// softmax ever could, while the drift safety rail prevents catastrophic divergence.
//
// The signal: minimize H(attention) = -sum(p * log(p))
// Equivalent to maximizing sharpness = -H = sum(p * log(p))
//
// Parameters:
//   - key: layer identifier
//   - logits: current attention logits
//   - seqK, seqQ: dimensions
//
// Returns (sharpness, drifted). If drifted=true, the KAN was reverted to its
// graduation checkpoint because it strayed too far.
func (s *ShadowTrainer) Phase2Step(key string, logits []float32, seqK, seqQ int) (float64, bool) {
	s.mu.Lock()
	state, ok := s.layers[key]
	if !ok || !state.phase2Active {
		s.mu.Unlock()
		return 0, false
	}
	state.phase2Steps++
	s.mu.Unlock()

	// Only evolve every N steps
	if state.phase2Steps%s.cfg.Phase2EveryN != 0 {
		return state.emaSharpness, false
	}

	// Current KAN output (using a snapshot for thread safety)
	coeffs := state.kan.GetCoefficients()
	scratch := NewLayerFromWeights(s.cfg, coeffs)
	kanOut := scratch.Forward(logits, seqK, seqQ)
	baseSharpness := sharpness(kanOut, seqK, seqQ)

	// Finite-difference gradient estimation against sharpness
	// Uses scratch layer to avoid mutating the live KAN during perturbation
	grads := make([]float64, len(coeffs))
	eps := s.cfg.GradEpsilon

	for i := range coeffs {
		perturbed := make([]float32, len(coeffs))
		copy(perturbed, coeffs)
		perturbed[i] += eps
		scratch.UpdateCoefficients(perturbed)
		kanOutPlus := scratch.Forward(logits, seqK, seqQ)
		sharpPlus := sharpness(kanOutPlus, seqK, seqQ)

		// We want to MAXIMIZE sharpness, so gradient ascent:
		// grad = (sharp+ - sharp) / eps
		g := (sharpPlus - baseSharpness) / float64(eps)
		if math.IsNaN(g) || math.IsInf(g, 0) {
			g = 0
		}
		grads[i] = g
	}

	// Adam update for Phase 2 (gradient ASCENT -- add instead of subtract)
	adam := state.phase2Adam
	adam.t++
	beta1 := s.cfg.AdamBeta1
	beta2 := s.cfg.AdamBeta2
	epsAdam := s.cfg.AdamEpsilon
	lr := float64(s.cfg.Phase2LearningRate)

	bc1 := 1.0 - math.Pow(beta1, float64(adam.t))
	bc2 := 1.0 - math.Pow(beta2, float64(adam.t))

	newCoeffs := make([]float32, len(coeffs))
	for i := range coeffs {
		adam.m[i] = beta1*adam.m[i] + (1.0-beta1)*grads[i]
		adam.v[i] = beta2*adam.v[i] + (1.0-beta2)*grads[i]*grads[i]

		mHat := adam.m[i] / bc1
		vHat := adam.v[i] / bc2

		// ASCENT: add the update (maximizing sharpness)
		update := lr * mHat / (math.Sqrt(vHat) + epsAdam)
		if math.IsNaN(update) || math.IsInf(update, 0) {
			update = 0
		}
		newCoeffs[i] = coeffs[i] + float32(update)
	}

	// Apply geometric mean normalization + redistribution
	state.kan.UpdateCoefficients(newCoeffs)

	// Check drift: compute KL divergence from graduation checkpoint
	newOut := state.kan.Forward(logits, seqK, seqQ)

	// Get graduation output for comparison
	gradLayer := NewLayerFromWeights(s.cfg, state.graduationWeights)
	gradOut := gradLayer.Forward(logits, seqK, seqQ)

	drift := klDivergence(gradOut, newOut, seqK, seqQ)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Safety rail: revert if drifted too far
	if drift > s.cfg.Phase2MaxDrift {
		slog.Warn("KAN Phase 2 drift exceeded, reverting to graduation checkpoint",
			"layer", key, "drift", drift, "max", s.cfg.Phase2MaxDrift)
		state.kan.UpdateCoefficients(state.graduationWeights)
		state.phase2Adam = newAdamState(len(state.graduationWeights))
		state.effectiveScale = 1.0 // Reverted to graduation = back to softmax
		return state.emaSharpness, true
	}

	// Update sharpness tracking and effective scale
	newSharpness := sharpness(newOut, seqK, seqQ)
	if state.emaSharpness == 0 {
		state.emaSharpness = newSharpness
	} else {
		state.emaSharpness = 0.95*state.emaSharpness + 0.05*newSharpness
	}

	// Recompute effective scale as ratio of current slope to graduation slope.
	// This captures just the Phase 2 sharpening delta.
	if state.graduationSlope > 0 {
		state.effectiveScale = rawSlope(state.kan) / state.graduationSlope
	}

	if state.phase2Steps%100 == 0 {
		slog.Debug("KAN Phase 2 evolution", "layer", key, "step", state.phase2Steps,
			"sharpness", state.emaSharpness, "drift", drift, "effective_scale", state.effectiveScale)
	}

	return state.emaSharpness, false
}

// IsPhase2Active returns whether a layer is in Phase 2 self-evolution.
func (s *ShadowTrainer) IsPhase2Active(key string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if state, ok := s.layers[key]; ok {
		return state.phase2Active
	}
	return false
}

// sharpness computes the negative entropy of attention weights (higher = sharper).
// sharpness = sum(p * log(p)) for each row, averaged across rows.
// For a perfect one-hot distribution, sharpness = 0.
// For uniform distribution, sharpness = -log(n) (most negative).
func sharpness(weights []float32, seqK, seqQ int) float64 {
	totalSharpness := 0.0
	for q := 0; q < seqQ; q++ {
		rowStart := q * seqK
		rowEnd := rowStart + seqK
		if rowEnd > len(weights) {
			rowEnd = len(weights)
		}

		rowSharpness := 0.0
		for i := rowStart; i < rowEnd; i++ {
			p := float64(weights[i])
			if p > 1e-10 {
				rowSharpness += p * math.Log(p)
			}
		}
		totalSharpness += rowSharpness
	}

	if seqQ > 0 {
		totalSharpness /= float64(seqQ)
	}
	return totalSharpness
}

// klDivergence computes KL(P || Q) = sum(p * log(p/q)) averaged across rows.
// Used as a drift metric between graduation checkpoint and current output.
func klDivergence(p, q []float32, seqK, seqQ int) float64 {
	totalKL := 0.0
	for row := 0; row < seqQ; row++ {
		rowStart := row * seqK
		rowEnd := rowStart + seqK
		if rowEnd > len(p) || rowEnd > len(q) {
			break
		}

		rowKL := 0.0
		for i := rowStart; i < rowEnd; i++ {
			pv := float64(p[i])
			qv := float64(q[i])
			if pv > 1e-10 && qv > 1e-10 {
				rowKL += pv * math.Log(pv/qv)
			}
		}
		totalKL += rowKL
	}

	if seqQ > 0 {
		totalKL /= float64(seqQ)
	}
	return totalKL
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
