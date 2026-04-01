package kan

// Config holds configuration for the Geometric KAN attention mechanism.
type Config struct {
	// Enabled activates the KAN shadow training mode
	Enabled bool

	// NumBasis is the number of B-spline basis functions (default: 8)
	NumBasis int

	// Order is the B-spline order (default: 3, cubic)
	Order int

	// GridMin is the lower bound of the B-spline grid (default: -5.0)
	GridMin float32

	// GridMax is the upper bound of the B-spline grid (default: 5.0)
	GridMax float32

	// LearningRate is the step size for Adam optimizer (default: 0.001)
	LearningRate float32

	// GradEpsilon for finite-difference gradient estimation (default: 1e-5)
	GradEpsilon float32

	// AdamBeta1 is the exponential decay rate for Adam's first moment (default: 0.9)
	AdamBeta1 float64

	// AdamBeta2 is the exponential decay rate for Adam's second moment (default: 0.999)
	AdamBeta2 float64

	// AdamEpsilon is the small constant for numerical stability in Adam (default: 1e-8)
	AdamEpsilon float64

	// ConvergenceThreshold is the EMA loss below which a layer is considered converged (default: 1e-4)
	ConvergenceThreshold float32

	// ConvergenceWindow is the number of consecutive steps below threshold needed (default: 50)
	ConvergenceWindow int

	// TrainEveryN only trains every N-th inference step to reduce overhead (default: 1)
	TrainEveryN int

	// HotSwapEnabled enables automatic replacement of softmax with KAN after convergence
	HotSwapEnabled bool

	// Phase2Enabled enables self-evolution after graduation. When true, the KAN
	// continues to adapt its weights at inference time using a self-supervised
	// signal (attention sharpness / output confidence) instead of MSE-to-softmax.
	Phase2Enabled bool

	// Phase2LearningRate is a separate (typically smaller) learning rate for
	// Phase 2 self-evolution to avoid destabilizing the graduated KAN. (default: 0.0001)
	Phase2LearningRate float32

	// Phase2EveryN controls how often Phase 2 adaptation runs (default: 10).
	// Higher = less overhead, slower adaptation.
	Phase2EveryN int

	// Phase2MaxDrift is the maximum allowed KL divergence between the current
	// KAN output and the graduated checkpoint. If drift exceeds this, the KAN
	// reverts to the checkpoint. Safety rail. (default: 0.1)
	Phase2MaxDrift float64

	// MaxHeads is the maximum number of cooperative KAN heads per layer.
	// Additional heads are spawned dynamically when loss plateaus.
	// Each head specializes on a different part of the error surface,
	// and they combine additively in log-space before exp-normalize.
	// (default: 3)
	MaxHeads int

	// PlateauWindow is the number of training steps without significant
	// improvement before a new head is spawned. (default: 200)
	PlateauWindow int

	// PlateauImprovement is the minimum relative EMA loss improvement
	// to reset the plateau counter. 0.05 = 5% improvement. (default: 0.05)
	PlateauImprovement float64

	// SavePath is the directory for serialized KAN parameters (default: ~/.ollama/kan/)
	SavePath string
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Enabled:              true,
		NumBasis:             8,
		Order:                3,
		GridMin:              -5.0,
		GridMax:              5.0,
		LearningRate:         0.001,
		GradEpsilon:          1e-5,
		AdamBeta1:            0.9,
		AdamBeta2:            0.999,
		AdamEpsilon:          1e-8,
		ConvergenceThreshold: 1e-4,
		ConvergenceWindow:    50,
		TrainEveryN:          1,
		HotSwapEnabled:       true,
		Phase2Enabled:        true,
		Phase2LearningRate:   0.0001,
		Phase2EveryN:         10,
		Phase2MaxDrift:       0.1,
		MaxHeads:             3,
		PlateauWindow:        200,
		PlateauImprovement:   0.05,
	}
}
