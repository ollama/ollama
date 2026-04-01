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

	// LearningRate for finite-difference SGD updates (default: 0.001)
	LearningRate float32

	// GradEpsilon for finite-difference gradient estimation (default: 1e-5)
	GradEpsilon float32

	// ConvergenceThreshold is the EMA loss below which a layer is considered converged (default: 1e-4)
	ConvergenceThreshold float32

	// ConvergenceWindow is the number of consecutive steps below threshold needed (default: 50)
	ConvergenceWindow int

	// TrainEveryN only trains every N-th inference step to reduce overhead (default: 1)
	TrainEveryN int

	// HotSwapEnabled enables automatic replacement of softmax with KAN after convergence
	HotSwapEnabled bool

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
		ConvergenceThreshold: 1e-4,
		ConvergenceWindow:    50,
		TrainEveryN:          1,
		HotSwapEnabled:       true,
	}
}
