// Package sqlite provides a SQLite-backed ML backend that uses the GGML
// compute infrastructure but reads model data from SQLite databases.
//
// This enables:
// - Granular tensor access (load only what's needed)
// - Access tracking for training (know which tensors were used)
// - Incremental updates (update individual tensors, not entire model)
package sqlite

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/fs"
	fssqlite "github.com/ollama/ollama/fs/sqlite"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/backend/ggml"
)

// Backend wraps the GGML backend but uses SQLite for model data.
type Backend struct {
	*ggml.Backend
	model *fssqlite.Model
}

// New creates a new SQLite-backed backend.
func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	// Check if this is a SQLite database
	if !strings.HasSuffix(modelPath, ".db") && !strings.HasSuffix(modelPath, ".sqlite") {
		return nil, fmt.Errorf("not a SQLite model file: %s", modelPath)
	}

	slog.Info("Opening SQLite model", "path", modelPath)

	// Open the SQLite model
	model, err := fssqlite.NewModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open SQLite model: %w", err)
	}

	slog.Info("SQLite model loaded",
		"architecture", model.KV().Architecture(),
		"tensors", len(model.Tensors().Items()),
		"size", model.Length(),
	)

	// For now, we need to use the GGML backend's compute infrastructure
	// This is a placeholder - full integration requires modifying ggml.New
	// to accept a data source interface
	//
	// TODO: Implement full SQLite backend integration
	// For phase 1, we'll use a hybrid approach where we:
	// 1. Create tensor structures from SQLite metadata
	// 2. Load tensor data from SQLite into GGML tensors
	// 3. Use GGML for compute

	return &Backend{
		model: model,
	}, nil
}

// Config returns the model configuration from SQLite.
func (b *Backend) Config() fs.Config {
	return b.model.KV()
}

// Close closes the backend and SQLite connection.
func (b *Backend) Close() {
	if b.Backend != nil {
		b.Backend.Close()
	}
	if b.model != nil {
		b.model.Close()
	}
}

// Model returns the underlying SQLite model for direct access.
func (b *Backend) Model() *fssqlite.Model {
	return b.model
}

// GetAccessedTensors returns the IDs of tensors accessed during forward pass.
func (b *Backend) GetAccessedTensors() []int {
	return b.model.GetAccessedTensors()
}

// ClearAccessTracking clears tensor access tracking.
func (b *Backend) ClearAccessTracking() {
	b.model.ClearAccessTracking()
}

// UpdateTensor updates a specific tensor's data (for training).
func (b *Backend) UpdateTensor(tensorID int, data []byte) error {
	return b.model.UpdateTensor(tensorID, data)
}

// Load loads model data from SQLite into compute backend.
func (b *Backend) Load(ctx context.Context, progress func(float32)) error {
	// TODO: Implement tensor loading from SQLite to GGML
	// This requires:
	// 1. For each tensor in SQLite:
	//    - Query tensor data from tensor_data table
	//    - Copy data to corresponding GGML tensor
	// 2. Track which tensors were loaded for access tracking
	return fmt.Errorf("SQLite backend Load not yet implemented - use hybrid approach")
}

func init() {
	ml.RegisterBackend("sqlite", func(modelPath string, params ml.BackendParams) (ml.Backend, error) {
		return New(modelPath, params)
	})
}
