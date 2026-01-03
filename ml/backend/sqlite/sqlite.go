// Package sqlite provides a SQLite-backed ML backend that uses the GGML
// compute infrastructure but reads/writes model data from SQLite databases.
//
// This enables:
// - Access tracking (know which tensors were used in forward pass)
// - Granular updates (UPDATE specific tensor rows, not bulk file rewrite)
// - Proper relational structure (indexed, queryable, normalized)
package sqlite

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/backend/ggml"
	"golang.org/x/sync/errgroup"
)

// Backend wraps GGML compute with SQLite data access.
type Backend struct {
	// Embed the GGML backend for compute operations
	*ggml.Backend

	// SQLite data source
	dataSource *ggml.SQLiteDataSource

	// Config from SQLite metadata
	config *SQLiteConfig
}

// SQLiteConfig implements fs.Config backed by SQLite metadata.
type SQLiteConfig struct {
	metadata map[string]string
	vocab    []string
	arch     string
}

func (c *SQLiteConfig) Architecture() string {
	return c.arch
}

func (c *SQLiteConfig) String(key string, defaultValue ...string) string {
	// Try exact key first
	if v, ok := c.metadata[key]; ok {
		return v
	}
	// Try with architecture prefix
	if !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		if v, ok := c.metadata[c.arch+"."+key]; ok {
			return v
		}
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

func (c *SQLiteConfig) Uint(key string, defaultValue ...uint32) uint32 {
	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return 0
	}
	v, _ := strconv.ParseUint(s, 10, 32)
	return uint32(v)
}

func (c *SQLiteConfig) Float(key string, defaultValue ...float32) float32 {
	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return 0
	}
	v, _ := strconv.ParseFloat(s, 32)
	return float32(v)
}

func (c *SQLiteConfig) Bool(key string, defaultValue ...bool) bool {
	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return false
	}
	v, _ := strconv.ParseBool(s)
	return v
}

func (c *SQLiteConfig) Strings(key string, defaultValue ...[]string) []string {
	// Special case: vocabulary tokens
	if key == "tokenizer.ggml.tokens" && len(c.vocab) > 0 {
		return c.vocab
	}

	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	var result []string
	json.Unmarshal([]byte(s), &result)
	return result
}

func (c *SQLiteConfig) Ints(key string, defaultValue ...[]int32) []int32 {
	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	var result []int32
	json.Unmarshal([]byte(s), &result)
	return result
}

func (c *SQLiteConfig) Floats(key string, defaultValue ...[]float32) []float32 {
	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	var result []float32
	json.Unmarshal([]byte(s), &result)
	return result
}

func (c *SQLiteConfig) Bools(key string, defaultValue ...[]bool) []bool {
	s := c.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	var result []bool
	json.Unmarshal([]byte(s), &result)
	return result
}

// New creates a new SQLite-backed backend.
func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	slog.Info("Opening SQLite model", "path", modelPath)

	// Open SQLite data source
	ds, err := ggml.NewSQLiteDataSource(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open SQLite model: %w", err)
	}

	// Load metadata for config
	metadata, err := ds.GetAllMetadata()
	if err != nil {
		ds.Close()
		return nil, fmt.Errorf("failed to load metadata: %w", err)
	}

	// Load vocabulary
	vocab, err := ds.GetVocabulary()
	if err != nil {
		ds.Close()
		return nil, fmt.Errorf("failed to load vocabulary: %w", err)
	}

	arch := metadata["general.architecture"]
	if arch == "" {
		arch = "llama" // default
	}

	config := &SQLiteConfig{
		metadata: metadata,
		vocab:    vocab,
		arch:     arch,
	}

	tensors := ds.AllTensors()

	slog.Info("SQLite model loaded",
		"architecture", arch,
		"tensors", len(tensors),
		"vocab_size", len(vocab),
		"total_bytes", format.HumanBytes2(ds.TotalBytes()),
	)

	// Note: Full GGML backend integration would require creating tensor structures
	// and loading data here. For now, we store the data source for later use.

	return &Backend{
		dataSource: ds,
		config:     config,
	}, nil
}

// Config returns the model configuration from SQLite.
func (b *Backend) Config() fs.Config {
	return b.config
}

// Close closes the backend.
func (b *Backend) Close() {
	if b.Backend != nil {
		b.Backend.Close()
	}
	if b.dataSource != nil {
		b.dataSource.Close()
	}
}

// Load loads model data from SQLite into compute backend.
func (b *Backend) Load(ctx context.Context, progress func(float32)) error {
	if b.Backend == nil {
		return fmt.Errorf("GGML backend not initialized - use LoadTensors first")
	}

	// Get tensors to load
	tensors := b.dataSource.AllTensors()
	totalBytes := b.dataSource.TotalBytes()

	var doneBytes atomic.Uint64

	g, loadCtx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.GOMAXPROCS(0))

	for _, t := range tensors {
		tensorName := t.Name
		g.Go(func() error {
			// Get tensor reader from SQLite
			reader, size, err := b.dataSource.GetTensorReader(tensorName)
			if err != nil {
				return fmt.Errorf("failed to get tensor reader for %s: %w", tensorName, err)
			}

			// Read tensor data
			data := make([]byte, size)
			if _, err := io.ReadFull(reader, data); err != nil {
				return fmt.Errorf("failed to read tensor %s: %w", tensorName, err)
			}

			// Get the GGML tensor and set its data
			ggmlTensor := b.Backend.Get(tensorName)
			if ggmlTensor == nil {
				slog.Warn("tensor not found in GGML backend", "name", tensorName)
				return nil
			}

			// Copy data to GGML tensor
			// Note: This requires access to internal GGML tensor pointer
			// which would need to be exposed from the ggml.Backend

			if loadCtx.Err() != nil {
				return loadCtx.Err()
			}

			if progress != nil {
				done := doneBytes.Add(uint64(size))
				progress(float32(done) / float32(totalBytes))
			}

			return nil
		})
	}

	return g.Wait()
}

// DataSource returns the SQLite data source for direct access.
func (b *Backend) DataSource() *ggml.SQLiteDataSource {
	return b.dataSource
}

// --- Training Support ---

// GetAccessedTensors returns IDs of tensors accessed during forward pass.
func (b *Backend) GetAccessedTensors() []int {
	return b.dataSource.GetAccessedTensorIDs()
}

// ClearAccessTracking clears tensor access tracking.
func (b *Backend) ClearAccessTracking() {
	b.dataSource.ClearAccessTracking()
}

// UpdateTensor updates a specific tensor's data in SQLite.
func (b *Backend) UpdateTensor(tensorID int, data []byte) error {
	return b.dataSource.UpdateTensor(tensorID, data)
}

// UpdateTensorByName updates a tensor by name.
func (b *Backend) UpdateTensorByName(name string, data []byte) error {
	return b.dataSource.UpdateTensorByName(name, data)
}

// BatchUpdateTensors updates multiple tensors in a single transaction.
func (b *Backend) BatchUpdateTensors(updates []ggml.TensorUpdate) error {
	return b.dataSource.BatchUpdate(updates)
}

// EnableTrainingMode reopens the database with read-write access.
func (b *Backend) EnableTrainingMode() error {
	return b.dataSource.OpenReadWrite()
}

// SaveUpdatedTensors saves all tensors that were accessed (and presumably modified).
// getTensorData should return the current tensor data from VRAM.
func (b *Backend) SaveUpdatedTensors(getTensorData func(tensorID int) ([]byte, error)) error {
	return b.dataSource.UpdateAccessedTensors(getTensorData)
}

func init() {
	ml.RegisterBackend("sqlite", func(modelPath string, params ml.BackendParams) (ml.Backend, error) {
		return New(modelPath, params)
	})
}
