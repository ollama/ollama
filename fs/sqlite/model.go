// model.go provides a compatibility layer that makes SQLite database look like
// the GGML model interface, allowing the existing backend to use SQLite data.
package sqlite

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

// Model wraps a SQLite database to provide the same interface as fsggml.GGML.
// This allows the existing GGML backend to use SQLite as a data source.
type Model struct {
	db     *DB
	kv     *KV
	length int64
}

// NewModel creates a Model from a SQLite database path.
func NewModel(dbPath string) (*Model, error) {
	db, err := Open(dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite model: %w", err)
	}

	// Calculate total size (approximate from tensor bytes)
	tensors, err := db.LoadTensors()
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to load tensors: %w", err)
	}

	var totalBytes int64
	for _, t := range tensors {
		totalBytes += t.ByteSize
	}

	return &Model{
		db:     db,
		kv:     &KV{db: db},
		length: totalBytes,
	}, nil
}

// NewModelReadWrite creates a Model with read-write access for training.
func NewModelReadWrite(dbPath string) (*Model, error) {
	db, err := OpenReadWrite(dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite model: %w", err)
	}

	tensors, err := db.LoadTensors()
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to load tensors: %w", err)
	}

	var totalBytes int64
	for _, t := range tensors {
		totalBytes += t.ByteSize
	}

	return &Model{
		db:     db,
		kv:     &KV{db: db},
		length: totalBytes,
	}, nil
}

// Close closes the underlying database.
func (m *Model) Close() error {
	return m.db.Close()
}

// DB returns the underlying SQLite database for direct access.
func (m *Model) DB() *DB {
	return m.db
}

// Length returns the total model size in bytes.
func (m *Model) Length() int64 {
	return m.length
}

// KV returns the key-value metadata interface.
func (m *Model) KV() *KV {
	return m.kv
}

// Tensors returns the tensor list interface.
func (m *Model) Tensors() *Tensors {
	return &Tensors{db: m.db}
}

// --- KV implements fs.Config and fsggml.KV-like interface ---

// KV provides key-value metadata access from SQLite, compatible with fs.Config.
type KV struct {
	db *DB
}

// Architecture returns the model architecture name.
func (kv *KV) Architecture() string {
	return kv.String("general.architecture", "unknown")
}

// String returns a string metadata value.
func (kv *KV) String(key string, defaultValue ...string) string {
	return kv.db.String(key, defaultValue...)
}

// Uint returns a uint32 metadata value.
func (kv *KV) Uint(key string, defaultValue ...uint32) uint32 {
	return kv.db.Uint(key, defaultValue...)
}

// Float returns a float32 metadata value.
func (kv *KV) Float(key string, defaultValue ...float32) float32 {
	return kv.db.Float(key, defaultValue...)
}

// Bool returns a bool metadata value.
func (kv *KV) Bool(key string, defaultValue ...bool) bool {
	return kv.db.Bool(key, defaultValue...)
}

// Strings returns a string array metadata value.
func (kv *KV) Strings(key string, defaultValue ...[]string) []string {
	// Special case: vocabulary tokens come from vocab table
	if key == "tokenizer.ggml.tokens" {
		vocab := kv.db.GetVocabulary()
		tokens, err := vocab.GetAllTokens()
		if err == nil && len(tokens) > 0 {
			return tokens
		}
	}

	return kv.db.Strings(key, defaultValue...)
}

// Ints returns an int32 array metadata value.
func (kv *KV) Ints(key string, defaultValue ...[]int32) []int32 {
	return kv.db.Ints(key, defaultValue...)
}

// Floats returns a float32 array metadata value.
func (kv *KV) Floats(key string, defaultValue ...[]float32) []float32 {
	return kv.db.Floats(key, defaultValue...)
}

// Bools returns a bool array metadata value.
func (kv *KV) Bools(key string, defaultValue ...[]bool) []bool {
	return kv.db.Bools(key, defaultValue...)
}

// BlockCount returns the number of transformer blocks.
func (kv *KV) BlockCount() uint64 {
	return uint64(kv.Uint("block_count"))
}

// FileType returns the model file type.
func (kv *KV) FileType() fsggml.FileType {
	return fsggml.FileType(kv.Uint("general.file_type"))
}

// --- Tensors provides tensor list access ---

// Tensors provides access to tensor metadata from SQLite.
type Tensors struct {
	db     *DB
	offset uint64 // Always 0 for SQLite (data is in separate table)
}

// Offset returns the tensor data offset (always 0 for SQLite).
func (t *Tensors) Offset() uint64 {
	return 0
}

// Items returns all tensor metadata as fsggml.Tensor compatible structs.
func (t *Tensors) Items() []*fsggml.Tensor {
	tensors, err := t.db.LoadTensors()
	if err != nil {
		return nil
	}

	result := make([]*fsggml.Tensor, len(tensors))
	for i, ti := range tensors {
		result[i] = &fsggml.Tensor{
			Name:   ti.Name,
			Kind:   uint32(ti.DTypeID),
			Offset: 0, // Not used for SQLite - we query by ID
			Shape:  ti.Dims,
		}
	}
	return result
}

// GroupLayers returns tensors grouped by layer.
func (t *Tensors) GroupLayers() map[string][]*fsggml.Tensor {
	items := t.Items()
	groups := make(map[string][]*fsggml.Tensor)

	for _, tensor := range items {
		// Extract layer key from tensor name
		parts := strings.Split(tensor.Name, ".")
		var key string
		for i, part := range parts {
			// Check if this part is a layer number
			if _, err := strconv.Atoi(part); err == nil && i > 0 {
				key = parts[i-1]
				break
			}
		}
		if key == "" {
			// No layer number found, use first part
			if len(parts) > 0 {
				key = parts[0]
			}
		}
		groups[key] = append(groups[key], tensor)
	}

	return groups
}

// --- TensorReader provides tensor data access ---

// TensorReader provides io.Reader interface for tensor data.
type TensorReader struct {
	db       *DB
	tensorID int
	data     []byte
	offset   int
}

// NewTensorReader creates a reader for a tensor's data.
func (m *Model) NewTensorReader(name string) (*TensorReader, *TensorInfo, error) {
	ti, err := m.db.GetTensorByName(name)
	if err != nil {
		return nil, nil, err
	}
	if ti == nil {
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}

	data, err := m.db.ReadTensorData(ti.ID)
	if err != nil {
		return nil, nil, err
	}

	return &TensorReader{
		db:       m.db,
		tensorID: ti.ID,
		data:     data,
		offset:   0,
	}, ti, nil
}

// Read implements io.Reader.
func (r *TensorReader) Read(p []byte) (n int, err error) {
	if r.offset >= len(r.data) {
		return 0, io.EOF
	}

	n = copy(p, r.data[r.offset:])
	r.offset += n
	return n, nil
}

// Size returns the total data size.
func (r *TensorReader) Size() int64 {
	return int64(len(r.data))
}

// --- Metadata helpers ---

// SetMetadata sets a metadata value (for training configuration updates).
func (m *Model) SetMetadata(key, value string) error {
	_, err := m.db.db.Exec(
		"INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
		key, value,
	)
	return err
}

// SetMetadataJSON sets a JSON-encoded metadata value.
func (m *Model) SetMetadataJSON(key string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return m.SetMetadata(key, string(data))
}

// GetAccessedTensors returns tensor IDs accessed since last clear.
func (m *Model) GetAccessedTensors() []int {
	return m.db.GetAccessedTensorIDs()
}

// ClearAccessTracking clears access tracking.
func (m *Model) ClearAccessTracking() {
	m.db.ClearAccessTracking()
}

// UpdateTensor updates a tensor's data.
func (m *Model) UpdateTensor(tensorID int, data []byte) error {
	return m.db.UpdateTensorData(tensorID, data)
}
