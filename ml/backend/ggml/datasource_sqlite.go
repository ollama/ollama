// datasource_sqlite.go implements TensorDataSource for SQLite databases.
package ggml

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	fsggml "github.com/ollama/ollama/fs/ggml"
	_ "github.com/mattn/go-sqlite3"
)

// SQLiteDataSource implements TensorDataSource for SQLite model databases.
type SQLiteDataSource struct {
	db   *sql.DB
	path string

	// Cached tensor metadata
	tensors    []*fsggml.Tensor
	tensorMap  map[string]*fsggml.Tensor
	tensorIDs  map[string]int // tensor name -> database ID
	totalBytes uint64

	// Access tracking for training
	accessedMu sync.Mutex
	accessed   map[int]bool

	mu sync.RWMutex
}

// NewSQLiteDataSource creates a data source from a SQLite database.
func NewSQLiteDataSource(path string) (*SQLiteDataSource, error) {
	db, err := sql.Open("sqlite3", path+"?cache=shared&mode=ro")
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite: %w", err)
	}

	// Enable optimizations
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to set pragmas: %w", err)
	}

	ds := &SQLiteDataSource{
		db:        db,
		path:      path,
		tensorMap: make(map[string]*fsggml.Tensor),
		tensorIDs: make(map[string]int),
		accessed:  make(map[int]bool),
	}

	// Load tensor metadata
	if err := ds.loadTensorMetadata(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to load tensor metadata: %w", err)
	}

	return ds, nil
}

func (s *SQLiteDataSource) loadTensorMetadata() error {
	rows, err := s.db.Query(`
		SELECT id, name, dims, dtype_id, byte_size
		FROM tensors
		ORDER BY id
	`)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var id int
		var name, dimsStr string
		var dtypeID int
		var byteSize int64

		if err := rows.Scan(&id, &name, &dimsStr, &dtypeID, &byteSize); err != nil {
			return err
		}

		// Parse dimensions
		var dims []uint64
		if dimsStr != "" {
			if err := json.Unmarshal([]byte(dimsStr), &dims); err != nil {
				return fmt.Errorf("failed to parse dims for %s: %w", name, err)
			}
		}

		tensor := &fsggml.Tensor{
			Name:   name,
			Kind:   uint32(dtypeID),
			Offset: 0, // Not used for SQLite
			Shape:  dims,
		}

		s.tensors = append(s.tensors, tensor)
		s.tensorMap[name] = tensor
		s.tensorIDs[name] = id
		s.totalBytes += uint64(byteSize)
	}

	return rows.Err()
}

// GetTensorReader returns a reader for tensor data from SQLite.
func (s *SQLiteDataSource) GetTensorReader(name string) (io.Reader, int64, error) {
	s.mu.RLock()
	id, ok := s.tensorIDs[name]
	s.mu.RUnlock()

	if !ok {
		return nil, 0, fmt.Errorf("tensor not found: %s", name)
	}

	// Track access for training
	s.accessedMu.Lock()
	s.accessed[id] = true
	s.accessedMu.Unlock()

	// Read tensor data from database
	var data []byte
	err := s.db.QueryRow("SELECT data FROM tensor_data WHERE tensor_id = ?", id).Scan(&data)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read tensor data for %s: %w", name, err)
	}

	return bytes.NewReader(data), int64(len(data)), nil
}

// TensorInfo returns tensor metadata.
func (s *SQLiteDataSource) TensorInfo(name string) *fsggml.Tensor {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tensorMap[name]
}

// AllTensors returns all tensor metadata.
func (s *SQLiteDataSource) AllTensors() []*fsggml.Tensor {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tensors
}

// TotalBytes returns total tensor data size.
func (s *SQLiteDataSource) TotalBytes() uint64 {
	return s.totalBytes
}

// Close closes the database connection.
func (s *SQLiteDataSource) Close() error {
	return s.db.Close()
}

// GetAccessedTensorIDs returns IDs of tensors accessed since last clear.
func (s *SQLiteDataSource) GetAccessedTensorIDs() []int {
	s.accessedMu.Lock()
	defer s.accessedMu.Unlock()

	ids := make([]int, 0, len(s.accessed))
	for id := range s.accessed {
		ids = append(ids, id)
	}
	return ids
}

// ClearAccessTracking clears the access tracking.
func (s *SQLiteDataSource) ClearAccessTracking() {
	s.accessedMu.Lock()
	s.accessed = make(map[int]bool)
	s.accessedMu.Unlock()
}

// GetTensorID returns the database ID for a tensor name.
func (s *SQLiteDataSource) GetTensorID(name string) (int, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	id, ok := s.tensorIDs[name]
	return id, ok
}

// DB returns the underlying database for direct access (e.g., for updates).
func (s *SQLiteDataSource) DB() *sql.DB {
	return s.db
}

// Path returns the database file path.
func (s *SQLiteDataSource) Path() string {
	return s.path
}

// --- Metadata access for fs.Config compatibility ---

// GetMetadata returns a metadata value from the database.
func (s *SQLiteDataSource) GetMetadata(key string) (string, error) {
	var value string
	err := s.db.QueryRow("SELECT value FROM metadata WHERE key = ?", key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return value, err
}

// GetAllMetadata returns all metadata as a map.
func (s *SQLiteDataSource) GetAllMetadata() (map[string]string, error) {
	rows, err := s.db.Query("SELECT key, value FROM metadata")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[string]string)
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return nil, err
		}
		result[key] = value
	}
	return result, rows.Err()
}

// GetVocabulary returns all tokens ordered by ID.
func (s *SQLiteDataSource) GetVocabulary() ([]string, error) {
	rows, err := s.db.Query("SELECT token_string FROM vocab ORDER BY token_id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tokens []string
	for rows.Next() {
		var token string
		if err := rows.Scan(&token); err != nil {
			return nil, err
		}
		tokens = append(tokens, token)
	}
	return tokens, rows.Err()
}

// --- Training Update Support ---

// UpdateTensor writes updated tensor data back to the database.
// This is the key operation for training - granular updates instead of bulk rewrite.
func (s *SQLiteDataSource) UpdateTensor(tensorID int, data []byte) error {
	_, err := s.db.Exec(
		"UPDATE tensor_data SET data = ? WHERE tensor_id = ?",
		data, tensorID,
	)
	return err
}

// UpdateTensorByName writes updated tensor data by name.
func (s *SQLiteDataSource) UpdateTensorByName(name string, data []byte) error {
	id, ok := s.GetTensorID(name)
	if !ok {
		return fmt.Errorf("tensor not found: %s", name)
	}
	return s.UpdateTensor(id, data)
}

// BatchUpdate performs multiple tensor updates in a single transaction.
// This is more efficient for updating multiple tensors after a training step.
type TensorUpdate struct {
	TensorID int
	Data     []byte
}

func (s *SQLiteDataSource) BatchUpdate(updates []TensorUpdate) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare("UPDATE tensor_data SET data = ? WHERE tensor_id = ?")
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, u := range updates {
		if _, err := stmt.Exec(u.Data, u.TensorID); err != nil {
			return fmt.Errorf("failed to update tensor %d: %w", u.TensorID, err)
		}
	}

	return tx.Commit()
}

// UpdateAccessedTensors updates only the tensors that were accessed during forward pass.
// This is the core training loop optimization:
// 1. Forward pass tracks which tensors were read
// 2. Backward pass computes gradients only for those tensors
// 3. This function persists only the changed tensors
func (s *SQLiteDataSource) UpdateAccessedTensors(getTensorData func(tensorID int) ([]byte, error)) error {
	accessed := s.GetAccessedTensorIDs()
	if len(accessed) == 0 {
		return nil
	}

	updates := make([]TensorUpdate, 0, len(accessed))
	for _, id := range accessed {
		data, err := getTensorData(id)
		if err != nil {
			return fmt.Errorf("failed to get updated data for tensor %d: %w", id, err)
		}
		updates = append(updates, TensorUpdate{TensorID: id, Data: data})
	}

	return s.BatchUpdate(updates)
}

// OpenReadWrite reopens the database with read-write access for training.
func (s *SQLiteDataSource) OpenReadWrite() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Close read-only connection
	if err := s.db.Close(); err != nil {
		return err
	}

	// Reopen with read-write
	db, err := sql.Open("sqlite3", s.path+"?cache=shared")
	if err != nil {
		return err
	}

	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"); err != nil {
		db.Close()
		return err
	}

	s.db = db
	return nil
}
