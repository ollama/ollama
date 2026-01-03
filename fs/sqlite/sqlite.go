// Package sqlite provides SQLite-backed model storage for ollama.
// This replaces GGUF file access with proper relational database queries,
// enabling granular tensor access and incremental updates for training.
package sqlite

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// DB wraps a SQLite database containing model weights and metadata.
type DB struct {
	db   *sql.DB
	path string

	// Cached metadata for fs.Config implementation
	metadataCache map[string]string
	metadataMu    sync.RWMutex

	// Cached tensor info
	tensorsCache []*TensorInfo
	tensorsMu    sync.RWMutex

	// Track accessed tensors for training
	accessedTensors map[int]bool
	accessMu        sync.Mutex
}

// TensorInfo holds metadata about a tensor from the tensors table.
type TensorInfo struct {
	ID           int
	Name         string
	Layer        int
	Component    string
	Subcomponent string
	Dims         []uint64
	DType        string
	DTypeID      int
	NumElements  int64
	ByteSize     int64
}

// Open opens a SQLite model database.
func Open(path string) (*DB, error) {
	sqlDB, err := sql.Open("sqlite3", path+"?mode=ro&cache=shared")
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite db: %w", err)
	}

	// Enable WAL mode for better concurrent read performance
	if _, err := sqlDB.Exec("PRAGMA journal_mode=WAL"); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("failed to set WAL mode: %w", err)
	}

	db := &DB{
		db:              sqlDB,
		path:            path,
		metadataCache:   make(map[string]string),
		accessedTensors: make(map[int]bool),
	}

	// Preload metadata cache
	if err := db.loadMetadataCache(); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("failed to load metadata: %w", err)
	}

	return db, nil
}

// OpenReadWrite opens a SQLite model database for read-write access.
func OpenReadWrite(path string) (*DB, error) {
	sqlDB, err := sql.Open("sqlite3", path+"?cache=shared")
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite db: %w", err)
	}

	// Enable WAL mode for better concurrent access
	if _, err := sqlDB.Exec("PRAGMA journal_mode=WAL"); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("failed to set WAL mode: %w", err)
	}

	db := &DB{
		db:              sqlDB,
		path:            path,
		metadataCache:   make(map[string]string),
		accessedTensors: make(map[int]bool),
	}

	// Preload metadata cache
	if err := db.loadMetadataCache(); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("failed to load metadata: %w", err)
	}

	return db, nil
}

func (db *DB) loadMetadataCache() error {
	rows, err := db.db.Query("SELECT key, value FROM metadata")
	if err != nil {
		return err
	}
	defer rows.Close()

	db.metadataMu.Lock()
	defer db.metadataMu.Unlock()

	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return err
		}
		db.metadataCache[key] = value
	}

	return rows.Err()
}

// Close closes the database connection.
func (db *DB) Close() error {
	return db.db.Close()
}

// Path returns the database file path.
func (db *DB) Path() string {
	return db.path
}

// --- fs.Config interface implementation ---

// Architecture returns the model architecture name.
func (db *DB) Architecture() string {
	return db.String("general.architecture", "unknown")
}

// String returns a string metadata value.
func (db *DB) String(key string, defaultValue ...string) string {
	db.metadataMu.RLock()
	val, ok := db.metadataCache[key]
	db.metadataMu.RUnlock()

	if ok {
		return val
	}

	// Try with architecture prefix
	arch := db.metadataCache["general.architecture"]
	if arch != "" && !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		db.metadataMu.RLock()
		val, ok = db.metadataCache[arch+"."+key]
		db.metadataMu.RUnlock()
		if ok {
			return val
		}
	}

	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

// Uint returns a uint32 metadata value.
func (db *DB) Uint(key string, defaultValue ...uint32) uint32 {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return 0
	}

	val, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return 0
	}
	return uint32(val)
}

// Float returns a float32 metadata value.
func (db *DB) Float(key string, defaultValue ...float32) float32 {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return 0
	}

	val, err := strconv.ParseFloat(s, 32)
	if err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return 0
	}
	return float32(val)
}

// Bool returns a bool metadata value.
func (db *DB) Bool(key string, defaultValue ...bool) bool {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return false
	}

	val, err := strconv.ParseBool(s)
	if err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return false
	}
	return val
}

// Strings returns a string array metadata value.
// The value is expected to be stored as JSON array in metadata table.
func (db *DB) Strings(key string, defaultValue ...[]string) []string {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}

	var result []string
	if err := json.Unmarshal([]byte(s), &result); err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	return result
}

// Ints returns an int32 array metadata value.
func (db *DB) Ints(key string, defaultValue ...[]int32) []int32 {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}

	var result []int32
	if err := json.Unmarshal([]byte(s), &result); err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	return result
}

// Floats returns a float32 array metadata value.
func (db *DB) Floats(key string, defaultValue ...[]float32) []float32 {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}

	var result []float32
	if err := json.Unmarshal([]byte(s), &result); err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	return result
}

// Bools returns a bool array metadata value.
func (db *DB) Bools(key string, defaultValue ...[]bool) []bool {
	s := db.String(key)
	if s == "" {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}

	var result []bool
	if err := json.Unmarshal([]byte(s), &result); err != nil {
		if len(defaultValue) > 0 {
			return defaultValue[0]
		}
		return nil
	}
	return result
}

// --- Tensor access ---

// LoadTensors loads all tensor metadata from the database.
func (db *DB) LoadTensors() ([]*TensorInfo, error) {
	db.tensorsMu.Lock()
	defer db.tensorsMu.Unlock()

	if db.tensorsCache != nil {
		return db.tensorsCache, nil
	}

	rows, err := db.db.Query(`
		SELECT id, name, layer, component, subcomponent, dims, dtype, dtype_id, n_elements, byte_size
		FROM tensors
		ORDER BY id
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tensors []*TensorInfo
	for rows.Next() {
		var t TensorInfo
		var dimsStr string
		var layer, component, subcomponent sql.NullString

		err := rows.Scan(
			&t.ID, &t.Name, &layer, &component, &subcomponent,
			&dimsStr, &t.DType, &t.DTypeID, &t.NumElements, &t.ByteSize,
		)
		if err != nil {
			return nil, err
		}

		if layer.Valid {
			t.Layer, _ = strconv.Atoi(layer.String)
		}
		if component.Valid {
			t.Component = component.String
		}
		if subcomponent.Valid {
			t.Subcomponent = subcomponent.String
		}

		// Parse dims JSON
		if dimsStr != "" {
			if err := json.Unmarshal([]byte(dimsStr), &t.Dims); err != nil {
				return nil, fmt.Errorf("failed to parse dims for tensor %s: %w", t.Name, err)
			}
		}

		tensors = append(tensors, &t)
	}

	db.tensorsCache = tensors
	return tensors, rows.Err()
}

// GetTensorByName returns tensor info by name.
func (db *DB) GetTensorByName(name string) (*TensorInfo, error) {
	tensors, err := db.LoadTensors()
	if err != nil {
		return nil, err
	}

	for _, t := range tensors {
		if t.Name == name {
			return t, nil
		}
	}
	return nil, nil
}

// ReadTensorData reads the raw tensor data blob.
func (db *DB) ReadTensorData(tensorID int) ([]byte, error) {
	// Track access for training
	db.accessMu.Lock()
	db.accessedTensors[tensorID] = true
	db.accessMu.Unlock()

	var data []byte
	err := db.db.QueryRow("SELECT data FROM tensor_data WHERE tensor_id = ?", tensorID).Scan(&data)
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor data for id %d: %w", tensorID, err)
	}
	return data, nil
}

// ReadTensorDataByName reads tensor data by tensor name.
func (db *DB) ReadTensorDataByName(name string) ([]byte, error) {
	t, err := db.GetTensorByName(name)
	if err != nil {
		return nil, err
	}
	if t == nil {
		return nil, fmt.Errorf("tensor not found: %s", name)
	}
	return db.ReadTensorData(t.ID)
}

// --- Vocabulary access ---

// Vocabulary represents the token vocabulary.
type Vocabulary struct {
	db *DB
}

// GetVocabulary returns access to the vocabulary.
func (db *DB) GetVocabulary() *Vocabulary {
	return &Vocabulary{db: db}
}

// TokenCount returns the number of tokens in the vocabulary.
func (v *Vocabulary) TokenCount() (int, error) {
	var count int
	err := v.db.db.QueryRow("SELECT COUNT(*) FROM vocab").Scan(&count)
	return count, err
}

// GetTokenString returns the string for a token ID.
func (v *Vocabulary) GetTokenString(tokenID int) (string, error) {
	var tokenString string
	err := v.db.db.QueryRow("SELECT token_string FROM vocab WHERE token_id = ?", tokenID).Scan(&tokenString)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return tokenString, err
}

// GetTokenID returns the token ID for a string.
func (v *Vocabulary) GetTokenID(tokenString string) (int, error) {
	var tokenID int
	err := v.db.db.QueryRow("SELECT token_id FROM vocab WHERE token_string = ?", tokenString).Scan(&tokenID)
	if err == sql.ErrNoRows {
		return -1, nil
	}
	return tokenID, err
}

// GetAllTokens returns all token strings ordered by ID.
func (v *Vocabulary) GetAllTokens() ([]string, error) {
	rows, err := v.db.db.Query("SELECT token_string FROM vocab ORDER BY token_id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tokens []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		tokens = append(tokens, s)
	}
	return tokens, rows.Err()
}

// --- Training support ---

// GetAccessedTensorIDs returns the IDs of tensors accessed since last clear.
func (db *DB) GetAccessedTensorIDs() []int {
	db.accessMu.Lock()
	defer db.accessMu.Unlock()

	ids := make([]int, 0, len(db.accessedTensors))
	for id := range db.accessedTensors {
		ids = append(ids, id)
	}
	return ids
}

// ClearAccessTracking clears the accessed tensor tracking.
func (db *DB) ClearAccessTracking() {
	db.accessMu.Lock()
	db.accessedTensors = make(map[int]bool)
	db.accessMu.Unlock()
}

// UpdateTensorData updates tensor data in the database.
func (db *DB) UpdateTensorData(tensorID int, data []byte) error {
	_, err := db.db.Exec("UPDATE tensor_data SET data = ? WHERE tensor_id = ?", data, tensorID)
	return err
}

// BeginTransaction starts a transaction for batch updates.
func (db *DB) BeginTransaction() (*sql.Tx, error) {
	return db.db.Begin()
}
