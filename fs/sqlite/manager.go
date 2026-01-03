// Package sqlite provides multi-database management for activity-based model loading.
// Each inference activity can assemble its own set of databases dynamically.
package sqlite

import (
	"database/sql"
	"fmt"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// DatabaseManager handles multiple database connections for activity-based loading.
// Databases can be attached/detached dynamically to assemble working sets.
type DatabaseManager struct {
	mu sync.RWMutex

	// Primary connection - used for ATTACH operations
	primary *sql.DB

	// Named database connections
	databases map[string]*ManagedDB

	// Attached databases (via ATTACH DATABASE)
	attached map[string]string // alias -> path
}

// ManagedDB represents a database in the manager.
type ManagedDB struct {
	DB       *sql.DB
	Path     string
	Alias    string
	ReadOnly bool
	InRAM    bool // Loaded to :memory: for speed
}

// NewDatabaseManager creates a new database manager.
// The primary path is the main database that other databases attach to.
func NewDatabaseManager(primaryPath string) (*DatabaseManager, error) {
	// Open primary with shared cache for attached database queries
	primary, err := sql.Open("sqlite3", primaryPath+"?cache=shared&mode=rwc")
	if err != nil {
		return nil, fmt.Errorf("failed to open primary database: %w", err)
	}

	// Enable WAL for better concurrent access
	if _, err := primary.Exec("PRAGMA journal_mode=WAL"); err != nil {
		primary.Close()
		return nil, fmt.Errorf("failed to enable WAL: %w", err)
	}

	return &DatabaseManager{
		primary:   primary,
		databases: make(map[string]*ManagedDB),
		attached:  make(map[string]string),
	}, nil
}

// AttachDatabase attaches a database file to the primary connection.
// This allows cross-database queries using the alias as schema prefix.
// Example: SELECT * FROM weights.tensors JOIN vocab.tokens ...
func (m *DatabaseManager) AttachDatabase(path, alias string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.attached[alias]; exists {
		return fmt.Errorf("database alias %q already attached", alias)
	}

	query := fmt.Sprintf("ATTACH DATABASE '%s' AS %s", path, alias)
	if _, err := m.primary.Exec(query); err != nil {
		return fmt.Errorf("failed to attach database %s: %w", path, err)
	}

	m.attached[alias] = path
	return nil
}

// DetachDatabase detaches a previously attached database.
func (m *DatabaseManager) DetachDatabase(alias string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.attached[alias]; !exists {
		return fmt.Errorf("database alias %q not attached", alias)
	}

	query := fmt.Sprintf("DETACH DATABASE %s", alias)
	if _, err := m.primary.Exec(query); err != nil {
		return fmt.Errorf("failed to detach database %s: %w", alias, err)
	}

	delete(m.attached, alias)
	return nil
}

// LoadToMemory loads a database file entirely into memory for faster access.
// Returns the in-memory database handle.
func (m *DatabaseManager) LoadToMemory(path, alias string) (*ManagedDB, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Open source database
	src, err := sql.Open("sqlite3", path+"?mode=ro")
	if err != nil {
		return nil, fmt.Errorf("failed to open source database: %w", err)
	}
	defer src.Close()

	// Create in-memory database
	mem, err := sql.Open("sqlite3", ":memory:?cache=shared")
	if err != nil {
		return nil, fmt.Errorf("failed to create memory database: %w", err)
	}

	// Use SQLite backup API via ATTACH + INSERT
	// Attach the source database to memory db
	attachQuery := fmt.Sprintf("ATTACH DATABASE '%s' AS src", path)
	if _, err := mem.Exec(attachQuery); err != nil {
		mem.Close()
		return nil, fmt.Errorf("failed to attach source: %w", err)
	}

	// Get list of tables from source
	rows, err := mem.Query("SELECT name, sql FROM src.sqlite_master WHERE type='table' AND sql IS NOT NULL")
	if err != nil {
		mem.Close()
		return nil, fmt.Errorf("failed to list tables: %w", err)
	}

	var tables []struct{ name, sql string }
	for rows.Next() {
		var t struct{ name, sql string }
		if err := rows.Scan(&t.name, &t.sql); err != nil {
			rows.Close()
			mem.Close()
			return nil, err
		}
		tables = append(tables, t)
	}
	rows.Close()

	// Create tables and copy data
	for _, t := range tables {
		// Create table structure
		if _, err := mem.Exec(t.sql); err != nil {
			mem.Close()
			return nil, fmt.Errorf("failed to create table %s: %w", t.name, err)
		}

		// Copy data
		copyQuery := fmt.Sprintf("INSERT INTO main.%s SELECT * FROM src.%s", t.name, t.name)
		if _, err := mem.Exec(copyQuery); err != nil {
			mem.Close()
			return nil, fmt.Errorf("failed to copy table %s: %w", t.name, err)
		}
	}

	// Detach source
	if _, err := mem.Exec("DETACH DATABASE src"); err != nil {
		mem.Close()
		return nil, fmt.Errorf("failed to detach source: %w", err)
	}

	managed := &ManagedDB{
		DB:       mem,
		Path:     path,
		Alias:    alias,
		ReadOnly: true,
		InRAM:    true,
	}

	m.databases[alias] = managed
	return managed, nil
}

// OpenDatabase opens a database connection without loading to memory.
func (m *DatabaseManager) OpenDatabase(path, alias string, readOnly bool) (*ManagedDB, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.databases[alias]; exists {
		return nil, fmt.Errorf("database alias %q already open", alias)
	}

	mode := "rw"
	if readOnly {
		mode = "ro"
	}

	db, err := sql.Open("sqlite3", path+"?mode="+mode+"&cache=shared")
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to enable WAL: %w", err)
	}

	managed := &ManagedDB{
		DB:       db,
		Path:     path,
		Alias:    alias,
		ReadOnly: readOnly,
		InRAM:    false,
	}

	m.databases[alias] = managed
	return managed, nil
}

// GetDatabase returns a managed database by alias.
func (m *DatabaseManager) GetDatabase(alias string) *ManagedDB {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.databases[alias]
}

// Primary returns the primary database connection.
// Use this for cross-database queries on attached databases.
func (m *DatabaseManager) Primary() *sql.DB {
	return m.primary
}

// Query executes a query on the primary connection (works across attached DBs).
func (m *DatabaseManager) Query(query string, args ...interface{}) (*sql.Rows, error) {
	return m.primary.Query(query, args...)
}

// Exec executes a statement on the primary connection.
func (m *DatabaseManager) Exec(query string, args ...interface{}) (sql.Result, error) {
	return m.primary.Exec(query, args...)
}

// CloseDatabase closes and removes a managed database.
func (m *DatabaseManager) CloseDatabase(alias string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	db, exists := m.databases[alias]
	if !exists {
		return fmt.Errorf("database alias %q not found", alias)
	}

	if err := db.DB.Close(); err != nil {
		return err
	}

	delete(m.databases, alias)
	return nil
}

// Close closes all database connections.
func (m *DatabaseManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Detach all attached databases
	for alias := range m.attached {
		m.primary.Exec(fmt.Sprintf("DETACH DATABASE %s", alias))
	}
	m.attached = make(map[string]string)

	// Close all managed databases
	for alias, db := range m.databases {
		db.DB.Close()
		delete(m.databases, alias)
	}

	// Close primary
	return m.primary.Close()
}

// ListAttached returns the list of attached database aliases.
func (m *DatabaseManager) ListAttached() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	aliases := make([]string, 0, len(m.attached))
	for alias := range m.attached {
		aliases = append(aliases, alias)
	}
	return aliases
}

// ListDatabases returns the list of managed database aliases.
func (m *DatabaseManager) ListDatabases() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	aliases := make([]string, 0, len(m.databases))
	for alias := range m.databases {
		aliases = append(aliases, alias)
	}
	return aliases
}
