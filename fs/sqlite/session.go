// session.go provides activity-based database sessions for inference.
// Each session assembles its own working set of databases.
package sqlite

import (
	"database/sql"
	"fmt"
	"sync"
)

// Session represents an inference activity with its assembled databases.
type Session struct {
	mu      sync.RWMutex
	id      string
	manager *DatabaseManager

	// Databases loaded for this session
	loaded map[string]*ManagedDB

	// Query helpers bound to this session
	tensors *TensorQuery
	vocab   *VocabQuery

	// Custom query templates for this session
	templates map[string]*QueryTemplate

	// Session-specific settings
	settings map[string]interface{}
}

// SessionConfig defines what databases to load for a session.
type SessionConfig struct {
	// Primary model database path
	ModelDB string

	// Additional databases to attach (path -> alias)
	AttachDBs map[string]string

	// Databases to load entirely into memory (path -> alias)
	MemoryDBs map[string]string

	// Initial settings
	Settings map[string]interface{}
}

// NewSession creates a new inference session with the given configuration.
func NewSession(id string, config *SessionConfig) (*Session, error) {
	manager, err := NewDatabaseManager(config.ModelDB)
	if err != nil {
		return nil, fmt.Errorf("failed to create database manager: %w", err)
	}

	s := &Session{
		id:        id,
		manager:   manager,
		loaded:    make(map[string]*ManagedDB),
		templates: make(map[string]*QueryTemplate),
		settings:  make(map[string]interface{}),
	}

	// Copy initial settings
	for k, v := range config.Settings {
		s.settings[k] = v
	}

	// Attach additional databases
	for path, alias := range config.AttachDBs {
		if err := manager.AttachDatabase(path, alias); err != nil {
			s.Close()
			return nil, fmt.Errorf("failed to attach %s: %w", alias, err)
		}
	}

	// Load memory databases
	for path, alias := range config.MemoryDBs {
		db, err := manager.LoadToMemory(path, alias)
		if err != nil {
			s.Close()
			return nil, fmt.Errorf("failed to load %s to memory: %w", alias, err)
		}
		s.loaded[alias] = db
	}

	// Initialize query helpers on primary
	s.tensors = NewTensorQuery(manager.Primary())
	s.vocab = NewVocabQuery(manager.Primary())

	return s, nil
}

// ID returns the session identifier.
func (s *Session) ID() string {
	return s.id
}

// Primary returns the primary database connection.
func (s *Session) Primary() *sql.DB {
	return s.manager.Primary()
}

// Manager returns the underlying database manager.
func (s *Session) Manager() *DatabaseManager {
	return s.manager
}

// Tensors returns the tensor query helper.
func (s *Session) Tensors() *TensorQuery {
	return s.tensors
}

// Vocab returns the vocabulary query helper.
func (s *Session) Vocab() *VocabQuery {
	return s.vocab
}

// Query creates a new query builder on the session's primary database.
func (s *Session) Query() *QueryBuilder {
	return Query(s.manager.Primary())
}

// SetTemplate registers a named query template for this session.
func (s *Session) SetTemplate(name, query string) *QueryTemplate {
	s.mu.Lock()
	defer s.mu.Unlock()

	t := NewTemplate(s.manager.Primary(), query)
	s.templates[name] = t
	return t
}

// Template returns a registered query template.
func (s *Session) Template(name string) *QueryTemplate {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.templates[name]
}

// Set stores a session setting.
func (s *Session) Set(key string, value interface{}) {
	s.mu.Lock()
	s.settings[key] = value
	s.mu.Unlock()
}

// Get retrieves a session setting.
func (s *Session) Get(key string) (interface{}, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.settings[key]
	return v, ok
}

// GetString retrieves a string setting with default.
func (s *Session) GetString(key, defaultValue string) string {
	if v, ok := s.Get(key); ok {
		if str, ok := v.(string); ok {
			return str
		}
	}
	return defaultValue
}

// GetInt retrieves an int setting with default.
func (s *Session) GetInt(key string, defaultValue int) int {
	if v, ok := s.Get(key); ok {
		if i, ok := v.(int); ok {
			return i
		}
	}
	return defaultValue
}

// AttachDatabase attaches an additional database to this session.
func (s *Session) AttachDatabase(path, alias string) error {
	return s.manager.AttachDatabase(path, alias)
}

// DetachDatabase detaches a database from this session.
func (s *Session) DetachDatabase(alias string) error {
	return s.manager.DetachDatabase(alias)
}

// LoadToMemory loads a database to memory for this session.
func (s *Session) LoadToMemory(path, alias string) (*ManagedDB, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db, err := s.manager.LoadToMemory(path, alias)
	if err != nil {
		return nil, err
	}

	s.loaded[alias] = db
	return db, nil
}

// GetLoaded returns a database loaded for this session.
func (s *Session) GetLoaded(alias string) *ManagedDB {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.loaded[alias]
}

// Close ends the session and releases all resources.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.loaded = nil
	s.templates = nil
	s.settings = nil

	return s.manager.Close()
}

// --- Session pool for managing multiple concurrent sessions ---

// SessionPool manages multiple inference sessions.
type SessionPool struct {
	mu       sync.RWMutex
	sessions map[string]*Session
}

// NewSessionPool creates a new session pool.
func NewSessionPool() *SessionPool {
	return &SessionPool{
		sessions: make(map[string]*Session),
	}
}

// Create creates a new session with the given configuration.
func (p *SessionPool) Create(id string, config *SessionConfig) (*Session, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.sessions[id]; exists {
		return nil, fmt.Errorf("session %s already exists", id)
	}

	s, err := NewSession(id, config)
	if err != nil {
		return nil, err
	}

	p.sessions[id] = s
	return s, nil
}

// Get returns an existing session by ID.
func (p *SessionPool) Get(id string) *Session {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.sessions[id]
}

// Close closes and removes a session.
func (p *SessionPool) Close(id string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	s, exists := p.sessions[id]
	if !exists {
		return fmt.Errorf("session %s not found", id)
	}

	if err := s.Close(); err != nil {
		return err
	}

	delete(p.sessions, id)
	return nil
}

// CloseAll closes all sessions.
func (p *SessionPool) CloseAll() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for id, s := range p.sessions {
		s.Close()
		delete(p.sessions, id)
	}
}

// List returns all session IDs.
func (p *SessionPool) List() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	ids := make([]string, 0, len(p.sessions))
	for id := range p.sessions {
		ids = append(ids, id)
	}
	return ids
}
