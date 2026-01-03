package server

import (
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"reflect"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

// MCPSessionManager manages active MCP sessions with automatic cleanup.
// This is the runtime component that tracks active connections.
type MCPSessionManager struct {
	mu          sync.RWMutex
	sessions    map[string]*MCPSession // session ID -> session
	ttl         time.Duration          // session timeout
	stopCleanup chan struct{}          // signals cleanup goroutine to stop
}

// MCPSession wraps an MCPManager with session metadata
type MCPSession struct {
	*MCPManager
	lastAccess time.Time
	sessionID  string
	configs    []api.MCPServerConfig
}

var (
	globalSessionManager *MCPSessionManager
	sessionManagerOnce   sync.Once
)

// GetMCPSessionManager returns the singleton MCP session manager
func GetMCPSessionManager() *MCPSessionManager {
	sessionManagerOnce.Do(func() {
		globalSessionManager = &MCPSessionManager{
			sessions:    make(map[string]*MCPSession),
			ttl:         30 * time.Minute, // Sessions expire after 30 min
			stopCleanup: make(chan struct{}),
		}
		// Start cleanup goroutine
		go globalSessionManager.cleanupExpired()
	})
	return globalSessionManager
}

// GetOrCreateManager gets existing or creates new MCP manager for session
func (sm *MCPSessionManager) GetOrCreateManager(sessionID string, configs []api.MCPServerConfig) (*MCPManager, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Check if session exists and configs match
	if session, exists := sm.sessions[sessionID]; exists {
		if configsMatch(session.configs, configs) {
			session.lastAccess = time.Now()
			slog.Debug("Reusing existing MCP session", "session", sessionID, "clients", len(session.clients))
			return session.MCPManager, nil
		}
		// Configs changed, shutdown old session
		slog.Info("MCP configs changed, recreating session", "session", sessionID)
		session.Shutdown()
		delete(sm.sessions, sessionID)
	}

	// Create new session
	slog.Info("Creating new MCP session", "session", sessionID, "configs", len(configs))
	manager := NewMCPManager(10)
	for _, config := range configs {
		if err := manager.AddServer(config); err != nil {
			slog.Warn("Failed to add MCP server", "name", config.Name, "error", err)
		}
	}

	sm.sessions[sessionID] = &MCPSession{
		MCPManager: manager,
		lastAccess: time.Now(),
		sessionID:  sessionID,
		configs:    configs,
	}

	return manager, nil
}

// GetManagerForToolsPath creates a manager for a tools directory path.
// It uses the definitions system to get auto-enabled servers for the path.
func (sm *MCPSessionManager) GetManagerForToolsPath(model string, toolsPath string) (*MCPManager, error) {
	// Generate consistent session ID for model + tools path
	sessionID := generateToolsSessionID(model, toolsPath)

	// Use definitions to get auto-enabled servers (single source of truth)
	defs, err := LoadMCPDefinitions()
	if err != nil {
		return nil, err
	}

	ctx := AutoEnableContext{ToolsPath: toolsPath}
	configs := defs.GetAutoEnableServers(ctx)

	return sm.GetOrCreateManager(sessionID, configs)
}

// cleanupExpired removes expired sessions
func (sm *MCPSessionManager) cleanupExpired() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-sm.stopCleanup:
			return
		case <-ticker.C:
			sm.mu.Lock()
			now := time.Now()
			for sessionID, session := range sm.sessions {
				if now.Sub(session.lastAccess) > sm.ttl {
					slog.Info("Cleaning up expired MCP session", "session", sessionID)
					session.Shutdown()
					delete(sm.sessions, sessionID)
				}
			}
			sm.mu.Unlock()
		}
	}
}

// Shutdown closes all sessions and stops the cleanup goroutine
func (sm *MCPSessionManager) Shutdown() {
	// Signal cleanup goroutine to stop
	close(sm.stopCleanup)

	sm.mu.Lock()
	defer sm.mu.Unlock()

	slog.Info("Shutting down MCP session manager", "sessions", len(sm.sessions))
	for sessionID, session := range sm.sessions {
		slog.Debug("Shutting down session", "session", sessionID)
		session.Shutdown()
	}
	sm.sessions = make(map[string]*MCPSession)
}

// configsMatch checks if two sets of MCP configs are equivalent
func configsMatch(a, b []api.MCPServerConfig) bool {
	if len(a) != len(b) {
		return false
	}
	return reflect.DeepEqual(a, b)
}

// generateToolsSessionID creates a consistent session ID for model + tools path
func generateToolsSessionID(model, toolsPath string) string {
	h := sha256.New()
	h.Write([]byte(model))
	h.Write([]byte(toolsPath))
	return "tools-" + hex.EncodeToString(h.Sum(nil))[:16]
}

// GenerateSessionID creates a session ID based on the request
func GenerateSessionID(req api.ChatRequest) string {
	// If explicit session ID provided
	if req.SessionID != "" {
		return req.SessionID
	}

	// For interactive mode with tools path
	if req.ToolsPath != "" {
		return generateToolsSessionID(req.Model, req.ToolsPath)
	}

	// Default: use request-specific ID (no persistence)
	h := sha256.New()
	h.Write([]byte(time.Now().Format(time.RFC3339Nano)))
	h.Write([]byte(req.Model))
	return "req-" + hex.EncodeToString(h.Sum(nil))[:16]
}
