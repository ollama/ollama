package auth

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func setupTestDir(t *testing.T) string {
	t.Helper()

	// Create a temporary directory for testing
	tmpDir := t.TempDir()

	// Create .ollama subdirectory
	ollamaDir := filepath.Join(tmpDir, ".ollama")
	if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
		t.Fatalf("failed to create .ollama dir: %v", err)
	}

	// Override home directory for tests (platform-specific)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", tmpDir)
	} else {
		t.Setenv("HOME", tmpDir)
	}

	return tmpDir
}

func TestSetSignInState(t *testing.T) {
	_ = setupTestDir(t)

	state := &SignInState{
		Name:  "testuser",
		Email: "test@example.com",
	}

	err := SetSignInState(state)
	if err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Verify file was created
	home, _ := os.UserHomeDir()
	statePath := filepath.Join(home, ".ollama", signInStateFile)

	data, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatalf("failed to read state file: %v", err)
	}

	var savedState SignInState
	if err := json.Unmarshal(data, &savedState); err != nil {
		t.Fatalf("failed to unmarshal state: %v", err)
	}

	if savedState.Name != "testuser" {
		t.Errorf("expected name 'testuser', got '%s'", savedState.Name)
	}

	if savedState.Email != "test@example.com" {
		t.Errorf("expected email 'test@example.com', got '%s'", savedState.Email)
	}

	if savedState.CachedAt.IsZero() {
		t.Error("expected CachedAt to be set, got zero time")
	}

	// Verify CachedAt is recent (within last minute)
	if time.Since(savedState.CachedAt) > time.Minute {
		t.Errorf("CachedAt is too old: %v", savedState.CachedAt)
	}
}

func TestSetSignInState_Overwrites(t *testing.T) {
	_ = setupTestDir(t)

	// Set initial state
	state1 := &SignInState{Name: "user1", Email: "user1@example.com"}
	if err := SetSignInState(state1); err != nil {
		t.Fatalf("first SetSignInState failed: %v", err)
	}

	// Overwrite with new state
	state2 := &SignInState{Name: "user2", Email: "user2@example.com"}
	if err := SetSignInState(state2); err != nil {
		t.Fatalf("second SetSignInState failed: %v", err)
	}

	// Verify only new state exists
	readState, err := GetSignInState()
	if err != nil {
		t.Fatalf("GetSignInState failed: %v", err)
	}

	if readState.Name != "user2" {
		t.Errorf("expected name 'user2', got '%s'", readState.Name)
	}
}

func TestGetSignInState(t *testing.T) {
	_ = setupTestDir(t)

	// First set a state
	originalState := &SignInState{
		Name:  "testuser",
		Email: "test@example.com",
	}
	if err := SetSignInState(originalState); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Now read it back
	readState, err := GetSignInState()
	if err != nil {
		t.Fatalf("GetSignInState failed: %v", err)
	}

	if readState.Name != originalState.Name {
		t.Errorf("expected name '%s', got '%s'", originalState.Name, readState.Name)
	}

	if readState.Email != originalState.Email {
		t.Errorf("expected email '%s', got '%s'", originalState.Email, readState.Email)
	}
}

func TestGetSignInState_NoFile(t *testing.T) {
	_ = setupTestDir(t)

	// Try to read without any file existing
	state, err := GetSignInState()
	if err == nil {
		t.Error("expected error when file doesn't exist, got nil")
	}
	if state != nil {
		t.Errorf("expected nil state, got %+v", state)
	}
}

func TestGetSignInState_InvalidJSON(t *testing.T) {
	tmpDir := setupTestDir(t)

	// Write invalid JSON to the state file
	statePath := filepath.Join(tmpDir, ".ollama", signInStateFile)
	if err := os.WriteFile(statePath, []byte("not valid json"), 0o600); err != nil {
		t.Fatalf("failed to write invalid json: %v", err)
	}

	state, err := GetSignInState()
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
	if state != nil {
		t.Errorf("expected nil state for invalid JSON, got %+v", state)
	}
}

func TestClearSignInState(t *testing.T) {
	_ = setupTestDir(t)

	// First set a state
	state := &SignInState{Name: "testuser", Email: "test@example.com"}
	if err := SetSignInState(state); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Verify file exists
	home, _ := os.UserHomeDir()
	statePath := filepath.Join(home, ".ollama", signInStateFile)
	if _, err := os.Stat(statePath); os.IsNotExist(err) {
		t.Fatal("state file should exist before clearing")
	}

	// Clear the state
	if err := ClearSignInState(); err != nil {
		t.Fatalf("ClearSignInState failed: %v", err)
	}

	// Verify file is gone
	if _, err := os.Stat(statePath); !os.IsNotExist(err) {
		t.Error("state file should be deleted after clearing")
	}
}

func TestClearSignInState_NoFile(t *testing.T) {
	_ = setupTestDir(t)

	// Clear when no file exists should not error
	err := ClearSignInState()
	if err != nil {
		t.Errorf("ClearSignInState should not error when file doesn't exist: %v", err)
	}
}

func TestClearSignInState_Idempotent(t *testing.T) {
	_ = setupTestDir(t)

	// Set a state first
	state := &SignInState{Name: "testuser", Email: "test@example.com"}
	if err := SetSignInState(state); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Clear multiple times should not error
	for i := range 3 {
		if err := ClearSignInState(); err != nil {
			t.Errorf("ClearSignInState iteration %d failed: %v", i, err)
		}
	}
}

func TestIsSignedIn(t *testing.T) {
	_ = setupTestDir(t)

	// Initially not signed in
	if IsSignedIn() {
		t.Error("should not be signed in initially")
	}

	// Set a state with a name
	state := &SignInState{Name: "testuser", Email: "test@example.com"}
	if err := SetSignInState(state); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Now should be signed in
	if !IsSignedIn() {
		t.Error("should be signed in after setting state")
	}

	// Clear the state
	if err := ClearSignInState(); err != nil {
		t.Fatalf("ClearSignInState failed: %v", err)
	}

	// Should not be signed in after clearing
	if IsSignedIn() {
		t.Error("should not be signed in after clearing state")
	}
}

func TestIsSignedIn_EmptyName(t *testing.T) {
	tmpDir := setupTestDir(t)

	// Write a state with empty name directly
	state := SignInState{
		Name:     "",
		Email:    "test@example.com",
		CachedAt: time.Now(),
	}
	data, _ := json.Marshal(state)
	statePath := filepath.Join(tmpDir, ".ollama", signInStateFile)
	if err := os.WriteFile(statePath, data, 0o600); err != nil {
		t.Fatalf("failed to write state: %v", err)
	}

	// Should not be signed in with empty name
	if IsSignedIn() {
		t.Error("should not be signed in with empty name")
	}
}

func TestSetSignInState_AtomicWrite(t *testing.T) {
	tmpDir := setupTestDir(t)

	state := &SignInState{Name: "testuser", Email: "test@example.com"}
	if err := SetSignInState(state); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Verify temp file is cleaned up
	tmpPath := filepath.Join(tmpDir, ".ollama", signInStateFile+".tmp")
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Error("temp file should be cleaned up after atomic write")
	}

	// Verify final file exists
	statePath := filepath.Join(tmpDir, ".ollama", signInStateFile)
	if _, err := os.Stat(statePath); os.IsNotExist(err) {
		t.Error("final state file should exist")
	}
}

func TestSetSignInState_FilePermissions(t *testing.T) {
	tmpDir := setupTestDir(t)

	state := &SignInState{Name: "testuser", Email: "test@example.com"}
	if err := SetSignInState(state); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	statePath := filepath.Join(tmpDir, ".ollama", signInStateFile)
	info, err := os.Stat(statePath)
	if err != nil {
		t.Fatalf("failed to stat state file: %v", err)
	}

	// Check file permissions (should be 0600 - owner read/write only)
	perm := info.Mode().Perm()
	if perm != 0o600 {
		t.Errorf("expected permissions 0600, got %04o", perm)
	}
}

func TestRoundTrip(t *testing.T) {
	_ = setupTestDir(t)

	// Test full round trip: set -> get -> clear -> get
	original := &SignInState{
		Name:  "roundtrip_user",
		Email: "roundtrip@example.com",
	}

	// Set
	if err := SetSignInState(original); err != nil {
		t.Fatalf("SetSignInState failed: %v", err)
	}

	// Get and verify
	retrieved, err := GetSignInState()
	if err != nil {
		t.Fatalf("GetSignInState failed: %v", err)
	}

	if retrieved.Name != original.Name {
		t.Errorf("name mismatch: expected '%s', got '%s'", original.Name, retrieved.Name)
	}
	if retrieved.Email != original.Email {
		t.Errorf("email mismatch: expected '%s', got '%s'", original.Email, retrieved.Email)
	}

	// Clear
	if err := ClearSignInState(); err != nil {
		t.Fatalf("ClearSignInState failed: %v", err)
	}

	// Get should fail now
	_, err = GetSignInState()
	if err == nil {
		t.Error("GetSignInState should fail after clear")
	}
}
