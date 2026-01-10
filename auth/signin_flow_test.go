package auth

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestWhoamiHandlerFlow simulates the WhoamiHandler logic flow
func TestWhoamiHandlerFlow(t *testing.T) {
	_ = setupTestDir(t)

	// Scenario 1: No local cache - should indicate need for network call
	t.Run("NoCache_RequiresNetwork", func(t *testing.T) {
		state, err := GetSignInState()
		if err == nil && state != nil && state.Name != "" {
			t.Error("should not have cached state initially")
		}
		// In real WhoamiHandler, this would trigger a network call
	})

	// Scenario 2: Simulate successful sign-in from network
	t.Run("CacheAfterNetworkSuccess", func(t *testing.T) {
		// Simulate receiving user from ollama.com
		networkUser := &SignInState{
			Name:  "networkuser",
			Email: "network@example.com",
		}

		// Cache the result (as WhoamiHandler would)
		if err := SetSignInState(networkUser); err != nil {
			t.Fatalf("SetSignInState failed: %v", err)
		}

		// Verify it's cached
		if !IsSignedIn() {
			t.Error("should be signed in after caching")
		}
	})

	// Scenario 3: Subsequent calls use cache (no network)
	t.Run("SubsequentCalls_UseCache", func(t *testing.T) {
		state, err := GetSignInState()
		if err != nil {
			t.Fatalf("GetSignInState failed: %v", err)
		}

		if state.Name != "networkuser" {
			t.Errorf("expected cached name 'networkuser', got '%s'", state.Name)
		}

		// In real WhoamiHandler, this would skip the network call and return cached data
	})

	// Scenario 4: Sign-out clears cache
	t.Run("SignOut_ClearsCache", func(t *testing.T) {
		// Simulate SignoutHandler clearing cache
		if err := ClearSignInState(); err != nil {
			t.Fatalf("ClearSignInState failed: %v", err)
		}

		if IsSignedIn() {
			t.Error("should not be signed in after sign-out")
		}
	})

	// Scenario 5: After sign-out, next call requires network
	t.Run("AfterSignOut_RequiresNetwork", func(t *testing.T) {
		state, err := GetSignInState()
		if err == nil && state != nil && state.Name != "" {
			t.Error("should require network after sign-out")
		}
	})
}

// TestOfflineScenarios tests behavior when offline
func TestOfflineScenarios(t *testing.T) {
	_ = setupTestDir(t)

	t.Run("Offline_WithCache_Works", func(t *testing.T) {
		// Pre-populate cache (simulate previous sign-in)
		state := &SignInState{Name: "cacheduser", Email: "cached@example.com"}
		if err := SetSignInState(state); err != nil {
			t.Fatalf("SetSignInState failed: %v", err)
		}

		// Simulate offline check - should work with cache
		cached, err := GetSignInState()
		if err != nil {
			t.Fatalf("should work offline with cache: %v", err)
		}

		if cached.Name != "cacheduser" {
			t.Errorf("expected 'cacheduser', got '%s'", cached.Name)
		}

		if !IsSignedIn() {
			t.Error("should report signed in with cache")
		}
	})

	t.Run("Offline_WithoutCache_Fails", func(t *testing.T) {
		// Clear cache first
		ClearSignInState()

		// Offline without cache should indicate not signed in
		if IsSignedIn() {
			t.Error("should not be signed in offline without cache")
		}

		_, err := GetSignInState()
		if err == nil {
			t.Error("should get error when no cache exists")
		}
	})
}

// TestMultipleSessions tests overwriting sessions
func TestMultipleSessions(t *testing.T) {
	_ = setupTestDir(t)

	t.Run("NewSignIn_OverwritesOld", func(t *testing.T) {
		// First user signs in
		user1 := &SignInState{Name: "user1", Email: "user1@example.com"}
		SetSignInState(user1)

		// Different user signs in (should overwrite)
		user2 := &SignInState{Name: "user2", Email: "user2@example.com"}
		SetSignInState(user2)

		// Should have user2's data
		state, _ := GetSignInState()
		if state.Name != "user2" {
			t.Errorf("expected 'user2', got '%s'", state.Name)
		}
	})
}

// TestEdgeCases tests various edge cases
func TestEdgeCases(t *testing.T) {
	_ = setupTestDir(t)

	t.Run("EmptyName_NotSignedIn", func(t *testing.T) {
		// User with empty name should not count as signed in
		state := &SignInState{Name: "", Email: "noname@example.com"}
		SetSignInState(state)

		if IsSignedIn() {
			t.Error("empty name should not count as signed in")
		}
	})

	t.Run("SpecialCharactersInName", func(t *testing.T) {
		state := &SignInState{
			Name:  "user with spaces & symbols!@#$%",
			Email: "special@example.com",
		}
		if err := SetSignInState(state); err != nil {
			t.Fatalf("failed to set state with special chars: %v", err)
		}

		read, err := GetSignInState()
		if err != nil {
			t.Fatalf("failed to read state: %v", err)
		}

		if read.Name != state.Name {
			t.Errorf("name mismatch with special chars")
		}
	})

	t.Run("UnicodeInName", func(t *testing.T) {
		state := &SignInState{
			Name:  "用户名 🎉 émojis",
			Email: "unicode@example.com",
		}
		if err := SetSignInState(state); err != nil {
			t.Fatalf("failed to set state with unicode: %v", err)
		}

		read, err := GetSignInState()
		if err != nil {
			t.Fatalf("failed to read state: %v", err)
		}

		if read.Name != state.Name {
			t.Errorf("name mismatch with unicode")
		}
	})

	t.Run("VeryLongEmail", func(t *testing.T) {
		longEmail := ""
		for range 1000 {
			longEmail += "a"
		}
		longEmail += "@example.com"

		state := &SignInState{Name: "user", Email: longEmail}
		if err := SetSignInState(state); err != nil {
			t.Fatalf("failed to set state with long email: %v", err)
		}

		read, err := GetSignInState()
		if err != nil {
			t.Fatalf("failed to read state: %v", err)
		}

		if read.Email != longEmail {
			t.Error("email mismatch with long value")
		}
	})
}

// TestConcurrentAccess tests race conditions
func TestConcurrentAccess(t *testing.T) {
	_ = setupTestDir(t)

	t.Run("ConcurrentReads", func(t *testing.T) {
		// Set up initial state
		state := &SignInState{Name: "concurrent", Email: "concurrent@example.com"}
		SetSignInState(state)

		// Multiple concurrent reads should all succeed
		done := make(chan bool, 10)
		for range 10 {
			go func() {
				read, err := GetSignInState()
				if err != nil {
					t.Errorf("concurrent read failed: %v", err)
				}
				if read.Name != "concurrent" {
					t.Errorf("wrong name in concurrent read")
				}
				done <- true
			}()
		}

		for range 10 {
			<-done
		}
	})

	t.Run("ConcurrentWrites", func(t *testing.T) {
		// Multiple concurrent writes - last one should win
		done := make(chan bool, 10)
		for range 10 {
			go func() {
				state := &SignInState{
					Name:  "user",
					Email: "user@example.com",
				}
				SetSignInState(state)
				done <- true
			}()
		}

		for range 10 {
			<-done
		}

		// Should have some valid state
		if !IsSignedIn() {
			t.Error("should be signed in after concurrent writes")
		}
	})
}

// TestFileSystemEdgeCases tests filesystem-related edge cases
func TestFileSystemEdgeCases(t *testing.T) {
	t.Run("DirectoryIsFile", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create a file where .ollama directory should be
		ollamaPath := filepath.Join(tmpDir, ".ollama")
		if err := os.WriteFile(ollamaPath, []byte("not a directory"), 0o600); err != nil {
			t.Fatalf("failed to create blocking file: %v", err)
		}

		// Override home directory
		if runtime.GOOS == "windows" {
			t.Setenv("USERPROFILE", tmpDir)
		} else {
			t.Setenv("HOME", tmpDir)
		}

		// Try to write - should fail because .ollama is a file, not a directory
		state := &SignInState{Name: "newuser", Email: "new@example.com"}
		err := SetSignInState(state)
		if err == nil {
			t.Error("should fail when .ollama is a file instead of directory")
		}
	})
}
