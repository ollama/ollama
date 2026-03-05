//go:build windows || darwin

package store

import (
	"path/filepath"
	"testing"
)

func TestStore(t *testing.T) {
	s, cleanup := setupTestStore(t)
	defer cleanup()

	t.Run("default id", func(t *testing.T) {
		// ID should be automatically generated
		id, err := s.ID()
		if err != nil {
			t.Fatal(err)
		}
		if id == "" {
			t.Error("expected non-empty ID")
		}

		// Verify ID is persisted
		id2, err := s.ID()
		if err != nil {
			t.Fatal(err)
		}
		if id != id2 {
			t.Errorf("expected ID %s, got %s", id, id2)
		}
	})

	t.Run("has completed first run", func(t *testing.T) {
		// Default should be false (hasn't completed first run yet)
		hasCompleted, err := s.HasCompletedFirstRun()
		if err != nil {
			t.Fatal(err)
		}
		if hasCompleted {
			t.Error("expected has completed first run to be false by default")
		}

		if err := s.SetHasCompletedFirstRun(true); err != nil {
			t.Fatal(err)
		}

		hasCompleted, err = s.HasCompletedFirstRun()
		if err != nil {
			t.Fatal(err)
		}
		if !hasCompleted {
			t.Error("expected has completed first run to be true")
		}
	})

	t.Run("settings", func(t *testing.T) {
		sc := Settings{
			Expose:     true,
			Browser:    true,
			Survey:     true,
			Models:     "/tmp/models",
			Agent:      true,
			Tools:      false,
			WorkingDir: "/tmp/work",
		}

		if err := s.SetSettings(sc); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}
		// Compare fields individually since Models might get a default
		if loaded.Expose != sc.Expose || loaded.Browser != sc.Browser ||
			loaded.Agent != sc.Agent || loaded.Survey != sc.Survey ||
			loaded.Tools != sc.Tools || loaded.WorkingDir != sc.WorkingDir {
			t.Errorf("expected %v, got %v", sc, loaded)
		}
	})

	t.Run("window size", func(t *testing.T) {
		if err := s.SetWindowSize(1024, 768); err != nil {
			t.Fatal(err)
		}

		width, height, err := s.WindowSize()
		if err != nil {
			t.Fatal(err)
		}
		if width != 1024 || height != 768 {
			t.Errorf("expected 1024x768, got %dx%d", width, height)
		}
	})

	t.Run("create and retrieve chat", func(t *testing.T) {
		chat := NewChat("test-chat-1")
		chat.Title = "Test Chat"

		chat.Messages = append(chat.Messages, NewMessage("user", "Hello", nil))
		chat.Messages = append(chat.Messages, NewMessage("assistant", "Hi there!", &MessageOptions{
			Model: "llama4",
		}))

		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("failed to save chat: %v", err)
		}

		retrieved, err := s.Chat("test-chat-1")
		if err != nil {
			t.Fatalf("failed to retrieve chat: %v", err)
		}

		if retrieved.ID != chat.ID {
			t.Errorf("expected ID %s, got %s", chat.ID, retrieved.ID)
		}
		if retrieved.Title != chat.Title {
			t.Errorf("expected title %s, got %s", chat.Title, retrieved.Title)
		}
		if len(retrieved.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(retrieved.Messages))
		}
		if retrieved.Messages[0].Content != "Hello" {
			t.Errorf("expected first message 'Hello', got %s", retrieved.Messages[0].Content)
		}
		if retrieved.Messages[1].Content != "Hi there!" {
			t.Errorf("expected second message 'Hi there!', got %s", retrieved.Messages[1].Content)
		}
	})

	t.Run("list chats", func(t *testing.T) {
		chat2 := NewChat("test-chat-2")
		chat2.Title = "Another Chat"
		chat2.Messages = append(chat2.Messages, NewMessage("user", "Test", nil))

		if err := s.SetChat(*chat2); err != nil {
			t.Fatalf("failed to save chat: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("failed to list chats: %v", err)
		}

		if len(chats) != 2 {
			t.Fatalf("expected 2 chats, got %d", len(chats))
		}
	})

	t.Run("delete chat", func(t *testing.T) {
		if err := s.DeleteChat("test-chat-1"); err != nil {
			t.Fatalf("failed to delete chat: %v", err)
		}

		// Verify it's gone
		_, err := s.Chat("test-chat-1")
		if err == nil {
			t.Error("expected error retrieving deleted chat")
		}

		// Verify other chat still exists
		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("failed to list chats: %v", err)
		}
		if len(chats) != 1 {
			t.Fatalf("expected 1 chat after deletion, got %d", len(chats))
		}
	})
}

// setupTestStore creates a temporary store for testing
func setupTestStore(t *testing.T) (*Store, func()) {
	t.Helper()

	tmpDir := t.TempDir()

	// Override legacy config path to ensure no migration happens
	oldLegacyConfigPath := legacyConfigPath
	legacyConfigPath = filepath.Join(tmpDir, "config.json")

	s := &Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}

	cleanup := func() {
		s.Close()
		legacyConfigPath = oldLegacyConfigPath
	}

	return s, cleanup
}
