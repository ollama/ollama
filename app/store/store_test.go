//go:build windows || darwin

package store

import (
	"fmt"
	"os"
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

	t.Run("settings default home view is launch", func(t *testing.T) {
		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "launch" {
			t.Fatalf("expected default LastHomeView to be launch, got %q", loaded.LastHomeView)
		}
	})

	t.Run("settings empty home view falls back to launch", func(t *testing.T) {
		if err := s.SetSettings(Settings{LastHomeView: ""}); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "launch" {
			t.Fatalf("expected empty LastHomeView to fall back to launch, got %q", loaded.LastHomeView)
		}
	})

	t.Run("settings disabled home view falls back to launch", func(t *testing.T) {
		if err := s.SetSettings(Settings{LastHomeView: "claude-desktop"}); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "launch" {
			t.Fatalf("expected disabled LastHomeView to fall back to launch, got %q", loaded.LastHomeView)
		}
	})

	t.Run("settings codex app home view is accepted", func(t *testing.T) {
		if err := s.SetSettings(Settings{LastHomeView: "codex-app"}); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "codex-app" {
			t.Fatalf("expected codex-app LastHomeView to be preserved, got %q", loaded.LastHomeView)
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

func TestDeleteAllChats(t *testing.T) {
	t.Run("deletes all chats and their messages", func(t *testing.T) {
		s, cleanup := setupTestStore(t)
		defer cleanup()

		// Create several chats with messages
		for i, id := range []string{"chat-a", "chat-b", "chat-c"} {
			chat := NewChat(id)
			chat.Messages = append(chat.Messages, NewMessage("user", fmt.Sprintf("msg %d", i), nil))
			if err := s.SetChat(*chat); err != nil {
				t.Fatalf("failed to save chat %s: %v", id, err)
			}
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("failed to list chats: %v", err)
		}
		if len(chats) != 3 {
			t.Fatalf("expected 3 chats before delete, got %d", len(chats))
		}

		if err := s.DeleteAllChats(); err != nil {
			t.Fatalf("DeleteAllChats() error = %v", err)
		}

		chats, err = s.Chats()
		if err != nil {
			t.Fatalf("failed to list chats after delete: %v", err)
		}
		if len(chats) != 0 {
			t.Fatalf("expected 0 chats after DeleteAllChats, got %d", len(chats))
		}
	})

	t.Run("empty store is a no-op", func(t *testing.T) {
		s, cleanup := setupTestStore(t)
		defer cleanup()

		if err := s.DeleteAllChats(); err != nil {
			t.Fatalf("DeleteAllChats() on empty store returned error: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("failed to list chats: %v", err)
		}
		if len(chats) != 0 {
			t.Fatalf("expected 0 chats, got %d", len(chats))
		}
	})

	t.Run("cascades to messages, tool_calls, and attachments", func(t *testing.T) {
		s, cleanup := setupTestStore(t)
		defer cleanup()

		chat := NewChat("cascade-test")
		chat.Messages = append(chat.Messages,
			NewMessage("user", "hello", nil),
			NewMessage("assistant", "hi", &MessageOptions{Model: "llama4"}),
		)
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("failed to save chat: %v", err)
		}

		if err := s.DeleteAllChats(); err != nil {
			t.Fatalf("DeleteAllChats() error = %v", err)
		}

		// The chat should be gone
		_, err := s.Chat("cascade-test")
		if err == nil {
			t.Error("expected error retrieving deleted chat, got nil")
		}

		// Verify messages were cascade-deleted via the database directly
		var msgCount int
		if err := s.db.conn.QueryRow("SELECT COUNT(*) FROM messages").Scan(&msgCount); err != nil {
			t.Fatalf("failed to count messages: %v", err)
		}
		if msgCount != 0 {
			t.Errorf("expected 0 messages after DeleteAllChats, got %d", msgCount)
		}
	})

	t.Run("image directory is removed", func(t *testing.T) {
		s, cleanup := setupTestStore(t)
		defer cleanup()

		// Create the images directory to simulate prior usage
		imgDir := s.ImgDir()
		if err := os.MkdirAll(imgDir, 0o755); err != nil {
			t.Fatalf("failed to create image dir: %v", err)
		}

		if err := s.DeleteAllChats(); err != nil {
			t.Fatalf("DeleteAllChats() error = %v", err)
		}

		if _, err := os.Stat(imgDir); !os.IsNotExist(err) {
			t.Errorf("expected image directory to be removed, but it still exists")
		}
	})

	t.Run("image directory missing is not an error", func(t *testing.T) {
		s, cleanup := setupTestStore(t)
		defer cleanup()

		// Don't create the image directory — DeleteAllChats should still succeed
		if err := s.DeleteAllChats(); err != nil {
			t.Fatalf("DeleteAllChats() with missing image dir returned error: %v", err)
		}
	})

	t.Run("store remains usable after delete", func(t *testing.T) {
		s, cleanup := setupTestStore(t)
		defer cleanup()

		chat := NewChat("before")
		chat.Messages = append(chat.Messages, NewMessage("user", "hello", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("failed to save initial chat: %v", err)
		}

		if err := s.DeleteAllChats(); err != nil {
			t.Fatalf("DeleteAllChats() error = %v", err)
		}

		// Should be able to create a new chat after deleting all
		newChat := NewChat("after")
		newChat.Messages = append(newChat.Messages, NewMessage("user", "world", nil))
		if err := s.SetChat(*newChat); err != nil {
			t.Fatalf("failed to save new chat after DeleteAllChats: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("failed to list chats: %v", err)
		}
		if len(chats) != 1 {
			t.Fatalf("expected 1 chat after re-creating, got %d", len(chats))
		}
		if chats[0].ID != "after" {
			t.Errorf("expected new chat ID 'after', got %q", chats[0].ID)
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
