//go:build windows || darwin

package store

import (
	"fmt"
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

func TestStore_ClearChatHistory(t *testing.T) {
	t.Run("basic chat clearing", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		chat1 := NewChat("chat-clear-1")
		chat1.Title = "First"
		chat1.Messages = append(chat1.Messages, NewMessage("user", "Hello", nil))
		chat2 := NewChat("chat-clear-2")
		chat2.Title = "Second"
		chat2.Messages = append(chat2.Messages, NewMessage("user", "Hi", nil))

		if err := s.SetChat(*chat1); err != nil {
			t.Fatalf("SetChat chat1: %v", err)
		}
		if err := s.SetChat(*chat2); err != nil {
			t.Fatalf("SetChat chat2: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 2 {
			t.Errorf("expected 2 chats before ClearChatHistory, got %d", len(chats))
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chats, err = s.Chats()
		if err != nil {
			t.Fatalf("Chats after clear: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after ClearChatHistory, got %d", len(chats))
		}
	})

	t.Run("empty database", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory on empty db should not fail: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats, got %d", len(chats))
		}
	})

	t.Run("multiple calls idempotent", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("first ClearChatHistory: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("second ClearChatHistory: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("third ClearChatHistory: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after multiple clears, got %d", len(chats))
		}
	})

	t.Run("settings table not affected", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		originalSettings := Settings{
			Expose:           true,
			Browser:          true,
			Survey:           false,
			Models:           "/custom/models",
			Agent:            true,
			Tools:            true,
			WorkingDir:       "/custom/work",
			ContextLength:    8192,
			TurboEnabled:     true,
			WebSearchEnabled: true,
			SelectedModel:    "llama4",
			SidebarOpen:      true,
			ThinkEnabled:     true,
			ThinkLevel:       "high",
		}

		if err := s.SetSettings(originalSettings); err != nil {
			t.Fatalf("SetSettings: %v", err)
		}

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test message", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after clear, got %d", len(chats))
		}

		loadedSettings, err := s.Settings()
		if err != nil {
			t.Fatalf("Settings: %v", err)
		}

		if loadedSettings.Expose != originalSettings.Expose {
			t.Errorf("Expose changed: expected %v, got %v", originalSettings.Expose, loadedSettings.Expose)
		}
		if loadedSettings.Browser != originalSettings.Browser {
			t.Errorf("Browser changed: expected %v, got %v", originalSettings.Browser, loadedSettings.Browser)
		}
		if loadedSettings.Survey != originalSettings.Survey {
			t.Errorf("Survey changed: expected %v, got %v", originalSettings.Survey, loadedSettings.Survey)
		}
		if loadedSettings.Agent != originalSettings.Agent {
			t.Errorf("Agent changed: expected %v, got %v", originalSettings.Agent, loadedSettings.Agent)
		}
		if loadedSettings.Tools != originalSettings.Tools {
			t.Errorf("Tools changed: expected %v, got %v", originalSettings.Tools, loadedSettings.Tools)
		}
		if loadedSettings.WorkingDir != originalSettings.WorkingDir {
			t.Errorf("WorkingDir changed: expected %v, got %v", originalSettings.WorkingDir, loadedSettings.WorkingDir)
		}
		if loadedSettings.ContextLength != originalSettings.ContextLength {
			t.Errorf("ContextLength changed: expected %v, got %v", originalSettings.ContextLength, loadedSettings.ContextLength)
		}
		if loadedSettings.TurboEnabled != originalSettings.TurboEnabled {
			t.Errorf("TurboEnabled changed: expected %v, got %v", originalSettings.TurboEnabled, loadedSettings.TurboEnabled)
		}
		if loadedSettings.WebSearchEnabled != originalSettings.WebSearchEnabled {
			t.Errorf("WebSearchEnabled changed: expected %v, got %v", originalSettings.WebSearchEnabled, loadedSettings.WebSearchEnabled)
		}
		if loadedSettings.SelectedModel != originalSettings.SelectedModel {
			t.Errorf("SelectedModel changed: expected %v, got %v", originalSettings.SelectedModel, loadedSettings.SelectedModel)
		}
		if loadedSettings.SidebarOpen != originalSettings.SidebarOpen {
			t.Errorf("SidebarOpen changed: expected %v, got %v", originalSettings.SidebarOpen, loadedSettings.SidebarOpen)
		}
		if loadedSettings.ThinkEnabled != originalSettings.ThinkEnabled {
			t.Errorf("ThinkEnabled changed: expected %v, got %v", originalSettings.ThinkEnabled, loadedSettings.ThinkEnabled)
		}
		if loadedSettings.ThinkLevel != originalSettings.ThinkLevel {
			t.Errorf("ThinkLevel changed: expected %v, got %v", originalSettings.ThinkLevel, loadedSettings.ThinkLevel)
		}
	})

	t.Run("users table not affected", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		originalUser := User{
			Name:  "Test User",
			Email: "test@example.com",
			Plan:  "pro",
		}

		if err := s.SetUser(originalUser); err != nil {
			t.Fatalf("SetUser: %v", err)
		}

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test message", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after clear, got %d", len(chats))
		}

		loadedUser, err := s.User()
		if err != nil {
			t.Fatalf("User: %v", err)
		}
		if loadedUser == nil {
			t.Fatal("expected user data to remain, got nil")
		}

		if loadedUser.Name != originalUser.Name {
			t.Errorf("Name changed: expected %v, got %v", originalUser.Name, loadedUser.Name)
		}
		if loadedUser.Email != originalUser.Email {
			t.Errorf("Email changed: expected %v, got %v", originalUser.Email, loadedUser.Email)
		}
		if loadedUser.Plan != originalUser.Plan {
			t.Errorf("Plan changed: expected %v, got %v", originalUser.Plan, loadedUser.Plan)
		}
	})

	t.Run("messages cascade deleted", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		chat := NewChat("test-chat")
		chat.Title = "Test"
		chat.Messages = append(chat.Messages, NewMessage("user", "Message 1", nil))
		chat.Messages = append(chat.Messages, NewMessage("assistant", "Response 1", nil))
		chat.Messages = append(chat.Messages, NewMessage("user", "Message 2", nil))

		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		messageCountBefore := countRows(t, s.db, "messages")
		if messageCountBefore != 3 {
			t.Errorf("expected 3 messages before clear, got %d", messageCountBefore)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		messageCountAfter := countRows(t, s.db, "messages")
		if messageCountAfter != 0 {
			t.Errorf("expected 0 messages after clear (CASCADE), got %d", messageCountAfter)
		}

		chatCountAfter := countRows(t, s.db, "chats")
		if chatCountAfter != 0 {
			t.Errorf("expected 0 chats after clear, got %d", chatCountAfter)
		}
	})

	t.Run("tool_calls cascade deleted", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		chat := NewChat("test-chat")
		chat.Title = "Test with Tools"

		toolCalls := []ToolCall{
			{
				Type: "function",
				Function: ToolFunction{
					Name:      "get_weather",
					Arguments: `{"location": "San Francisco"}`,
				},
			},
			{
				Type: "function",
				Function: ToolFunction{
					Name:      "search",
					Arguments: `{"query": "ollama"}`,
				},
			},
		}

		chat.Messages = append(chat.Messages, NewMessage("user", "What's the weather?", nil))
		chat.Messages = append(chat.Messages, NewMessage("assistant", "Let me check", &MessageOptions{
			ToolCalls: toolCalls,
		}))

		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		toolCallCountBefore := countRows(t, s.db, "tool_calls")
		if toolCallCountBefore != 2 {
			t.Errorf("expected 2 tool_calls before clear, got %d", toolCallCountBefore)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		toolCallCountAfter := countRows(t, s.db, "tool_calls")
		if toolCallCountAfter != 0 {
			t.Errorf("expected 0 tool_calls after clear (CASCADE), got %d", toolCallCountAfter)
		}

		chatCountAfter := countRows(t, s.db, "chats")
		if chatCountAfter != 0 {
			t.Errorf("expected 0 chats after clear, got %d", chatCountAfter)
		}
	})

	t.Run("attachments cascade deleted", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		chat := NewChat("test-chat")
		chat.Title = "Test with Attachments"

		attachments := []File{
			{
				Filename: "test1.txt",
				Data:     []byte("test data 1"),
			},
			{
				Filename: "test2.txt",
				Data:     []byte("test data 2"),
			},
			{
				Filename: "test3.jpg",
				Data:     []byte("fake image data"),
			},
		}

		chat.Messages = append(chat.Messages, NewMessage("user", "Here are some files", &MessageOptions{
			Attachments: attachments,
		}))
		chat.Messages = append(chat.Messages, NewMessage("assistant", "Got them", nil))

		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		attachmentCountBefore := countRows(t, s.db, "attachments")
		if attachmentCountBefore != 3 {
			t.Errorf("expected 3 attachments before clear, got %d", attachmentCountBefore)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		attachmentCountAfter := countRows(t, s.db, "attachments")
		if attachmentCountAfter != 0 {
			t.Errorf("expected 0 attachments after clear (CASCADE), got %d", attachmentCountAfter)
		}

		chatCountAfter := countRows(t, s.db, "chats")
		if chatCountAfter != 0 {
			t.Errorf("expected 0 chats after clear, got %d", chatCountAfter)
		}
	})

	t.Run("window size not affected", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		if err := s.SetWindowSize(1920, 1080); err != nil {
			t.Fatalf("SetWindowSize: %v", err)
		}

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after clear, got %d", len(chats))
		}

		width, height, err := s.WindowSize()
		if err != nil {
			t.Fatalf("WindowSize: %v", err)
		}
		if width != 1920 || height != 1080 {
			t.Errorf("WindowSize changed: expected 1920x1080, got %dx%d", width, height)
		}
	})

	t.Run("integration test", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		settings := Settings{
			Expose:           true,
			Browser:          true,
			Models:           "/test/models",
			Agent:            true,
			ContextLength:    16384,
			TurboEnabled:     true,
			WebSearchEnabled: false,
			SelectedModel:    "llama4:latest",
			ThinkEnabled:     true,
		}
		if err := s.SetSettings(settings); err != nil {
			t.Fatalf("SetSettings: %v", err)
		}

		user := User{
			Name:  "Integration Test User",
			Email: "integration@test.com",
			Plan:  "enterprise",
		}
		if err := s.SetUser(user); err != nil {
			t.Fatalf("SetUser: %v", err)
		}

		if err := s.SetWindowSize(2560, 1440); err != nil {
			t.Fatalf("SetWindowSize: %v", err)
		}

		chat1 := NewChat("chat-1")
		chat1.Title = "Chat with Everything"
		chat1.Messages = append(chat1.Messages, NewMessage("user", "File upload test", &MessageOptions{
			Attachments: []File{
				{Filename: "doc.pdf", Data: []byte("pdf data")},
			},
		}))
		chat1.Messages = append(chat1.Messages, NewMessage("assistant", "Tool use test", &MessageOptions{
			ToolCalls: []ToolCall{
				{
					Type: "function",
					Function: ToolFunction{
						Name:      "calculator",
						Arguments: `{"expr": "2+2"}`,
					},
				},
			},
		}))

		chat2 := NewChat("chat-2")
		chat2.Title = "Another Chat"
		chat2.Messages = append(chat2.Messages, NewMessage("user", "Simple message", nil))

		if err := s.SetChat(*chat1); err != nil {
			t.Fatalf("SetChat chat1: %v", err)
		}
		if err := s.SetChat(*chat2); err != nil {
			t.Fatalf("SetChat chat2: %v", err)
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		chatCountBefore := countRows(t, s.db, "chats")
		messageCountBefore := countRows(t, s.db, "messages")
		toolCallCountBefore := countRows(t, s.db, "tool_calls")
		attachmentCountBefore := countRows(t, s.db, "attachments")

		if chatCountBefore != 2 {
			t.Errorf("expected 2 chats before clear, got %d", chatCountBefore)
		}
		if messageCountBefore != 3 {
			t.Errorf("expected 3 messages before clear, got %d", messageCountBefore)
		}
		if toolCallCountBefore != 1 {
			t.Errorf("expected 1 tool_call before clear, got %d", toolCallCountBefore)
		}
		if attachmentCountBefore != 1 {
			t.Errorf("expected 1 attachment before clear, got %d", attachmentCountBefore)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chatCountAfter := countRows(t, s.db, "chats")
		messageCountAfter := countRows(t, s.db, "messages")
		toolCallCountAfter := countRows(t, s.db, "tool_calls")
		attachmentCountAfter := countRows(t, s.db, "attachments")

		if chatCountAfter != 0 {
			t.Errorf("expected 0 chats after clear, got %d", chatCountAfter)
		}
		if messageCountAfter != 0 {
			t.Errorf("expected 0 messages after clear (CASCADE), got %d", messageCountAfter)
		}
		if toolCallCountAfter != 0 {
			t.Errorf("expected 0 tool_calls after clear (CASCADE), got %d", toolCallCountAfter)
		}
		if attachmentCountAfter != 0 {
			t.Errorf("expected 0 attachments after clear (CASCADE), got %d", attachmentCountAfter)
		}

		loadedSettings, err := s.Settings()
		if err != nil {
			t.Fatalf("Settings after clear: %v", err)
		}
		if loadedSettings.Expose != settings.Expose ||
			loadedSettings.Browser != settings.Browser ||
			loadedSettings.Agent != settings.Agent ||
			loadedSettings.ContextLength != settings.ContextLength ||
			loadedSettings.TurboEnabled != settings.TurboEnabled ||
			loadedSettings.SelectedModel != settings.SelectedModel ||
			loadedSettings.ThinkEnabled != settings.ThinkEnabled {
			t.Errorf("Settings were modified after ClearChatHistory")
		}

		loadedUser, err := s.User()
		if err != nil {
			t.Fatalf("User after clear: %v", err)
		}
		if loadedUser == nil {
			t.Fatal("expected user to remain after clear")
		}
		if loadedUser.Name != user.Name || loadedUser.Email != user.Email || loadedUser.Plan != user.Plan {
			t.Errorf("User data was modified after ClearChatHistory")
		}

		width, height, err := s.WindowSize()
		if err != nil {
			t.Fatalf("WindowSize after clear: %v", err)
		}
		if width != 2560 || height != 1440 {
			t.Errorf("WindowSize changed after clear: expected 2560x1440, got %dx%d", width, height)
		}
	})

	t.Run("multiple chats with complex data", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		for i := 1; i <= 5; i++ {
			chat := NewChat(fmt.Sprintf("chat-%d", i))
			chat.Title = fmt.Sprintf("Chat %d", i)

			for j := 1; j <= 3; j++ {
				chat.Messages = append(chat.Messages, NewMessage("user", fmt.Sprintf("Message %d", j), &MessageOptions{
					Attachments: []File{
						{Filename: fmt.Sprintf("file-%d-%d.txt", i, j), Data: []byte(fmt.Sprintf("data %d %d", i, j))},
					},
				}))
				chat.Messages = append(chat.Messages, NewMessage("assistant", fmt.Sprintf("Response %d", j), &MessageOptions{
					ToolCalls: []ToolCall{
						{
							Type: "function",
							Function: ToolFunction{
								Name:      "test_tool",
								Arguments: fmt.Sprintf(`{"arg": "%d-%d"}`, i, j),
							},
						},
					},
				}))
			}

			if err := s.SetChat(*chat); err != nil {
				t.Fatalf("SetChat chat-%d: %v", i, err)
			}
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		chatCountBefore := countRows(t, s.db, "chats")
		messageCountBefore := countRows(t, s.db, "messages")
		toolCallCountBefore := countRows(t, s.db, "tool_calls")
		attachmentCountBefore := countRows(t, s.db, "attachments")

		if chatCountBefore != 5 {
			t.Errorf("expected 5 chats, got %d", chatCountBefore)
		}
		if messageCountBefore != 30 {
			t.Errorf("expected 30 messages (5 chats * 6 msgs), got %d", messageCountBefore)
		}
		if toolCallCountBefore != 15 {
			t.Errorf("expected 15 tool_calls, got %d", toolCallCountBefore)
		}
		if attachmentCountBefore != 15 {
			t.Errorf("expected 15 attachments, got %d", attachmentCountBefore)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chatCountAfter := countRows(t, s.db, "chats")
		messageCountAfter := countRows(t, s.db, "messages")
		toolCallCountAfter := countRows(t, s.db, "tool_calls")
		attachmentCountAfter := countRows(t, s.db, "attachments")

		if chatCountAfter != 0 {
			t.Errorf("expected 0 chats after clear, got %d", chatCountAfter)
		}
		if messageCountAfter != 0 {
			t.Errorf("expected 0 messages after clear (CASCADE), got %d", messageCountAfter)
		}
		if toolCallCountAfter != 0 {
			t.Errorf("expected 0 tool_calls after clear (CASCADE), got %d", toolCallCountAfter)
		}
		if attachmentCountAfter != 0 {
			t.Errorf("expected 0 attachments after clear (CASCADE), got %d", attachmentCountAfter)
		}
	})

	t.Run("device id and first run status not affected", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		if err := s.SetHasCompletedFirstRun(true); err != nil {
			t.Fatalf("SetHasCompletedFirstRun: %v", err)
		}

		originalID, err := s.ID()
		if err != nil {
			t.Fatalf("ID: %v", err)
		}
		if originalID == "" {
			t.Fatal("expected non-empty ID")
		}

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after clear, got %d", len(chats))
		}

		loadedID, err := s.ID()
		if err != nil {
			t.Fatalf("ID after clear: %v", err)
		}
		if loadedID != originalID {
			t.Errorf("ID changed: expected %v, got %v", originalID, loadedID)
		}

		hasCompleted, err := s.HasCompletedFirstRun()
		if err != nil {
			t.Fatalf("HasCompletedFirstRun after clear: %v", err)
		}
		if !hasCompleted {
			t.Error("HasCompletedFirstRun should remain true after clear")
		}
	})

	t.Run("clear with corrupted database path fails gracefully", func(t *testing.T) {
		s := &Store{DBPath: "/nonexistent/path/to/db.sqlite"}

		err := s.ClearChatHistory()
		if err == nil {
			t.Error("expected error with invalid DB path")
		}
	})

	t.Run("settings table schema preserved after clear", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		settingsCountBefore := countRows(t, s.db, "settings")
		if settingsCountBefore != 1 {
			t.Errorf("expected 1 settings row, got %d", settingsCountBefore)
		}

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		settingsCountAfter := countRows(t, s.db, "settings")
		if settingsCountAfter != 1 {
			t.Errorf("settings table affected: expected 1 row, got %d", settingsCountAfter)
		}
	})

	t.Run("users table schema preserved after clear", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		user1 := User{Name: "User 1", Email: "user1@test.com", Plan: "free"}
		if err := s.SetUser(user1); err != nil {
			t.Fatalf("SetUser: %v", err)
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		usersCountBefore := countRows(t, s.db, "users")
		if usersCountBefore != 1 {
			t.Errorf("expected 1 user row, got %d", usersCountBefore)
		}

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Test", nil))
		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		usersCountAfter := countRows(t, s.db, "users")
		if usersCountAfter != 1 {
			t.Errorf("users table affected: expected 1 row, got %d", usersCountAfter)
		}

		chats, err := s.Chats()
		if err != nil {
			t.Fatalf("Chats: %v", err)
		}
		if len(chats) != 0 {
			t.Errorf("expected 0 chats after clear, got %d", len(chats))
		}
	})

	t.Run("no orphaned records after clear", func(t *testing.T) {
		s := &Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
		defer s.Close()

		chat := NewChat("test-chat")
		chat.Messages = append(chat.Messages, NewMessage("user", "Message with attachments", &MessageOptions{
			Attachments: []File{{Filename: "test.txt", Data: []byte("data")}},
		}))
		chat.Messages = append(chat.Messages, NewMessage("assistant", "Response", &MessageOptions{
			ToolCalls: []ToolCall{{Type: "function", Function: ToolFunction{Name: "test", Arguments: "{}"}}},
		}))

		if err := s.SetChat(*chat); err != nil {
			t.Fatalf("SetChat: %v", err)
		}

		if err := s.ensureDB(); err != nil {
			t.Fatalf("ensureDB: %v", err)
		}

		if err := s.ClearChatHistory(); err != nil {
			t.Fatalf("ClearChatHistory: %v", err)
		}

		orphanedMessages := countRowsWithCondition(t, s.db, "messages", "chat_id NOT IN (SELECT id FROM chats)")
		if orphanedMessages != 0 {
			t.Errorf("found %d orphaned messages", orphanedMessages)
		}

		orphanedToolCalls := countRowsWithCondition(t, s.db, "tool_calls", "message_id NOT IN (SELECT id FROM messages)")
		if orphanedToolCalls != 0 {
			t.Errorf("found %d orphaned tool_calls", orphanedToolCalls)
		}

		orphanedAttachments := countRowsWithCondition(t, s.db, "attachments", "message_id NOT IN (SELECT id FROM messages)")
		if orphanedAttachments != 0 {
			t.Errorf("found %d orphaned attachments", orphanedAttachments)
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
