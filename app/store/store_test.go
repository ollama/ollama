//go:build windows || darwin

package store

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/onboarding"
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

	t.Run("settings default home view is chat", func(t *testing.T) {
		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "chat" {
			t.Fatalf("expected default LastHomeView to be chat, got %q", loaded.LastHomeView)
		}
	})

	t.Run("settings empty home view falls back to chat", func(t *testing.T) {
		if err := s.SetSettings(Settings{LastHomeView: ""}); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "chat" {
			t.Fatalf("expected empty LastHomeView to fall back to chat, got %q", loaded.LastHomeView)
		}
	})

	t.Run("settings retired home view falls back to chat", func(t *testing.T) {
		if err := s.SetSettings(Settings{LastHomeView: "claude-desktop"}); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "chat" {
			t.Fatalf("expected retired LastHomeView to fall back to chat, got %q", loaded.LastHomeView)
		}
	})

	t.Run("settings integration home view falls back to chat", func(t *testing.T) {
		if err := s.SetSettings(Settings{LastHomeView: "codex-app"}); err != nil {
			t.Fatal(err)
		}

		loaded, err := s.Settings()
		if err != nil {
			t.Fatal(err)
		}

		if loaded.LastHomeView != "chat" {
			t.Fatalf("expected integration LastHomeView to fall back to chat, got %q", loaded.LastHomeView)
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

func TestOnboardingVersionRoundTrip(t *testing.T) {
	s, cleanup := setupTestStore(t)
	defer cleanup()

	settings, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if settings.OnboardingVersion != 0 {
		t.Fatalf("expected onboarding version 0 by default, got %d", settings.OnboardingVersion)
	}

	settings.OnboardingVersion = 1
	if err := s.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	loaded, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.OnboardingVersion != 1 {
		t.Fatalf("expected onboarding version 1, got %d", loaded.OnboardingVersion)
	}
}

func setupPairedOnboarding(t *testing.T) *Store {
	t.Helper()
	s, cleanup := setupTestStore(t)
	t.Cleanup(cleanup)
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("LOCALAPPDATA", filepath.Join(home, "AppData", "Local"))
	previous := defaultDBPath
	defaultDBPath = onboarding.AppDatabasePath()
	t.Cleanup(func() { defaultDBPath = previous })
	s.DBPath = ""
	return s
}

func TestSettingsSurviveInvalidOnboardingMarker(t *testing.T) {
	s, cleanup := setupTestStore(t)
	t.Cleanup(cleanup)
	marker := filepath.Join(filepath.Dir(s.DBPath), "onboarding-v1.completed")
	if err := os.MkdirAll(marker, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := s.SetSettings(Settings{SelectedModel: "saved-model"}); err != nil {
		t.Fatal(err)
	}
	settings, err := s.Settings()
	if err != nil || settings.SelectedModel != "saved-model" || settings.OnboardingVersion != 0 {
		t.Fatalf("invalid marker must preserve readable database settings: %+v, %v", settings, err)
	}
}

func TestAppCompletionReachesCLI(t *testing.T) {
	for _, source := range []string{"app", "existing database"} {
		t.Run(source, func(t *testing.T) {
			s := setupPairedOnboarding(t)
			if err := s.ensureDB(); err != nil {
				t.Fatal(err)
			}
			if needed, err := config.NeedsWelcome(); err != nil || !needed {
				t.Fatalf("unfinished app skipped CLI onboarding: %v, %v", needed, err)
			}
			settings := Settings{OnboardingVersion: CurrentOnboardingVersion}
			save := s.SetSettings
			if source == "existing database" {
				save = s.db.setSettings // Older app completed without publishing a marker.
			}
			if err := save(settings); err != nil {
				t.Fatal(err)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if needed, err := config.NeedsWelcome(); err != nil || needed {
				t.Fatalf("app completion did not reach CLI: %v, %v", needed, err)
			}
		})
	}
}

func TestLegacyAppCompletionReachesCLI(t *testing.T) {
	for _, schema := range []int{1, 16, 17, 0} {
		t.Run(fmt.Sprint(schema), func(t *testing.T) {
			s := setupPairedOnboarding(t)
			if err := s.ensureDB(); err != nil {
				t.Fatal(err)
			}
			if _, err := s.db.conn.Exec("ALTER TABLE settings DROP COLUMN onboarding_version"); err != nil {
				t.Fatal(err)
			}
			if _, err := s.db.conn.Exec("UPDATE settings SET schema_version = ?", schema); err != nil {
				t.Fatal(err)
			}
			wantWelcome := schema < 1 || schema > 16
			if needed, err := config.NeedsWelcome(); err != nil || needed != wantWelcome {
				t.Fatalf("welcome needed=%v, err=%v; want %v", needed, err, wantWelcome)
			}
			var unchanged int
			if err := s.db.conn.QueryRow("SELECT schema_version FROM settings WHERE id = 1").Scan(&unchanged); err != nil || unchanged != schema {
				t.Fatalf("CLI changed the app schema: %d, %v", unchanged, err)
			}
			var columns int
			if err := s.db.conn.QueryRow("SELECT count(*) FROM pragma_table_info('settings') WHERE name = 'onboarding_version'").Scan(&columns); err != nil || columns != 0 {
				t.Fatalf("CLI migrated the app database: columns=%d, err=%v", columns, err)
			}
		})
	}
}

func TestCLICompletionReachesApp(t *testing.T) {
	for _, installed := range []bool{false, true} {
		s := setupPairedOnboarding(t)
		stale := Settings{SelectedModel: "saved-model"}
		if installed {
			if err := s.SetSettings(stale); err != nil {
				t.Fatal(err)
			}
		}
		if err := config.CompleteWelcome(); err != nil {
			t.Fatal(err)
		}
		if !installed {
			if _, err := os.Stat(defaultDBPath); !os.IsNotExist(err) {
				t.Fatal("CLI completion must not create the app database")
			}
		}
		settings, err := s.Settings()
		if err != nil || settings.OnboardingVersion != CurrentOnboardingVersion {
			t.Fatalf("app did not import CLI completion: %+v, %v", settings, err)
		}
		if installed && settings.SelectedModel != stale.SelectedModel {
			t.Fatal("completion changed app settings")
		}
		if err := s.SetSettings(stale); err != nil {
			t.Fatal(err)
		}
		if saved, err := s.Settings(); err != nil || saved.OnboardingVersion != CurrentOnboardingVersion {
			t.Fatalf("stale settings reset completion: %+v, %v", saved, err)
		}
	}
}

func TestClaudeDesktopUsedRoundTrip(t *testing.T) {
	s, cleanup := setupTestStore(t)
	defer cleanup()

	settings, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if settings.ClaudeDesktopUsed {
		t.Fatal("expected Claude Desktop history to be false by default")
	}

	settings.ClaudeDesktopUsed = true
	if err := s.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	loaded, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.ClaudeDesktopUsed {
		t.Fatal("expected Claude Desktop history to persist")
	}
}

func TestCodexDesktopUsedPreservedBySettings(t *testing.T) {
	s, cleanup := setupTestStore(t)
	defer cleanup()

	settings, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.Browser = true
	settings.ClaudeDesktopUsed = true
	settings.CodexDesktopUsed = true
	if err := s.SetSettings(settings); err != nil {
		t.Fatal(err)
	}
	saved, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if saved.CodexDesktopUsed {
		t.Fatal("ordinary settings save acknowledged the intro")
	}
	settings.CodexDesktopUsed = false
	if saved != settings {
		t.Fatal("ordinary settings save lost unrelated settings")
	}

	for range 2 {
		if err := s.MarkCodexDesktopUsed(); err != nil {
			t.Fatal(err)
		}
	}
	saved, err = s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	want := settings
	want.CodexDesktopUsed = true
	if saved != want {
		t.Fatal("acknowledgment did not preserve unrelated settings")
	}

	settings.Browser = false
	if err := s.SetSettings(settings); err != nil {
		t.Fatal(err)
	}
	saved, err = s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	want.Browser = false
	if saved != want {
		t.Fatal("stale settings save lost acknowledgment or the requested setting")
	}
}

func TestSpeechSettingsRoundTrip(t *testing.T) {
	s, cleanup := setupTestStore(t)
	defer cleanup()

	settings, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if settings.SpeechVoice != "" || settings.SpeechRate != 1 || settings.SpeechVolume != 1 || settings.SpeechAutoRead {
		t.Fatalf("unexpected speech defaults: %+v", settings)
	}

	settings.SpeechVoice = "com.example.voice"
	settings.SpeechRate = 1.5
	settings.SpeechVolume = 0.4
	settings.SpeechAutoRead = true
	if err := s.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	saved, err := s.Settings()
	if err != nil {
		t.Fatal(err)
	}
	if saved.SpeechVoice != settings.SpeechVoice || saved.SpeechRate != settings.SpeechRate || saved.SpeechVolume != settings.SpeechVolume || saved.SpeechAutoRead != settings.SpeechAutoRead {
		t.Fatalf("speech settings did not round trip: got %+v, want %+v", saved, settings)
	}
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
