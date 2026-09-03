package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/codexproxy"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/types/model"
)

func withCodexAppPlatform(t *testing.T, goos string) {
	t.Helper()
	old := codexAppGOOS
	codexAppGOOS = goos
	t.Cleanup(func() {
		codexAppGOOS = old
	})
}

func withCodexAppRouterHealth(t *testing.T, health func() error) {
	t.Helper()
	old := codexAppRouterHealth
	codexAppRouterHealth = health
	t.Cleanup(func() {
		codexAppRouterHealth = old
	})
}

func withCodexAppProcessHooks(t *testing.T, isRunning func() bool, quit func() error, open func() error) {
	t.Helper()
	oldIsRunning := codexAppIsRunning
	oldQuit := codexAppQuitApp
	oldOpen := codexAppOpenApp
	oldOpenPath := codexAppOpenPath
	oldOpenStart := codexAppOpenStart
	oldForceQuit := codexAppForceQuit
	oldHasWindow := codexAppHasWindow
	oldRunPath := codexAppRunPath
	oldStartID := codexAppStartID
	oldCanOpenID := codexAppCanOpenID
	oldExitTimeout := codexAppExitTimeout
	oldForceExitTimeout := codexAppForceExitTimeout
	codexAppIsRunning = isRunning
	codexAppHasWindow = isRunning
	codexAppQuitApp = quit
	codexAppOpenApp = func([]string) error { return open() }
	t.Cleanup(func() {
		codexAppIsRunning = oldIsRunning
		codexAppQuitApp = oldQuit
		codexAppOpenApp = oldOpen
		codexAppOpenPath = oldOpenPath
		codexAppOpenStart = oldOpenStart
		codexAppForceQuit = oldForceQuit
		codexAppHasWindow = oldHasWindow
		codexAppRunPath = oldRunPath
		codexAppStartID = oldStartID
		codexAppCanOpenID = oldCanOpenID
		codexAppExitTimeout = oldExitTimeout
		codexAppForceExitTimeout = oldForceExitTimeout
	})
}

func TestCodexAppIntegration(t *testing.T) {
	c := &CodexApp{}

	t.Run("implements runner", func(t *testing.T) {
		var _ Runner = c
	})
	t.Run("implements supported integration", func(t *testing.T) {
		var _ SupportedIntegration = c
	})
	t.Run("implements managed single model", func(t *testing.T) {
		var _ ManagedSingleModel = c
	})
	t.Run("receives model list", func(t *testing.T) {
		var _ ManagedModelListConfigurer = c
	})
	t.Run("onboarding is noninteractive", func(t *testing.T) {
		var _ ManagedInteractiveOnboarding = c
		if c.RequiresInteractiveOnboarding() {
			t.Fatal("Codex App onboarding should only mark launch config")
		}
	})
	t.Run("implements restore", func(t *testing.T) {
		var _ RestorableIntegration = c
		var _ RestoreHintIntegration = c
		var _ ConfigurationSuccessIntegration = c
		var _ RestoreSuccessIntegration = c
	})
}

func TestCodexAppNativeCatalogUsesDebugModelsInScratchHome(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(configPath), "auth.json"), []byte("test-auth"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(configPath), "models_cache.json"), []byte(`{"models":[{"slug":"cached"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}

	oldExecutable := codexAppCodexExecutable
	oldRun := codexAppRunDebugModels
	codexAppCodexExecutable = func() (string, error) { return "/test/codex", nil }
	codexAppRunDebugModels = func(executable, codexHome string, bundled bool) ([]byte, error) {
		if executable != "/test/codex" || codexHome == "" || bundled {
			t.Fatalf("debug models call = executable %q, CODEX_HOME %q, bundled %v", executable, codexHome, bundled)
		}
		for name, want := range map[string]string{
			"auth.json":         "test-auth",
			"models_cache.json": `{"models":[{"slug":"cached"}]}`,
		} {
			data, err := os.ReadFile(filepath.Join(codexHome, name))
			if err != nil || string(data) != want {
				t.Fatalf("scratch %s = %q, %v; want %q", name, data, err, want)
			}
		}
		return []byte(`{"models":[{"slug":"gpt-live","priority":1}]}`), nil
	}
	t.Cleanup(func() {
		codexAppCodexExecutable = oldExecutable
		codexAppRunDebugModels = oldRun
	})

	data, err := defaultCodexAppNativeModelCatalog(configPath)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := parseCodexAppModelCatalog(data)
	if err != nil {
		t.Fatal(err)
	}
	slug, err := codexAppRawCatalogSlug(catalog.Models[0])
	if err != nil || slug != "gpt-live" {
		t.Fatalf("native catalog slug = %q, %v; want gpt-live", slug, err)
	}
}

func TestCodexAppNativeCatalogFallsBackToCacheThenBundled(t *testing.T) {
	t.Run("cache", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(filepath.Dir(configPath), "models_cache.json"), []byte(`{"client_version":"test","models":[{"slug":"gpt-cached"}]}`), 0o600); err != nil {
			t.Fatal(err)
		}

		oldExecutable := codexAppCodexExecutable
		oldRun := codexAppRunDebugModels
		codexAppCodexExecutable = func() (string, error) { return "/test/codex", nil }
		codexAppRunDebugModels = func(_ string, _ string, bundled bool) ([]byte, error) {
			if bundled {
				t.Fatal("bundled fallback should not run when cache is valid")
			}
			return nil, errors.New("offline")
		}
		t.Cleanup(func() {
			codexAppCodexExecutable = oldExecutable
			codexAppRunDebugModels = oldRun
		})

		data, err := defaultCodexAppNativeModelCatalog(configPath)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), `"gpt-cached"`) || strings.Contains(string(data), "client_version") {
			t.Fatalf("normalized cached catalog = %s", data)
		}
	})

	t.Run("bundled", func(t *testing.T) {
		configPath := filepath.Join(t.TempDir(), ".codex", "config.toml")
		oldExecutable := codexAppCodexExecutable
		oldRun := codexAppRunDebugModels
		codexAppCodexExecutable = func() (string, error) { return "/test/codex", nil }
		codexAppRunDebugModels = func(_ string, _ string, bundled bool) ([]byte, error) {
			if !bundled {
				return nil, errors.New("offline")
			}
			return []byte(`{"models":[{"slug":"gpt-bundled"}]}`), nil
		}
		t.Cleanup(func() {
			codexAppCodexExecutable = oldExecutable
			codexAppRunDebugModels = oldRun
		})

		data, err := defaultCodexAppNativeModelCatalog(configPath)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), `"gpt-bundled"`) {
			t.Fatalf("bundled catalog = %s", data)
		}
	})
}

func TestCodexAppRouterHealth(t *testing.T) {
	t.Run("ready", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != codexproxy.PathPrefix+"/_health" {
				t.Fatalf("health path = %q", r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()
		t.Setenv("OLLAMA_HOST", server.URL)

		if err := defaultCodexAppRouterHealth(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("old server", func(t *testing.T) {
		server := httptest.NewServer(http.NotFoundHandler())
		defer server.Close()
		t.Setenv("OLLAMA_HOST", server.URL)

		err := defaultCodexAppRouterHealth()
		if err == nil || !strings.Contains(err.Error(), "does not include ChatGPT routing") {
			t.Fatalf("health error = %v", err)
		}
	})
}

func TestCodexAppDesktopChecksRouterBeforeChangingProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	healthErr := errors.New("router unavailable")
	withCodexAppRouterHealth(t, func() error { return healthErr })

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}

	openCalls := 0
	quitCalls := 0
	oldCanOpenID := codexAppCanOpenID
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() { codexAppCanOpenID = oldCanOpenID })
	withCodexAppProcessHooks(t,
		func() bool { return true },
		func() error { quitCalls++; return nil },
		func() error { openCalls++; return nil },
	)

	err := (&CodexApp{}).UseOllamaFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b"), false)
	if !errors.Is(err, healthErr) {
		t.Fatalf("UseOllamaFromDesktop error = %v", err)
	}
	if quitCalls != 0 || openCalls != 0 {
		t.Fatalf("process calls = quit %d, open %d; want none", quitCalls, openCalls)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != original {
		t.Fatalf("config changed before router health check: %q", data)
	}
}

func TestCodexAppDesktopRequiresRestartConfirmationBeforeChangingProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}

	openCalls := 0
	quitCalls := 0
	oldCanOpenID := codexAppCanOpenID
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() { codexAppCanOpenID = oldCanOpenID })
	withCodexAppProcessHooks(t,
		func() bool { return true },
		func() error { quitCalls++; return nil },
		func() error { openCalls++; return nil },
	)

	err := (&CodexApp{}).UseOllamaFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b"), false)
	if !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("UseOllamaFromDesktop error = %v, want restart confirmation", err)
	}
	if quitCalls != 0 || openCalls != 0 {
		t.Fatalf("process calls = quit %d, open %d; want none", quitCalls, openCalls)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != original {
		t.Fatalf("config changed before restart confirmation: %q", data)
	}
}

func TestCodexAppDesktopRequiresConsentBeforeStoppingLegacyProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}

	oldCanOpenID := codexAppCanOpenID
	oldProfileRunning := codexAppProfileIsRunning
	oldStopProfile := codexAppStopProfile
	codexAppCanOpenID = func() bool { return true }
	codexAppProfileIsRunning = func() bool { return true }
	stopCalls := 0
	codexAppStopProfile = func() error { stopCalls++; return nil }
	t.Cleanup(func() {
		codexAppCanOpenID = oldCanOpenID
		codexAppProfileIsRunning = oldProfileRunning
		codexAppStopProfile = oldStopProfile
	})
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error { return nil },
	)

	err := (&CodexApp{}).UseOllamaFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b"), false)
	if !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("UseOllamaFromDesktop error = %v, want restart confirmation", err)
	}
	if stopCalls != 0 {
		t.Fatalf("legacy profile stops before confirmation = %d, want none", stopCalls)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != original {
		t.Fatalf("config changed before legacy profile consent: %q", data)
	}
}

func TestCodexAppDesktopRechecksRestartConfirmationBeforeProfileWrite(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}

	runningChecks := 0
	openCalls := 0
	quitCalls := 0
	oldCanOpenID := codexAppCanOpenID
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() { codexAppCanOpenID = oldCanOpenID })
	withCodexAppProcessHooks(t,
		func() bool {
			runningChecks++
			return runningChecks > 1
		},
		func() error { quitCalls++; return nil },
		func() error { openCalls++; return nil },
	)

	err := (&CodexApp{}).UseOllamaFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b"), false)
	if !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("UseOllamaFromDesktop error = %v, want late restart confirmation", err)
	}
	if quitCalls != 0 || openCalls != 0 {
		t.Fatalf("process calls = quit %d, open %d; want none", quitCalls, openCalls)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != original {
		t.Fatalf("config changed after ChatGPT opened during preparation: %q", data)
	}
}

func TestCodexAppDesktopUsesAndRestoresRegularProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	authPath := filepath.Join(filepath.Dir(configPath), "auth.json")
	auth := []byte(`{"tokens":{"access_token":"same-profile"}}`)
	if err := os.WriteFile(authPath, auth, 0o600); err != nil {
		t.Fatal(err)
	}

	openCalls := 0
	oldCanOpenID := codexAppCanOpenID
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() { codexAppCanOpenID = oldCanOpenID })
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error { openCalls++; return nil },
	)

	app := &CodexApp{}
	if err := app.UseOllamaFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b", "glm-5.3-flash:cloud"), false); err != nil {
		t.Fatal(err)
	}
	if !app.OllamaConfigured() || app.CurrentModel() != "qwen3:8b" {
		t.Fatalf("regular profile was not configured for Ollama")
	}
	if got, err := os.ReadFile(authPath); err != nil || string(got) != string(auth) {
		t.Fatalf("shared auth changed: %q, %v", got, err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}

	if err := app.RestoreFromDesktop(false); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(data)) != strings.TrimSpace(original) {
		t.Fatalf("restored config = %q, want %q", data, original)
	}
	if openCalls != 1 {
		t.Fatalf("stopped ChatGPT should remain stopped on restore; open calls = %d", openCalls)
	}
}

func TestCodexAppDesktopAppliesProfileAfterRunningAppExits(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}

	running := true
	events := []string{}
	oldCanOpenID := codexAppCanOpenID
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() { codexAppCanOpenID = oldCanOpenID })
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			data, err := os.ReadFile(configPath)
			if err != nil {
				return err
			}
			if string(data) != original {
				return fmt.Errorf("config changed before ChatGPT exited: %s", data)
			}
			events = append(events, "quit")
			running = false
			return nil
		},
		func() error {
			events = append(events, "open")
			running = true
			return nil
		},
	)

	if err := (&CodexApp{}).UseOllamaFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b"), true); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(events, []string{"quit", "open"}) {
		t.Fatalf("events = %v, want quit then open", events)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := codexRootStringValueOK(string(data), codexRootModelProviderKey); ok {
		t.Fatalf("model provider = %q, want omitted built-in default after exit:\n%s", got, data)
	}
	if got := codexRootStringValue(string(data), codexRootOpenAIBaseURLKey); got != "http://127.0.0.1:11434/api/codex/v1" {
		t.Fatalf("openai_base_url = %q, want Codex router after exit:\n%s", got, data)
	}
}

func TestCodexAppDesktopRestartsExistingProfileWithoutRebuildingCatalog(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	app := &CodexApp{}
	if err := app.ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	original, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	running := true
	quitCalls := 0
	openCalls := 0
	oldCanOpenID := codexAppCanOpenID
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() { codexAppCanOpenID = oldCanOpenID })
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return nil
		},
		func() error {
			openCalls++
			running = true
			return nil
		},
	)

	if err := app.RestartFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("RestartFromDesktop error = %v, want restart confirmation", err)
	}
	if quitCalls != 0 || openCalls != 0 {
		t.Fatalf("process calls before confirmation = quit %d, open %d", quitCalls, openCalls)
	}
	if err := app.RestartFromDesktop(true); err != nil {
		t.Fatal(err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("process calls = quit %d, open %d; want one restart", quitCalls, openCalls)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(data, original) {
		t.Fatal("restart changed the existing ChatGPT profile")
	}
}

func TestCodexAppRestoreForShutdownDoesNotReopenChatGPT(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\n"
	if err := os.WriteFile(configPath, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := (&CodexApp{}).ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatal(err)
	}

	running := true
	openCalls := 0
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error { running = false; return nil },
		func() error { openCalls++; return nil },
	)
	if err := (&CodexApp{}).RestoreForShutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if openCalls != 0 {
		t.Fatalf("shutdown restore reopened ChatGPT %d times", openCalls)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(data)) != strings.TrimSpace(original) {
		t.Fatalf("shutdown restore config = %q, want %q", data, original)
	}
}

func TestCodexAppCountsOnlyOllamaRequestsInRegularProfile(t *testing.T) {
	setTestHome(t, t.TempDir())
	requestLine := func(at time.Time, kinds ...string) []byte {
		line, err := json.Marshal(map[string]any{
			"timestamp": at,
			"type":      "response_item",
			"payload": map[string]any{
				"type": "message",
				"role": "user",
				"internal_chat_message_metadata_passthrough": map[string]any{
					"content_item_kinds": kinds,
				},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return append(line, '\n')
	}
	turnContextLine := func(at time.Time, model string) []byte {
		line, err := json.Marshal(map[string]any{
			"timestamp": at,
			"type":      "turn_context",
			"payload": map[string]any{
				"model": model,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return append(line, '\n')
	}

	if err := resetCodexAppRegularProfileRequestCount(); err != nil {
		t.Fatal(err)
	}
	startPath, err := codexAppRegularProfileSessionStartPath()
	if err != nil {
		t.Fatal(err)
	}
	startData, err := os.ReadFile(startPath)
	if err != nil {
		t.Fatal(err)
	}
	start, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(string(startData)))
	if err != nil {
		t.Fatal(err)
	}
	configPath, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	regularSessionPath := filepath.Join(filepath.Dir(configPath), "sessions", "2026", "08", "28", "rollout.jsonl")
	if err := os.MkdirAll(filepath.Dir(regularSessionPath), 0o700); err != nil {
		t.Fatal(err)
	}
	routingCatalog := []byte(`{"models":[{"slug":"qwen3:8b"}]}`)
	if err := os.WriteFile(codexAppRoutingCatalogPathForConfig(configPath), routingCatalog, 0o600); err != nil {
		t.Fatal(err)
	}
	regularLines := append(turnContextLine(start.Add(time.Second), "qwen3:8b"), requestLine(start.Add(2*time.Second), "user.text")...)
	regularLines = append(regularLines, turnContextLine(start.Add(3*time.Second), "gpt-5.6-sol")...)
	regularLines = append(regularLines, requestLine(start.Add(4*time.Second), "user.text")...)
	regularLines = append(regularLines, turnContextLine(start.Add(5*time.Second), "qwen3:8b:latest")...)
	regularLines = append(regularLines, requestLine(start.Add(6*time.Second), "user.text")...)
	if err := os.WriteFile(regularSessionPath, regularLines, 0o600); err != nil {
		t.Fatal(err)
	}
	if got := codexAppRegularProfileRequestCount(); got != 2 {
		t.Fatalf("regular profile Ollama request count = %d, want 2", got)
	}

	file, err := os.OpenFile(regularSessionPath, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	nativeTurn := append(turnContextLine(start.Add(7*time.Second), "gpt-5.6-sol"), requestLine(start.Add(8*time.Second), "user.text")...)
	if _, err := file.Write(nativeTurn); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if got := codexAppRegularProfileRequestCount(); got != 2 {
		t.Fatalf("regular profile count after native request = %d, want 2", got)
	}

	file, err = os.OpenFile(regularSessionPath, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	ollamaTurn := append(turnContextLine(start.Add(9*time.Second), "qwen3:8b"), requestLine(start.Add(10*time.Second), "user.text")...)
	if _, err := file.Write(ollamaTurn); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if got := codexAppRegularProfileRequestCount(); got != 3 {
		t.Fatalf("incremental regular profile Ollama request count = %d, want 3", got)
	}
}

func TestCodexAppLegacyProfileCleanupRejectsStalePID(t *testing.T) {
	setTestHome(t, t.TempDir())
	withCodexAppPlatform(t, "darwin")
	if err := writeCodexAppOllamaProfilePID(424242); err != nil {
		t.Fatal(err)
	}

	oldExecutable := codexAppProfileExecutable
	oldCommand := codexAppProcessCommand
	codexAppProfileExecutable = func() (string, error) { return "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT", nil }
	codexAppProcessCommand = func(int) (string, error) { return "/usr/bin/unrelated --serve", nil }
	t.Cleanup(func() {
		codexAppProfileExecutable = oldExecutable
		codexAppProcessCommand = oldCommand
	})

	err := defaultCodexAppStopOllamaProfile()
	if err == nil || !strings.Contains(err.Error(), "identity could not be verified") {
		t.Fatalf("stale PID error = %v, want identity verification", err)
	}
	if _, ok := codexAppOllamaProfilePID(); ok {
		t.Fatal("stale PID was not cleared")
	}
}

func TestCodexAppSupportedPlatforms(t *testing.T) {
	for _, goos := range []string{"darwin", "windows"} {
		t.Run(goos, func(t *testing.T) {
			withCodexAppPlatform(t, goos)
			if err := codexAppSupported(); err != nil {
				t.Fatalf("codexAppSupported returned error: %v", err)
			}
		})
	}

	t.Run("linux unsupported", func(t *testing.T) {
		withCodexAppPlatform(t, "linux")
		err := codexAppSupported()
		if err == nil || !strings.Contains(err.Error(), "macOS and Windows") {
			t.Fatalf("codexAppSupported error = %v, want platform message", err)
		}
	})
}

func TestCodexAppWindowsAppPathCandidates(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	local := filepath.Join(t.TempDir(), "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)

	exe := filepath.Join(local, "Codex", "app-26.429.30905", "Codex.exe")
	if err := os.MkdirAll(filepath.Dir(exe), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(exe, []byte{}, 0o644); err != nil {
		t.Fatal(err)
	}

	if got := codexAppAppPath(); got != exe {
		t.Fatalf("codexAppAppPath = %q, want %q", got, exe)
	}
}

func TestCodexAppInstalledUsesWindowsStartMenuFallback(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	t.Setenv("LOCALAPPDATA", filepath.Join(t.TempDir(), "LocalAppData"))

	oldStartID := codexAppStartID
	oldIsRunning := codexAppIsRunning
	codexAppStartID = func() string { return "OpenAI.Codex_12345!App" }
	codexAppIsRunning = func() bool { return false }
	t.Cleanup(func() {
		codexAppStartID = oldStartID
		codexAppIsRunning = oldIsRunning
	})

	if !codexAppInstalled() {
		t.Fatal("expected Windows Start menu app id to count as installed")
	}
}

func TestCodexAppInstalledUsesMacBundleIDFallback(t *testing.T) {
	withCodexAppPlatform(t, "darwin")

	oldCanOpenID := codexAppCanOpenID
	oldStat := codexAppStat
	codexAppCanOpenID = func() bool { return true }
	codexAppStat = func(string) (os.FileInfo, error) { return nil, os.ErrNotExist }
	t.Cleanup(func() {
		codexAppCanOpenID = oldCanOpenID
		codexAppStat = oldStat
	})

	if !codexAppInstalled() {
		t.Fatal("expected macOS LaunchServices bundle id fallback to count as installed")
	}
}

func TestChatGPTMissingAppGivesDownloadRecovery(t *testing.T) {
	withCodexAppPlatform(t, "darwin")

	oldCanOpenID := codexAppCanOpenID
	oldStat := codexAppStat
	codexAppCanOpenID = func() bool { return false }
	codexAppStat = func(string) (os.FileInfo, error) { return nil, os.ErrNotExist }
	t.Cleanup(func() {
		codexAppCanOpenID = oldCanOpenID
		codexAppStat = oldStat
	})

	err := EnsureIntegrationInstalled(chatGPTIntegrationName, &CodexApp{})
	if err == nil {
		t.Fatal("expected missing ChatGPT install error")
	}
	if !strings.Contains(err.Error(), "chatgpt is not installed") || !strings.Contains(err.Error(), "https://chatgpt.com/download") {
		t.Fatalf("missing-app error = %q, want ChatGPT download recovery", err)
	}
}

func TestCodexAppConfigureAddsOllamaModelsToBuiltInProvider(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"profile = \"default\"\n" +
		"model = \"gpt-5.5\"\n\n" +
		"[profiles.default]\n" +
		"model = \"gpt-5.5\"\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CodexApp{}
	if err := c.ConfigureWithModels("llama3.2", testLaunchModels("llama3.2", "qwen3:8b")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)

	for _, want := range []string{
		`model = "llama3.2"`,
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath),
		`openai_base_url = "http://127.0.0.1:9999/api/codex/v1"`,
		`[profiles.default]`,
	} {
		if !strings.Contains(content, want) {
			t.Fatalf("expected config to contain %q, got:\n%s", want, content)
		}
	}
	if got, ok := codexRootStringValueOK(content, "profile"); ok {
		t.Fatalf("legacy root profile should be removed, got %q in:\n%s", got, content)
	}
	if got, ok := codexRootStringValueOK(content, codexRootModelProviderKey); ok {
		t.Fatalf("root model_provider = %q, want omitted built-in default in:\n%s", got, content)
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) {
		t.Fatalf("legacy app profile section should not be generated, got:\n%s", content)
	}
	if strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("custom app provider should not be generated, got:\n%s", content)
	}
	if got := c.CurrentModel(); got != "llama3.2" {
		t.Fatalf("CurrentModel = %q, want llama3.2", got)
	}

	restoreData, err := os.ReadFile(codexAppRestoreStatePath())
	if err != nil {
		t.Fatalf("expected restore state: %v", err)
	}
	if !strings.Contains(string(restoreData), `"profile": "default"`) {
		t.Fatalf("expected restore state to remember default profile, got %s", restoreData)
	}
	catalogData, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatalf("expected model catalog: %v", err)
	}
	var catalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(catalogData, &catalog); err != nil {
		t.Fatalf("catalog should be valid JSON: %v", err)
	}
	if got := catalogSlugs(catalog.Models); strings.Join(got, ",") != "llama3.2,qwen3:8b,gpt-5.6-sol" {
		t.Fatalf("catalog slugs = %v, want Ollama models followed by native models", got)
	}
	ollamaEntry := catalog.Models[0]
	for key, want := range map[string]any{
		"shell_type":                           "unified_exec",
		"supported_in_api":                     true,
		"include_skills_usage_instructions":    true,
		"include_plugin_usage_instructions":    true,
		"include_apps_usage_instructions":      true,
		"supports_reasoning_summary_parameter": false,
	} {
		if got := ollamaEntry[key]; got != want {
			t.Fatalf("Ollama catalog %s = %#v, want %#v", key, got, want)
		}
	}
	truncationPolicy, ok := ollamaEntry["truncation_policy"].(map[string]any)
	if !ok || truncationPolicy["mode"] != "tokens" {
		t.Fatalf("Ollama truncation policy = %#v, want token mode", ollamaEntry["truncation_policy"])
	}
	if _, ok := ollamaEntry["tool_mode"]; ok {
		t.Fatalf("Ollama catalog must not opt into a native-only code tool mode: %#v", ollamaEntry)
	}
}

func TestCodexAppReasoningEffortsExposeOffAndMaxWithoutDroppingUserChoices(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		efforts []string
		want    []string
	}{
		{
			name:    "new desktop setting",
			text:    `model = "gpt-5.6-sol"` + "\n",
			efforts: []string{"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"},
			want:    []string{"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"},
		},
		{
			name: "existing section and multiline value",
			text: "[desktop]\n" +
				"enabled-reasoning-efforts = [\n" +
				"  \"low\",\n" +
				"  \"high\",\n" +
				"]\n" +
				`theme = "system"` + "\n",
			efforts: []string{"none", "low", "high", "max"},
			want:    []string{"none", "low", "high", "max"},
		},
		{
			name:    "existing dotted root value",
			text:    `desktop.enabled-reasoning-efforts = ["low", "ultra"]` + "\n",
			efforts: []string{"none", "low", "max", "ultra"},
			want:    []string{"none", "low", "max", "ultra"},
		},
		{
			name:    "section without trailing newline",
			text:    "[desktop]\n" + `theme = "system"`,
			efforts: []string{"none", "medium", "max"},
			want:    []string{"none", "medium", "max"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			updated := codexAppSetReasoningEfforts(tt.text, tt.efforts)
			config, err := codexParseConfig(updated)
			if err != nil {
				t.Fatalf("updated config is invalid: %v\n%s", err, updated)
			}
			got, ok := codexAppConfigReasoningEfforts(config)
			if !ok || !slices.Equal(got, tt.want) {
				t.Fatalf("reasoning efforts = %v, %v; want %v", got, ok, tt.want)
			}
			if count := strings.Count(updated, codexAppReasoningEffortsKey); count != 1 {
				t.Fatalf("reasoning setting count = %d, want 1 in:\n%s", count, updated)
			}
			if strings.Contains(tt.text, `theme = "system"`) {
				if theme, ok := config.String("desktop", "theme"); !ok || theme != "system" {
					t.Fatalf("desktop theme = %q, %v; want preserved", theme, ok)
				}
			}
		})
	}

	if got, want := codexAppMergeReasoningEfforts([]string{"low", "high", "ultra", "persistent"}), []string{"none", "low", "high", "max", "ultra", "persistent"}; !slices.Equal(got, want) {
		t.Fatalf("merged reasoning efforts = %v, want %v", got, want)
	}
	if got, want := codexAppReasoningEffortsForConfig(""), []string{"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}; !slices.Equal(got, want) {
		t.Fatalf("default desktop reasoning efforts = %v, want %v", got, want)
	}
}

func TestCodexAppConfigureUsesAppSpecificProfileWithoutTouchingCLIProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`profile = "default"` + "\n\n" +
		"[profiles.ollama-launch]\n" +
		`model = "cli-model"` + "\n" +
		`openai_base_url = "http://cli.invalid/v1/"` + "\n" +
		`model_provider = "ollama-launch"` + "\n\n" +
		"[model_providers.ollama-launch]\n" +
		`name = "CLI Ollama"` + "\n" +
		`base_url = "http://cli.invalid/v1/"` + "\n" +
		`wire_api = "responses"` + "\n\n" +
		"[profiles.default]\n" +
		`model = "gpt-5.5"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if got, ok := codexRootStringValueOK(content, "profile"); ok {
		t.Fatalf("legacy root profile should be removed, got %q in:\n%s", got, content)
	}
	if got := codexSectionStringValue(content, codexProfileHeader(), "openai_base_url"); got != "http://cli.invalid/v1/" {
		t.Fatalf("CLI profile base URL = %q, want preserved CLI URL in:\n%s", got, content)
	}
	if got := codexSectionStringValue(content, codexProviderHeader(), "name"); got != "CLI Ollama" {
		t.Fatalf("CLI provider name = %q, want preserved CLI provider in:\n%s", got, content)
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) {
		t.Fatalf("legacy app profile section should not be generated, got:\n%s", content)
	}
	if got := codexRootStringValue(content, "model"); got != "llama3.2" {
		t.Fatalf("root model = %q, want llama3.2", got)
	}
	if got := codexRootStringValue(content, codexRootOpenAIBaseURLKey); got != "http://127.0.0.1:9999/api/codex/v1" {
		t.Fatalf("openai_base_url = %q, want loopback Codex router", got)
	}
	if got, ok := codexRootStringValueOK(content, codexRootModelProviderKey); ok {
		t.Fatalf("model_provider = %q, want omitted built-in default", got)
	}
	if strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("custom app provider should be removed, got:\n%s", content)
	}
	assertBackupContains(t, filepath.Join(fileutil.BackupDir(), codexAppIntegrationName, "config.toml.*"), `profile = "default"`)
}

func TestCodexAppConfigureIsIdempotentAndPreservesUnrelatedProvider(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "[model_providers.custom]\n" +
		`name = "Custom"` + "\n" +
		`base_url = "https://example.invalid/v1"` + "\n" +
		`env_key = "CUSTOM_API_KEY"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	app := &CodexApp{}
	models := testLaunchModels("llama3.2", "qwen3:8b")
	if err := app.ConfigureWithModels("llama3.2", models); err != nil {
		t.Fatalf("first ConfigureWithModels returned error: %v", err)
	}
	first, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels("llama3.2", models); err != nil {
		t.Fatalf("second ConfigureWithModels returned error: %v", err)
	}
	second, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(second) != string(first) {
		t.Fatalf("rerun changed generated config:\nfirst:\n%s\nsecond:\n%s", first, second)
	}

	parsed, err := codexParseConfig(string(second))
	if err != nil {
		t.Fatal(err)
	}
	if got := parsed.ProviderString("custom", "env_key"); got != "CUSTOM_API_KEY" {
		t.Fatalf("custom provider env_key = %q, want preserved value", got)
	}
	if parsed.Exists("model_providers", codexAppProfileName, "env_key") {
		t.Fatalf("managed local Ollama provider should not require an API key:\n%s", second)
	}
	if got := app.CurrentModel(); got != "llama3.2" {
		t.Fatalf("CurrentModel = %q, want llama3.2", got)
	}
}

func TestCodexAppConfigurePersistsAutoReviewModel(t *testing.T) {
	for _, test := range []struct {
		name       string
		configured string
		want       string
	}{
		{name: "selected model default", want: "glm-5.3:cloud"},
		{name: "native explicit", configured: "native"},
		{name: "chatgpt explicit", configured: "chatgpt"},
		{name: "selected cloud model", configured: "selected", want: "glm-5.3:cloud"},
		{name: "ollama alias", configured: "ollama", want: "glm-5.3:cloud"},
		{name: "explicit configured model", configured: "qwen3:8b", want: "qwen3:8b"},
		{name: "explicit configured cloud model", configured: "deepseek-v4-flash:cloud", want: "deepseek-v4-flash:cloud"},
	} {
		t.Run(test.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			setTestHome(t, tmpDir)
			t.Setenv(codexAppAutoReviewModelEnv, test.configured)

			if err := (&CodexApp{}).ConfigureWithModels("glm-5.3:cloud", testLaunchModels("glm-5.3:cloud", "deepseek-v4-flash:cloud", "qwen3:8b")); err != nil {
				t.Fatal(err)
			}

			data, err := os.ReadFile(codexAppRoutingCatalogPathForConfig(filepath.Join(tmpDir, ".codex", "config.toml")))
			if err != nil {
				t.Fatal(err)
			}
			var catalog struct {
				AutoReviewModel string `json:"auto_review_model"`
			}
			if err := json.Unmarshal(data, &catalog); err != nil {
				t.Fatal(err)
			}
			if catalog.AutoReviewModel != test.want {
				t.Fatalf("auto_review_model = %q, want %q", catalog.AutoReviewModel, test.want)
			}
		})
	}
}

func TestCodexAppConfigureRejectsUnknownAutoReviewModelBeforeWriting(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv(codexAppAutoReviewModelEnv, "missing-model")

	err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2"))
	if err == nil || !strings.Contains(err.Error(), codexAppAutoReviewModelEnv) {
		t.Fatalf("ConfigureWithModels error = %v, want invalid Auto-review model", err)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, ".codex")); !os.IsNotExist(err) {
		t.Fatalf("invalid Auto-review model wrote configuration: %v", err)
	}
}

func TestCodexCLIConfigRefreshLeavesCodexAppConfigActive(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	appModels := testLaunchModels("llama3.2", "gemma4")
	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", appModels); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	appCatalogPath := mustCodexAppModelCatalogPath(t)
	if err := ensureCodexConfig("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatalf("ensureCodexConfig returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if got, ok := codexRootStringValueOK(content, "profile"); ok {
		t.Fatalf("CLI config refresh should not activate a root profile, got %q in:\n%s", got, content)
	}
	for key, want := range map[string]string{
		"model":              "llama3.2",
		"model_catalog_json": appCatalogPath,
		"openai_base_url":    "http://127.0.0.1:9999/api/codex/v1",
	} {
		if got := codexRootStringValue(content, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, content)
		}
	}
	if got, ok := codexRootStringValueOK(content, codexRootModelProviderKey); ok {
		t.Fatalf("root model_provider = %q, want omitted built-in default in:\n%s", got, content)
	}
	if strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("additive app config should not leave a custom provider section:\n%s", content)
	}
	cliCatalogPath := filepath.Join(tmpDir, ".codex", "model.json")
	if strings.Contains(content, codexProfileHeader()) {
		t.Fatalf("CLI legacy profile section should not be generated, got:\n%s", content)
	}
	if strings.Contains(content, codexProviderHeader()) {
		t.Fatalf("CLI provider should be isolated from app root config, got:\n%s", content)
	}

	cliProfilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
	cliProfileData, err := os.ReadFile(cliProfilePath)
	if err != nil {
		t.Fatalf("CLI profile config not created: %v", err)
	}
	cliProfile := string(cliProfileData)
	for key, want := range map[string]string{
		"model":              "qwen3:8b",
		"model_provider":     codexProfileName,
		"model_catalog_json": cliCatalogPath,
	} {
		if got := codexRootStringValue(cliProfile, key); got != want {
			t.Fatalf("CLI profile %s = %q, want %q in:\n%s", key, got, want, cliProfile)
		}
	}
	if got := codexSectionStringValue(cliProfile, codexProviderHeader(), "base_url"); got != "http://127.0.0.1:9999/v1/" {
		t.Fatalf("CLI profile provider base URL = %q", got)
	}

	appCatalogData, err := os.ReadFile(appCatalogPath)
	if err != nil {
		t.Fatal(err)
	}
	var appCatalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(appCatalogData, &appCatalog); err != nil {
		t.Fatalf("app catalog should be valid JSON: %v", err)
	}
	if got := catalogSlugs(appCatalog.Models); strings.Join(got, ",") != "llama3.2,gemma4,gpt-5.6-sol" {
		t.Fatalf("app catalog slugs = %v, want Ollama models followed by the native catalog", got)
	}
	routingCatalogData, err := os.ReadFile(codexAppRoutingCatalogPathForConfig(configPath))
	if err != nil {
		t.Fatal(err)
	}
	var routingCatalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(routingCatalogData, &routingCatalog); err != nil {
		t.Fatalf("routing catalog should be valid JSON: %v", err)
	}
	if got := catalogSlugs(routingCatalog.Models); strings.Join(got, ",") != "llama3.2,gemma4" {
		t.Fatalf("routing catalog slugs = %v, want only the selected Ollama models", got)
	}

	cliCatalogData, err := os.ReadFile(cliCatalogPath)
	if err != nil {
		t.Fatal(err)
	}
	var cliCatalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(cliCatalogData, &cliCatalog); err != nil {
		t.Fatalf("CLI catalog should be valid JSON: %v", err)
	}
	if got := catalogSlugs(cliCatalog.Models); strings.Join(got, ",") != "qwen3:8b" {
		t.Fatalf("CLI catalog slugs = %v, want qwen3:8b", got)
	}
}

func TestCodexAppConfigureUsesConnectableHostForUnspecifiedBindAddress(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if strings.Contains(content, "0.0.0.0") {
		t.Fatalf("config should not write bind-only host, got:\n%s", content)
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) {
		t.Fatalf("legacy app profile section should not be generated, got:\n%s", content)
	}
	if got := codexRootStringValue(content, codexRootOpenAIBaseURLKey); got != "http://127.0.0.1:11434/api/codex/v1" {
		t.Fatalf("openai_base_url = %q, want connectable loopback router URL", got)
	}
	if got, ok := codexRootStringValueOK(content, codexRootModelProviderKey); ok {
		t.Fatalf("root model_provider = %q, want omitted built-in default", got)
	}
}

func TestCodexAppConfigureRejectsMalformedTomlBeforeSideEffects(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "profile = \n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2"))
	if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
		t.Fatalf("ConfigureWithModels error = %v, want invalid TOML", err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("malformed config should be left untouched, got:\n%s", data)
	}
	if _, err := os.Stat(codexAppRestoreStatePath()); !os.IsNotExist(err) {
		t.Fatalf("restore state should not be written before config validation, err=%v", err)
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("model catalog should not be written before config validation, err=%v", err)
	}
}

func TestCodexAppConfigureRejectsMalformedTomlEvenWithExistingRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "[profiles.ollama-launch\nmodel = \"llama3.2\"\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"default","had_model":true,"model":"gpt-5.5","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2"))
	if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
		t.Fatalf("ConfigureWithModels error = %v, want invalid TOML", err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("malformed config should be left untouched, got:\n%s", data)
	}
	stateData, err := os.ReadFile(codexAppRestoreStatePath())
	if err != nil {
		t.Fatal(err)
	}
	if string(stateData) != restoreState {
		t.Fatalf("restore state should be left untouched, got:\n%s", stateData)
	}
}

func TestCodexAppCurrentModelRequiresManagedActiveProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "" +
		"profile = \"default\"\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		"model = \"llama3.2\"\n" +
		fmt.Sprintf("model_provider = %q\n\n", codexAppProfileName) +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		"base_url = \"http://127.0.0.1:11434/v1/\"\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := (&CodexApp{}).CurrentModel(); got != "" {
		t.Fatalf("CurrentModel = %q, want empty when active profile is not managed", got)
	}
}

func TestCodexAppCurrentModelRecognizesManagedRootProviderForms(t *testing.T) {
	for _, tt := range []struct {
		name         string
		providerLine string
		want         string
	}{
		{name: "omitted default provider", want: "qwen3:8b"},
		{name: "legacy explicit provider", providerLine: `model_provider = "openai"` + "\n", want: "qwen3:8b"},
		{name: "different explicit provider", providerLine: `model_provider = "custom"` + "\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			setTestHome(t, tmpDir)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

			configPath := filepath.Join(tmpDir, ".codex", "config.toml")
			if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
				t.Fatal(err)
			}
			content := "" +
				`model = "qwen3:8b"` + "\n" +
				tt.providerLine +
				fmt.Sprintf(`model_catalog_json = %q`, mustWriteCodexAppTestCatalog(t, "qwen3:8b")) + "\n" +
				`openai_base_url = "http://127.0.0.1:11434/api/codex/v1"` + "\n"
			if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := (&CodexApp{}).CurrentModel(); got != tt.want {
				t.Fatalf("CurrentModel = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCodexAppOllamaConfiguredKeepsOffSwitchWhenCatalogIsMissing(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "" +
		`model = "glm-5.3-flash:cloud"` + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, codexAppModelCatalogPathForConfig(configPath)) + "\n" +
		`openai_base_url = "http://127.0.0.1:11434/api/codex/v1"` + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	app := &CodexApp{}
	if got := app.CurrentModel(); got != "" {
		t.Fatalf("CurrentModel = %q, want empty without a usable catalog", got)
	}
	if !app.OllamaConfigured() {
		t.Fatal("OllamaConfigured = false, want the damaged managed profile to remain removable")
	}
}

func TestCodexAppCurrentModelRequiresHealthyCatalog(t *testing.T) {
	for _, tt := range []struct {
		name           string
		rootCatalog    bool
		profileCatalog bool
		writeCatalog   bool
		catalogData    string
	}{
		{
			name:           "missing catalog reference",
			rootCatalog:    false,
			profileCatalog: true,
			writeCatalog:   true,
			catalogData:    `{"models":[{"slug":"llama3.2"}]}`,
		},
		{
			name:           "deleted catalog file",
			rootCatalog:    true,
			profileCatalog: true,
			writeCatalog:   false,
			catalogData:    `{"models":[{"slug":"llama3.2"}]}`,
		},
		{
			name:           "missing profile catalog reference",
			rootCatalog:    true,
			profileCatalog: false,
			writeCatalog:   true,
			catalogData:    `{"models":[{"slug":"llama3.2"}]}`,
		},
		{
			name:           "corrupt catalog file",
			rootCatalog:    true,
			profileCatalog: true,
			writeCatalog:   true,
			catalogData:    `{"models":`,
		},
		{
			name:           "empty catalog",
			rootCatalog:    true,
			profileCatalog: true,
			writeCatalog:   true,
			catalogData:    `{"models":[]}`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			setTestHome(t, tmpDir)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

			configPath := filepath.Join(tmpDir, ".codex", "config.toml")
			if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
				t.Fatal(err)
			}
			catalogPath := mustCodexAppModelCatalogPath(t)
			if tt.writeCatalog {
				if err := os.WriteFile(catalogPath, []byte(tt.catalogData), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			var rootCatalogLine, profileCatalogLine string
			if tt.rootCatalog {
				rootCatalogLine = fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n"
			}
			if tt.profileCatalog {
				profileCatalogLine = fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n"
			}
			content := "" +
				`model = "llama3.2"` + "\n" +
				fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
				rootCatalogLine + "\n" +
				codexProfileHeaderFor(codexAppProfileName) + "\n" +
				fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
				profileCatalogLine + "\n" +
				codexProviderHeaderFor(codexAppProfileName) + "\n" +
				`base_url = "http://127.0.0.1:11434/v1/"` + "\n"
			if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := (&CodexApp{}).CurrentModel(); got != "" {
				t.Fatalf("CurrentModel = %q, want empty when catalog is unhealthy", got)
			}
		})
	}
}

func TestCodexAppCurrentModelDetectsDriftedModel(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	catalogPath := mustWriteCodexAppTestCatalog(t, "llama3.2")
	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "" +
		`model = "gpt-5.5"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`name = "Ollama"` + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n" +
		`wire_api = "responses"` + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := (&CodexApp{}).CurrentModel(); got != "" {
		t.Fatalf("CurrentModel = %q, want empty when model has drifted from the Ollama catalog", got)
	}
}

func TestCodexAppCurrentModelAcceptsLatestSuffixDrift(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	catalogPath := mustWriteCodexAppTestCatalog(t, "llama3.2")
	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "" +
		`model = "llama3.2:latest"` + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n" +
		`openai_base_url = "http://127.0.0.1:11434/api/codex/v1"` + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := (&CodexApp{}).CurrentModel(); got != "llama3.2:latest" {
		t.Fatalf("CurrentModel = %q, want llama3.2:latest (:latest suffix should not be treated as drift)", got)
	}
}

func TestCodexAppConfigurePopulatesCatalogFromEnrichedModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	models := []LaunchModel{
		{Name: "gemma4", ContextLength: 65536 + len("gemma4"), Capabilities: []model.Capability{model.CapabilityVision, model.CapabilityThinking}, ToolCapable: true},
		{Name: "qwen3:8b"},
		{Name: "llama3.2"},
	}
	if err := (&CodexApp{}).ConfigureWithModels("gemma4", models); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatal(err)
	}
	var catalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatalf("catalog should be valid JSON: %v", err)
	}

	if got := catalogSlugs(catalog.Models); strings.Join(got, ",") != "gemma4,qwen3:8b,llama3.2,gpt-5.6-sol" {
		t.Fatalf("catalog slugs = %v, want enriched Ollama models plus native models", got)
	}
	for _, model := range catalog.Models[:3] {
		slug, _ := model["slug"].(string)
		if model["display_name"] != slug {
			t.Fatalf("display_name should match slug for %q: %v", slug, model["display_name"])
		}
		if model["visibility"] != "list" {
			t.Fatalf("visibility for %q = %v, want list", slug, model["visibility"])
		}
		levels, ok := model["supported_reasoning_levels"].([]any)
		if !ok {
			t.Fatalf("supported_reasoning_levels for %q = %T, want list", slug, model["supported_reasoning_levels"])
		}
		if slug == "gemma4" {
			if model["default_reasoning_level"] != "medium" {
				t.Fatalf("default_reasoning_level for %q = %v, want medium", slug, model["default_reasoning_level"])
			}
			wantEfforts := []string{"none", "medium"}
			gotEfforts := make([]string, 0, len(levels))
			for _, level := range levels {
				entry, ok := level.(map[string]any)
				if !ok {
					t.Fatalf("reasoning level for %q = %T, want object", slug, level)
				}
				effort, _ := entry["effort"].(string)
				gotEfforts = append(gotEfforts, effort)
				if description, _ := entry["description"].(string); description == "" {
					t.Fatalf("reasoning level %q for %q has no description", effort, slug)
				}
			}
			if strings.Join(gotEfforts, ",") != strings.Join(wantEfforts, ",") {
				t.Fatalf("reasoning levels for %q = %v, want %v", slug, gotEfforts, wantEfforts)
			}
		} else {
			if model["default_reasoning_level"] != nil {
				t.Fatalf("default_reasoning_level for %q = %v, want nil", slug, model["default_reasoning_level"])
			}
			if len(levels) != 0 {
				t.Fatalf("supported_reasoning_levels for %q = %v, want none", slug, levels)
			}
		}
		wantContext := float64(128000)
		wantModalities := []string{"text"}
		if slug == "gemma4" {
			wantContext = float64(65536 + len(slug))
			wantModalities = []string{"text", "image"}
		}
		if model["context_window"] != wantContext {
			t.Fatalf("context_window for %q = %v, want %v", slug, model["context_window"], wantContext)
		}
		if got := catalogInputModalities(model); strings.Join(got, ",") != strings.Join(wantModalities, ",") {
			t.Fatalf("input_modalities for %q = %v, want %v", slug, got, wantModalities)
		}
		wantTools := slug == "gemma4"
		if got := model["supports_search_tool"]; got != wantTools {
			t.Fatalf("supports_search_tool for %q = %v, want %v", slug, got, wantTools)
		}
		if got := model["supports_parallel_tool_calls"]; got != wantTools {
			t.Fatalf("supports_parallel_tool_calls for %q = %v, want %v", slug, got, wantTools)
		}
		if got := model["web_search_tool_type"]; got != "text" {
			t.Fatalf("web_search_tool_type for %q = %v, want text", slug, got)
		}
	}

	configPath, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	routingData, err := os.ReadFile(codexAppRoutingCatalogPathForConfig(configPath))
	if err != nil {
		t.Fatal(err)
	}
	var routingCatalog struct {
		Models []struct {
			Slug     string `json:"slug"`
			Thinking struct {
				Supported bool     `json:"supported"`
				Levels    []string `json:"levels"`
			} `json:"thinking"`
		} `json:"models"`
	}
	if err := json.Unmarshal(routingData, &routingCatalog); err != nil {
		t.Fatalf("routing catalog should be valid JSON: %v", err)
	}
	if len(routingCatalog.Models) != 3 {
		t.Fatalf("routing catalog models = %#v, want 3 selected Ollama models", routingCatalog.Models)
	}
	gemmaThinking := routingCatalog.Models[0].Thinking
	if routingCatalog.Models[0].Slug != "gemma4" || !gemmaThinking.Supported || !slices.Equal(gemmaThinking.Levels, []string{"none", "medium"}) {
		t.Fatalf("gemma4 routing thinking = %+v, want binary off/medium metadata", gemmaThinking)
	}
	for _, routed := range routingCatalog.Models[1:] {
		if routed.Thinking.Supported || len(routed.Thinking.Levels) != 0 {
			t.Fatalf("non-thinking model %q routing metadata = %+v, want unsupported", routed.Slug, routed.Thinking)
		}
	}
}

func TestCodexAppThinkingLevelsUseVerifiedContracts(t *testing.T) {
	tests := []struct {
		name        string
		modelName   string
		family      string
		thinking    bool
		wantInitial string
		wantLevels  []string
	}{
		{name: "non-thinking model"},
		{name: "binary fallback", thinking: true, wantInitial: "medium", wantLevels: []string{"none", "medium"}},
		{name: "recommended GLM 5.3 Flash", modelName: "glm-5.3-flash:cloud", wantInitial: "max", wantLevels: []string{"low", "high", "max"}},
		{name: "recommended GLM 5.3", modelName: "glm-5.3:cloud", wantInitial: "max", wantLevels: []string{"low", "high", "max"}},
		{name: "recommended DeepSeek V4 Flash", modelName: "deepseek-v4-flash:cloud", wantInitial: "high", wantLevels: []string{"none", "high", "max"}},
		{name: "recommended Gemma 4 cloud", modelName: "gemma4:31b-cloud", wantInitial: "medium", wantLevels: []string{"none", "medium"}},
		{name: "recommended Gemma 4 local", modelName: "gemma4:26b", wantInitial: "medium", wantLevels: []string{"none", "medium"}},
		{name: "similar unverified tag uses fallback", modelName: "glm-5.3-flash:custom", thinking: true, wantInitial: "medium", wantLevels: []string{"none", "medium"}},
		{name: "GLM 5.3 family fallback", family: "glm5_next", thinking: true, wantInitial: "max", wantLevels: []string{"low", "high", "max"}},
		{name: "GPT-OSS family", family: "gpt-oss", thinking: true, wantInitial: "medium", wantLevels: []string{"low", "medium", "high"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			launchModel := LaunchModel{Name: tt.modelName}
			launchModel.Details.Family = tt.family
			if tt.thinking {
				launchModel.Capabilities = []model.Capability{model.CapabilityThinking}
			}
			metadata := codexAppModelMetadataFromLaunchModel(launchModel)
			gotInitial, gotLevels := metadata.defaultThinkingLevel, metadata.thinkingLevels
			if gotInitial != tt.wantInitial || !slices.Equal(gotLevels, tt.wantLevels) {
				t.Fatalf("thinking levels = %q, %v; want %q, %v", gotInitial, gotLevels, tt.wantInitial, tt.wantLevels)
			}
			if metadata.supportsThinking != (len(tt.wantLevels) > 0) {
				t.Fatalf("supports thinking = %v, want %v", metadata.supportsThinking, len(tt.wantLevels) > 0)
			}
		})
	}
}

func TestDefaultCodexAppRunDebugModelsUsesScratchWorkingDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}
	executable := filepath.Join(t.TempDir(), "codex-fixture")
	if err := os.WriteFile(executable, []byte("#!/bin/sh\npwd\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	scratch := t.TempDir()
	output, err := defaultCodexAppRunDebugModels(executable, scratch, false)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.TrimSpace(string(output))
	gotInfo, gotErr := os.Stat(got)
	scratchInfo, scratchErr := os.Stat(scratch)
	if gotErr != nil || scratchErr != nil || !os.SameFile(gotInfo, scratchInfo) {
		t.Fatalf("working directory = %q, want scratch CODEX_HOME %q", got, scratch)
	}
}

func TestCodexAppConfigureCatalogIncludesExactSelectedModel(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	models := []LaunchModel{
		{Name: "llama3.2:latest", ContextLength: 65_536},
		{Name: "qwen3:8b"},
	}
	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", models); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	configPath, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := codexRootStringValue(string(configData), codexRootModelKey); got != "llama3.2" {
		t.Fatalf("root model = %q, want llama3.2", got)
	}

	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatal(err)
	}
	var catalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatalf("catalog should be valid JSON: %v", err)
	}
	if got := catalogSlugs(catalog.Models); strings.Join(got, ",") != "llama3.2,qwen3:8b,gpt-5.6-sol" {
		t.Fatalf("catalog slugs = %v, want exact selected model plus native models without :latest duplicate", got)
	}
	if got := catalog.Models[0]["context_window"]; got != float64(65_536) {
		t.Fatalf("selected model context_window = %v, want 65536", got)
	}
}

func TestCodexAppConfigureUpgradesLegacyRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "odc-resp-dev"` + "\n\n" +
		"[model_providers.odc-resp-dev]\n" +
		`base_url = "https://example.invalid/v1/"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(`{"had_profile":false}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	state, err := loadCodexAppRestoreState()
	if err != nil {
		t.Fatal(err)
	}
	if state.HadProfile {
		t.Fatalf("HadProfile = true, want legacy false")
	}
	if !state.HadModel || state.Model != "gpt-5.5" {
		t.Fatalf("model restore state = (%v, %q), want previous root model", state.HadModel, state.Model)
	}
	if !state.HadModelProvider || state.ModelProvider != "odc-resp-dev" {
		t.Fatalf("model provider restore state = (%v, %q), want previous root provider", state.HadModelProvider, state.ModelProvider)
	}
}

func TestCodexAppConfigureMigratesLegacyManagedConfigWithoutPollutingRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	catalogPath := mustCodexAppModelCatalogPath(t)
	existing := "" +
		fmt.Sprintf(`profile = %q`, codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`name = "Ollama"` + "\n" +
		`base_url = "http://127.0.0.1:9999/v1/"` + "\n" +
		`wire_api = "responses"` + "\n\n" +
		"[profiles.default]\n" +
		`model = "gpt-5.5"` + "\n\n" +
		"[desktop]\n" +
		`enabled-reasoning-efforts = ["low", "high"]` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(`{"had_profile":true,"profile":"default"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CodexApp{}
	if err := c.ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	state, err := loadCodexAppRestoreState()
	if err != nil {
		t.Fatal(err)
	}
	if !state.HadProfile || state.Profile != "default" {
		t.Fatalf("profile restore state = (%v, %q), want default", state.HadProfile, state.Profile)
	}
	if state.HadModel || state.HadModelProvider || state.HadModelCatalogJSON {
		t.Fatalf("legacy restore state should not capture managed root values: %+v", state)
	}
	if !state.HadDesktopReasoningEfforts || !slices.Equal(state.DesktopReasoningEfforts, []string{"low", "high"}) {
		t.Fatalf("desktop reasoning restore state = (%v, %v), want original user choices", state.HadDesktopReasoningEfforts, state.DesktopReasoningEfforts)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	migrated := string(data)
	if got, ok := codexRootStringValueOK(migrated, "profile"); ok {
		t.Fatalf("legacy root profile should be removed during migration, got %q in:\n%s", got, migrated)
	}
	if strings.Contains(migrated, codexProfileHeaderFor(codexAppProfileName)) {
		t.Fatalf("legacy app profile section should be removed during migration, got:\n%s", migrated)
	}

	if err := c.Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err = os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	restored := string(data)
	if got := codexRootStringValue(restored, "profile"); got != "default" {
		t.Fatalf("root profile = %q, want default in:\n%s", got, restored)
	}
	for _, key := range []string{"model", "model_provider", "model_catalog_json"} {
		if got, ok := codexRootStringValueOK(restored, key); ok {
			t.Fatalf("root %s should be removed on restore, got %q in:\n%s", key, got, restored)
		}
	}
	if strings.Contains(restored, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(restored, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("owned app config should be removed on restore, got:\n%s", restored)
	}
	restoredConfig, err := codexParseConfig(restored)
	if err != nil {
		t.Fatal(err)
	}
	if efforts, ok := codexAppConfigReasoningEfforts(restoredConfig); !ok || !slices.Equal(efforts, []string{"low", "high"}) {
		t.Fatalf("restored desktop reasoning efforts = %v, %v; want [low high]", efforts, ok)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreRestoresPreviousProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"profile = \"default\"\n" +
		"model = \"gpt-5.5\"\n" +
		"model_provider = \"openai\"\n" +
		"model_catalog_json = \"/tmp/original-catalog.json\"\n" +
		"openai_base_url = \"https://api.openai.com/v1\"\n\n" +
		"[profiles.default]\n" +
		"model = \"gpt-5.5\"\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CodexApp{}
	if err := c.ConfigureWithModels("llama3.2", testLaunchModels("llama3.2")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}
	if err := c.Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `profile = "default"`) || strings.Contains(string(data), fmt.Sprintf(`profile = %q`, codexAppProfileName)) {
		t.Fatalf("restore should restore previous active profile, got:\n%s", data)
	}
	restored := string(data)
	if strings.Contains(restored, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(restored, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("restore should remove owned app sections, got:\n%s", restored)
	}
	for key, want := range map[string]string{
		"profile":            "default",
		"model":              "gpt-5.5",
		"model_provider":     "openai",
		"model_catalog_json": "/tmp/original-catalog.json",
		"openai_base_url":    "https://api.openai.com/v1",
	} {
		if got := codexRootStringValue(restored, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, restored)
		}
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
	if _, err := os.Stat(codexAppRestoreStatePath()); !os.IsNotExist(err) {
		t.Fatalf("restore state should be removed, got err=%v", err)
	}
}

func TestCodexAppRestoreRestoresDesktopReasoningEffortsExactly(t *testing.T) {
	tests := []struct {
		name         string
		desktopValue string
		wantOriginal []string
	}{
		{
			name:         "existing user choices",
			desktopValue: `enabled-reasoning-efforts = ["minimal", "high", "persistent"]` + "\n",
			wantOriginal: []string{"minimal", "high", "persistent"},
		},
		{name: "setting originally absent"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			setTestHome(t, tmpDir)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
			withCodexAppPlatform(t, "darwin")
			withCodexAppProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { return nil })

			configPath := filepath.Join(tmpDir, ".codex", "config.toml")
			if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
				t.Fatal(err)
			}
			existing := `model = "gpt-5.6-sol"` + "\n\n" +
				"[desktop]\n" +
				tt.desktopValue +
				`theme = "system"` + "\n"
			if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
				t.Fatal(err)
			}

			app := &CodexApp{}
			if err := app.ConfigureWithModels("gemma4", []LaunchModel{{
				Name:         "gemma4",
				Capabilities: []model.Capability{model.CapabilityThinking},
			}}); err != nil {
				t.Fatalf("ConfigureWithModels returned error: %v", err)
			}

			managedData, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			managedConfig, err := codexParseConfig(string(managedData))
			if err != nil {
				t.Fatal(err)
			}
			managedEfforts, ok := codexAppConfigReasoningEfforts(managedConfig)
			if !ok || !slices.Contains(managedEfforts, "none") || !slices.Contains(managedEfforts, "max") {
				t.Fatalf("managed reasoning efforts = %v, %v; want none and max enabled", managedEfforts, ok)
			}

			if err := app.Restore(); err != nil {
				t.Fatalf("Restore returned error: %v", err)
			}
			restoredData, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			restoredConfig, err := codexParseConfig(string(restoredData))
			if err != nil {
				t.Fatal(err)
			}
			gotOriginal, hadOriginal := codexAppConfigReasoningEfforts(restoredConfig)
			if tt.wantOriginal == nil {
				if hadOriginal {
					t.Fatalf("restored reasoning efforts = %v, want setting removed", gotOriginal)
				}
			} else if !hadOriginal || !slices.Equal(gotOriginal, tt.wantOriginal) {
				t.Fatalf("restored reasoning efforts = %v, %v; want %v", gotOriginal, hadOriginal, tt.wantOriginal)
			}
			if theme, ok := restoredConfig.String("desktop", "theme"); !ok || theme != "system" {
				t.Fatalf("desktop theme = %q, %v; want preserved", theme, ok)
			}
		})
	}
}

func TestCodexAppRestorePreservesNativeModelSelectedWhileConnected(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	catalogPath := codexAppModelCatalogPathForConfig(configPath)
	routingPath := codexAppRoutingCatalogPathForConfig(configPath)
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	state := codexAppRestoreState{
		HadModel:            true,
		Model:               "gpt-5.5",
		HadModelProvider:    true,
		ModelProvider:       "openai",
		HadOpenAIBaseURL:    true,
		OpenAIBaseURL:       "https://api.openai.com/v1",
		HadModelCatalogJSON: false,
	}
	for _, tt := range []struct {
		name      string
		current   string
		routing   string
		wantModel string
	}{
		{name: "native selection", current: "gpt-5.6-sol", routing: `{"models":[{"slug":"glm-5.3-flash:cloud"}]}`, wantModel: "gpt-5.6-sol"},
		{name: "Ollama selection", current: "glm-5.3-flash:cloud", routing: `{"models":[{"slug":"glm-5.3-flash:cloud"}]}`, wantModel: "gpt-5.5"},
		{name: "damaged routing catalog", current: "glm-5.3-flash:cloud", routing: `{`, wantModel: "gpt-5.5"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(routingPath, []byte(tt.routing), 0o600); err != nil {
				t.Fatal(err)
			}
			managed := fmt.Sprintf("model = %q\nmodel_catalog_json = %q\nopenai_base_url = %q\n", tt.current, catalogPath, codexAppProxyBaseURL())
			restored := codexAppRestoreRootValues(managed, state)
			if got := codexRootStringValue(restored, codexRootModelKey); got != tt.wantModel {
				t.Fatalf("restored model = %q, want %q in:\n%s", got, tt.wantModel, restored)
			}
		})
	}
}

func TestCodexAppRestoreMissingConfigRemovesRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"stale","had_model":true,"model":"old","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	if _, err := os.Stat(codexAppRestoreStatePath()); !os.IsNotExist(err) {
		t.Fatalf("restore state should be removed when config is missing, got err=%v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppConfigureMissingConfigReplacesStaleRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"stale","had_model":true,"model":"old","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2")); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	state, err := loadCodexAppRestoreState()
	if err != nil {
		t.Fatal(err)
	}
	if state.HadProfile || state.HadModel || state.HadModelProvider || state.HadModelCatalogJSON {
		t.Fatalf("restore state = %+v, want empty snapshot when config was missing", state)
	}
}

func TestCodexAppConfigureRefreshesRestoreStateAfterManualProfileSwitch(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	initial := "" +
		`profile = "default"` + "\n" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		"[profiles.default]\n" +
		`model = "gpt-5.5"` + "\n"
	if err := os.WriteFile(configPath, []byte(initial), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", testLaunchModels("llama3.2")); err != nil {
		t.Fatalf("first ConfigureWithModels returned error: %v", err)
	}

	manual := "" +
		`profile = "manual"` + "\n" +
		`model = "manual-model"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		"[profiles.manual]\n" +
		`model = "manual-model"` + "\n"
	if err := os.WriteFile(configPath, []byte(manual), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatalf("second ConfigureWithModels returned error: %v", err)
	}
	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	restored := string(data)
	for key, want := range map[string]string{
		"profile":        "manual",
		"model":          "manual-model",
		"model_provider": "openai",
	} {
		if got := codexRootStringValue(restored, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, restored)
		}
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreRejectsMalformedTomlWithoutWriting(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "model = \"unterminated\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(`{"had_profile":false,"had_model":false,"had_model_provider":false,"had_model_catalog_json":false}`), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&CodexApp{}).Restore()
	if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
		t.Fatalf("Restore error = %v, want invalid TOML", err)
	}
	catalogPath, pathErr := codexAppModelCatalogPath()
	if pathErr != nil {
		t.Fatal(pathErr)
	}
	for _, want := range []string{
		"Restore did not complete",
		"Codex config: " + configPath,
		"Restore state: " + codexAppRestoreStatePath(),
		"Model catalog: " + catalogPath,
		"Backups: " + filepath.Join(fileutil.BackupDir(), codexAppIntegrationName),
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Restore error missing %q:\n%v", want, err)
		}
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("malformed config should be left untouched, got:\n%s", data)
	}
	if _, err := os.Stat(codexAppRestoreStatePath()); err != nil {
		t.Fatalf("restore state should remain after failed restore: %v", err)
	}
}

func TestCodexAppRestoreWithoutStateRemovesManagedRootModel(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	existing := "" +
		fmt.Sprintf(`profile = %q`, codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, []byte(`{"models":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	for _, key := range []string{"profile", "model", "model_provider", "model_catalog_json"} {
		if got, ok := codexRootStringValueOK(content, key); ok {
			t.Fatalf("root %s should be removed, got %q in:\n%s", key, got, content)
		}
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("owned app sections should be removed, got:\n%s", content)
	}
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("owned catalog should be removed when unused, err=%v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreDoesNotStompUserChangedRootConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`profile = "manual"` + "\n" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n\n" +
		"[profiles.manual]\n" +
		`model = "gpt-5.5"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, []byte(`{"models":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"default","had_model":true,"model":"old","had_model_provider":true,"model_provider":"old-provider","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	for key, want := range map[string]string{
		"profile":        "manual",
		"model":          "gpt-5.5",
		"model_provider": "openai",
	} {
		if got := codexRootStringValue(content, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, content)
		}
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("owned app sections should be removed when no longer active, got:\n%s", content)
	}
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("owned catalog should be removed when unused, err=%v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreDoesNotTreatCLIProfileAsOwned(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error { return nil },
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`profile = "ollama-launch"` + "\n" +
		`model = "cli-model"` + "\n" +
		`model_provider = "ollama-launch"` + "\n\n" +
		"[profiles.ollama-launch]\n" +
		`model = "cli-model"` + "\n" +
		`openai_base_url = "http://cli.invalid/v1/"` + "\n" +
		`model_provider = "ollama-launch"` + "\n\n" +
		"[model_providers.ollama-launch]\n" +
		`name = "CLI Ollama"` + "\n" +
		`base_url = "http://cli.invalid/v1/"` + "\n" +
		`wire_api = "responses"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"default","had_model":true,"model":"gpt-5.5","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("CLI Codex profile should be left untouched, got:\n%s", data)
	}
}

func TestCodexAppRunRestartsRunningAppWhenConfirmed(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	var quitCalls, openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return nil
		},
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&CodexApp{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("quit/open calls = %d/%d, want 1/1", quitCalls, openCalls)
	}
}

func TestCodexAppRunWaitsForGracefulExitBeforeReopening(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	oldSleep := codexAppSleep
	t.Cleanup(func() {
		codexAppSleep = oldSleep
	})

	running := true
	var quitCalls, openCalls, sleepCalls int
	codexAppSleep = func(time.Duration) {
		sleepCalls++
		if sleepCalls == 2 {
			running = false
		}
	}
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			return nil
		},
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&CodexApp{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("quit/open calls = %d/%d, want 1/1", quitCalls, openCalls)
	}
	if sleepCalls == 0 {
		t.Fatal("expected restart to wait for Codex to exit before reopening")
	}
}

func TestCodexAppRunCtrlCAbortsEntireRestartFlow(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	oldSleep := codexAppSleep
	oldDefaultSpinner := DefaultSpinner
	t.Cleanup(func() {
		codexAppSleep = oldSleep
		DefaultSpinner = oldDefaultSpinner
	})

	// Simulate the user pressing Ctrl+C during the graceful-exit wait: the
	// shared spinner's cancellation channel is closed on the first poll,
	// which only happens inside the wait loop.
	cancel := make(chan struct{})
	var spinnerStopped bool
	codexAppSleep = func(time.Duration) {
		select {
		case <-cancel:
		default:
			close(cancel)
		}
	}
	DefaultSpinner = func(string) *Spinner {
		return NewSpinner(func() { spinnerStopped = true }, cancel)
	}

	var calls []string
	withCodexAppProcessHooks(t,
		func() bool { return true }, // app stays "running" so the wait polls
		func() error { calls = append(calls, "quit"); return nil },
		func() error { calls = append(calls, "open"); return nil },
	)
	codexAppExitTimeout = 5 * time.Second
	codexAppForceQuit = func() error {
		calls = append(calls, "force")
		return nil
	}

	err := (&CodexApp{}).Run("qwen3.5", nil, nil)
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("Run error = %v, want ErrCancelled", err)
	}
	if !spinnerStopped {
		t.Fatal("expected the shared spinner to be stopped on cancel")
	}
	// The flow must abort after quit: no force-quit, no reopen, despite the app
	// still being "running" (which would otherwise trigger the force-quit path).
	want := []string{"quit"}
	if !slices.Equal(calls, want) {
		t.Fatalf("calls = %v, want the whole flow to abort after quit: %v", calls, want)
	}
}

func TestCodexAppRunForceStopsMacAfterGracefulTimeout(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	calls := make([]string, 0)
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			calls = append(calls, "quit")
			return nil
		},
		func() error {
			calls = append(calls, "open")
			return nil
		},
	)
	codexAppExitTimeout = 0
	codexAppForceQuit = func() error {
		calls = append(calls, "force")
		running = false
		return nil
	}

	if err := (&CodexApp{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	want := []string{"quit", "force", "open"}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
}

func TestCodexAppRunReturnsMacForceStopError(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	withCodexAppProcessHooks(t,
		func() bool { return true },
		func() error { return nil },
		func() error {
			t.Fatal("app should not reopen when force stop fails")
			return nil
		},
	)
	codexAppExitTimeout = 0
	codexAppForceQuit = func() error {
		return fmt.Errorf("operation not permitted")
	}

	err := (&CodexApp{}).Run("qwen3.5", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "force stop ChatGPT") || !strings.Contains(err.Error(), "operation not permitted") {
		t.Fatalf("Run error = %v, want force stop failure", err)
	}
}

func TestCodexAppRunOpensOnWindowsWhenNotRunning(t *testing.T) {
	withCodexAppPlatform(t, "windows")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&CodexApp{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRunRestartsWindowsStartAppID(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	var quitCalls, openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return nil
		},
		func() error {
			openCalls++
			return nil
		},
	)

	codexAppStartID = func() string { return "OpenAI.Codex_2p2nqsd0c76g0!App" }
	codexAppRunPath = func() string {
		return `C:\Program Files\WindowsApps\OpenAI.Codex_26.429.8261.0_x64__2p2nqsd0c76g0\app\Codex.exe`
	}
	var openedStartID, openedPath string
	codexAppOpenStart = func(appID string) error {
		openedStartID = appID
		return nil
	}
	codexAppOpenPath = func(path string) error {
		openedPath = path
		return nil
	}

	if err := (&CodexApp{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 {
		t.Fatalf("quit calls = %d, want 1", quitCalls)
	}
	if openedStartID != "OpenAI.Codex_2p2nqsd0c76g0!App" {
		t.Fatalf("opened Start AppID = %q", openedStartID)
	}
	if openedPath != "" {
		t.Fatalf("opened path = %q, want Start AppID path only", openedPath)
	}
}

func TestCodexAppRunForceStopsWindowsBackgroundProcessesBeforeReopening(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	windowOpen := true
	running := true
	calls := make([]string, 0)
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			calls = append(calls, "quit")
			windowOpen = false
			return nil
		},
		func() error {
			t.Fatal("open app fallback should not be used")
			return nil
		},
	)
	codexAppHasWindow = func() bool { return windowOpen }
	codexAppForceQuit = func() error {
		calls = append(calls, "force")
		running = false
		return nil
	}
	codexAppStartID = func() string { return "OpenAI.Codex_2p2nqsd0c76g0!App" }
	codexAppOpenStart = func(appID string) error {
		calls = append(calls, "open:"+appID)
		return nil
	}

	if err := (&CodexApp{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	want := []string{"quit", "force", "open:OpenAI.Codex_2p2nqsd0c76g0!App"}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
}

func TestCodexAppRunReturnsWindowsForceStopError(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	windowOpen := true
	withCodexAppProcessHooks(t,
		func() bool { return true },
		func() error {
			windowOpen = false
			return nil
		},
		func() error {
			t.Fatal("open app fallback should not be used")
			return nil
		},
	)
	codexAppHasWindow = func() bool { return windowOpen }
	codexAppForceQuit = func() error {
		return fmt.Errorf("access denied")
	}
	codexAppOpenStart = func(string) error {
		t.Fatal("app should not reopen when force stop fails")
		return nil
	}

	err := (&CodexApp{}).Run("qwen3.5", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "force stop ChatGPT") || !strings.Contains(err.Error(), "access denied") {
		t.Fatalf("Run error = %v, want force stop failure", err)
	}
}

func TestCodexAppRunRejectsExtraArgs(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	err := (&CodexApp{}).Run("qwen3.5", nil, []string{"--foo"})
	if err == nil || !strings.Contains(err.Error(), "does not accept extra arguments") {
		t.Fatalf("Run error = %v, want extra args rejection", err)
	}
}

func TestCodexAppProcessMatchesMainAndAppServer(t *testing.T) {
	for _, command := range []string{
		"/Applications/ChatGPT.app/Contents/MacOS/ChatGPT",
		"/Applications/ChatGPT.app/Contents/Resources/codex app-server --analytics-default-enabled",
		"/Applications/Codex.app/Contents/MacOS/Codex",
		"/Applications/Codex.app/Contents/Resources/codex app-server --analytics-default-enabled",
		`C:\Users\parth\AppData\Local\Programs\Codex\Codex.exe`,
		`"C:\Users\parth\AppData\Local\Codex\app-26.429.30905\resources\codex.exe" app-server --analytics-default-enabled`,
		`"C:\Users\parth\AppData\Local\openai-codex-electron\resources\codex.exe" "app-server"`,
	} {
		if !codexAppProcessMatches(command) {
			t.Fatalf("expected command to match Codex App process: %s", command)
		}
	}

	for _, command := range []string{
		"/Applications/ChatGPT.app/Contents/Frameworks/ChatGPT Helper.app/Contents/MacOS/ChatGPT Helper",
		"/Applications/Codex.app/Contents/Frameworks/Codex Helper.app/Contents/MacOS/Codex Helper",
		"/Applications/Codex.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler",
		`"C:\Program Files\WindowsApps\OpenAI.Codex_26.429.8261.0_x64__2p2nqsd0c76g0\app\Codex.exe" --type=renderer --user-data-dir="C:\Users\parth\AppData\Roaming\Codex"`,
		`"C:\Program Files\WindowsApps\OpenAI.Codex_26.429.8261.0_x64__2p2nqsd0c76g0\app\Codex.exe" --type=crashpad-handler`,
	} {
		if codexAppProcessMatches(command) {
			t.Fatalf("expected helper command not to match Codex App process: %s", command)
		}
	}
}

func TestCodexAppCandidatesIncludeChatGPT(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	candidates := codexAppDarwinAppCandidates()
	if len(candidates) == 0 || candidates[0] != "/Applications/ChatGPT.app" {
		t.Fatalf("darwin candidates = %v, want ChatGPT first", candidates)
	}
	if !slices.Contains(candidates, "/Applications/Codex.app") {
		t.Fatalf("darwin candidates = %v, want legacy Codex app", candidates)
	}

	withCodexAppPlatform(t, "windows")
	local := filepath.Join(t.TempDir(), "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	if candidates := codexAppWindowsAppCandidates(); !slices.Contains(candidates, filepath.Join(local, "Programs", "ChatGPT", "ChatGPT.exe")) {
		t.Fatalf("windows candidates = %v, want ChatGPT app", candidates)
	}
}

func catalogSlugs(models []map[string]any) []string {
	slugs := make([]string, 0, len(models))
	for _, model := range models {
		if slug, _ := model["slug"].(string); slug != "" {
			slugs = append(slugs, slug)
		}
	}
	return slugs
}

func catalogInputModalities(entry map[string]any) []string {
	raw, _ := entry["input_modalities"].([]any)
	modalities := make([]string, 0, len(raw))
	for _, item := range raw {
		if modality, _ := item.(string); modality != "" {
			modalities = append(modalities, modality)
		}
	}
	return modalities
}

func mustCodexAppModelCatalogPath(t *testing.T) string {
	t.Helper()
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	return catalogPath
}

func mustWriteCodexAppTestCatalog(t *testing.T, slugs ...string) string {
	t.Helper()
	catalogPath := mustCodexAppModelCatalogPath(t)
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
		t.Fatal(err)
	}
	models := make([]map[string]string, 0, len(slugs))
	for _, slug := range slugs {
		models = append(models, map[string]string{"slug": slug})
	}
	data, err := json.Marshal(map[string]any{"models": models})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	configPath, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRoutingCatalogPathForConfig(configPath), data, 0o644); err != nil {
		t.Fatal(err)
	}
	return catalogPath
}
