package launch

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"
)

func codexAppCLIRestoreFixture(t *testing.T) (*CodexApp, string, map[string]string) {
	t.Helper()
	app, configPath := codexAppRestoreFixture(t)
	original := map[string]string{
		"config.toml": "model = \"gpt-5.5\"\nmodel_provider = \"openai\"\napproval_policy = \"on-request\"\n" +
			"[model_providers.custom]\nname = \"Custom\"\nbase_url = \"https://custom.invalid/v1\"\n" +
			"[mcp_servers.fixture]\ncommand = \"fixture-mcp\"\n[desktop]\ntheme = \"dark\"\n",
		"auth.json":              `{"tokens":{"access_token":"fixture-user-auth"}}`,
		"history.jsonl":          "fixture history\n",
		"sessions/fixture.jsonl": "fixture conversation\n",
	}
	for name, content := range original {
		path := filepath.Join(filepath.Dir(configPath), name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := app.ConfigureWithModels("qwen3:8b", testLaunchModels("qwen3:8b")); err != nil {
		t.Fatal(err)
	}
	oldSpinner := DefaultSpinner
	DefaultSpinner = func(string) *Spinner { return NewSpinner(func() {}, nil) }
	t.Cleanup(func() { DefaultSpinner = oldSpinner })
	return app, configPath, original
}

func codexAppRestoreFixtureFiles(t *testing.T, configPath string) map[string]string {
	t.Helper()
	home := filepath.Dir(filepath.Dir(configPath))
	files := make(map[string]string)
	if err := filepath.WalkDir(home, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(path)
			files[strings.TrimPrefix(path, home)] = "symlink:" + target
			return err
		}
		data, err := os.ReadFile(path)
		files[strings.TrimPrefix(path, home)] = string(data)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return files
}

func TestLaunchCmdChatGPTRestoreConfirmsBeforeChanges(t *testing.T) {
	promptErr := errors.New("fixture prompt failure")
	for _, mode := range []string{"decline", "cancel", "prompt error", "headless", "legacy decline"} {
		t.Run(mode, func(t *testing.T) {
			_, configPath, _ := codexAppCLIRestoreFixture(t)
			before := codexAppRestoreFixtureFiles(t, configPath)
			withInteractiveSession(t, mode != "headless")
			withCodexAppProcessHooks(t,
				func() bool { return mode != "legacy decline" },
				func() error { t.Error("quit before consent"); return nil },
				func() error { t.Error("open after declining restore"); return nil },
			)
			codexAppCanOpenID = func(context.Context) bool { return true }
			oldProfileRunning, oldStopProfile := codexAppProfileIsRunning, codexAppStopProfile
			codexAppProfileIsRunning = func(context.Context) (bool, error) { return mode == "legacy decline", nil }
			codexAppStopProfile = func(context.Context) error { t.Error("legacy quit before consent"); return nil }
			t.Cleanup(func() {
				codexAppProfileIsRunning, codexAppStopProfile = oldProfileRunning, oldStopProfile
			})

			oldConfirm := DefaultConfirmPrompt
			promptCalls := 0
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
				promptCalls++
				if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
					t.Errorf("files changed before confirmation (-before +after):\n%s", diff)
				}
				switch mode {
				case "cancel":
					return false, ErrCancelled
				case "prompt error":
					return false, promptErr
				default:
					return false, nil
				}
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
			cmd := LaunchCmd(func(*cobra.Command, []string) error {
				t.Fatal("restore must not require the server")
				return nil
			}, nil)
			cmd.SilenceErrors, cmd.SilenceUsage = true, true
			cmd.SetArgs([]string{"chatgpt", "--restore"})
			var err error
			stderr := captureStderr(t, func() { err = cmd.Execute() })
			switch mode {
			case "headless":
				if err == nil || !strings.Contains(err.Error(), "--yes") {
					t.Fatalf("headless restore error = %v, want confirmation guidance", err)
				}
			case "prompt error":
				if !errors.Is(err, promptErr) {
					t.Fatalf("restore error = %v, want prompt error", err)
				}
			default:
				if err != nil {
					t.Fatal(err)
				}
			}
			wantPrompts := 1
			if mode == "headless" {
				wantPrompts = 0
			}
			if promptCalls != wantPrompts || strings.Contains(stderr, codexAppRestoreSuccess) {
				t.Fatalf("prompt calls = %d, want %d; restore output = %q", promptCalls, wantPrompts, stderr)
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("cancelled restore changed files (-before +after):\n%s", diff)
			}
		})
	}
}

func TestCodexAppCLIRestorePreservesSettingsAndHistory(t *testing.T) {
	for _, mode := range []string{"closed", "darwin running", "windows running", "headless --yes"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, original := codexAppCLIRestoreFixture(t)
			before := codexAppRestoreFixtureFiles(t, configPath)
			withInteractiveSession(t, mode != "headless --yes")
			running := mode != "closed"
			quitRequested := false
			var calls []string
			reopen := func() error {
				if app.OllamaConfigured() || app.RestorePending() {
					t.Error("reopened before restoring the profile")
				}
				calls = append(calls, "open")
				return nil
			}
			withCodexAppProcessHooks(t,
				func() bool { return running },
				func() error {
					if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
						t.Fatalf("files changed before quit (-before +after):\n%s", diff)
					}
					calls = append(calls, "quit")
					quitRequested = true
					return nil
				},
				reopen,
			)
			codexAppCanOpenID = func(context.Context) bool { return true }
			oldSleep := codexAppSleep
			codexAppSleep = func(time.Duration) {
				if !quitRequested {
					t.Fatal("wait before quit request")
				}
				if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
					t.Fatalf("files changed while ChatGPT was running (-before +after):\n%s", diff)
				}
				calls = append(calls, "wait")
				running = false
			}
			t.Cleanup(func() { codexAppSleep = oldSleep })
			if mode == "windows running" {
				withCodexAppPlatform(t, "windows")
				codexAppStartID = func(context.Context) string { return "fixture.ChatGPT!App" }
				codexAppOpenStart = func(id string) error {
					if id != "fixture.ChatGPT!App" {
						t.Fatalf("restart app ID = %q", id)
					}
					return reopen()
				}
			}
			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
				calls = append(calls, "confirm")
				return true, nil
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
			cmd := LaunchCmd(func(*cobra.Command, []string) error {
				t.Fatal("restore must not require the server")
				return nil
			}, nil)
			args := []string{"chatgpt", "--restore"}
			if mode == "headless --yes" {
				args = append(args, "--yes")
			} else {
				// ChatGPT can keep its own quit-confirmation prompt open longer
				// than the automatic exit timeout. The CLI should keep waiting.
				codexAppExitTimeout = 0
			}
			cmd.SetArgs(args)
			var err error
			stderr := captureStderr(t, func() { err = cmd.Execute() })
			if err != nil {
				t.Fatal(err)
			}
			if strings.Count(stderr, codexAppRestoreSuccess) != 1 {
				t.Fatalf("restore output = %q, want one success message", stderr)
			}
			var wantCalls []string
			if mode != "closed" {
				wantCalls = []string{"confirm", "quit", "wait", "open"}
				if mode == "headless --yes" {
					wantCalls = wantCalls[1:]
				}
			}
			if !slices.Equal(calls, wantCalls) {
				t.Fatalf("lifecycle calls = %v, want %v", calls, wantCalls)
			}
			for name, want := range original {
				data, err := os.ReadFile(filepath.Join(filepath.Dir(configPath), name))
				if err != nil {
					t.Fatal(err)
				}
				if name == "config.toml" {
					gotConfig, err := codexParseConfigText(string(data))
					if err != nil {
						t.Fatal(err)
					}
					wantConfig, err := codexParseConfigText(want)
					if err != nil {
						t.Fatal(err)
					}
					if diff := cmp.Diff(wantConfig, gotConfig); diff != "" {
						t.Errorf("settings changed (-before +after):\n%s", diff)
					}
				} else if string(data) != want {
					t.Errorf("restore changed %s", name)
				}
			}
			if app.OllamaConfigured() || app.RestorePending() {
				t.Fatal("restore left Ollama configuration or pending cleanup")
			}
		})
	}
}

func TestCodexAppCLIRestoreInterruptedQuitPreservesFiles(t *testing.T) {
	quitErr := errors.New("fixture quit failure")
	for _, mode := range []string{"quit failure", "timeout", "ctrl-c"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, _ := codexAppCLIRestoreFixture(t)
			withInteractiveSession(t, mode == "ctrl-c")
			before := codexAppRestoreFixtureFiles(t, configPath)
			restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
			defer restoreConfirm()
			cancel := make(chan struct{})
			spinnerStopped := false
			DefaultSpinner = func(string) *Spinner {
				return NewSpinner(func() { spinnerStopped = true }, cancel)
			}
			withCodexAppProcessHooks(t,
				func() bool { return true },
				func() error {
					if mode == "quit failure" {
						return quitErr
					}
					return nil
				},
				func() error { t.Error("reopened after interrupted quit"); return nil },
			)
			codexAppExitTimeout = 20 * time.Millisecond
			if mode == "quit failure" {
				codexAppExitTimeout = time.Second
			}
			if mode == "ctrl-c" {
				oldSleep := codexAppSleep
				codexAppSleep = func(time.Duration) {
					select {
					case <-cancel:
					default:
						close(cancel)
					}
				}
				t.Cleanup(func() { codexAppSleep = oldSleep })
			}
			err := app.Restore()
			if mode == "quit failure" && !errors.Is(err, quitErr) ||
				mode == "ctrl-c" && !errors.Is(err, ErrCancelled) ||
				mode == "timeout" && (err == nil || !strings.Contains(err.Error(), "Timed out waiting for ChatGPT")) {
				t.Fatalf("restore error = %v for %s", err, mode)
			}
			if !spinnerStopped {
				t.Error("restart spinner was not stopped")
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("interrupted quit changed files (-before +after):\n%s", diff)
			}
		})
	}
}

func TestCodexAppCLIRestoreBoundsQuitRequest(t *testing.T) {
	for _, mode := range []string{"timeout", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, _ := codexAppCLIRestoreFixture(t)
			before := codexAppRestoreFixtureFiles(t, configPath)
			withInteractiveSession(t, mode == "cancel")
			restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
			defer restoreConfirm()
			withCodexAppProcessHooks(t,
				func() bool { return true },
				func() error { return nil },
				func() error { t.Error("reopened after interrupted quit request"); return nil },
			)
			codexAppQuitApp = defaultCodexAppQuitApp
			codexAppExitTimeout = 20 * time.Millisecond
			// This child only simulates a quit request waiting for an app dialog.
			// PATH contains no real osascript, so no desktop app can be contacted.
			stubCodexAppProcessCommands(t, map[string]string{
				"osascript": "exec /bin/sleep 1\n",
			})
			cancel := make(chan struct{})
			DefaultSpinner = func(string) *Spinner { return NewSpinner(func() {}, cancel) }
			if mode == "cancel" {
				timer := time.AfterFunc(100*time.Millisecond, func() { close(cancel) })
				t.Cleanup(func() { timer.Stop() })
			}
			start := time.Now()
			err := app.Restore()
			if mode == "cancel" && !errors.Is(err, ErrCancelled) ||
				mode == "timeout" && (err == nil || !strings.Contains(err.Error(), "Timed out waiting for ChatGPT")) {
				t.Fatalf("restore error = %v for %s", err, mode)
			}
			if elapsed := time.Since(start); elapsed >= 750*time.Millisecond {
				t.Errorf("quit request ignored %s for %v", mode, elapsed)
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("interrupted request changed files (-before +after):\n%s", diff)
			}
		})
	}
}

func TestCodexAppCLIRestoreLegacyQuit(t *testing.T) {
	for _, mode := range []string{"exit", "timeout", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, _ := codexAppCLIRestoreFixture(t)
			withInteractiveSession(t, mode == "cancel")
			restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
			defer restoreConfirm()
			marker := filepath.Join(t.TempDir(), "quit-request")
			t.Setenv("OLLAMA_TEST_LEGACY_QUIT_MARKER", marker)
			script := "printf '%s' \"$2\" > \"$OLLAMA_TEST_LEGACY_QUIT_MARKER.tmp\"\n" +
				"/bin/mv \"$OLLAMA_TEST_LEGACY_QUIT_MARKER.tmp\" \"$OLLAMA_TEST_LEGACY_QUIT_MARKER\"\n"
			if mode != "exit" {
				script += "exec /bin/sleep 1\n"
			}
			stubCodexAppProcessCommands(t, map[string]string{"osascript": script})
			// Even if quit regresses to sending a signal, this PID belongs only
			// to an owned sleep fixture, never a desktop app or unrelated process.
			process, err := os.StartProcess("/bin/sleep", []string{"sleep", "5"}, &os.ProcAttr{})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = process.Kill(); _, _ = process.Wait() })
			oldExecutable, oldCommand := codexAppProfileExecutable, codexAppProcessCommand
			oldRunning, oldStop := codexAppProfileIsRunning, codexAppStopProfile
			codexAppProfileExecutable = func() (string, error) { return "/fixture/ChatGPT", nil }
			userData, err := codexAppOllamaProfileUserDataDir()
			if err != nil {
				t.Fatal(err)
			}
			codexAppProcessCommand = func(_ context.Context, pid int) (string, error) {
				if pid != process.Pid {
					t.Fatalf("unexpected process lookup %d", pid)
				}
				if _, err := os.Stat(marker); mode == "exit" && err == nil {
					return "", os.ErrNotExist
				}
				return "/fixture/ChatGPT --user-data-dir=" + userData, nil
			}
			codexAppProfileIsRunning, codexAppStopProfile = defaultCodexAppOllamaProfileIsRunning, defaultCodexAppStopOllamaProfile
			t.Cleanup(func() {
				codexAppProfileExecutable, codexAppProcessCommand = oldExecutable, oldCommand
				codexAppProfileIsRunning, codexAppStopProfile = oldRunning, oldStop
			})
			if err := writeCodexAppOllamaProfilePID(process.Pid); err != nil {
				t.Fatal(err)
			}
			before := codexAppRestoreFixtureFiles(t, configPath)
			openCalls := 0
			withCodexAppProcessHooks(t,
				func() bool { return false },
				func() error { t.Error("regular app was already closed"); return nil },
				func() error { openCalls++; return nil },
			)
			codexAppExitTimeout = time.Second
			if mode == "timeout" {
				codexAppExitTimeout = 20 * time.Millisecond
			}
			cancel := make(chan struct{})
			DefaultSpinner = func(string) *Spinner { return NewSpinner(func() {}, cancel) }
			if mode == "cancel" {
				cancelled := make(chan struct{})
				go func() {
					defer close(cancelled)
					deadline := time.Now().Add(time.Second)
					for time.Now().Before(deadline) {
						if _, err := os.Stat(marker); err == nil {
							break
						}
						time.Sleep(time.Millisecond)
					}
					close(cancel)
				}()
				t.Cleanup(func() { <-cancelled })
			}
			err = app.Restore()
			request, readErr := os.ReadFile(marker)
			if readErr == nil && !strings.Contains(string(request), strconv.Itoa(process.Pid)) ||
				readErr != nil && mode != "timeout" {
				t.Fatalf("quit request did not target the owned legacy PID: %q, %v", request, readErr)
			}
			if mode == "exit" {
				if err != nil || openCalls != 1 || app.OllamaConfigured() || app.RestorePending() {
					t.Fatalf("restored legacy client: error %v, open calls %d, connected %v, pending %v", err, openCalls, app.OllamaConfigured(), app.RestorePending())
				}
				return
			}
			if mode == "cancel" && !errors.Is(err, ErrCancelled) ||
				mode == "timeout" && (err == nil || !strings.Contains(err.Error(), "Timed out waiting for ChatGPT")) {
				t.Fatalf("legacy quit error = %v for %s", err, mode)
			}
			if openCalls != 0 {
				t.Fatal("reopened after interrupted legacy quit")
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("interrupted legacy quit changed files (-before +after):\n%s", diff)
			}
		})
	}
}

func TestCodexAppProfileChangeFinishesAfterCommit(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	running := true
	var calls []string
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error { calls = append(calls, "quit"); running = false; return nil },
		func() error { calls = append(calls, "open"); return nil },
	)
	cancel := make(chan struct{})
	ctx, stop := codexAppExitContext(time.Second, cancel)
	defer stop()
	err := codexAppApplyProfileChange(ctx, func() error {
		calls = append(calls, "commit")
		close(cancel)
		return nil
	}, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(calls, []string{"quit", "commit", "open"}) {
		t.Fatalf("calls = %v, want quit, commit, open", calls)
	}
}

func TestCodexAppCLIRestoreObservesCancelBeforeMutation(t *testing.T) {
	oldProcs := runtime.GOMAXPROCS(1)
	t.Cleanup(func() { runtime.GOMAXPROCS(oldProcs) })
	app, configPath, _ := codexAppCLIRestoreFixture(t)
	before := codexAppRestoreFixtureFiles(t, configPath)
	cancel := make(chan struct{})
	running := true
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error { running = false; close(cancel); return nil },
		func() error { t.Error("reopened after pre-mutation cancellation"); return nil },
	)
	err := app.restore(true, cancel, time.Second)
	if !errors.Is(err, ErrCancelled) {
		t.Errorf("restore error = %v, want cancellation", err)
	}
	if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
		t.Fatalf("changed files after cancellation (-before +after):\n%s", diff)
	}
}

func TestCodexAppCLIRestoreFailureReopensRunningClient(t *testing.T) {
	for _, mode := range []string{"closed", "regular", "legacy", "legacy reopen failure"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, _ := codexAppCLIRestoreFixture(t)
			withInteractiveSession(t, false)
			restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
			defer restoreConfirm()
			if err := os.WriteFile(configPath, []byte("model = \"unterminated\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			before := codexAppRestoreFixtureFiles(t, configPath)
			running := mode == "regular"
			legacyRunning := strings.HasPrefix(mode, "legacy")
			oldRunning, oldStop := codexAppProfileIsRunning, codexAppStopProfile
			codexAppProfileIsRunning = func(context.Context) (bool, error) { return legacyRunning, nil }
			codexAppStopProfile = func(context.Context) error { legacyRunning = false; return nil }
			t.Cleanup(func() { codexAppProfileIsRunning, codexAppStopProfile = oldRunning, oldStop })
			openCalls := 0
			openErr := errors.New("fixture reopen failure")
			withCodexAppProcessHooks(t,
				func() bool { return running },
				func() error { running = false; return nil },
				func() error {
					openCalls++
					if mode == "legacy reopen failure" {
						return openErr
					}
					return nil
				},
			)
			err := app.Restore()
			if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
				t.Fatalf("restore error = %v, want invalid TOML", err)
			}
			wantOpen := 1
			if mode == "closed" {
				wantOpen = 0
			}
			if openCalls != wantOpen {
				t.Errorf("open calls = %d, want %d", openCalls, wantOpen)
			}
			if mode == "legacy reopen failure" && !errors.Is(err, openErr) {
				t.Errorf("restore error lost reopen failure: %v", err)
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("failed restore changed files (-before +after):\n%s", diff)
			}
		})
	}
}

func TestCodexAppLateLegacyStartRequiresConsent(t *testing.T) {
	for _, mode := range []string{"restore", "desktop update", "approved restore"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, _ := codexAppCLIRestoreFixture(t)
			before := codexAppRestoreFixtureFiles(t, configPath)
			withCodexAppRouterHealth(t, func() error { return nil })
			openCalls := 0
			withCodexAppProcessHooks(t, func() bool { return false },
				func() error { t.Error("regular app was closed"); return nil },
				func() error { openCalls++; return nil })
			codexAppCanOpenID = func(context.Context) bool { return true }
			oldRunning, oldStop := codexAppProfileIsRunning, codexAppStopProfile
			checks, stops := 0, 0
			codexAppProfileIsRunning = func(context.Context) (bool, error) { checks++; return checks > 1 && stops == 0, nil }
			codexAppStopProfile = func(context.Context) error { stops++; return nil }
			t.Cleanup(func() { codexAppProfileIsRunning, codexAppStopProfile = oldRunning, oldStop })
			var err error
			if mode == "desktop update" {
				err = app.UpdateOllamaModelsFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b"), false)
			} else {
				err = app.RestoreFromDesktop(mode == "approved restore")
			}
			if mode == "approved restore" {
				if err != nil || stops != 1 || openCalls != 1 {
					t.Fatalf("approved restore: error %v, stops %d, opens %d", err, stops, openCalls)
				}
				return
			}
			if !errors.Is(err, ErrCodexAppRestartConfirmationRequired) || stops != 0 || openCalls != 0 {
				t.Fatalf("late start bypassed consent: error %v, stops %d, opens %d", err, stops, openCalls)
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("changed files before consent (-before +after):\n%s", diff)
			}
		})
	}
}

func TestCodexAppExitWaitBoundsProcessProbes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process fixtures require a POSIX shell")
	}
	for _, probe := range []string{"pgrep", "ps", "powershell", "legacy ps"} {
		for _, mode := range []string{"timeout", "cancel"} {
			t.Run(probe+"/"+mode, func(t *testing.T) {
				_, configPath, _ := codexAppCLIRestoreFixture(t)
				withCodexAppProcessHooks(t, func() bool { return true },
					func() error { return nil }, func() error { t.Error("unexpected open"); return nil })
				codexAppIsRunning = defaultCodexAppIsRunning
				blocked := filepath.Join(t.TempDir(), "blocked-probe")
				t.Setenv("OLLAMA_TEST_BLOCKED_PROBE", blocked)
				slowProbe := ": > \"$OLLAMA_TEST_BLOCKED_PROBE\"\nexec /bin/sleep 2\n"
				commands := map[string]string{"pgrep": "exit 1\n", "ps": "exit 0\n", "osascript": "exit 0\n"}
				switch probe {
				case "pgrep", "ps":
					commands[probe] = slowProbe
				case "powershell":
					withCodexAppPlatform(t, "windows")
					commands["powershell.exe"] = slowProbe
				case "legacy ps":
					marker := filepath.Join(t.TempDir(), "first-probe")
					t.Setenv("OLLAMA_TEST_PROBE_MARKER", marker)
					commands["ps"] = "if [ -e \"$OLLAMA_TEST_PROBE_MARKER\" ]; then\n" + slowProbe + "fi\n" +
						": > \"$OLLAMA_TEST_PROBE_MARKER\"\nprintf '%s' \"$OLLAMA_TEST_PROCESS_COMMAND\"\n"
					// Only an owned fixture PID is ever eligible for a quit request.
					process, err := os.StartProcess("/bin/sleep", []string{"sleep", "5"}, &os.ProcAttr{})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { _ = process.Kill(); _, _ = process.Wait() })
					oldExecutable, oldCommand := codexAppProfileExecutable, codexAppProcessCommand
					codexAppProfileExecutable = func() (string, error) { return "/fixture/ChatGPT", nil }
					codexAppProcessCommand = defaultCodexAppProcessCommand
					t.Cleanup(func() { codexAppProfileExecutable, codexAppProcessCommand = oldExecutable, oldCommand })
					userData, err := codexAppOllamaProfileUserDataDir()
					if err != nil {
						t.Fatal(err)
					}
					t.Setenv("OLLAMA_TEST_PROCESS_COMMAND", "/fixture/ChatGPT --user-data-dir="+userData)
					if err := writeCodexAppOllamaProfilePID(process.Pid); err != nil {
						t.Fatal(err)
					}
				}
				stubCodexAppProcessCommands(t, commands)
				before := codexAppRestoreFixtureFiles(t, configPath)
				cancel := make(chan struct{})
				timeout := time.Second
				if mode == "cancel" {
					timeout = -1
					cancelled := make(chan struct{})
					go func() {
						defer close(cancelled)
						deadline := time.Now().Add(5 * time.Second)
						for time.Now().Before(deadline) {
							if _, err := os.Stat(blocked); err == nil {
								break
							}
							time.Sleep(time.Millisecond)
						}
						close(cancel)
					}()
					t.Cleanup(func() { <-cancelled })
				}
				start := time.Now()
				var err error
				if probe == "legacy ps" {
					ctx, stop := codexAppExitContext(timeout, cancel)
					defer stop()
					err = defaultCodexAppStopOllamaProfile(ctx)
					if err != nil {
						if contextErr := codexAppExitContextError(ctx); contextErr != nil {
							err = contextErr
						}
					}
				} else {
					ctx, stop := codexAppExitContext(timeout, cancel)
					defer stop()
					err = codexAppStopForRestart(ctx)
				}
				if mode == "cancel" && !errors.Is(err, ErrCancelled) ||
					mode == "timeout" && (err == nil || !strings.Contains(err.Error(), "Timed out waiting for ChatGPT")) {
					t.Errorf("probe ignored %s: %v", mode, err)
				}
				if elapsed := time.Since(start); elapsed >= 1750*time.Millisecond {
					t.Errorf("probe ignored %s for %v", mode, elapsed)
				}
				if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
					t.Fatalf("interrupted probe changed files (-before +after):\n%s", diff)
				}
			})
		}
	}
}

func TestCodexAppRestoreProbePhasesRespectExitContext(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("process fixtures require a POSIX shell")
	}
	for _, phase := range []string{"initial regular", "apply regular", "after legacy", "initial legacy", "legacy singleton", "windows start target", "windows path target"} {
		for _, mode := range []string{"timeout", "cancel"} {
			t.Run(phase+"/"+mode, func(t *testing.T) {
				app, configPath, _ := codexAppCLIRestoreFixture(t)
				withCodexAppProcessHooks(t, func() bool { return true },
					func() error { t.Error("quit after an interrupted status check"); return nil },
					func() error { t.Error("open after an interrupted status check"); return nil })
				checks := 0
				codexAppIsRunning = func(ctx context.Context) bool {
					checks++
					if strings.HasPrefix(phase, "windows") {
						return true
					}
					if phase == "initial legacy" || phase == "legacy singleton" ||
						checks == 1 && phase != "initial regular" {
						return false
					}
					return defaultCodexAppIsRunning(ctx)
				}
				oldRunning, oldStop := codexAppProfileIsRunning, codexAppStopProfile
				legacyStopped := false
				codexAppProfileIsRunning = func(context.Context) (bool, error) { return phase == "after legacy" && !legacyStopped, nil }
				codexAppStopProfile = func(context.Context) error { legacyStopped = true; return nil }
				t.Cleanup(func() { codexAppProfileIsRunning, codexAppStopProfile = oldRunning, oldStop })
				blocked := filepath.Join(t.TempDir(), "blocked-probe")
				t.Setenv("OLLAMA_TEST_BLOCKED_PROBE", blocked)
				slowOnce := "if [ -e \"$OLLAMA_TEST_BLOCKED_PROBE\" ]; then exit 1; fi\n" +
					": > \"$OLLAMA_TEST_BLOCKED_PROBE\"\nexec /bin/sleep 2\n"
				commands := map[string]string{"pgrep": slowOnce, "ps": "exit 0\n", "powershell.exe": slowOnce}
				if strings.HasPrefix(phase, "windows") {
					withCodexAppPlatform(t, "windows")
					codexAppStartID = func(context.Context) string { return "" }
					codexAppRunPath = func(context.Context) string { return "" }
					if phase == "windows start target" {
						codexAppStartID = defaultCodexAppStartAppID
					} else {
						codexAppRunPath = defaultCodexAppRunningAppPath
					}
				}
				if phase == "initial legacy" || phase == "legacy singleton" {
					process, err := os.StartProcess("/bin/sleep", []string{"sleep", "5"}, &os.ProcAttr{})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { _ = process.Kill(); _, _ = process.Wait() })
					oldExecutable, oldCommand := codexAppProfileExecutable, codexAppProcessCommand
					codexAppProfileExecutable = func() (string, error) { return "/fixture/ChatGPT", nil }
					codexAppProcessCommand = defaultCodexAppProcessCommand
					codexAppProfileIsRunning = defaultCodexAppOllamaProfileIsRunning
					t.Cleanup(func() { codexAppProfileExecutable, codexAppProcessCommand = oldExecutable, oldCommand })
					commands["ps"] = slowOnce
					if phase == "initial legacy" {
						if err := writeCodexAppOllamaProfilePID(process.Pid); err != nil {
							t.Fatal(err)
						}
					} else {
						userData, err := codexAppOllamaProfileUserDataDir()
						if err != nil {
							t.Fatal(err)
						}
						if err := os.MkdirAll(userData, 0o700); err != nil {
							t.Fatal(err)
						}
						for _, name := range []string{codexAppSingletonLockName, codexAppSingletonSocketName, codexAppSingletonCookieName} {
							if err := os.Symlink("fixture-"+strconv.Itoa(process.Pid), filepath.Join(userData, name)); err != nil {
								t.Fatal(err)
							}
						}
					}
				}
				stubCodexAppProcessCommands(t, commands)
				before := codexAppRestoreFixtureFiles(t, configPath)
				cancel := make(chan struct{})
				timeout := time.Second
				if mode == "cancel" {
					timeout = -1
					cancelled := make(chan struct{})
					go func() {
						defer close(cancelled)
						deadline := time.Now().Add(5 * time.Second)
						for time.Now().Before(deadline) {
							if _, err := os.Stat(blocked); err == nil {
								break
							}
							time.Sleep(time.Millisecond)
						}
						close(cancel)
					}()
					t.Cleanup(func() { <-cancelled })
				}
				start := time.Now()
				err := app.restore(true, cancel, timeout)
				if mode == "cancel" && !errors.Is(err, ErrCancelled) ||
					mode == "timeout" && (err == nil || !strings.Contains(err.Error(), "Timed out waiting for ChatGPT")) {
					t.Errorf("%s ignored %s: %v", phase, mode, err)
				}
				if elapsed := time.Since(start); elapsed >= 1750*time.Millisecond {
					t.Errorf("%s ignored %s for %v", phase, mode, elapsed)
				}
				if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
					t.Fatalf("interrupted restore changed files (-before +after):\n%s", diff)
				}
			})
		}
	}
}

func TestCodexAppDeclinedRestorePreservesLegacyMetadata(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("owned process fixture requires a POSIX executable")
	}
	for _, mode := range []string{"missing PID", "malformed PID", "probe error", "legacy-only probe error"} {
		t.Run(mode, func(t *testing.T) {
			app, configPath, _ := codexAppCLIRestoreFixture(t)
			withInteractiveSession(t, true)
			restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{})
			defer restorePolicy()
			withCodexAppProcessHooks(t, func() bool { return mode != "legacy-only probe error" },
				func() error { t.Error("quit before consent"); return nil },
				func() error { t.Error("open after declining"); return nil })
			process, err := os.StartProcess("/bin/sleep", []string{"sleep", "5"}, &os.ProcAttr{})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = process.Kill(); _, _ = process.Wait() })
			userData, err := codexAppOllamaProfileUserDataDir()
			if err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(userData, 0o700); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{codexAppSingletonLockName, codexAppSingletonSocketName, codexAppSingletonCookieName} {
				if err := os.Symlink("fixture-"+strconv.Itoa(process.Pid), filepath.Join(userData, name)); err != nil {
					t.Fatal(err)
				}
			}
			if mode != "missing PID" {
				if err := writeCodexAppOllamaProfilePID(process.Pid); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "malformed PID" {
				path, err := codexAppOllamaProfilePIDPath()
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte("not a process ID\n"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			probeErr := errors.New("fixture identity lookup failed")
			oldRunning, oldExecutable, oldCommand := codexAppProfileIsRunning, codexAppProfileExecutable, codexAppProcessCommand
			codexAppProfileIsRunning = defaultCodexAppOllamaProfileIsRunning
			codexAppProfileExecutable = func() (string, error) { return "/fixture/ChatGPT", nil }
			codexAppProcessCommand = func(context.Context, int) (string, error) {
				if strings.Contains(mode, "probe error") {
					return "", probeErr
				}
				return "/fixture/ChatGPT --user-data-dir=" + userData, nil
			}
			t.Cleanup(func() {
				codexAppProfileIsRunning, codexAppProfileExecutable, codexAppProcessCommand = oldRunning, oldExecutable, oldCommand
			})
			oldConfirm := DefaultConfirmPrompt
			prompts := 0
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { prompts++; return false, nil }
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
			before := codexAppRestoreFixtureFiles(t, configPath)
			err = app.Restore()
			if mode == "legacy-only probe error" {
				if !errors.Is(err, probeErr) || prompts != 0 {
					t.Errorf("lost identity error: %v, prompts %d", err, prompts)
				}
			} else if !errors.Is(err, ErrCancelled) || prompts != 1 {
				t.Errorf("declined restore: error %v, prompts %d", err, prompts)
			}
			if diff := cmp.Diff(before, codexAppRestoreFixtureFiles(t, configPath)); diff != "" {
				t.Fatalf("preflight changed legacy metadata (-before +after):\n%s", diff)
			}
		})
	}
}
