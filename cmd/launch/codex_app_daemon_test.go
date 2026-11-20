package launch

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestCodexAppDesktopRestartsManagedDaemon(t *testing.T) {
	stopFailure := errors.New("daemon shutdown failed")
	for _, tt := range []struct {
		name      string
		running   bool
		confirmed bool
		repair    bool
		stopErr   error
		wantErr   error
		want      []string
	}{
		{name: "save and restart", running: true, confirmed: true, want: []string{"quit", "daemon-stop", "open"}},
		{name: "restart saved profile", running: true, confirmed: true, repair: true, want: []string{"quit", "daemon-stop", "open"}},
		{name: "confirmation required", running: true, wantErr: ErrCodexAppRestartConfirmationRequired},
		{name: "daemon failure reopens unchanged profile", running: true, confirmed: true, stopErr: stopFailure, wantErr: stopFailure, want: []string{"quit", "daemon-stop", "open"}},
		{name: "stopped save stays stopped"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			home := t.TempDir()
			setTestHome(t, home)
			withCodexAppPlatform(t, "darwin")
			withCodexAppRouterHealth(t, func() error { return nil })
			app := &CodexApp{}
			if err := app.Configure("qwen3:8b"); err != nil {
				t.Fatal(err)
			}
			configPath := filepath.Join(home, ".codex", "config.toml")
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			catalogPath := codexAppModelCatalogPathForConfig(configPath)
			var events []string
			running := tt.running
			withCodexAppProcessHooks(t,
				func() bool { return running },
				func() error { events = append(events, "quit"); running = false; return nil },
				func() error {
					events = append(events, "open")
					if tt.stopErr == nil && !tt.repair {
						data, err := os.ReadFile(catalogPath)
						if err != nil || !strings.Contains(string(data), "gemma4:latest") {
							t.Fatalf("app reopened before catalog save: %v", err)
						}
					}
					return nil
				},
			)
			codexAppCanOpenID = func(context.Context) bool { return true }
			codexAppStopDaemon = func() error {
				if running {
					t.Fatal("daemon stopped before the desktop app exited")
				}
				current, err := os.ReadFile(configPath)
				if err != nil || !slices.Equal(current, before) {
					t.Fatal("profile changed before daemon shutdown")
				}
				events = append(events, "daemon-stop")
				return tt.stopErr
			}
			if tt.repair {
				err = app.RestartFromDesktop(tt.confirmed)
			} else {
				err = app.UpdateOllamaModelsFromDesktop("qwen3:8b", testLaunchModels("qwen3:8b", "gemma4:latest"), tt.confirmed)
			}
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("error = %v, want %v", err, tt.wantErr)
			}
			if !slices.Equal(events, tt.want) {
				t.Fatalf("restart steps = %v, want %v", events, tt.want)
			}
			if tt.wantErr != nil {
				after, readErr := os.ReadFile(configPath)
				if readErr != nil || !slices.Equal(after, before) {
					t.Fatal("failed or unconfirmed restart changed the profile")
				}
			}
		})
	}
}

func TestCodexAppManagedDaemonStopCommand(t *testing.T) {
	for _, tt := range []struct {
		name   string
		script string
		ok     bool
	}{
		{name: "stopped", script: `printf '{"status":"stopped"}'`, ok: true},
		{name: "already stopped", script: `printf '{"status":"notRunning"}'`, ok: true},
		{name: "legacy client", script: `echo "error: unrecognized subcommand 'daemon'" >&2; exit 2`, ok: true},
		{name: "legacy argument parser", script: `echo "error: unexpected argument 'daemon' found" >&2; exit 2`, ok: true},
		{name: "unrelated argument error", script: `echo "error: invalid configuration" >&2; exit 2`},
		{name: "stop failed", script: `echo "fixture-private-marker" >&2; exit 1`},
		{name: "unexpected status", script: `printf '{"status":"running"}'`},
		{name: "malformed result", script: `printf 'not-json'`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			home := t.TempDir()
			setTestHome(t, home)
			t.Setenv("CODEX_HOME", filepath.Join(home, "unrelated-profile"))
			argsPath := filepath.Join(home, "args")
			profilePath := filepath.Join(home, "profile")
			t.Setenv("DAEMON_TEST_ARGS", argsPath)
			t.Setenv("DAEMON_TEST_PROFILE", profilePath)
			stubCodexAppProcessCommands(t, map[string]string{"codex": "printf '%s\\n' \"$@\" > \"$DAEMON_TEST_ARGS\"\nprintf '%s' \"$CODEX_HOME\" > \"$DAEMON_TEST_PROFILE\"\n" + tt.script})
			executable, err := exec.LookPath("codex")
			if err != nil {
				t.Fatal(err)
			}
			old := codexAppCodexExecutable
			codexAppCodexExecutable = func() (string, error) { return executable, nil }
			t.Cleanup(func() { codexAppCodexExecutable = old })
			err = stopCodexAppManagedDaemon(context.Background())
			if (err == nil) != tt.ok {
				t.Fatalf("error = %v, want success %v", err, tt.ok)
			}
			if err != nil && strings.Contains(err.Error(), "fixture-private-marker") {
				t.Fatal("command stderr leaked through the error")
			}
			args, readErr := os.ReadFile(argsPath)
			if readErr != nil || string(args) != "app-server\ndaemon\nstop\n" {
				t.Fatalf("args = %q, error %v", args, readErr)
			}
			profile, readErr := os.ReadFile(profilePath)
			if readErr != nil || string(profile) != filepath.Join(home, ".codex") {
				t.Fatalf("daemon profile = %q, error %v", profile, readErr)
			}
		})
	}
}

func TestCodexAppManagedDaemonStopTimeout(t *testing.T) {
	setTestHome(t, t.TempDir())
	stubCodexAppProcessCommands(t, map[string]string{"codex": "exec /bin/sleep 30"})
	executable, err := exec.LookPath("codex")
	if err != nil {
		t.Fatal(err)
	}
	old := codexAppCodexExecutable
	codexAppCodexExecutable = func() (string, error) { return executable, nil }
	t.Cleanup(func() { codexAppCodexExecutable = old })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := stopCodexAppManagedDaemon(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stop error = %v, want deadline exceeded", err)
	}
}

func TestCodexAppManagedDaemonMissingCLI(t *testing.T) {
	old := codexAppCodexExecutable
	codexAppCodexExecutable = func() (string, error) { return "", exec.ErrNotFound }
	t.Cleanup(func() { codexAppCodexExecutable = old })
	if err := stopCodexAppManagedDaemon(context.Background()); err != nil {
		t.Fatalf("legacy client without a CLI: %v", err)
	}
	codexAppCodexExecutable = func() (string, error) { return "", os.ErrPermission }
	if err := stopCodexAppManagedDaemon(context.Background()); !errors.Is(err, os.ErrPermission) {
		t.Fatalf("lookup failure = %v, want permission error", err)
	}
}

func TestCodexAppBundledDaemonExecutable(t *testing.T) {
	for _, platform := range []string{"darwin", "windows"} {
		for _, modern := range []bool{false, true} {
			layout := "legacy"
			if modern {
				layout = "packaged"
			}
			t.Run(platform+"/"+layout, func(t *testing.T) {
				home := t.TempDir()
				setTestHome(t, home)
				withCodexAppPlatform(t, platform)
				t.Setenv("PATH", t.TempDir())
				appPath := filepath.Join(home, "Applications", "ChatGPT.app")
				resources := filepath.Join(appPath, "Contents", "Resources")
				name := "codex"
				if platform == "windows" {
					local := filepath.Join(home, "Local")
					t.Setenv("LOCALAPPDATA", local)
					appPath = filepath.Join(local, "Programs", "ChatGPT", "ChatGPT.exe")
					resources = filepath.Join(filepath.Dir(appPath), "resources")
					name = "codex.exe"
					if err := os.MkdirAll(filepath.Dir(appPath), 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(appPath, nil, 0o600); err != nil {
						t.Fatal(err)
					}
				}
				if err := os.MkdirAll(resources, 0o755); err != nil {
					t.Fatal(err)
				}
				legacy := filepath.Join(resources, name)
				if err := os.WriteFile(legacy, nil, 0o600); err != nil {
					t.Fatal(err)
				}
				want := legacy
				if modern {
					want = filepath.Join(resources, "codex-cli", "bin", name)
					if err := os.MkdirAll(filepath.Dir(want), 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(want, nil, 0o600); err != nil {
						t.Fatal(err)
					}
				}
				oldStat := codexAppStat
				codexAppStat = func(path string) (os.FileInfo, error) {
					if !strings.HasPrefix(path, home+string(filepath.Separator)) {
						return nil, os.ErrNotExist
					}
					return os.Stat(path)
				}
				t.Cleanup(func() { codexAppStat = oldStat })
				got, err := defaultCodexAppCodexExecutable()
				if err != nil || got != want {
					t.Fatalf("executable = %q, error %v; want %q", got, err, want)
				}
			})
		}
	}
}
