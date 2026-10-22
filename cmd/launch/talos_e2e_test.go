package launch

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

// This opt-in test downloads and executes the public, signature-verifying Talos
// installer in a temporary prefix. Only the model server is a fixture; the
// installer, config CLI, Python runtime, chat and event log are real.
// Run with TALOS_TEST_INSTALLER=1 go test -timeout 20m ./cmd/launch -run TestTalosReleasedInstaller -v.
func TestTalosReleasedInstaller(t *testing.T) {
	if os.Getenv("TALOS_TEST_INSTALLER") != "1" {
		t.Skip("set TALOS_TEST_INSTALLER=1 to test the published installer")
	}
	if runtime.GOOS == "windows" {
		t.Skip("Talos installer supports macOS and Linux")
	}
	for _, name := range []string{"bash", "curl", "python3"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Fatalf("installer prerequisite %s: %v", name, err)
		}
	}
	// Keep the evidence tied to the reviewed release, even after the public
	// installer advances. Its own checksum and signature checks still run.
	const version = "0.20.0-beta.5"
	const archiveSHA256 = "ddd24ce4d26f79640fe7ce9e9770a9aa770e8c73700895212fa253772cd63059"
	archive, err := exec.Command("curl", "-fsSL", "--max-time", "120", "https://talos-agent.ch/dist/talos-"+version+".tar.gz").Output()
	if err != nil {
		t.Fatalf("download reviewed release: %v", err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(archive)); got != archiveSHA256 {
		t.Fatalf("reviewed release archive checksum: got %s, want %s", got, archiveSHA256)
	}
	tmp := t.TempDir()
	clearTalosEnvVars(t)
	withLauncherHooks(t)
	prefix := filepath.Join(tmp, "installed-talos")
	t.Setenv("TALOS_PREFIX", prefix)
	t.Setenv("TALOS_BIN_DIR", filepath.Join(tmp, "commands"))
	t.Setenv("TALOS_SECRETS_ENV", filepath.Join(tmp, "empty-secrets.env"))
	t.Setenv("NO_COLOR", "1")
	oldLookPath := talosLookPath
	talosLookPath = func(name string) (string, error) {
		if name == "talos" {
			return "", exec.ErrNotFound
		}
		return exec.LookPath(name)
	}
	t.Cleanup(func() { talosLookPath = oldLookPath })
	DefaultConfirmPrompt = func(_ string, _ ConfirmOptions) (bool, error) { return true, nil }
	if err := (&Talos{}).ensureInstalled(); err != nil {
		t.Fatal(err)
	}
	installedVersion, err := exec.Command(filepath.Join(prefix, "bin", "talos"), "--version").CombinedOutput()
	if err != nil || strings.TrimSpace(string(installedVersion)) != version {
		t.Fatalf("installed release version: %v, %q", err, installedVersion)
	}

	var calls, invalid atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			invalid.Add(1)
		}
		switch {
		case r.Method == "GET" && r.URL.Path == "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"data":[{"id":"fixture-model"}]}`)
		case r.Method == "POST" && r.URL.Path == "/v1/chat/completions":
			var body struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Model != "fixture-model" {
				invalid.Add(1)
				http.Error(w, "unexpected model request", http.StatusBadRequest)
				return
			}
			calls.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"LAUNCHER_E2E_OK\"}}]}\n\ndata: [DONE]\n\n")
		default:
			invalid.Add(1)
			http.Error(w, "unexpected endpoint", http.StatusNotFound)
		}
	}))
	t.Cleanup(server.Close)
	t.Setenv("TALOS_BASE_URL_OLLAMA", server.URL+"/v1")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	caller := filepath.Join(tmp, "caller")
	if err := os.Mkdir(caller, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(caller)
	wrapper := filepath.Join(prefix, "bin", "talos")

	for _, mode := range []string{"prefix-wrapper", "path-symlink", "legacy-venv"} {
		t.Run(mode, func(t *testing.T) {
			if mode == "path-symlink" {
				before := talosLookPath
				talosLookPath = func(name string) (string, error) {
					if name == "talos" {
						return filepath.Join(tmp, "commands", "talos"), nil
					}
					return exec.LookPath(name)
				}
				t.Cleanup(func() { talosLookPath = before })
				t.Setenv("TALOS_PREFIX", filepath.Join(tmp, "not-the-selected-installation"))
			}
			if mode == "legacy-venv" {
				if err := os.Rename(wrapper, wrapper+".test-disabled"); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := os.Rename(wrapper+".test-disabled", wrapper); err != nil {
						t.Error(err)
					}
				})
			}
			if err := (&Talos{}).Configure("fixture-model"); err != nil {
				t.Fatal(err)
			}
			if got := (&Talos{}).CurrentModel(); got != "fixture-model" {
				t.Fatalf("configuration read-back: %q", got)
			}
			argv, err := (&Talos{}).command()
			if err != nil {
				t.Fatal(err)
			}
			cmd := talosAttachedCommand(argv, "chat")
			cmd.Stdin = strings.NewReader("Reply with LAUNCHER_E2E_OK.\nexit\n")
			var output bytes.Buffer
			cmd.Stdout, cmd.Stderr = &output, &output
			if err := cmd.Run(); err != nil {
				t.Fatalf("chat failed: %v\n%s", err, output.String())
			}
			if !strings.Contains(output.String(), "LAUNCHER_E2E_OK") {
				t.Fatalf("reply missing:\n%s", output.String())
			}
		})
	}
	if invalid.Load() != 0 || calls.Load() != 3 {
		t.Fatalf("model transport: %d calls, %d invalid requests", calls.Load(), invalid.Load())
	}
	python := filepath.Join(prefix, ".venv", "bin", "python")
	check := `import sqlite3,sys;d=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True);assert d.execute("select count(*) from events where type='done'").fetchone()[0]==3;print('Three completed chat turns persisted')`
	out, err := exec.Command(python, "-B", "-c", check, filepath.Join(prefix, "data", "eventlog.db")).CombinedOutput()
	if err != nil {
		t.Fatalf("event-log read-back: %v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

// This separately exercises an existing real Homebrew keg, without installing,
// upgrading or changing it. Only the temporary operator profile is writable.
func TestTalosHomebrewRuntime(t *testing.T) {
	keg := os.Getenv("TALOS_TEST_HOMEBREW_PREFIX")
	if keg == "" {
		t.Skip("set TALOS_TEST_HOMEBREW_PREFIX to test an existing Homebrew keg")
	}
	for _, custom := range []bool{false, true} {
		t.Run(fmt.Sprintf("custom-profile=%t", custom), func(t *testing.T) {
			talosHomebrewRuntime(t, keg, custom)
		})
	}
}

func talosHomebrewRuntime(t *testing.T, keg string, custom bool) {
	t.Helper()
	clearTalosEnvVars(t)
	home := t.TempDir()
	t.Setenv("HOME", home)
	withTalosUserHome(t, home)
	profile := filepath.Join(home, ".talos", "homebrew")
	if custom {
		profile = filepath.Join(home, "private-profile")
		t.Setenv("TALOS_PROFILE_DIR", profile)
	}
	// Model the private directory created by the operator's interactive setup;
	// this test does not approve terminal identity or emulate setup prompts.
	if err := os.MkdirAll(profile, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	launcher := filepath.Join(keg, "bin", "talos")
	prefix, ok := talosWrapperPrefix(launcher)
	if !ok || prefix != filepath.Join(keg, "libexec") {
		t.Fatalf("not a recognized Homebrew keg: %s", keg)
	}
	before := talosLookPath
	talosLookPath = func(name string) (string, error) {
		if name == "talos" {
			return launcher, nil
		}
		return exec.LookPath(name)
	}
	t.Cleanup(func() { talosLookPath = before })
	if err := (&Talos{}).Configure("homebrew-fixture-model"); err != nil {
		t.Fatal(err)
	}
	if got := (&Talos{}).CurrentModel(); got != "homebrew-fixture-model" {
		t.Fatalf("managed configuration read-back: %q", got)
	}
	paths := (&Talos{}).Paths()
	if len(paths) != 1 || paths[0] != filepath.Join(profile, "talos.env") {
		t.Fatalf("managed configuration path: %v", paths)
	}
	info, err := os.Stat(paths[0])
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("private managed config: %v, %v", info, err)
	}
	if _, err := os.Stat(filepath.Join(keg, "libexec", "talos.env")); !os.IsNotExist(err) {
		t.Fatalf("managed configuration must not enter the keg: %v", err)
	}
}
