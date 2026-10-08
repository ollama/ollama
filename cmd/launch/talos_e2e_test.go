package launch

import (
	"bytes"
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
// Run with TALOS_TEST_INSTALLER=1 go test ./cmd/launch -run TestTalosReleasedInstaller -v.
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
