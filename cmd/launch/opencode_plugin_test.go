package launch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/spf13/cobra"
)

func TestLaunchCmdOpenCodeRequiresServer(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell fake executable")
	}
	setTestHome(t, t.TempDir())
	withInteractiveSession(t, true)
	bin := t.TempDir()
	t.Setenv("PATH", bin)
	t.Setenv("OLLAMA_HOST", "")
	t.Setenv(openCodeStartupEnv, "")
	t.Setenv("OPENCODE_CONFIG_DIR", t.TempDir())
	executable := `#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\n' "$OLLAMA_OPENCODE_TEST_VERSION"
  exit 0
fi
if [ "$1" = "plugin" ]; then
  [ "$2" = "list" ] || exit 1
  printf '%s\n' 'ID                  VERSION  SOURCE' 'ollama.integration  0.1.0    test-installed-plugin'
  exit 0
fi
printf '%s' "$OLLAMA_OPENCODE_STARTUP" > "$OLLAMA_OPENCODE_TEST_OUTPUT"
`
	if err := os.WriteFile(filepath.Join(bin, "opencode"), []byte(executable), 0o755); err != nil {
		t.Fatal(err)
	}
	offline := errors.New("Ollama is offline")
	for _, tc := range []struct {
		name, version, payload string
		args                   []string
		heartbeatErr           error
	}{
		{name: "plain", version: "2.4.1"},
		{name: "selected model", version: "2.4.1", args: []string{"--model", "example-model"}, payload: `{"model":"example-model:latest"}`},
		{name: "selected cloud model", version: "2.4.1", args: []string{"--model", "example-model:cloud"}, payload: `{"model":"example-model:cloud","checkModelAccess":true}`},
		{name: "offline", version: "2.4.1", heartbeatErr: offline},
		{name: "selected model offline", version: "2.4.1", args: []string{"--model", "example-model"}, heartbeatErr: offline},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := filepath.Join(t.TempDir(), "startup")
			t.Setenv("OLLAMA_OPENCODE_TEST_OUTPUT", output)
			t.Setenv("OLLAMA_OPENCODE_TEST_VERSION", tc.version)
			checked := false
			cmd := LaunchCmd(func(*cobra.Command, []string) error {
				checked = true
				return tc.heartbeatErr
			}, nil)
			cmd.SetArgs(append([]string{"opencode"}, tc.args...))
			if err := cmd.Execute(); !checked || !errors.Is(err, tc.heartbeatErr) {
				t.Fatalf("checked server = %v, launch error = %v, want %v", checked, err, tc.heartbeatErr)
			}
			payload, err := os.ReadFile(output)
			if tc.heartbeatErr != nil {
				if !os.IsNotExist(err) {
					t.Fatalf("OpenCode started while Ollama was unavailable: %v", err)
				}
			} else if err != nil || string(payload) != tc.payload {
				t.Fatalf("OpenCode startup payload = %q, %v, want %q", payload, err, tc.payload)
			}
		})
	}
}

func TestOpenCodePluginLaunch(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")
	for _, req := range []IntegrationLaunchRequest{
		{},
		{ModelOverride: "qwen3.8:27b"},
		{ExtraArgs: []string{"--continue"}},
		{ExtraArgs: []string{"-c"}},
		{ExtraArgs: []string{"--session", "ses_123"}},
		{ExtraArgs: []string{"-s", "ses_123"}},
		{ExtraArgs: []string{"--session=ses_123"}},
	} {
		if !canLaunchOpenCodePlugin(req) {
			t.Errorf("request should use the plugin: %+v", req)
		}
	}
	for _, req := range []IntegrationLaunchRequest{
		{ExtraArgs: []string{"run", "test"}},
		{ExtraArgs: []string{"--session"}},
		{ForceConfigure: true},
		{ConfigureOnly: true},
		{Restore: true},
	} {
		if canLaunchOpenCodePlugin(req) {
			t.Errorf("request should keep the existing launcher path: %+v", req)
		}
	}
	t.Setenv("OLLAMA_HOST", "http://localhost:18181")
	if canLaunchOpenCodePlugin(IntegrationLaunchRequest{}) {
		t.Fatal("explicit Ollama host should keep the existing launcher path")
	}
}

func TestOpenCodeStartupModel(t *testing.T) {
	catalog := "opencode/llama3.2\r\nollama/llama3.2:latest\r\nollama/example/model:latest\r\nollama/example/model\r\n"
	for requested, want := range map[string]string{
		"llama3.2":             "llama3.2:latest",
		"llama3.2:other":       "",
		"example/model":        "example/model",
		"example/absent":       "",
		"example/model:latest": "example/model:latest",
	} {
		if got := openCodeStartupModel(catalog, requested); got != want {
			t.Errorf("requested model %q resolved to %q, want %q", requested, got, want)
		}
	}
}

func TestEnsureOpenCodePlugin(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell fake executable")
	}
	bin := filepath.Join(t.TempDir(), "opencode")
	executable := `#!/bin/sh
case "$1 $2" in
  'plugin list') printf '%s\n' "$OLLAMA_OPENCODE_TEST_LIST" ;;
  'plugin add')
    printf '%s' "$3" > "$OLLAMA_OPENCODE_TEST_OUTPUT"
    exit "$OLLAMA_OPENCODE_TEST_ADD_STATUS"
    ;;
  *) exit 2 ;;
esac
`
	if err := os.WriteFile(bin, []byte(executable), 0o755); err != nil {
		t.Fatal(err)
	}

	legacyContents := openCodePluginHeader + "\n// legacy server-only plugin\n"
	for _, tc := range []struct {
		name, listing, legacy, addStatus string
		wantAdd, wantArchive             bool
	}{
		{name: "installs missing package", listing: "ollama.integration.extra  1  another-package", addStatus: "0", wantAdd: true},
		{name: "reuses installed package", listing: "ollama.integration  0.1.0  " + openCodePluginSource},
		{name: "reuses local package", listing: "ollama.integration  local  /config/plugins/ollama.integration"},
		{name: "archives managed predecessor", legacy: legacyContents, addStatus: "0", wantAdd: true, wantArchive: true},
		{name: "restores predecessor after failed install", legacy: legacyContents, addStatus: "1", wantAdd: true},
		{name: "keeps user files", legacy: "// user plugin\n", addStatus: "0", wantAdd: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			configDir := filepath.Join(t.TempDir(), "OpenCode config")
			plugins := filepath.Join(configDir, "plugins")
			legacy := filepath.Join(plugins, "ollama-thinking.js")
			listing := tc.listing
			if tc.legacy != "" {
				if err := os.MkdirAll(plugins, 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(legacy, []byte(tc.legacy), 0o600); err != nil {
					t.Fatal(err)
				}
				if tc.legacy == legacyContents {
					listing = "ollama.integration  local  " + legacy
				}
			}
			output := filepath.Join(t.TempDir(), "install-argument")
			t.Setenv("OPENCODE_CONFIG_DIR", configDir)
			t.Setenv("OLLAMA_OPENCODE_TEST_LIST", listing)
			t.Setenv("OLLAMA_OPENCODE_TEST_OUTPUT", output)
			t.Setenv("OLLAMA_OPENCODE_TEST_ADD_STATUS", tc.addStatus)

			if err := ensureOpenCodePlugin(context.Background(), bin); (err != nil) != (tc.addStatus == "1") {
				t.Fatalf("ensure plugin returned %v", err)
			}
			argument, err := os.ReadFile(output)
			if tc.wantAdd {
				if err != nil || string(argument) != openCodePluginSource {
					t.Fatalf("installed source = %q, %v, want %q", argument, err, openCodePluginSource)
				}
			} else if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("plugin add was called despite the installed plugin: %v", err)
			}
			if tc.wantArchive {
				if _, err := os.Stat(legacy); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("legacy plugin remains discoverable: %v", err)
				}
				backups, err := filepath.Glob(filepath.Join(plugins, ".ollama-thinking.*.bak"))
				if err != nil || len(backups) != 1 {
					t.Fatalf("expected one archived legacy plugin, got %v, %v", backups, err)
				}
				legacy = backups[0]
			}
			if tc.legacy != "" {
				if contents, err := os.ReadFile(legacy); err != nil || string(contents) != tc.legacy {
					t.Fatalf("legacy plugin contents changed: %v", err)
				}
			}
		})
	}
}
