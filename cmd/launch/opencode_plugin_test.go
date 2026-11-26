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

	for _, tc := range []struct {
		name, listing, addStatus string
		wantAdd                  bool
	}{
		{name: "installs missing package", listing: "ollama.integration.extra  1  another-package", addStatus: "0", wantAdd: true},
		{name: "reuses installed package", listing: "ollama.integration  0.1.0  " + openCodePluginSource},
		{name: "reuses resolved package on startup", listing: "-  4012112  " + openCodePluginSource},
		{name: "reuses local package", listing: "ollama.integration  local  /config/plugins/ollama.integration"},
		{name: "surfaces install failure", listing: "No plugins found", addStatus: "1", wantAdd: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := filepath.Join(t.TempDir(), "install-argument")
			t.Setenv("OLLAMA_OPENCODE_TEST_LIST", tc.listing)
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
		})
	}
}
