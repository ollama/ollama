package launch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestLaunchCmdOpenCode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell fake executable")
	}
	setTestHome(t, t.TempDir())
	bin := t.TempDir()
	t.Setenv("PATH", bin)
	t.Setenv("OLLAMA_HOST", "")
	t.Setenv(openCodeStartupEnv, "")
	t.Setenv("OPENCODE_CONFIG_CONTENT", "")
	executable := `#!/bin/sh
printf '%s\n' "$*" >> "$OLLAMA_OPENCODE_TEST_CALLS"
if [ "$1" = "--version" ]; then
  printf '%s\n' "$OLLAMA_OPENCODE_TEST_VERSION"
  exit 0
fi
if [ "$1" = "plugin" ]; then
  [ "$2" = "list" ] || exit 1
  printf '%s\n' 'ollama.integration  0.1.1  test-installed-plugin'
  exit 0
fi
[ -z "$OPENCODE_CONFIG_CONTENT" ] || exit 3
printf '%s' "$OLLAMA_OPENCODE_STARTUP" > "$OLLAMA_OPENCODE_TEST_OUTPUT"
printf '%s\n' "$@" > "$OLLAMA_OPENCODE_TEST_OUTPUT.args"
`
	if err := os.WriteFile(filepath.Join(bin, "opencode"), []byte(executable), 0o755); err != nil {
		t.Fatal(err)
	}
	offline := errors.New("Ollama is offline")
	for _, tc := range []struct {
		name, version, payload, wantArgs, wantErr string
		args                                      []string
		heartbeatErr                              error
		setupOnly, headless                       bool
	}{
		{name: "plain"},
		{name: "selected model", args: []string{"--model", "example-model"}, payload: `{"model":"example-model:latest"}`},
		{name: "selected cloud model", args: []string{"--model", "example-model:cloud"}, payload: `{"model":"example-model:cloud","checkModelAccess":true}`},
		{name: "auto approve", args: []string{"--yes", "--model", "example-model"}, payload: `{"model":"example-model:latest","autoApprove":true}`},
		{name: "resume", args: []string{"--", "--continue"}, wantArgs: "--continue"},
		{name: "headless with model", args: []string{"--yes", "--model", "example-model", "--", "run", "hello"}, wantArgs: "run\n--model\nollama/example-model:latest\nhello", headless: true},
		{name: "configure only", args: []string{"--config"}, setupOnly: true},
		{name: "configure with model", args: []string{"--config", "--model", "example-model"}, wantErr: "--config installs"},
		{name: "older version", version: "1.4.1", wantErr: "requires OpenCode 2.4.1"},
		{name: "offline", heartbeatErr: offline},
		{name: "selected model offline", args: []string{"--model", "example-model"}, heartbeatErr: offline},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withInteractiveSession(t, !tc.headless)
			dir := t.TempDir()
			output := filepath.Join(dir, "startup")
			calls := filepath.Join(dir, "calls")
			t.Setenv("OLLAMA_OPENCODE_TEST_OUTPUT", output)
			t.Setenv("OLLAMA_OPENCODE_TEST_CALLS", calls)
			version := tc.version
			if version == "" {
				version = "2.4.1"
			}
			t.Setenv("OLLAMA_OPENCODE_TEST_VERSION", version)
			checked := false
			cmd := LaunchCmd(func(*cobra.Command, []string) error {
				checked = true
				return tc.heartbeatErr
			}, nil)
			cmd.SetArgs(append([]string{"opencode"}, tc.args...))
			err := cmd.Execute()
			if !checked {
				t.Fatal("launcher skipped the Ollama server check")
			}
			if tc.heartbeatErr != nil {
				if !errors.Is(err, tc.heartbeatErr) {
					t.Fatalf("launch error = %v, want %v", err, tc.heartbeatErr)
				}
				if _, err := os.Stat(calls); !os.IsNotExist(err) {
					t.Fatalf("launcher invoked OpenCode after the server check failed: %v", err)
				}
				return
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("launch error = %v, want %q", err, tc.wantErr)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			payload, err := os.ReadFile(output)
			if tc.setupOnly || tc.wantErr != "" {
				if !os.IsNotExist(err) {
					t.Fatalf("OpenCode started unexpectedly: %v", err)
				}
				if tc.setupOnly {
					if data, err := os.ReadFile(calls); err != nil || !strings.Contains(string(data), "plugin list") {
						t.Fatalf("setup did not check for the OpenCode plugin: %q, %v", data, err)
					}
				}
				return
			}
			if err != nil || string(payload) != tc.payload {
				t.Fatalf("OpenCode startup payload = %q, %v, want %q", payload, err, tc.payload)
			}
			if args, err := os.ReadFile(output + ".args"); err != nil || strings.TrimSpace(string(args)) != tc.wantArgs {
				t.Fatalf("OpenCode arguments = %q, %v, want %q", args, err, tc.wantArgs)
			}
		})
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
