package launch

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func stubCodexAppProcessCommands(t *testing.T, commands map[string]string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("test commands require a POSIX shell")
	}

	binDir := t.TempDir()
	for name, script := range commands {
		if err := os.WriteFile(filepath.Join(binDir, name), []byte("#!/bin/sh\n"+script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// Never fall back to the real osascript or another system command.
	t.Setenv("PATH", binDir)
}

func TestDefaultCodexAppIsRunningProcesses(t *testing.T) {
	otherPID := os.Getpid() + 1
	for _, tt := range []struct {
		name     string
		goos     string
		output   string
		exitCode string
		want     bool
	}{
		{name: "no processes", goos: "darwin"},
		{name: "apps stopped or absent", goos: "darwin", output: "1 /sbin/launchd\n100 /Applications/Ollama.app/Contents/MacOS/Ollama\n"},
		{name: "ChatGPT main", goos: "darwin", output: fmt.Sprintf("%d /Applications/ChatGPT.app/Contents/MacOS/ChatGPT\n", otherPID), want: true},
		{name: "Codex main", goos: "darwin", output: fmt.Sprintf("%d /Applications/Codex.app/Contents/MacOS/Codex\n", otherPID), want: true},
		{name: "ChatGPT app server", goos: "darwin", output: fmt.Sprintf("%d /Applications/ChatGPT.app/Contents/Resources/codex app-server --analytics-default-enabled\n", otherPID), want: true},
		{name: "Codex app server", goos: "darwin", output: fmt.Sprintf("%d /Applications/Codex.app/Contents/Resources/codex app-server --analytics-default-enabled\n", otherPID), want: true},
		{name: "ChatGPT helper only", goos: "darwin", output: "105 /Applications/ChatGPT.app/Contents/Frameworks/ChatGPT Helper.app/Contents/MacOS/ChatGPT Helper\n"},
		{name: "Codex helpers only", goos: "darwin", output: "106 /Applications/Codex.app/Contents/Frameworks/Codex Helper.app/Contents/MacOS/Codex Helper\n107 /Applications/Codex.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler\n"},
		{name: "current process", goos: "darwin", output: fmt.Sprintf("%d /Applications/ChatGPT.app/Contents/MacOS/ChatGPT\n", os.Getpid())},
		{name: "malformed process list", goos: "darwin", output: "\n108\ninvalid /Applications/Codex.app/Contents/MacOS/Codex\n"},
		{name: "failed process lookup", goos: "darwin", output: fmt.Sprintf("%d /Applications/ChatGPT.app/Contents/MacOS/ChatGPT\n", otherPID), exitCode: "1"},
		{name: "Windows stopped", goos: "windows"},
		{name: "Windows running", goos: "windows", output: fmt.Sprintf("%d\r\n", otherPID), want: true},
		{name: "Windows current process", goos: "windows", output: fmt.Sprintf("%d\r\n", os.Getpid())},
		{name: "unsupported platform", goos: "linux", output: fmt.Sprintf("%d /Applications/Codex.app/Contents/MacOS/Codex\n", otherPID)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			withCodexAppPlatform(t, tt.goos)
			command := "ps"
			if tt.goos == "windows" {
				command = "powershell.exe"
			}
			stubCodexAppProcessCommands(t, map[string]string{
				command: "printf '%s' \"$OLLAMA_TEST_CODEX_APP_PROCESSES\"\nexit \"${OLLAMA_TEST_CODEX_APP_EXIT_CODE:-0}\"\n",
			})
			t.Setenv("OLLAMA_TEST_CODEX_APP_PROCESSES", tt.output)
			t.Setenv("OLLAMA_TEST_CODEX_APP_EXIT_CODE", tt.exitCode)

			if got := defaultCodexAppIsRunning(); got != tt.want {
				t.Fatalf("defaultCodexAppIsRunning() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDefaultCodexAppIsRunningDoesNotCallAppleScript(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	marker := filepath.Join(t.TempDir(), "osascript-called")
	t.Setenv("OLLAMA_TEST_CODEX_APP_OSASCRIPT_MARKER", marker)
	stubCodexAppProcessCommands(t, map[string]string{
		"ps": "exit 0\n",
		// Simulate slow System Events without contacting the real application.
		"osascript": "printf called > \"$OLLAMA_TEST_CODEX_APP_OSASCRIPT_MARKER\"\nexec /bin/sleep 1\n",
	})

	if defaultCodexAppIsRunning() {
		t.Fatal("reported a running app with no matching processes")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("running-state detection invoked osascript: %v", err)
	}
}

func TestCodexAppSlowProcessLookupPreservesRestartConfirmation(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	withCodexAppProcessHooks(t, defaultCodexAppIsRunning,
		func() error { t.Fatal("quit app without restart confirmation"); return nil },
		func() error { t.Fatal("open app without restart confirmation"); return nil },
	)
	t.Setenv("OLLAMA_TEST_CODEX_APP_PROCESSES", fmt.Sprintf("%d /Applications/ChatGPT.app/Contents/MacOS/ChatGPT", os.Getpid()+1))
	stubCodexAppProcessCommands(t, map[string]string{
		"ps": "/bin/sleep 2\nprintf '%s\\n' \"$OLLAMA_TEST_CODEX_APP_PROCESSES\"\n",
	})

	changed := false
	err := codexAppApplyProfileFromDesktop(func() error {
		changed = true
		return nil
	}, false, false)
	if !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Errorf("profile update error = %v, want restart confirmation", err)
	}
	if changed {
		t.Fatal("profile changed without restart confirmation after a slow process lookup")
	}
}
