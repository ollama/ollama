package launch

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestOpenCodeIntegration(t *testing.T) {
	o := &OpenCode{}

	t.Run("String", func(t *testing.T) {
		if got := o.String(); got != "OpenCode" {
			t.Errorf("String() = %q, want %q", got, "OpenCode")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = o
	})
}

func TestLookupCloudModelLimit(t *testing.T) {
	tests := []struct {
		name        string
		wantOK      bool
		wantContext int
		wantOutput  int
	}{
		{"glm-4.7", false, 0, 0},
		{"glm-4.7:cloud", true, 202_752, 131_072},
		{"glm-5:cloud", true, 202_752, 131_072},
		{"glm-5.1:cloud", true, 202_752, 131_072},
		{"gemma4:31b-cloud", true, 262_144, 131_072},
		{"gpt-oss:120b-cloud", true, 131_072, 131_072},
		{"gpt-oss:20b-cloud", true, 131_072, 131_072},
		{"kimi-k2.5", false, 0, 0},
		{"kimi-k2.5:cloud", true, 262_144, 262_144},
		{"deepseek-v3.2", false, 0, 0},
		{"deepseek-v3.2:cloud", true, 163_840, 65_536},
		{"qwen3.5", false, 0, 0},
		{"qwen3.5:cloud", true, 262_144, 32_768},
		{"qwen3-coder:480b", false, 0, 0},
		{"qwen3-coder:480b:cloud", true, 262_144, 65_536},
		{"qwen3-coder-next:cloud", true, 262_144, 32_768},
		{"llama3.2", false, 0, 0},
		{"unknown-model:cloud", false, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, ok := lookupCloudModelLimit(tt.name)
			if ok != tt.wantOK {
				t.Errorf("lookupCloudModelLimit(%q) ok = %v, want %v", tt.name, ok, tt.wantOK)
			}
			if ok {
				if l.Context != tt.wantContext {
					t.Errorf("context = %d, want %d", l.Context, tt.wantContext)
				}
				if l.Output != tt.wantOutput {
					t.Errorf("output = %d, want %d", l.Output, tt.wantOutput)
				}
			}
		})
	}
}

func TestFindOpenCode(t *testing.T) {
	oldGOOS := openCodeGOOS
	t.Cleanup(func() { openCodeGOOS = oldGOOS })

	t.Run("fallback to ~/.opencode/bin", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Ensure opencode is not on PATH
		t.Setenv("PATH", tmpDir)
		openCodeGOOS = runtime.GOOS

		// Without the fallback binary, findOpenCode should fail
		if _, ok := findOpenCode(); ok {
			t.Fatal("findOpenCode should fail when binary is not on PATH or in fallback location")
		}

		// Create a fake binary at the curl install fallback location
		binDir := filepath.Join(tmpDir, ".opencode", "bin")
		os.MkdirAll(binDir, 0o755)
		name := "opencode"
		if runtime.GOOS == "windows" {
			name = "opencode.exe"
		}
		fakeBin := filepath.Join(binDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)

		// Now findOpenCode should succeed via fallback
		path, ok := findOpenCode()
		if !ok {
			t.Fatal("findOpenCode should succeed with fallback binary")
		}
		if path != fakeBin {
			t.Errorf("findOpenCode = %q, want %q", path, fakeBin)
		}
	})
}

func TestEnsureOpenCodeInstalled(t *testing.T) {
	oldGOOS := openCodeGOOS
	t.Cleanup(func() { openCodeGOOS = oldGOOS })

	withConfirm := func(t *testing.T, fn func(prompt string) (bool, error)) {
		t.Helper()
		oldConfirm := DefaultConfirmPrompt
		DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
			return fn(prompt)
		}
		t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
	}

	t.Run("already installed", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		openCodeGOOS = runtime.GOOS
		writeFakeBinary(t, tmpDir, "opencode")

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		bin, err := ensureOpenCodeInstalled()
		if err != nil {
			t.Fatalf("ensureOpenCodeInstalled() error = %v", err)
		}
		if filepath.Base(bin) == "" {
			t.Fatalf("expected opencode binary path, got %q", bin)
		}
	})

	t.Run("missing dependencies", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		t.Setenv("PATH", t.TempDir())
		openCodeGOOS = "linux"

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		_, err := ensureOpenCodeInstalled()
		if err == nil || !strings.Contains(err.Error(), "required dependencies are missing") {
			t.Fatalf("expected missing dependency error, got %v", err)
		}
	})

	t.Run("missing and user declines install", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		openCodeGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")
		writeFakeBinary(t, tmpDir, "bash")

		withConfirm(t, func(prompt string) (bool, error) {
			if !strings.Contains(prompt, "OpenCode is not installed.") {
				t.Fatalf("unexpected prompt: %q", prompt)
			}
			return false, nil
		})

		_, err := ensureOpenCodeInstalled()
		if err == nil || !strings.Contains(err.Error(), "installation cancelled") {
			t.Fatalf("expected cancellation error, got %v", err)
		}
	})

	t.Run("missing and user confirms unix install succeeds", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		homeDir := t.TempDir()
		setTestHome(t, homeDir)
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		openCodeGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")

		installLog := filepath.Join(tmpDir, "bash.log")
		opencodePath := filepath.Join(homeDir, ".opencode", "bin", "opencode")
		bashScript := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "-c" ]; then
  /bin/mkdir -p %q
  /bin/cat > %q <<'EOS'
#!/bin/sh
exit 0
EOS
  /bin/chmod +x %q
fi
exit 0
`, installLog, filepath.Dir(opencodePath), opencodePath, opencodePath)
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte(bashScript), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureOpenCodeInstalled()
		if err != nil {
			t.Fatalf("ensureOpenCodeInstalled() error = %v", err)
		}
		if bin != opencodePath {
			t.Fatalf("bin = %q, want %q", bin, opencodePath)
		}

		logData, err := os.ReadFile(installLog)
		if err != nil {
			t.Fatalf("failed to read install log: %v", err)
		}
		if !strings.Contains(string(logData), openCodeInstallScript) {
			t.Fatalf("expected opencode install script in log, got:\n%s", string(logData))
		}
	})

	t.Run("missing and user confirms windows install succeeds", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		homeDir := t.TempDir()
		setTestHome(t, homeDir)
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		openCodeGOOS = "windows"

		installLog := filepath.Join(tmpDir, "npm.log")
		opencodePath := filepath.Join(homeDir, ".opencode", "bin", "opencode.exe")
		npmScript := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
/bin/mkdir -p %q
/bin/cat > %q <<'EOS'
@echo off
exit /b 0
EOS
/bin/chmod +x %q
exit 0
`, installLog, filepath.Dir(opencodePath), opencodePath, opencodePath)
		if err := os.WriteFile(filepath.Join(tmpDir, "npm"), []byte(npmScript), 0o755); err != nil {
			t.Fatalf("failed to write fake npm: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureOpenCodeInstalled()
		if err != nil {
			t.Fatalf("ensureOpenCodeInstalled() error = %v", err)
		}
		if bin != opencodePath {
			t.Fatalf("bin = %q, want %q", bin, opencodePath)
		}

		logData, err := os.ReadFile(installLog)
		if err != nil {
			t.Fatalf("failed to read install log: %v", err)
		}
		if !strings.Contains(string(logData), "install -g @opencode/cli@2") {
			t.Fatalf("expected npm install command in log, got:\n%s", string(logData))
		}
	})

	t.Run("install command fails", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		openCodeGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		_, err := ensureOpenCodeInstalled()
		if err == nil || !strings.Contains(err.Error(), "failed to install opencode") {
			t.Fatalf("expected install failure error, got %v", err)
		}
	})
}

func TestOpenCodeInstallerCommand(t *testing.T) {
	tests := []struct {
		name      string
		goos      string
		wantBin   string
		wantParts []string
		wantErr   bool
	}{
		{
			name:      "linux",
			goos:      "linux",
			wantBin:   "bash",
			wantParts: []string{"-c", "set -o pipefail", "https://opencode.ai/v2/install"},
		},
		{
			name:      "darwin",
			goos:      "darwin",
			wantBin:   "bash",
			wantParts: []string{"-c", "set -o pipefail", "https://opencode.ai/v2/install"},
		},
		{
			name:      "windows",
			goos:      "windows",
			wantBin:   "npm",
			wantParts: []string{"install", "-g", "@opencode/cli@2"},
		},
		{
			name:    "unsupported",
			goos:    "plan9",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin, args, err := openCodeInstallerCommand(tt.goos)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("openCodeInstallerCommand() error = %v", err)
			}
			if bin != tt.wantBin {
				t.Fatalf("bin = %q, want %q", bin, tt.wantBin)
			}
			joined := strings.Join(args, " ")
			for _, want := range tt.wantParts {
				if !strings.Contains(joined, want) {
					t.Fatalf("args %q missing %q", joined, want)
				}
			}
		})
	}
}
