package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func assertKimiBinPath(t *testing.T, bin string) {
	t.Helper()
	base := strings.ToLower(filepath.Base(bin))
	if !strings.HasPrefix(base, "kimi") {
		t.Fatalf("bin = %q, want path to kimi executable", bin)
	}
}

func TestKimiIntegration(t *testing.T) {
	k := &Kimi{}

	t.Run("String", func(t *testing.T) {
		if got := k.String(); got != "Kimi Code CLI" {
			t.Errorf("String() = %q, want %q", got, "Kimi Code CLI")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = k
	})
}

func TestKimiArgs(t *testing.T) {
	k := &Kimi{}

	got := k.args(`{"foo":"bar"}`, []string{"--quiet", "--print"})
	want := []string{"--config", `{"foo":"bar"}`, "--quiet", "--print"}
	if !slices.Equal(got, want) {
		t.Fatalf("args() = %v, want %v", got, want)
	}
}

func TestWindowsPathToWSL(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		want  string
		valid bool
	}{
		{
			name:  "user profile path",
			in:    `C:\Users\parth`,
			want:  filepath.Join("/mnt", "c", "Users", "parth"),
			valid: true,
		},
		{
			name:  "path with trailing slash",
			in:    `D:\tools\bin\`,
			want:  filepath.Join("/mnt", "d", "tools", "bin"),
			valid: true,
		},
		{
			name:  "non windows path",
			in:    "/home/parth",
			valid: false,
		},
		{
			name:  "empty",
			in:    "",
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := windowsPathToWSL(tt.in)
			if !tt.valid {
				if got != "" {
					t.Fatalf("windowsPathToWSL(%q) = %q, want empty", tt.in, got)
				}
				return
			}
			if got != tt.want {
				t.Fatalf("windowsPathToWSL(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestFindKimiBinaryFallbacks(t *testing.T) {
	oldGOOS := kimiGOOS
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	t.Run("linux/ubuntu uv tool path", func(t *testing.T) {
		homeDir := t.TempDir()
		setTestHome(t, homeDir)
		t.Setenv("PATH", t.TempDir())
		kimiGOOS = "linux"

		target := filepath.Join(homeDir, ".local", "share", "uv", "tools", "kimi-cli", "bin", "kimi")
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatalf("failed to create candidate dir: %v", err)
		}
		if err := os.WriteFile(target, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatalf("failed to write kimi candidate: %v", err)
		}

		got, err := findKimiBinary()
		if err != nil {
			t.Fatalf("findKimiBinary() error = %v", err)
		}
		if got != target {
			t.Fatalf("findKimiBinary() = %q, want %q", got, target)
		}
	})

	t.Run("windows appdata uv bin", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		t.Setenv("PATH", t.TempDir())
		kimiGOOS = "windows"

		appDataDir := t.TempDir()
		t.Setenv("APPDATA", appDataDir)
		t.Setenv("LOCALAPPDATA", "")

		target := filepath.Join(appDataDir, "uv", "bin", "kimi.cmd")
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatalf("failed to create candidate dir: %v", err)
		}
		if err := os.WriteFile(target, []byte("@echo off\r\nexit /b 0\r\n"), 0o755); err != nil {
			t.Fatalf("failed to write kimi candidate: %v", err)
		}

		got, err := findKimiBinary()
		if err != nil {
			t.Fatalf("findKimiBinary() error = %v", err)
		}
		if got != target {
			t.Fatalf("findKimiBinary() = %q, want %q", got, target)
		}
	})
}

func TestValidateKimiPassthroughArgs_RejectsConflicts(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "--config", args: []string{"--config", "{}"}, want: "--config"},
		{name: "--config=", args: []string{"--config={}"}, want: "--config={"},
		{name: "--config-file", args: []string{"--config-file", "x.toml"}, want: "--config-file"},
		{name: "--config-file=", args: []string{"--config-file=x.toml"}, want: "--config-file=x.toml"},
		{name: "--model", args: []string{"--model", "foo"}, want: "--model"},
		{name: "--model=", args: []string{"--model=foo"}, want: "--model=foo"},
		{name: "-m", args: []string{"-m", "foo"}, want: "-m"},
		{name: "-m=", args: []string{"-m=foo"}, want: "-m=foo"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateKimiPassthroughArgs(tt.args)
			if err == nil {
				t.Fatalf("expected error for args %v", tt.args)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestBuildKimiInlineConfig(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	cfg, err := buildKimiInlineConfig("llama3.2", 65536)
	if err != nil {
		t.Fatalf("buildKimiInlineConfig() error = %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(cfg), &parsed); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}

	if parsed["default_model"] != "ollama" {
		t.Fatalf("default_model = %v, want ollama", parsed["default_model"])
	}

	providers, ok := parsed["providers"].(map[string]any)
	if !ok {
		t.Fatalf("providers missing or wrong type: %T", parsed["providers"])
	}
	ollamaProvider, ok := providers["ollama"].(map[string]any)
	if !ok {
		t.Fatalf("providers.ollama missing or wrong type: %T", providers["ollama"])
	}
	if ollamaProvider["type"] != "openai_legacy" {
		t.Fatalf("provider type = %v, want openai_legacy", ollamaProvider["type"])
	}
	if ollamaProvider["base_url"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("provider base_url = %v, want http://127.0.0.1:11434/v1", ollamaProvider["base_url"])
	}
	if ollamaProvider["api_key"] != "ollama" {
		t.Fatalf("provider api_key = %v, want ollama", ollamaProvider["api_key"])
	}

	models, ok := parsed["models"].(map[string]any)
	if !ok {
		t.Fatalf("models missing or wrong type: %T", parsed["models"])
	}
	ollamaModel, ok := models["ollama"].(map[string]any)
	if !ok {
		t.Fatalf("models.ollama missing or wrong type: %T", models["ollama"])
	}
	if ollamaModel["provider"] != "ollama" {
		t.Fatalf("model provider = %v, want ollama", ollamaModel["provider"])
	}
	if ollamaModel["model"] != "llama3.2" {
		t.Fatalf("model model = %v, want llama3.2", ollamaModel["model"])
	}
	if ollamaModel["max_context_size"] != float64(65536) {
		t.Fatalf("model max_context_size = %v, want 65536", ollamaModel["max_context_size"])
	}
}

func TestBuildKimiInlineConfig_UsesConnectableHostForUnspecifiedBind(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	cfg, err := buildKimiInlineConfig("llama3.2", 65536)
	if err != nil {
		t.Fatalf("buildKimiInlineConfig() error = %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(cfg), &parsed); err != nil {
		t.Fatalf("config is not valid JSON: %v", err)
	}

	providers, ok := parsed["providers"].(map[string]any)
	if !ok {
		t.Fatalf("providers missing or wrong type: %T", parsed["providers"])
	}

	ollamaProvider, ok := providers["ollama"].(map[string]any)
	if !ok {
		t.Fatalf("providers.ollama missing or wrong type: %T", providers["ollama"])
	}
	if got, _ := ollamaProvider["base_url"].(string); got != "http://127.0.0.1:11434/v1" {
		t.Fatalf("provider base_url = %q, want %q", got, "http://127.0.0.1:11434/v1")
	}
}

func TestResolveKimiMaxContextSize(t *testing.T) {
	t.Run("uses cloud limit when known", func(t *testing.T) {
		got := resolveKimiMaxContextSize("kimi-k2.5:cloud")
		if got != 262_144 {
			t.Fatalf("resolveKimiMaxContextSize() = %d, want 262144", got)
		}
	})

	t.Run("uses model show context length for local models", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/show" {
				http.NotFound(w, r)
				return
			}
			fmt.Fprint(w, `{"model_info":{"llama.context_length":131072}}`)
		}))
		defer srv.Close()
		t.Setenv("OLLAMA_HOST", srv.URL)

		got := resolveKimiMaxContextSize("llama3.2")
		if got != 131_072 {
			t.Fatalf("resolveKimiMaxContextSize() = %d, want 131072", got)
		}
	})

	t.Run("falls back to default when show fails", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		defer srv.Close()
		t.Setenv("OLLAMA_HOST", srv.URL)

		oldTimeout := kimiModelShowTimeout
		kimiModelShowTimeout = 100 * 1000 * 1000 // 100ms
		t.Cleanup(func() { kimiModelShowTimeout = oldTimeout })

		got := resolveKimiMaxContextSize("llama3.2")
		if got != kimiDefaultMaxContextSize {
			t.Fatalf("resolveKimiMaxContextSize() = %d, want %d", got, kimiDefaultMaxContextSize)
		}
	})
}

func TestKimiRun_RejectsConflictingArgsBeforeInstall(t *testing.T) {
	k := &Kimi{}

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect install prompt, got %q", prompt)
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	err := k.Run("llama3.2", []string{"--model", "other"})
	if err == nil || !strings.Contains(err.Error(), "--model") {
		t.Fatalf("expected conflict error mentioning --model, got %v", err)
	}
}

func TestKimiRun_PassesInlineConfigAndExtraArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	logPath := filepath.Join(tmpDir, "kimi-args.log")
	script := fmt.Sprintf(`#!/bin/sh
for arg in "$@"; do
  printf "%%s\n" "$arg" >> %q
done
exit 0
`, logPath)
	if err := os.WriteFile(filepath.Join(tmpDir, "kimi"), []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake kimi: %v", err)
	}
	t.Setenv("PATH", tmpDir)

	srv := httptest.NewServer(http.NotFoundHandler())
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	k := &Kimi{}
	if err := k.Run("llama3.2", []string{"--quiet", "--print"}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read args log: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) < 4 {
		t.Fatalf("expected at least 4 args, got %v", lines)
	}
	if lines[0] != "--config" {
		t.Fatalf("first arg = %q, want --config", lines[0])
	}

	var cfg map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &cfg); err != nil {
		t.Fatalf("config arg is not valid JSON: %v", err)
	}
	providers := cfg["providers"].(map[string]any)
	ollamaProvider := providers["ollama"].(map[string]any)
	if ollamaProvider["type"] != "openai_legacy" {
		t.Fatalf("provider type = %v, want openai_legacy", ollamaProvider["type"])
	}

	if lines[2] != "--quiet" || lines[3] != "--print" {
		t.Fatalf("extra args = %v, want [--quiet --print]", lines[2:])
	}
}

func TestEnsureKimiInstalled(t *testing.T) {
	oldGOOS := kimiGOOS
	t.Cleanup(func() { kimiGOOS = oldGOOS })

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
		writeFakeBinary(t, tmpDir, "kimi")
		kimiGOOS = runtime.GOOS

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		bin, err := ensureKimiInstalled()
		if err != nil {
			t.Fatalf("ensureKimiInstalled() error = %v", err)
		}
		assertKimiBinPath(t, bin)
	})

	t.Run("missing dependencies", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		kimiGOOS = "linux"

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		_, err := ensureKimiInstalled()
		if err == nil || !strings.Contains(err.Error(), "required dependencies are missing") {
			t.Fatalf("expected missing dependency error, got %v", err)
		}
	})

	t.Run("missing and user declines install", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		writeFakeBinary(t, tmpDir, "curl")
		writeFakeBinary(t, tmpDir, "bash")
		kimiGOOS = "linux"

		withConfirm(t, func(prompt string) (bool, error) {
			if !strings.Contains(prompt, "Kimi is not installed.") {
				t.Fatalf("unexpected prompt: %q", prompt)
			}
			return false, nil
		})

		_, err := ensureKimiInstalled()
		if err == nil || !strings.Contains(err.Error(), "installation cancelled") {
			t.Fatalf("expected cancellation error, got %v", err)
		}
	})

	t.Run("missing and user confirms install succeeds", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		kimiGOOS = "linux"

		writeFakeBinary(t, tmpDir, "curl")

		installLog := filepath.Join(tmpDir, "bash.log")
		kimiPath := filepath.Join(tmpDir, "kimi")
		bashScript := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "-c" ]; then
  /bin/cat > %q <<'EOS'
#!/bin/sh
exit 0
EOS
  /bin/chmod +x %q
fi
exit 0
`, installLog, kimiPath, kimiPath)
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte(bashScript), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureKimiInstalled()
		if err != nil {
			t.Fatalf("ensureKimiInstalled() error = %v", err)
		}
		assertKimiBinPath(t, bin)

		logData, err := os.ReadFile(installLog)
		if err != nil {
			t.Fatalf("failed to read install log: %v", err)
		}
		if !strings.Contains(string(logData), "https://code.kimi.com/install.sh") {
			t.Fatalf("expected install.sh command in log, got:\n%s", string(logData))
		}
	})

	t.Run("install succeeds and kimi is in home local bin without PATH update", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		homeDir := t.TempDir()
		setTestHome(t, homeDir)

		tmpBin := t.TempDir()
		t.Setenv("PATH", tmpBin)
		kimiGOOS = "linux"
		writeFakeBinary(t, tmpBin, "curl")

		installedKimi := filepath.Join(homeDir, ".local", "bin", "kimi")
		bashScript := fmt.Sprintf(`#!/bin/sh
if [ "$1" = "-c" ]; then
  /bin/mkdir -p %q
  /bin/cat > %q <<'EOS'
#!/bin/sh
exit 0
EOS
  /bin/chmod +x %q
fi
exit 0
`, filepath.Dir(installedKimi), installedKimi, installedKimi)
		if err := os.WriteFile(filepath.Join(tmpBin, "bash"), []byte(bashScript), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureKimiInstalled()
		if err != nil {
			t.Fatalf("ensureKimiInstalled() error = %v", err)
		}
		if bin != installedKimi {
			t.Fatalf("bin = %q, want %q", bin, installedKimi)
		}
	})

	t.Run("install command fails", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		kimiGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		_, err := ensureKimiInstalled()
		if err == nil || !strings.Contains(err.Error(), "failed to install kimi") {
			t.Fatalf("expected install failure error, got %v", err)
		}
	})

	t.Run("install succeeds but binary missing on PATH", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		kimiGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		_, err := ensureKimiInstalled()
		if err == nil || !strings.Contains(err.Error(), "binary was not found on PATH") {
			t.Fatalf("expected PATH guidance error, got %v", err)
		}
	})
}

func TestKimiInstallerCommand(t *testing.T) {
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
			wantParts: []string{"-c", "install.sh"},
		},
		{
			name:      "darwin",
			goos:      "darwin",
			wantBin:   "bash",
			wantParts: []string{"-c", "install.sh"},
		},
		{
			name:      "windows",
			goos:      "windows",
			wantBin:   "powershell",
			wantParts: []string{"-Command", "install.ps1"},
		},
		{
			name:    "unsupported",
			goos:    "freebsd",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin, args, err := kimiInstallerCommand(tt.goos)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("kimiInstallerCommand() error = %v", err)
			}
			if bin != tt.wantBin {
				t.Fatalf("bin = %q, want %q", bin, tt.wantBin)
			}
			joined := strings.Join(args, " ")
			for _, part := range tt.wantParts {
				if !strings.Contains(joined, part) {
					t.Fatalf("args %q missing %q", joined, part)
				}
			}
		})
	}
}
