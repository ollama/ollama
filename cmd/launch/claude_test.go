package launch

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestClaudeIntegration(t *testing.T) {
	c := &Claude{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Claude Code" {
			t.Errorf("String() = %q, want %q", got, "Claude Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
}

func TestClaudeFindPath(t *testing.T) {
	c := &Claude{}

	t.Run("finds claude in PATH", func(t *testing.T) {
		tmpDir := t.TempDir()
		name := "claude"
		if runtime.GOOS == "windows" {
			name = "claude.exe"
		}
		fakeBin := filepath.Join(tmpDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)
		t.Setenv("PATH", tmpDir)

		got, err := c.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fakeBin {
			t.Errorf("findPath() = %q, want %q", got, fakeBin)
		}
	})

	t.Run("falls back to ~/.claude/local/claude", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", t.TempDir()) // empty dir, no claude binary

		name := "claude"
		if runtime.GOOS == "windows" {
			name = "claude.exe"
		}
		fallback := filepath.Join(tmpDir, ".claude", "local", name)
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := c.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("falls back to ~/.local/bin/claude", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", t.TempDir()) // empty dir, no claude binary

		name := "claude"
		if runtime.GOOS == "windows" {
			name = "claude.exe"
		}
		fallback := filepath.Join(tmpDir, ".local", "bin", name)
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := c.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("returns error when neither PATH nor fallback exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", t.TempDir()) // empty dir, no claude binary

		_, err := c.findPath()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestEnsureClaudeInstalled(t *testing.T) {
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
		writeFakeBinary(t, tmpDir, "claude")

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		bin, err := ensureClaudeInstalled()
		if err != nil {
			t.Fatalf("ensureClaudeInstalled() error = %v", err)
		}
		if filepath.Base(bin) != "claude" && filepath.Base(bin) != "claude.cmd" {
			t.Fatalf("bin = %q, want claude binary", bin)
		}
	})

	t.Run("missing dependencies", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		t.Setenv("PATH", t.TempDir())

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		_, err := ensureClaudeInstalled()
		if err == nil || !strings.Contains(err.Error(), "required dependencies are missing") {
			t.Fatalf("expected missing dependency error, got %v", err)
		}
	})

	t.Run("missing and user declines install", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		writeClaudeInstallerDeps(t, tmpDir)

		withConfirm(t, func(prompt string) (bool, error) {
			if prompt != "Claude Code is not installed. Install now?" {
				t.Fatalf("unexpected prompt: %q", prompt)
			}
			return false, nil
		})

		_, err := ensureClaudeInstalled()
		if err == nil || !strings.Contains(err.Error(), "installation cancelled") {
			t.Fatalf("expected cancellation error, got %v", err)
		}
	})

	t.Run("missing and user confirms install succeeds", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		homeDir := t.TempDir()
		setTestHome(t, homeDir)
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)

		writeFakeBinary(t, tmpDir, "curl")

		installLog := filepath.Join(tmpDir, "bash.log")
		installedClaude := filepath.Join(homeDir, ".local", "bin", "claude")
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
`, installLog, filepath.Dir(installedClaude), installedClaude, installedClaude)
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte(bashScript), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureClaudeInstalled()
		if err != nil {
			t.Fatalf("ensureClaudeInstalled() error = %v", err)
		}
		if bin != installedClaude {
			t.Fatalf("bin = %q, want %q", bin, installedClaude)
		}

		logData, err := os.ReadFile(installLog)
		if err != nil {
			t.Fatalf("failed to read install log: %v", err)
		}
		if !strings.Contains(string(logData), "https://claude.ai/install.sh") {
			t.Fatalf("expected install.sh command in log, got:\n%s", string(logData))
		}
	})

	t.Run("install command fails", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		setTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		writeFakeBinary(t, tmpDir, "curl")
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		_, err := ensureClaudeInstalled()
		if err == nil || !strings.Contains(err.Error(), "failed to install claude") {
			t.Fatalf("expected install failure error, got %v", err)
		}
	})
}

func writeClaudeInstallerDeps(t *testing.T, dir string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		writeFakeBinary(t, dir, "powershell")
		return
	}
	writeFakeBinary(t, dir, "curl")
	writeFakeBinary(t, dir, "bash")
}

func TestClaudeInstallerCommand(t *testing.T) {
	tests := []struct {
		name    string
		goos    string
		wantBin string
		want    string
		wantErr string
	}{
		{
			name:    "unix",
			goos:    "linux",
			wantBin: "bash",
			want:    "curl -fsSL https://claude.ai/install.sh | bash",
		},
		{
			name:    "macos",
			goos:    "darwin",
			wantBin: "bash",
			want:    "curl -fsSL https://claude.ai/install.sh | bash",
		},
		{
			name:    "windows",
			goos:    "windows",
			wantBin: "powershell",
			want:    "irm https://claude.ai/install.ps1 | iex",
		},
		{
			name:    "unsupported",
			goos:    "plan9",
			wantErr: "unsupported platform",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin, args, err := claudeInstallerCommand(tt.goos)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("claudeInstallerCommand() error = %v", err)
			}
			if bin != tt.wantBin {
				t.Fatalf("bin = %q, want %q", bin, tt.wantBin)
			}
			if !slices.Contains(args, tt.want) {
				t.Fatalf("args = %v, want command containing %q", args, tt.want)
			}
		})
	}
}

func TestClaudeArgs(t *testing.T) {
	c := &Claude{}

	tests := []struct {
		name       string
		model      string
		args       []string
		wantPrefix []string
		wantExtra  []string
	}{
		{"with model", "llama3.2", nil, []string{"--model", "llama3.2"}, nil},
		{"empty model", "", nil, nil, nil},
		{"with model and verbose", "llama3.2", []string{"--verbose"}, []string{"--model", "llama3.2"}, []string{"--verbose"}},
		{"empty model with help", "", []string{"--help"}, nil, []string{"--help"}},
		{"with allowed tools", "llama3.2", []string{"--allowedTools", "Read,Write,Bash"}, []string{"--model", "llama3.2"}, []string{"--allowedTools", "Read,Write,Bash"}},
		{"with channels", "llama3.2", []string{"--channels", "plugin:telegram@claude-plugins-official"}, []string{"--model", "llama3.2"}, []string{"--channels", "plugin:telegram@claude-plugins-official"}},
		{"user settings remain last", "llama3.2", []string{"--settings", `{"env":{"CUSTOM":"1"}}`}, []string{"--model", "llama3.2"}, []string{"--settings", `{"env":{"CUSTOM":"1"}}`}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := c.args(tt.model, tt.args)
			if err != nil {
				t.Fatalf("args(%q, %v) error = %v", tt.model, tt.args, err)
			}
			if !slices.Equal(got[:len(tt.wantPrefix)], tt.wantPrefix) {
				t.Fatalf("args(%q, %v) prefix = %v, want %v", tt.model, tt.args, got[:len(tt.wantPrefix)], tt.wantPrefix)
			}
			settingsIndex := len(tt.wantPrefix)
			if got[settingsIndex] != "--settings" {
				t.Fatalf("args(%q, %v)[%d] = %q, want --settings", tt.model, tt.args, settingsIndex, got[settingsIndex])
			}
			var settings struct {
				Env map[string]string `json:"env"`
			}
			if err := json.Unmarshal([]byte(got[settingsIndex+1]), &settings); err != nil {
				t.Fatalf("settings are not valid JSON: %v", err)
			}
			if settings.Env["ANTHROPIC_BASE_URL"] != envconfig.Host().String() {
				t.Errorf("ANTHROPIC_BASE_URL = %q, want %q", settings.Env["ANTHROPIC_BASE_URL"], envconfig.Host().String())
			}
			if settings.Env["ANTHROPIC_DEFAULT_SONNET_MODEL"] != tt.model {
				t.Errorf("ANTHROPIC_DEFAULT_SONNET_MODEL = %q, want %q", settings.Env["ANTHROPIC_DEFAULT_SONNET_MODEL"], tt.model)
			}
			if gotExtra := got[settingsIndex+2:]; !slices.Equal(gotExtra, tt.wantExtra) {
				t.Errorf("args(%q, %v) extra = %v, want %v", tt.model, tt.args, gotExtra, tt.wantExtra)
			}
		})
	}
}

func TestClaudeEnvVars(t *testing.T) {
	c := &Claude{}

	envMap := func(envs []string) map[string]string {
		m := make(map[string]string)
		for _, e := range envs {
			k, v, _ := strings.Cut(e, "=")
			m[k] = v
		}
		return m
	}

	got := envMap(c.envVars("llama3.2"))
	for key, want := range map[string]string{
		"ANTHROPIC_BASE_URL":                  envconfig.Host().String(),
		"ANTHROPIC_API_KEY":                   "",
		"ANTHROPIC_AUTH_TOKEN":                "ollama",
		"CLAUDE_CODE_ATTRIBUTION_HEADER":      "0",
		"CLAUDE_CODE_TOTAL_TOKENS_REMINDER":   "off",
		"DISABLE_ERROR_REPORTING":             "1",
		"DISABLE_FEEDBACK_COMMAND":            "1",
		"CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
		"CLAUDE_CODE_USE_BEDROCK":             "",
		"CLAUDE_CODE_USE_VERTEX":              "",
		"CLAUDE_CODE_USE_FOUNDRY":             "",
		"CLAUDE_CODE_USE_MANTLE":              "",
		"ANTHROPIC_DEFAULT_OPUS_MODEL":        "llama3.2",
		"ANTHROPIC_DEFAULT_SONNET_MODEL":      "llama3.2",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL":       "llama3.2",
		"CLAUDE_CODE_SUBAGENT_MODEL":          "llama3.2",
	} {
		if got[key] != want {
			t.Errorf("%s = %q, want %q", key, got[key], want)
		}
	}

	// Both variables disable Claude Code feature-flag evaluation, which keeps Channels unavailable.
	for _, key := range []string{"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_TELEMETRY"} {
		if _, ok := got[key]; ok {
			t.Errorf("%s must not be set by Ollama", key)
		}
	}
}

func TestClaudeSettingsMatchEnvVars(t *testing.T) {
	c := &Claude{}

	for _, model := range []string{"llama3.2", "glm-5:cloud"} {
		t.Run(model, func(t *testing.T) {
			encoded, err := c.settings(model)
			if err != nil {
				t.Fatalf("settings(%q) error = %v", model, err)
			}

			var settings struct {
				Env map[string]string `json:"env"`
			}
			if err := json.Unmarshal([]byte(encoded), &settings); err != nil {
				t.Fatalf("settings(%q) are not valid JSON: %v", model, err)
			}

			want := make(map[string]string)
			for _, entry := range c.envVars(model) {
				key, value, ok := strings.Cut(entry, "=")
				if ok {
					want[key] = value
				}
			}
			if !maps.Equal(settings.Env, want) {
				t.Errorf("settings env = %v, want %v", settings.Env, want)
			}
		})
	}
}

func TestClaudeModelEnvVars(t *testing.T) {
	c := &Claude{}

	envMap := func(envs []string) map[string]string {
		m := make(map[string]string)
		for _, e := range envs {
			k, v, _ := strings.Cut(e, "=")
			m[k] = v
		}
		return m
	}

	t.Run("maps all Claude model env vars to the provided model", func(t *testing.T) {
		got := envMap(c.modelEnvVars("llama3.2"))
		if got["ANTHROPIC_DEFAULT_OPUS_MODEL"] != "llama3.2" {
			t.Errorf("OPUS = %q, want llama3.2", got["ANTHROPIC_DEFAULT_OPUS_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_SONNET_MODEL"] != "llama3.2" {
			t.Errorf("SONNET = %q, want llama3.2", got["ANTHROPIC_DEFAULT_SONNET_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_HAIKU_MODEL"] != "llama3.2" {
			t.Errorf("HAIKU = %q, want llama3.2", got["ANTHROPIC_DEFAULT_HAIKU_MODEL"])
		}
		if got["CLAUDE_CODE_SUBAGENT_MODEL"] != "llama3.2" {
			t.Errorf("SUBAGENT = %q, want llama3.2", got["CLAUDE_CODE_SUBAGENT_MODEL"])
		}
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want empty for local models", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})

	t.Run("supports empty model", func(t *testing.T) {
		got := envMap(c.modelEnvVars(""))
		if got["ANTHROPIC_DEFAULT_OPUS_MODEL"] != "" {
			t.Errorf("OPUS = %q, want empty", got["ANTHROPIC_DEFAULT_OPUS_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_SONNET_MODEL"] != "" {
			t.Errorf("SONNET = %q, want empty", got["ANTHROPIC_DEFAULT_SONNET_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_HAIKU_MODEL"] != "" {
			t.Errorf("HAIKU = %q, want empty", got["ANTHROPIC_DEFAULT_HAIKU_MODEL"])
		}
		if got["CLAUDE_CODE_SUBAGENT_MODEL"] != "" {
			t.Errorf("SUBAGENT = %q, want empty", got["CLAUDE_CODE_SUBAGENT_MODEL"])
		}
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want empty", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})

	t.Run("sets auto compact window for known cloud models", func(t *testing.T) {
		got := envMap(c.modelEnvVars("glm-5:cloud"))
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "202752" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want 202752", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})

	t.Run("does not set auto compact window for unknown cloud models", func(t *testing.T) {
		got := envMap(c.modelEnvVars("unknown-model:cloud"))
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want empty", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})
}
