package launch

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/pelletier/go-toml/v2"

	modelpkg "github.com/ollama/ollama/types/model"
)

func writeFakeKimiCode(t *testing.T, dir, logPath string) {
	t.Helper()
	script := fmt.Sprintf(`#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "2.3.3"
  exit 0
fi
for arg in "$@"; do
  printf "%%s\n" "$arg" >> %q
done
exit 0
`, logPath)
	if err := os.WriteFile(filepath.Join(dir, "kimi"), []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake kimi: %v", err)
	}
}

func readKimiCodeConfig(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read kimi config: %v", err)
	}
	cfg := map[string]any{}
	if err := toml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("kimi config is not valid TOML: %v\n%s", err, data)
	}
	return cfg
}

func kimiCodeConfigModel(t *testing.T, cfg map[string]any, alias string) map[string]any {
	t.Helper()
	models, _ := cfg["models"].(map[string]any)
	entry, ok := models[alias].(map[string]any)
	if !ok {
		t.Fatalf("models[%q] missing from config: %v", alias, models)
	}
	return entry
}

func stringList(v any) []string {
	items, _ := v.([]any)
	out := make([]string, 0, len(items))
	for _, item := range items {
		s, _ := item.(string)
		out = append(out, s)
	}
	return out
}

func TestKimiRun_KimiCodeWritesConfigAndSelectsModel(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	homeDir := t.TempDir()
	setTestHome(t, homeDir)
	binDir := t.TempDir()
	logPath := filepath.Join(binDir, "kimi-args.log")
	writeFakeKimiCode(t, binDir, logPath)
	t.Setenv("PATH", binDir)
	t.Setenv("KIMI_CODE_HOME", "")

	srv := httptest.NewServer(http.NotFoundHandler())
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	configPath := filepath.Join(homeDir, ".kimi-code", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := `default_model = "kimi-code/k3"
telemetry = false

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.kimi.com/coding/v1"

[models."kimi-code/k3"]
provider = "managed:kimi-code"
model = "k3"
max_context_size = 1048576

[models."ollama/stale:cloud"]
provider = "ollama"
model = "stale:cloud"
max_context_size = 4096
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	models := []LaunchModel{
		{
			Name:          "kimi-k3:cloud",
			ContextLength: 262144,
			ToolCapable:   true,
			Capabilities:  []modelpkg.Capability{modelpkg.CapabilityCompletion, modelpkg.CapabilityTools, modelpkg.CapabilityVision, modelpkg.CapabilityThinking},
			Thinking:      &modelpkg.Thinking{Values: []any{false, "low", "high", "max"}, Default: "max"},
		},
		{
			Name:          "qwen3-coder:30b",
			ContextLength: 32768,
			ToolCapable:   true,
			Capabilities:  []modelpkg.Capability{modelpkg.CapabilityCompletion, modelpkg.CapabilityTools},
		},
	}

	var gotPrompt string
	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		gotPrompt = prompt
		return true, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	k := &Kimi{}
	if err := k.Run("kimi-k3:cloud", models, []string{"-p", "hello"}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !strings.Contains(gotPrompt, configPath) {
		t.Fatalf("prompt = %q, want existing config confirmation for %s", gotPrompt, configPath)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read args log: %v", err)
	}
	gotArgs := strings.Split(strings.TrimSpace(string(data)), "\n")
	wantArgs := []string{"-m", "ollama/kimi-k3:cloud", "-p", "hello"}
	if !slices.Equal(gotArgs, wantArgs) {
		t.Fatalf("args = %v, want %v", gotArgs, wantArgs)
	}

	cfg := readKimiCodeConfig(t, configPath)
	if cfg["default_model"] != "kimi-code/k3" || cfg["telemetry"] != false {
		t.Fatalf("user settings not preserved: %v", cfg)
	}

	providers := cfg["providers"].(map[string]any)
	if _, ok := providers["managed:kimi-code"]; !ok {
		t.Fatalf("user provider removed: %v", providers)
	}
	ollama := providers["ollama"].(map[string]any)
	if ollama["type"] != "openai" || ollama["base_url"] != srv.URL+"/v1" || ollama["api_key"] != "ollama" {
		t.Fatalf("ollama provider = %v", ollama)
	}

	allModels := cfg["models"].(map[string]any)
	if _, ok := allModels["kimi-code/k3"]; !ok {
		t.Fatalf("user model removed: %v", allModels)
	}
	if _, ok := allModels["ollama/stale:cloud"]; !ok {
		t.Fatalf("previously written ollama model was removed: %v", allModels)
	}

	kimi := kimiCodeConfigModel(t, cfg, "ollama/kimi-k3:cloud")
	if kimi["model"] != "kimi-k3:cloud" || kimi["provider"] != "ollama" || kimi["max_context_size"] != int64(262144) {
		t.Fatalf("kimi entry = %v", kimi)
	}
	if got, want := stringList(kimi["capabilities"]), []string{"tool_use", "image_in", "thinking"}; !slices.Equal(got, want) {
		t.Fatalf("capabilities = %v, want %v", got, want)
	}
	if got, want := stringList(kimi["support_efforts"]), []string{"low", "high", "max"}; !slices.Equal(got, want) {
		t.Fatalf("support_efforts = %v, want %v", got, want)
	}
	if kimi["default_effort"] != "max" || kimi["off_effort"] != "none" {
		t.Fatalf("efforts = default %v off %v", kimi["default_effort"], kimi["off_effort"])
	}

	qwen := kimiCodeConfigModel(t, cfg, "ollama/qwen3-coder:30b")
	if got, want := stringList(qwen["capabilities"]), []string{"tool_use"}; !slices.Equal(got, want) {
		t.Fatalf("capabilities = %v, want %v", got, want)
	}
	if len(qwen) != 4 {
		t.Fatalf("non-thinking model has unexpected keys: %v", qwen)
	}
}

func TestKimiCodeModelEntry_ThinkingContracts(t *testing.T) {
	tests := []struct {
		name          string
		model         LaunchModel
		wantCaps      []string
		wantEfforts   []string
		wantDefault   any
		wantOffEffort any
	}{
		{
			name: "always thinking",
			model: LaunchModel{
				Name:          "minimax-m3:cloud",
				ContextLength: 200000,
				ToolCapable:   true,
				Capabilities:  []modelpkg.Capability{modelpkg.CapabilityTools, modelpkg.CapabilityThinking},
				Thinking:      &modelpkg.Thinking{Values: []any{true}, Default: true},
			},
			wantCaps:    []string{"tool_use", "thinking", "always_thinking"},
			wantEfforts: []string{"high"},
			wantDefault: "high",
		},
		{
			name: "on off thinking",
			model: LaunchModel{
				Name:          "qwen3.5:cloud",
				ContextLength: 262144,
				ToolCapable:   true,
				Capabilities:  []modelpkg.Capability{modelpkg.CapabilityTools, modelpkg.CapabilityThinking},
				Thinking:      &modelpkg.Thinking{Values: []any{false, true}, Default: true},
			},
			wantCaps:      []string{"tool_use", "thinking"},
			wantEfforts:   []string{"high"},
			wantDefault:   "high",
			wantOffEffort: "none",
		},
		{
			name: "thinking without metadata",
			model: LaunchModel{
				Name:          "local-thinker",
				ContextLength: 8192,
				ToolCapable:   true,
				Capabilities:  []modelpkg.Capability{modelpkg.CapabilityTools, modelpkg.CapabilityThinking},
			},
			wantCaps: []string{"tool_use", "thinking"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := kimiCodeModelEntry(tt.model)
			if got := stringList(entry["capabilities"]); !slices.Equal(got, tt.wantCaps) {
				t.Fatalf("capabilities = %v, want %v", got, tt.wantCaps)
			}
			if got := stringList(entry["support_efforts"]); !slices.Equal(got, tt.wantEfforts) {
				t.Fatalf("support_efforts = %v, want %v", got, tt.wantEfforts)
			}
			if entry["default_effort"] != tt.wantDefault {
				t.Fatalf("default_effort = %v, want %v", entry["default_effort"], tt.wantDefault)
			}
			if entry["off_effort"] != tt.wantOffEffort {
				t.Fatalf("off_effort = %v, want %v", entry["off_effort"], tt.wantOffEffort)
			}
		})
	}
}

func TestWriteKimiCodeConfig_UsesKimiCodeHome(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	if err := writeKimiCodeConfig("glm-5.3:cloud", []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 200000}}); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	cfg := readKimiCodeConfig(t, filepath.Join(kimiHome, "config.toml"))
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
	if cfg["default_model"] != "ollama/glm-5.3:cloud" {
		t.Fatalf("default_model = %v, want ollama/glm-5.3:cloud", cfg["default_model"])
	}
}

func TestWriteKimiCodeConfig_DefaultModel(t *testing.T) {
	tests := []struct {
		name     string
		existing string
		want     string
	}{
		{
			name: "replaces previous ollama default",
			existing: `default_model = "ollama/old:cloud"

[models."ollama/old:cloud"]
provider = "ollama"
model = "old:cloud"
`,
			want: "ollama/glm-5.3:cloud",
		},
		{
			name: "replaces dangling default",
			existing: `default_model = "gone"
`,
			want: "ollama/glm-5.3:cloud",
		},
		{
			name: "keeps user default",
			existing: `default_model = "kimi-code/k3"

[models."kimi-code/k3"]
provider = "managed:kimi-code"
model = "k3"
`,
			want: "kimi-code/k3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			kimiHome := t.TempDir()
			t.Setenv("KIMI_CODE_HOME", kimiHome)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
			configPath := filepath.Join(kimiHome, "config.toml")
			if err := os.WriteFile(configPath, []byte(tt.existing), 0o644); err != nil {
				t.Fatal(err)
			}

			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
				return true, nil
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

			if err := writeKimiCodeConfig("glm-5.3:cloud", []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 200000}}); err != nil {
				t.Fatalf("writeKimiCodeConfig() error = %v", err)
			}

			cfg := readKimiCodeConfig(t, configPath)
			if cfg["default_model"] != tt.want {
				t.Fatalf("default_model = %v, want %v", cfg["default_model"], tt.want)
			}
		})
	}
}

func TestKimiCodeModelEntry_FillsThinkingFromShow(t *testing.T) {
	old := lookupServerModelMetadata
	lookupServerModelMetadata = func(name string) (serverModelMetadata, bool) {
		if name != "beam:cloud" {
			t.Fatalf("unexpected show lookup for %q", name)
		}
		return serverModelMetadata{Thinking: &modelpkg.Thinking{Values: []any{"low", "medium", "high", "xhigh", "max"}, Default: "max"}}, true
	}
	t.Cleanup(func() { lookupServerModelMetadata = old })

	entry := kimiCodeModelEntry(LaunchModel{
		Name:          "beam:cloud",
		ContextLength: 1048576,
		ToolCapable:   true,
		Capabilities:  []modelpkg.Capability{modelpkg.CapabilityTools, modelpkg.CapabilityThinking},
	})
	if got, want := stringList(entry["support_efforts"]), []string{"low", "medium", "high", "xhigh", "max"}; !slices.Equal(got, want) {
		t.Fatalf("support_efforts = %v, want %v", got, want)
	}
	if entry["default_effort"] != "max" {
		t.Fatalf("default_effort = %v, want max", entry["default_effort"])
	}
	if got, want := stringList(entry["capabilities"]), []string{"tool_use", "thinking", "always_thinking"}; !slices.Equal(got, want) {
		t.Fatalf("capabilities = %v, want %v", got, want)
	}
}

func TestWriteKimiCodeConfig_AddsPrimaryWhenMissingFromModels(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	cfg := readKimiCodeConfig(t, filepath.Join(kimiHome, "config.toml"))
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
}

func TestWriteKimiCodeConfig_RejectsInvalidExistingConfig(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	configPath := filepath.Join(kimiHome, "config.toml")
	if err := os.WriteFile(configPath, []byte("not = [valid"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err == nil {
		t.Fatal("expected parse error")
	}
	data, _ := os.ReadFile(configPath)
	if string(data) != "not = [valid" {
		t.Fatalf("invalid config was overwritten: %q", data)
	}
}

func TestWriteKimiCodeConfig_DeclinedUpdateLeavesConfigUntouched(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `default_model = "kimi-code/k3"

[providers."managed:kimi-code"]
type = "kimi"
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	err := writeKimiCodeConfig("glm-5.3:cloud", []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 200000}})
	if err == nil {
		t.Fatal("expected error when update is declined")
	}
	if !strings.Contains(err.Error(), "--yes") {
		t.Fatalf("error = %v, want --yes hint", err)
	}
	data, readErr := os.ReadFile(configPath)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if string(data) != existing {
		t.Fatalf("config changed after decline: %q", data)
	}
}

func TestWriteKimiCodeConfig_PromptsBeforeUpdatingUserSettings(t *testing.T) {
	tests := []struct {
		name       string
		existing   string
		wantPrompt bool
	}{
		{
			name: "fresh config",
		},
		{
			name: "ollama entries only",
			existing: `default_model = "ollama/old:cloud"

[providers.ollama]
type = "openai"
base_url = "http://gpu-box:11434/v1"
api_key = "ollama"

[models."ollama/old:cloud"]
provider = "ollama"
model = "old:cloud"
`,
		},
		{
			name: "user provider",
			existing: `[providers."managed:kimi-code"]
type = "kimi"
`,
			wantPrompt: true,
		},
		{
			name: "user model",
			existing: `[models."kimi-code/k3"]
provider = "managed:kimi-code"
model = "k3"
`,
			wantPrompt: true,
		},
		{
			name:       "user setting",
			existing:   "telemetry = false\n",
			wantPrompt: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			kimiHome := t.TempDir()
			t.Setenv("KIMI_CODE_HOME", kimiHome)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
			configPath := filepath.Join(kimiHome, "config.toml")
			if tt.existing != "" {
				if err := os.WriteFile(configPath, []byte(tt.existing), 0o644); err != nil {
					t.Fatal(err)
				}
			}

			var gotPrompt string
			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
				gotPrompt = prompt
				return true, nil
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

			if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
				t.Fatalf("writeKimiCodeConfig() error = %v", err)
			}
			if (gotPrompt != "") != tt.wantPrompt {
				t.Fatalf("prompt = %q, wantPrompt %v", gotPrompt, tt.wantPrompt)
			}
			if tt.wantPrompt && !strings.Contains(gotPrompt, configPath) {
				t.Fatalf("prompt = %q, want confirmation for %s", gotPrompt, configPath)
			}
			cfg := readKimiCodeConfig(t, configPath)
			kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
			if providers := cfg["providers"].(map[string]any); providers["ollama"].(map[string]any)["base_url"] != "http://127.0.0.1:11434/v1" {
				t.Fatalf("base_url = %v, want local host", providers["ollama"])
			}
		})
	}
}

func TestWriteKimiCodeConfig_SkipsUnchangedRewrite(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `[providers."managed:kimi-code"]
type = "kimi"
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	prompts := 0
	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		prompts++
		return true, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	models := []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 200000}, {Name: "qwen3-coder:30b", ContextLength: 32768}}
	if err := writeKimiCodeConfig("glm-5.3:cloud", models); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}
	if prompts != 1 {
		t.Fatalf("prompts after first write = %d, want 1", prompts)
	}
	first, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeKimiCodeConfig("glm-5.3:cloud", models); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}
	if prompts != 1 {
		t.Fatalf("prompts after identical write = %d, want 1", prompts)
	}
	second, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(second) != string(first) {
		t.Fatal("config was rewritten even though nothing changed")
	}
	kimiCodeConfigModel(t, readKimiCodeConfig(t, configPath), "ollama/qwen3-coder:30b")
}

func TestWriteKimiCodeConfig_UsesResolvedPrimaryMetadata(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	configPath := filepath.Join(kimiHome, "config.toml")

	srv := httptest.NewServer(http.NotFoundHandler())
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	models := []LaunchModel{{
		Name:          "llama3.2:latest",
		ContextLength: 131072,
		ToolCapable:   true,
		Capabilities:  []modelpkg.Capability{modelpkg.CapabilityTools},
	}}
	if err := writeKimiCodeConfig("llama3.2", models); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	cfg := readKimiCodeConfig(t, configPath)
	if cfg["default_model"] != "ollama/llama3.2" {
		t.Fatalf("default_model = %v, want ollama/llama3.2", cfg["default_model"])
	}
	entry := kimiCodeConfigModel(t, cfg, "ollama/llama3.2")
	if entry["max_context_size"] != int64(131072) {
		t.Fatalf("max_context_size = %v, want 131072", entry["max_context_size"])
	}
	if got := stringList(entry["capabilities"]); !slices.Equal(got, []string{"tool_use"}) {
		t.Fatalf("capabilities = %v, want [tool_use]", got)
	}
	entries := cfg["models"].(map[string]any)
	if len(entries) != 1 {
		t.Fatalf("models = %v, want only ollama/llama3.2", entries)
	}
}

func TestWriteKimiCodeConfig_PreservesUnrelatedContent(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `# hand-tuned, keep this
telemetry = false

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.kimi.com/coding/v1"

[models."kimi-code/k3"]
provider = "managed:kimi-code"
model = "k3"
max_context_size = 1048576
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	raw, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"# hand-tuned, keep this",
		"telemetry = false",
		`[providers."managed:kimi-code"]`,
		`type = "kimi"`,
		`[models."kimi-code/k3"]`,
	} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("config lost %q:\n%s", want, raw)
		}
	}
	cfg := readKimiCodeConfig(t, configPath)
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
	if cfg["default_model"] != "ollama/glm-5.3:cloud" {
		t.Fatalf("default_model = %v, want ollama/glm-5.3:cloud", cfg["default_model"])
	}
}

func TestWriteKimiCodeConfig_PreservesMultilineHook(t *testing.T) {
	const command = `cat <<'CONFIG'
providers.ollama.api_key = "literal hook text"
[providers.ollama]
type = "openai"
[models."ollama/local".overrides]
max_context_size = 8192
CONFIG
`
	for _, delimiter := range []string{"'''", `"""`} {
		t.Run(delimiter, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			kimiHome := t.TempDir()
			t.Setenv("KIMI_CODE_HOME", kimiHome)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
			path := filepath.Join(kimiHome, "config.toml")
			existing := fmt.Sprintf("[[hooks]]\nevent = 'SessionStart'\ncommand = %s\n%s%s\n", delimiter, command, delimiter)
			if err := os.WriteFile(path, []byte(existing), 0o600); err != nil {
				t.Fatal(err)
			}
			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

			models := []LaunchModel{{Name: "local", ContextLength: 32768}}
			if err := writeKimiCodeConfig("local", models); err != nil {
				t.Fatal(err)
			}
			cfg := readKimiCodeConfig(t, path)
			hooks := cfg["hooks"].([]any)
			if got := hooks[0].(map[string]any)["command"]; got != command {
				t.Fatalf("hook command = %q, want %q", got, command)
			}
			first, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := writeKimiCodeConfig("local", models); err != nil {
				t.Fatal(err)
			}
			second, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if string(first) != string(second) {
				t.Fatal("unchanged configuration was rewritten")
			}
		})
	}
}

func TestKimiLegacyCLIArchived(t *testing.T) {
	tests := map[string]bool{
		"1.50.0": false,
		"1.51.0": false,
		"1.52.0": true,
		"1.53.1": true,
		"2.0.0":  true,
		"bogus":  false,
	}
	for version, want := range tests {
		if got := kimiLegacyCLIArchived(version); got != want {
			t.Errorf("kimiLegacyCLIArchived(%q) = %v, want %v", version, got, want)
		}
	}
}

func TestEnsureKimiInstalled_RejectsArchivedCLIWhenNotInteractive(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binaries")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = "linux"
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	setTestHome(t, t.TempDir())
	binDir := t.TempDir()
	t.Setenv("PATH", binDir)
	writeFakeBinary(t, binDir, "curl")
	writeFakeBinary(t, binDir, "bash")
	legacyPath := filepath.Join(binDir, "kimi")
	if err := os.WriteFile(legacyPath, []byte("#!/bin/sh\necho 'kimi, version 1.52.0'\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldInteractive := isInteractiveSession
	isInteractiveSession = func() bool { return false }
	t.Cleanup(func() { isInteractiveSession = oldInteractive })

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	_, err := ensureKimiInstalled()
	if err == nil || !strings.Contains(err.Error(), "no longer runs commands") {
		t.Fatalf("ensureKimiInstalled() error = %v, want archived kimi-cli error", err)
	}
	if !strings.Contains(err.Error(), "https://code.kimi.com/kimi-code/install.sh") {
		t.Fatalf("error = %v, want install command", err)
	}
}

func TestFindKimiBinary_PrefersKimiCodeInstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX executable names")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = runtime.GOOS
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	homeDir := t.TempDir()
	setTestHome(t, homeDir)
	pathDir := t.TempDir()
	writeFakeBinary(t, pathDir, "kimi")
	t.Setenv("PATH", pathDir)

	kimiCodeBin := filepath.Join(homeDir, ".kimi-code", "bin", "kimi")
	if err := os.MkdirAll(filepath.Dir(kimiCodeBin), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(kimiCodeBin, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	got, err := findKimiBinary()
	if err != nil {
		t.Fatalf("findKimiBinary() error = %v", err)
	}
	if got != kimiCodeBin {
		t.Fatalf("findKimiBinary() = %q, want %q", got, kimiCodeBin)
	}
}

func TestEnsureKimiInstalled_FallsBackToLegacyCLIOnDecline(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binaries")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = "linux"
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	setTestHome(t, t.TempDir())
	binDir := t.TempDir()
	t.Setenv("PATH", binDir)
	writeFakeBinary(t, binDir, "curl")
	writeFakeBinary(t, binDir, "bash")
	legacyPath := filepath.Join(binDir, "kimi")
	if err := os.WriteFile(legacyPath, []byte("#!/bin/sh\necho 'kimi, version 1.50.0'\necho 'kimi-cli is no longer maintained.'\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldInteractive := isInteractiveSession
	isInteractiveSession = func() bool { return true }
	t.Cleanup(func() { isInteractiveSession = oldInteractive })

	var gotPrompt string
	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		gotPrompt = prompt
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	got, err := ensureKimiInstalled()
	if err != nil {
		t.Fatalf("ensureKimiInstalled() error = %v", err)
	}
	if got != legacyPath {
		t.Fatalf("ensureKimiInstalled() = %q, want legacy %q", got, legacyPath)
	}
	if !strings.Contains(gotPrompt, "kimi-cli 1.50.0 is no longer maintained") {
		t.Fatalf("prompt = %q, want archived kimi-cli prompt", gotPrompt)
	}
}

func TestKimiRun_RejectsArchivedLegacyCLI(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	logPath := filepath.Join(tmpDir, "kimi-args.log")
	script := fmt.Sprintf(`#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "kimi, version 1.53.1"
  exit 0
fi
for arg in "$@"; do
  printf "%%s\n" "$arg" >> %q
done
exit 0
`, logPath)
	if err := os.WriteFile(filepath.Join(tmpDir, "kimi"), []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake kimi: %v", err)
	}
	writeFakeBinary(t, tmpDir, "curl")
	writeFakeBinary(t, tmpDir, "bash")
	t.Setenv("PATH", tmpDir)
	t.Setenv("KIMI_CODE_HOME", "")

	oldInteractive := isInteractiveSession
	isInteractiveSession = func() bool { return true }
	t.Cleanup(func() { isInteractiveSession = oldInteractive })

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	err := (&Kimi{}).Run("llama3.2", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "no longer runs commands") {
		t.Fatalf("Run() error = %v, want archived kimi-cli error", err)
	}
	if !strings.Contains(err.Error(), "https://code.kimi.com/kimi-code/install.sh") {
		t.Fatalf("error = %v, want install command", err)
	}
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Fatalf("archived kimi-cli should not be launched")
	}
	if _, err := os.Stat(filepath.Join(tmpDir, ".kimi-code", "config.toml")); !os.IsNotExist(err) {
		t.Fatalf("kimi code config should not be written for an archived kimi-cli")
	}
}

func TestEnsureKimiInstalled_KeepsLegacyCLIWhenNotInteractive(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binaries")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = "linux"
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	setTestHome(t, t.TempDir())
	binDir := t.TempDir()
	t.Setenv("PATH", binDir)
	writeFakeBinary(t, binDir, "curl")
	writeFakeBinary(t, binDir, "bash")
	legacyPath := filepath.Join(binDir, "kimi")
	if err := os.WriteFile(legacyPath, []byte("#!/bin/sh\necho 'kimi, version 1.50.0'\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldInteractive := isInteractiveSession
	isInteractiveSession = func() bool { return false }
	t.Cleanup(func() { isInteractiveSession = oldInteractive })

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	got, err := ensureKimiInstalled()
	if err != nil {
		t.Fatalf("ensureKimiInstalled() error = %v", err)
	}
	if got != legacyPath {
		t.Fatalf("ensureKimiInstalled() = %q, want legacy %q", got, legacyPath)
	}
}

func TestEnsureKimiInstalled_FallsBackWhenInstallerDepsMissing(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binaries")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = "linux"
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	setTestHome(t, t.TempDir())
	binDir := t.TempDir()
	t.Setenv("PATH", binDir)
	legacyPath := filepath.Join(binDir, "kimi")
	if err := os.WriteFile(legacyPath, []byte("#!/bin/sh\necho 'kimi, version 1.50.0'\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldInteractive := isInteractiveSession
	isInteractiveSession = func() bool { return true }
	t.Cleanup(func() { isInteractiveSession = oldInteractive })

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	got, err := ensureKimiInstalled()
	if err != nil {
		t.Fatalf("ensureKimiInstalled() error = %v", err)
	}
	if got != legacyPath {
		t.Fatalf("ensureKimiInstalled() = %q, want legacy %q", got, legacyPath)
	}
}

func TestWriteKimiCodeConfig_PreservesCustomOllamaAlias(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `default_model = "local-fast"

[models."local-fast"]
provider = "ollama"
model = "local-fast"
max_context_size = 8192
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	cfg := readKimiCodeConfig(t, configPath)
	if cfg["default_model"] != "local-fast" {
		t.Fatalf("default_model = %v, want user's local-fast", cfg["default_model"])
	}
	entry := kimiCodeConfigModel(t, cfg, "local-fast")
	if entry["max_context_size"] != int64(8192) {
		t.Fatalf("local-fast = %v, want preserved entry", entry)
	}
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
}

func TestWriteKimiCodeConfig_PreservesModelOverrides(t *testing.T) {
	tests := map[string]string{
		"inline": `[models."ollama/glm-5.3:cloud"]
provider = "ollama"
model = "glm-5.3:cloud"
max_context_size = 262144
overrides = { max_context_size = 8192 }
`,
		"separate table": `[models."ollama/glm-5.3:cloud"]
provider = "ollama"
model = "glm-5.3:cloud"
max_context_size = 262144

[models."ollama/glm-5.3:cloud".overrides]
max_context_size = 8192
`,
	}

	for name, existing := range tests {
		t.Run(name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			kimiHome := t.TempDir()
			t.Setenv("KIMI_CODE_HOME", kimiHome)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
			configPath := filepath.Join(kimiHome, "config.toml")
			if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
				t.Fatal(err)
			}

			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
				t.Fatalf("unexpected prompt: %q", prompt)
				return false, nil
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

			models := []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 262144}}
			if err := writeKimiCodeConfig("glm-5.3:cloud", models); err != nil {
				t.Fatalf("writeKimiCodeConfig() error = %v", err)
			}
			cfg := readKimiCodeConfig(t, configPath)
			entry := kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
			overrides, ok := entry["overrides"].(map[string]any)
			if !ok || overrides["max_context_size"] != int64(8192) {
				t.Fatalf("overrides = %v, want max_context_size 8192", entry["overrides"])
			}

			first, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			if err := writeKimiCodeConfig("glm-5.3:cloud", models); err != nil {
				t.Fatalf("writeKimiCodeConfig() error = %v", err)
			}
			second, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			if string(second) != string(first) {
				t.Fatalf("config was rewritten on an unchanged second pass")
			}
		})
	}
}

func TestWriteKimiCodeConfig_HandlesDottedProviderKeys(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `providers.ollama.type = "openai"
providers.ollama.base_url = "http://gpu-box:11434/v1"
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	raw, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "providers.ollama.") {
		t.Fatalf("dotted provider keys survived:\n%s", raw)
	}
	cfg := readKimiCodeConfig(t, configPath)
	providers := cfg["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	if ollama["base_url"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("base_url = %v, want local host", ollama["base_url"])
	}
}

func TestWriteKimiCodeConfig_HandlesDottedKeysInsideTables(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `[providers]
ollama.type = "openai"
ollama.base_url = "http://gpu-box:11434/v1"
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}
	cfg := readKimiCodeConfig(t, configPath)
	providers := cfg["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	if ollama["base_url"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("base_url = %v, want local host", ollama["base_url"])
	}
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
}

func TestWriteKimiCodeConfig_HandlesInlineProviderTable(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `providers = { ollama = { type = "openai", base_url = "http://gpu-box:11434/v1", api_key = "ollama" } }
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}
	cfg := readKimiCodeConfig(t, configPath)
	providers := cfg["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	if ollama["base_url"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("base_url = %v, want local host", ollama["base_url"])
	}
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
}

func TestWriteKimiCodeConfig_PreservesProviderAndModelFields(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	configPath := filepath.Join(kimiHome, "config.toml")
	existing := `[providers.ollama]
type = "openai"
base_url = "http://gpu-box:11434/v1"
api_key = "ollama"
custom_headers = { X-Proxy = "corp" }

[models."ollama/glm-5.3:cloud"]
provider = "ollama"
model = "glm-5.3:cloud"
max_context_size = 262144
max_output_size = 8192
display_name = "GLM (work)"
`
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	models := []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 262144}}
	if err := writeKimiCodeConfig("glm-5.3:cloud", models); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}

	cfg := readKimiCodeConfig(t, configPath)
	providers := cfg["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	headers, ok := ollama["custom_headers"].(map[string]any)
	if !ok || headers["X-Proxy"] != "corp" {
		t.Fatalf("custom_headers = %v, want preserved", ollama["custom_headers"])
	}
	entry := kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
	if entry["max_output_size"] != int64(8192) || entry["display_name"] != "GLM (work)" {
		t.Fatalf("model fields = %v, want max_output_size and display_name preserved", entry)
	}
	if ollama["base_url"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("base_url = %v, want local host", ollama["base_url"])
	}
}

func TestWriteKimiCodeConfig_PreservesProviderAuthentication(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config string
		key    string
		want   any
	}{
		{"inline key", `api_key = "custom-key"`, "api_key", "custom-key"},
		{"environment key", `api_key_env = "OLLAMA_API_KEY"`, "api_key_env", "OLLAMA_API_KEY"},
		{"OAuth", `oauth = { storage = "keyring", key = "ollama-auth" }`, "oauth", map[string]any{"storage": "keyring", "key": "ollama-auth"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			kimiHome := t.TempDir()
			t.Setenv("KIMI_CODE_HOME", kimiHome)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
			path := filepath.Join(kimiHome, "config.toml")
			if err := os.WriteFile(path, []byte("[providers.ollama]\ntype = 'openai'\n"+tc.config+"\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := writeKimiCodeConfig("local", []LaunchModel{{Name: "local", ContextLength: 32768}}); err != nil {
				t.Fatal(err)
			}
			cfg := readKimiCodeConfig(t, path)
			provider := cfg["providers"].(map[string]any)["ollama"].(map[string]any)
			if diff := cmp.Diff(tc.want, provider[tc.key]); diff != "" {
				t.Fatalf("authentication changed (-want +got):\n%s", diff)
			}
			for _, key := range []string{"api_key", "api_key_env", "oauth"} {
				if key != tc.key {
					if _, ok := provider[key]; ok {
						t.Fatalf("conflicting credential %s was added", key)
					}
				}
			}
		})
	}
}

func TestFindKimiBinary_SkipsNonExecutableInstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX executable bits")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = runtime.GOOS
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	homeDir := t.TempDir()
	setTestHome(t, homeDir)
	pathDir := t.TempDir()
	writeFakeBinary(t, pathDir, "kimi")
	t.Setenv("PATH", pathDir)

	kimiCodeBin := filepath.Join(homeDir, ".kimi-code", "bin", "kimi")
	if err := os.MkdirAll(filepath.Dir(kimiCodeBin), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(kimiCodeBin, []byte("#!/bin/sh\nexit 0\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := findKimiBinary()
	if err != nil {
		t.Fatalf("findKimiBinary() error = %v", err)
	}
	if got != filepath.Join(pathDir, "kimi") {
		t.Fatalf("findKimiBinary() = %q, want the executable on PATH", got)
	}
}

func TestWriteKimiCodeConfig_PrivateConfigDir(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("does not check POSIX permissions")
	}
	homeDir := t.TempDir()
	setTestHome(t, homeDir)
	t.Setenv("KIMI_CODE_HOME", "")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	if err := writeKimiCodeConfig("glm-5.3:cloud", nil); err != nil {
		t.Fatalf("writeKimiCodeConfig() error = %v", err)
	}
	info, err := os.Stat(filepath.Join(homeDir, ".kimi-code"))
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o700 {
		t.Fatalf("~/.kimi-code mode = %o, want 700", perm)
	}
}

func TestKimiConfigureOnlyWritesConfig(t *testing.T) {
	setTestHome(t, t.TempDir())
	kimiHome := t.TempDir()
	t.Setenv("KIMI_CODE_HOME", kimiHome)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return false, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	models := []LaunchModel{{Name: "glm-5.3:cloud", ContextLength: 200000}}
	if err := launchAfterConfiguration("kimi", &Kimi{}, "glm-5.3:cloud", models, IntegrationLaunchRequest{ConfigureOnly: true}); err != nil {
		t.Fatalf("launchAfterConfiguration() error = %v", err)
	}
	cfg := readKimiCodeConfig(t, filepath.Join(kimiHome, "config.toml"))
	kimiCodeConfigModel(t, cfg, "ollama/glm-5.3:cloud")
}

func TestEnsureKimiInstalled_RejectsStillArchivedAfterInstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binaries")
	}
	oldGOOS := kimiGOOS
	kimiGOOS = "linux"
	t.Cleanup(func() { kimiGOOS = oldGOOS })

	setTestHome(t, t.TempDir())
	binDir := t.TempDir()
	t.Setenv("PATH", binDir)
	writeFakeBinary(t, binDir, "curl")
	writeFakeBinary(t, binDir, "bash")
	legacyPath := filepath.Join(binDir, "kimi")
	if err := os.WriteFile(legacyPath, []byte("#!/bin/sh\necho 'kimi, version 1.52.0'\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldInteractive := isInteractiveSession
	isInteractiveSession = func() bool { return true }
	t.Cleanup(func() { isInteractiveSession = oldInteractive })

	oldConfirm := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}
	t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })

	_, err := ensureKimiInstalled()
	if err == nil || !strings.Contains(err.Error(), "no longer runs commands") {
		t.Fatalf("ensureKimiInstalled() error = %v, want archived kimi-cli error after install", err)
	}
}
