package launch

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPiDefaultModelScope(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		settings string
		global   string
		location string
		want     bool
	}{
		{name: "default Ollama view", want: true},
		{name: "continue retains native session selection", args: []string{"--continue"}},
		{name: "resume can select another project", args: []string{"--resume"}},
		{name: "explicit session can select another project", args: []string{"--session", "session.jsonl"}},
		{name: "fork can select another project", args: []string{"--fork", "session.jsonl"}},
		{name: "text mode retains default scope", args: []string{"--mode", "text"}, want: true},
		{name: "last valid mode is interactive", args: []string{"--mode", "json", "--mode", "text"}, want: true},
		{name: "last valid mode is headless", args: []string{"--mode", "text", "--mode", "rpc"}},
		{name: "explicit model", args: []string{"--model", "anthropic/claude"}},
		{name: "explicit provider", args: []string{"--provider", "anthropic"}},
		{name: "explicit scope", args: []string{"--models", "anthropic/*"}},
		{name: "native subcommand keeps its argument position", args: []string{"list"}},
		{name: "flag in prompt value", args: []string{"--system-prompt", "--models"}, want: true},
		{name: "Pi parses model after bare separator", args: []string{"--", "--model", "text"}},
		{name: "unrelated settings", settings: `{"editorPaddingX":3}`, want: true},
		{name: "global scope", settings: `{"enabledModels":["anthropic/*"]}`},
		{name: "explicit all models", settings: `{"enabledModels":[]}`},
		{name: "project scope", location: "project", settings: `{"enabledModels":["anthropic/*"]}`},
		{name: "project provider preference", location: "project", settings: `{"defaultProvider":"anthropic","defaultModel":"claude"}`},
		{name: "project Ollama preference", location: "project", settings: `{"defaultProvider":"ollama","defaultModel":"selected"}`, want: true},
		{name: "project Ollama overrides global provider", global: `{"defaultProvider":"anthropic"}`, location: "project", settings: `{"defaultProvider":"ollama"}`, want: true},
		{name: "project provider overrides global Ollama", global: `{"defaultProvider":"ollama"}`, location: "project", settings: `{"defaultProvider":"anthropic"}`},
		{name: "project provider retains global scope", global: `{"enabledModels":["anthropic/*"]}`, location: "project", settings: `{"defaultProvider":"ollama"}`},
		{name: "custom agent directory", location: "custom", settings: `{"enabledModels":["anthropic/*"]}`},
		{name: "tilde agent directory", location: "tilde", settings: `{"enabledModels":["anthropic/*"]}`},
		{name: "unreadable settings are left to Pi", location: "project", settings: `{"enabledModels":`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			setTestHome(t, home)
			t.Setenv("PI_CODING_AGENT_DIR", "")
			cwd := t.TempDir()
			t.Chdir(cwd)
			path := filepath.Join(home, ".pi", "agent", "settings.json")
			if tc.global != "" {
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(tc.global), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			switch tc.location {
			case "project":
				path = filepath.Join(cwd, ".pi", "settings.json")
			case "custom", "tilde":
				dir := filepath.Join(home, "custom-agent")
				t.Setenv("PI_CODING_AGENT_DIR", dir)
				if tc.location == "tilde" {
					t.Setenv("PI_CODING_AGENT_DIR", "~/custom-agent")
				}
				path = filepath.Join(dir, "settings.json")
			}
			if tc.settings != "" {
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(tc.settings), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if got := piUseDefaultModelScope(tc.args); got != tc.want {
				t.Fatalf("default scope = %v, want %v", got, tc.want)
			}
			if tc.settings != "" {
				data, err := os.ReadFile(path)
				if err != nil || string(data) != tc.settings {
					t.Fatalf("Pi settings changed: %q, %v", data, err)
				}
			}
		})
	}
}
