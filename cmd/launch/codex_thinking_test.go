package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestCodexThinkingWarning(t *testing.T) {
	named := &api.ModelRecommendationThinking{Values: []any{"low", "high", "max"}, Default: "max"}
	for _, tt := range []struct {
		name     string
		thinking *api.ModelRecommendationThinking
		effort   string
		want     string
	}{
		{"inherited medium", named, "medium", "max"},
		{"inherited xhigh", named, "xhigh", "max"},
		{"ultra prefers catalog max", &api.ModelRecommendationThinking{Values: []any{"max", "high"}, Default: "high"}, "ultra", "max"},
		{"ultra uses last supported level", &api.ModelRecommendationThinking{Values: []any{"low", "medium", "high"}, Default: "medium"}, "ultra", "high"},
		{"ultra respects catalog order", &api.ModelRecommendationThinking{Values: []any{"high", "low"}, Default: "high"}, "ultra", "low"},
		{"persistent wire alias uses model default", named, "persistent", "max"},
		{"supported low", named, "low", ""},
		{"supported high", named, "high", ""},
		{"supported max", named, "max", ""},
		{"unset uses model default", named, "", ""},
		{"supported xhigh is preserved", &api.ModelRecommendationThinking{Values: []any{"low", "xhigh"}, Default: "low"}, "xhigh", ""},
		{"model specific default", &api.ModelRecommendationThinking{Values: []any{"low", "medium", "high"}, Default: "medium"}, "max", "medium"},
		{"boolean alias is preserved", &api.ModelRecommendationThinking{Values: []any{false, true}, Default: false}, "medium", ""},
		{"explicit off keeps API semantics", named, "none", ""},
		{"unknown metadata", nil, "medium", ""},
		{"non-thinking model", &api.ModelRecommendationThinking{Values: []any{false}, Default: false}, "medium", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := codexThinkingWarning(LaunchModel{Name: "test:cloud", Thinking: tt.thinking}, tt.effort)
			if tt.want == "" {
				if got != "" {
					t.Fatalf("unexpected warning: %s", got)
				}
				return
			}
			if !strings.Contains(got, fmt.Sprintf("thinking level %q isn't supported by test:cloud", tt.effort)) ||
				!strings.Contains(got, fmt.Sprintf("Ollama will use %q.", tt.want)) {
				t.Fatalf("warning does not explain requested and resolved levels: %s", got)
			}
			if tt.thinking == named && !strings.Contains(got, "Supported levels: low, high, max.") {
				t.Fatalf("warning must list catalog levels: %s", got)
			}
		})
	}
}

func TestCodexConfigReadOptions(t *testing.T) {
	for _, tt := range []struct {
		name  string
		extra []string
		args  []string
		cwd   string
		ok    bool
	}{
		{"inherited", nil, nil, "", true},
		{"explicit choice", []string{"exec", "--json", "-c", `model_reasoning_effort="low"`, "Say OK"}, []string{"-c", `model_reasoning_effort="low"`}, "", true},
		{"last override wins", []string{"-cmodel_reasoning_effort=low", "--config=model_reasoning_effort=high"}, []string{"-cmodel_reasoning_effort=low", "--config=model_reasoning_effort=high"}, "", true},
		{"quoted key", []string{"--config", `"model_reasoning_effort" = "low"`}, []string{"--config", `"model_reasoning_effort" = "low"`}, "", true},
		{"config value is not a command", []string{"-c", "resume"}, []string{"-c", "resume"}, "", true},
		{"managed invocation", []string{"--profile", codexProfileName, "-c", `model_provider="ollama-launch"`, "-m", "test:cloud", "exec", "Say OK"}, []string{"-c", `model_provider="ollama-launch"`, "-c", `model="test:cloud"`}, "", true},
		{"unmanaged profile", []string{"--profile", "other"}, nil, "", false},
		{"working directory", []string{"exec", "-C", "project", "Say OK"}, nil, "project", true},
		{"attached directory", []string{"-Cproject"}, nil, "project", true},
		{"long directory", []string{"--cd=project"}, nil, "project", true},
		{"directory is not a command", []string{"--cd", "review"}, nil, "review", true},
		{"output file is not a command", []string{"exec", "--output-last-message", "review", "Say OK"}, nil, "", true},
		{"short output file is not a command", []string{"exec", "-o", "resume", "Say OK"}, nil, "", true},
		{"extra directory is not a command", []string{"exec", "--add-dir", "review", "Say OK"}, nil, "", true},
		{"schema file is not a command", []string{"exec", "--output-schema", "fork", "Say OK"}, nil, "", true},
		{"image paths are not commands", []string{"exec", "--image", "review", "resume", "--color", "never", "Say OK"}, nil, "", true},
		{"attached image paths are not commands", []string{"exec", "--image=review", "fork", "-c", `model_reasoning_effort="low"`}, []string{"-c", `model_reasoning_effort="low"`}, "", true},
		{"feature overrides", []string{"--enable", "test", "--disable=other"}, []string{"--enable", "test", "--disable=other"}, "", true},
		{"prompt after separator", []string{"exec", "--", "-cmodel_reasoning_effort=low"}, nil, "", true},
		{"command name as prompt", []string{"exec", "--", "review"}, nil, "", true},
		{"resume has saved settings", []string{"exec", "resume", "--last"}, nil, "", false},
		{"remote settings", []string{"--remote=ws://localhost:1234"}, nil, "", false},
		{"ignore user settings", []string{"exec", "--ignore-user-config"}, nil, "", false},
		{"worktree settings", []string{"--worktree"}, nil, "", false},
		{"help", []string{"--help"}, nil, "", false},
		{"help command", []string{"help"}, nil, "", false},
		{"exec help command", []string{"exec", "help"}, nil, "", false},
		{"MCP listing", []string{"mcp", "list"}, nil, "", false},
		{"doctor command", []string{"doctor"}, nil, "", false},
		{"cloud command alias", []string{"cloud-tasks", "list"}, nil, "", false},
		{"hidden tunnel command", []string{"tcp-tunnel"}, nil, "", false},
		{"hidden policy command", []string{"execpolicy", "check"}, nil, "", false},
		{"hidden proxy command", []string{"responses-api-proxy"}, nil, "", false},
		{"hidden relay command", []string{"stdio-to-uds", "/tmp/socket"}, nil, "", false},
		{"command word as exec prompt", []string{"exec", "doctor"}, nil, "", true},
		{"hidden command as exec prompt", []string{"exec", "responses-api-proxy"}, nil, "", true},
		{"command word as alias prompt", []string{"e", "mcp"}, nil, "", true},
		{"command word after separator", []string{"--", "help"}, nil, "", true},
		{"missing override", []string{"-c"}, nil, "", false},
		{"missing directory", []string{"--cd"}, nil, "", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			args, cwd, ok := codexConfigReadOptions(tt.extra)
			if !slices.Equal(args, tt.args) || cwd != tt.cwd || ok != tt.ok {
				t.Fatalf("got (%v, %q, %v), want (%v, %q, %v)", args, cwd, ok, tt.args, tt.cwd, tt.ok)
			}
		})
	}
}

func TestCodexReadThinking(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"success", "rpc error", "malformed", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			timeout := 5 * time.Second
			if mode == "timeout" {
				timeout = 100 * time.Millisecond
			}
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			cmd := exec.CommandContext(ctx, exe, "-test.run=^TestCodexThinkingConfigProcess$")
			cmd.Env = append(os.Environ(), "OLLAMA_CODEX_CONFIG_TEST="+mode)
			got, err := codexReadThinking(cmd, "/project")
			if mode == "success" {
				if err != nil || got != "medium" {
					t.Fatalf("got (%q, %v), want resolved medium", got, err)
				}
			} else if err == nil || got != "" {
				t.Fatalf("got (%q, %v), want no effort on failed read", got, err)
			}
			if err != nil && strings.Contains(err.Error(), "private config") {
				t.Fatal("config API error exposed private settings")
			}
			if cmd.ProcessState == nil {
				t.Fatal("config reader process was not reaped")
			}
		})
	}
}

// The helper speaks only the read-only config protocol. It fails if the caller
// asks to start a thread or omits the working directory or initialization.
func TestCodexThinkingConfigProcess(t *testing.T) {
	mode := os.Getenv("OLLAMA_CODEX_CONFIG_TEST")
	if mode == "" {
		return
	}
	decoder := json.NewDecoder(os.Stdin)
	for _, method := range []string{"initialize", "initialized", "config/read"} {
		var request struct {
			Method string `json:"method"`
			Params struct {
				Cwd string `json:"cwd"`
			} `json:"params"`
		}
		if err := decoder.Decode(&request); err != nil || request.Method != method {
			os.Exit(1)
		}
		if method == "initialize" {
			fmt.Println(`{"method":"notice","params":{}}`)
			fmt.Println(`{"id":0,"result":{}}`)
		}
		if method == "config/read" && request.Params.Cwd != "/project" {
			os.Exit(1)
		}
	}
	switch mode {
	case "success":
		fmt.Println(`{"id":1,"result":{"config":{"model_reasoning_effort":"medium"}}}`)
	case "rpc error":
		fmt.Println(`{"id":1,"error":{"message":"private config"}}`)
	case "malformed":
		fmt.Println(`invalid json`)
	case "timeout":
		time.Sleep(30 * time.Second)
	}
	io.Copy(io.Discard, os.Stdin)
	os.Exit(0)
}
