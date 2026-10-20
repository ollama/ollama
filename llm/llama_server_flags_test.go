package llm

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestNoUIFlagFromHelp(t *testing.T) {
	tests := []struct {
		name string
		help string
		err  error
		want string
	}{
		{
			name: "current llama-server",
			help: "--ui,  --webui, --no-ui, --no-webui     whether to enable the Web UI (default: enabled)\n",
			want: "--no-ui",
		},
		{
			name: "only mcp proxy spelling is not enough",
			help: "--ui-mcp-proxy, --webui-mcp-proxy, --no-ui-mcp-proxy, --no-webui-mcp-proxy\n--webui, --no-webui  whether to enable the Web UI\n",
			want: "--no-webui",
		},
		{
			name: "old llama-server",
			help: "--webui, --no-webui  whether to enable the Web UI\n",
			want: "--no-webui",
		},
		{
			name: "help failed",
			help: "--no-ui",
			err:  errors.New("exit status 1"),
			want: "--no-webui",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := noUIFlagFromHelp([]byte(tt.help), tt.err); got != tt.want {
				t.Fatalf("noUIFlagFromHelp() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLlamaServerNoUIFlagCachesPerBinary(t *testing.T) {
	exe := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(exe, []byte("v1"), 0o755); err != nil {
		t.Fatal(err)
	}

	calls := 0
	orig := llamaServerHelpFn
	llamaServerHelpFn = func(string) ([]byte, error) {
		calls++
		return []byte("--ui, --webui, --no-ui, --no-webui\n"), nil
	}
	t.Cleanup(func() { llamaServerHelpFn = orig })

	for range 3 {
		if got := LlamaServerNoUIFlag(exe); got != "--no-ui" {
			t.Fatalf("LlamaServerNoUIFlag() = %q, want --no-ui", got)
		}
	}
	if calls != 1 {
		t.Fatalf("help probes = %d, want 1", calls)
	}

	if got := LlamaServerNoUIFlag(filepath.Join(t.TempDir(), "missing")); got != "--no-webui" {
		t.Fatalf("missing binary flag = %q, want --no-webui", got)
	}
}
