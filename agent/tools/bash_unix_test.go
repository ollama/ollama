//go:build !windows

package tools

import (
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/agent"
)

func TestConfigureBashCommandSetsProcessGroup(t *testing.T) {
	cmd := exec.Command("bash", "-c", "true")
	configureBashCommand(cmd)
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.Setpgid {
		t.Fatalf("configureBashCommand should start bash in a new process group")
	}
}

func TestBashWaitDelayBoundsBackgroundOutputPipe(t *testing.T) {
	start := time.Now()
	result, err := (&Bash{}).Execute(context.Background(), agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": "sleep 5 & echo done",
	})
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > bashWaitDelay+2*time.Second {
		t.Fatalf("command elapsed = %s, want bounded near %s", elapsed, bashWaitDelay)
	}
	if !strings.Contains(result.Content, "done") {
		t.Fatalf("content = %q, want command output", result.Content)
	}
	if !strings.Contains(result.Content, "output pipes did not close") {
		t.Fatalf("content = %q, want wait delay message", result.Content)
	}
}
