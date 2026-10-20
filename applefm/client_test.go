package applefm

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestWaitUntilRunningReportsExit(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires a unix false command")
	}
	path, err := exec.LookPath("false")
	if err != nil {
		t.Skip(err)
	}
	t.Cleanup(func(old string) func() { return func() { fmPath = old } }(fmPath))
	fmPath = path

	c, err := NewClient("apple-foundation-model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Load(context.Background(), ml.SystemInfo{}, nil, false); err != nil {
		t.Fatal(err)
	}
	sockDir := c.sockDir

	if err := c.WaitUntilRunning(t.Context()); err == nil || !strings.Contains(err.Error(), "exited") {
		t.Fatalf("error = %v, want exit error", err)
	}
	if !c.HasExited() {
		t.Error("HasExited = false after exit")
	}

	c.Close()
	if _, err := os.Stat(sockDir); !os.IsNotExist(err) {
		t.Errorf("socket directory not removed: %v", err)
	}
}
