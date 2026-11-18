package applefm

import (
	"net/http"
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
	old := fmPath
	fmPath = path
	t.Cleanup(func() { fmPath = old })

	c := NewClient("apple-foundation-model")
	if _, err := c.Load(t.Context(), ml.SystemInfo{}, nil, false); err != nil {
		t.Fatal(err)
	}

	if err := c.WaitUntilRunning(t.Context()); err == nil || !strings.Contains(err.Error(), "exited") {
		t.Fatalf("error = %v, want exit error", err)
	}
	if !c.HasExited() {
		t.Error("HasExited = false after exit")
	}
	c.Close()
}

func TestWaitUntilRunningTimesOutOnHungHealthCheck(t *testing.T) {
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "1500ms")
	c := newFakeFM(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})

	err := c.WaitUntilRunning(t.Context())
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("error = %v, want timeout", err)
	}
}
