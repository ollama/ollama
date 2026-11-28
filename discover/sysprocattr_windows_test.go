//go:build windows

package discover

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestDiscoverSysProcAttrHasNoWindow(t *testing.T) {
	attr := discoverSysProcAttr()
	if attr == nil {
		t.Fatal("expected non-nil SysProcAttr on Windows")
	}
	if attr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("expected CREATE_NO_WINDOW flag, got %#x", attr.CreationFlags)
	}
}

// TestDiscoveryConsoleChild is the child half of TestDiscoveryConsoleNative.
// It runs only when invoked as a subprocess (OLLAMA_CONSOLE_TEST_CHILD=1) and
// verifies that the current process has no associated console window while
// still preserving stdout/stderr.
func TestDiscoveryConsoleChild(t *testing.T) {
	if os.Getenv("OLLAMA_CONSOLE_TEST_CHILD") != "1" {
		return
	}
	hwnd, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("GetConsoleWindow").Call()
	if hwnd != 0 {
		os.Exit(23)
	}
	os.Stdout.WriteString("console-absent\n")
	os.Stderr.WriteString("stderr-preserved\n")
	os.Exit(0)
}

// TestDiscoveryConsoleNative launches a real child process with the same
// SysProcAttr that the GPU-discovery launch sites use (discoverSysProcAttr),
// and asserts the child has no console window while stdout/stderr are intact.
// It verifies the helper's behavior (a child launched with discoverSysProcAttr
// gets no console window); it does not assert that every production call site
// applies the helper.
func TestDiscoveryConsoleNative(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestDiscoveryConsoleChild$")
	cmd.Env = append(os.Environ(), "OLLAMA_CONSOLE_TEST_CHILD=1")
	cmd.SysProcAttr = discoverSysProcAttr()

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("hidden child failed: %v: %s", err, out)
	}
	if !strings.Contains(string(out), "console-absent") {
		t.Fatalf("child launched with a console window; output=%s", out)
	}
	if !strings.Contains(string(out), "stderr-preserved") {
		t.Fatalf("stderr not preserved; output=%s", out)
	}
}
