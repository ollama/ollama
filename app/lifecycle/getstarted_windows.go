package lifecycle

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

func GetStarted() error {
	const CREATE_NEW_CONSOLE = 0x00000010
	var err error
	bannerScript := filepath.Join(AppDir, "ollama_welcome.ps1")
	args := []string{
		// TODO once we're signed, the execution policy bypass should be removed
		"powershell", "-noexit", "-ExecutionPolicy", "Bypass", "-nologo", "-file", bannerScript,
	}
	args[0], err = exec.LookPath(args[0])
	if err != nil {
		return err
	}

	// Make sure the script actually exists
	_, err = os.Stat(bannerScript)
	if err != nil {
		return fmt.Errorf("getting started banner script error %s", err)
	}

	slog.Info(fmt.Sprintf("opening getting started terminal with %v", args))
	attrs := &os.ProcAttr{
		Files: []*os.File{os.Stdin, os.Stdout, os.Stderr},
		Sys:   &syscall.SysProcAttr{CreationFlags: CREATE_NEW_CONSOLE, HideWindow: false},
	}
	proc, err := os.StartProcess(args[0], args, attrs)
	if err != nil {
		return fmt.Errorf("unable to start getting started shell %w", err)
	}

	slog.Debug(fmt.Sprintf("getting started terminal PID: %d", proc.Pid))
	return proc.Release()
}
