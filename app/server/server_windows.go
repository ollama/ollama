package server

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/windows"
)

var (
	pidFile       = filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "ollama.pid")
	serverLogPath = filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama", "server.log")
)

func commandContext(ctx context.Context, name string, arg ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, arg...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: windows.CREATE_NEW_PROCESS_GROUP,
	}

	return cmd
}

func terminate(proc *os.Process) error {
	dll, err := windows.LoadDLL("kernel32.dll")
	if err != nil {
		return err
	}
	defer dll.Release()

	pid := proc.Pid

	f, err := dll.FindProc("AttachConsole")
	if err != nil {
		return err
	}

	r1, _, err := f.Call(uintptr(pid))
	if r1 == 0 && err != syscall.ERROR_ACCESS_DENIED {
		return err
	}

	f, err = dll.FindProc("SetConsoleCtrlHandler")
	if err != nil {
		return err
	}

	r1, _, err = f.Call(0, 1)
	if r1 == 0 {
		return err
	}

	f, err = dll.FindProc("GenerateConsoleCtrlEvent")
	if err != nil {
		return err
	}

	r1, _, err = f.Call(windows.CTRL_BREAK_EVENT, uintptr(pid))
	if r1 == 0 {
		return err
	}

	r1, _, err = f.Call(windows.CTRL_C_EVENT, uintptr(pid))
	if r1 == 0 {
		return err
	}

	return nil
}

const STILL_ACTIVE = 259

func terminated(pid int) (bool, error) {
	hProcess, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION, false, uint32(pid))
	if err != nil {
		if errno, ok := err.(windows.Errno); ok && errno == windows.ERROR_INVALID_PARAMETER {
			return true, nil
		}
		return false, fmt.Errorf("failed to open process: %v", err)
	}
	defer windows.CloseHandle(hProcess)

	var exitCode uint32
	err = windows.GetExitCodeProcess(hProcess, &exitCode)
	if err != nil {
		return false, fmt.Errorf("failed to get exit code: %v", err)
	}

	if exitCode == STILL_ACTIVE {
		return false, nil
	}

	return true, nil
}

// reapServers kills all ollama processes except our own
func reapServers() error {
	// Get current process ID to avoid killing ourselves
	currentPID := os.Getpid()

	// Use wmic to find ollama processes
	cmd := exec.Command("wmic", "process", "where", "name='ollama.exe'", "get", "ProcessId")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	output, err := cmd.Output()
	if err != nil {
		// No ollama processes found
		slog.Debug("no ollama processes found")
		return nil //nolint:nilerr
	}

	lines := strings.Split(string(output), "\n")
	var pids []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == "ProcessId" {
			continue
		}

		if _, err := strconv.Atoi(line); err == nil {
			pids = append(pids, line)
		}
	}

	for _, pidStr := range pids {
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			continue
		}

		if pid == currentPID {
			continue
		}

		cmd := exec.Command("taskkill", "/F", "/PID", pidStr)
		if err := cmd.Run(); err != nil {
			slog.Warn("failed to kill ollama process", "pid", pid, "err", err)
		}
	}

	return nil
}
