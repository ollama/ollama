package lifecycle

import (
	"context"
	"fmt"
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

func getCmd(ctx context.Context, exePath string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, exePath, "serve")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: windows.CREATE_NEW_PROCESS_GROUP,
	}

	return cmd
}

func terminate(cmd *exec.Cmd) error {
	dll, err := windows.LoadDLL("kernel32.dll")
	if err != nil {
		return err
	}
	//nolint:errcheck
	defer dll.Release()

	pid := cmd.Process.Pid

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

func isProcessExited(pid int) (bool, error) {
	hProcess, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION, false, uint32(pid))
	if err != nil {
		return false, fmt.Errorf("failed to open process: %v", err)
	}
	//nolint:errcheck
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
