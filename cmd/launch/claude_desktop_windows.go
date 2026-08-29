//go:build windows

package launch

import (
	"context"
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

// powershellCommandContext prepares a powershell command that will not flash
// a console window. The desktop app runs as a windowsgui binary, so every
// child process would otherwise pop up a visible console; status polling
// invokes these commands several times per second.
func powershellCommandContext(ctx context.Context, script string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-Command", script)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: windows.CREATE_NO_WINDOW,
	}
	return cmd
}

func powershellCommand(script string) *exec.Cmd {
	return powershellCommandContext(context.Background(), script)
}

// claudeDesktopRunningScript reports the window-styled Claude process ID.
const claudeDesktopRunningScript = `(Get-Process claude -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1).Id`

func defaultClaudeDesktopProcessID(ctx context.Context) ([]byte, error) {
	return powershellCommandContext(ctx, claudeDesktopRunningScript).Output()
}

// openClaudeDesktopWindowsPath launches Claude without flashing a console.
func openClaudeDesktopWindowsPath(path string) error {
	return powershellCommand("Start-Process -FilePath " + quotePowerShellString(path)).Run()
}
