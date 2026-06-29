//go:build windows

package tools

import (
	"context"
	"os/exec"
	"strings"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var bashJobHandles sync.Map

func shellToolName() string {
	return "powershell"
}

func shellToolDescription() string {
	return "Execute a PowerShell command on the system. Use this to inspect files, run tests, and perform development tasks."
}

func shellCommandDescription() string {
	return "The PowerShell command to execute."
}

func newBashCommand(ctx context.Context, command, cwdPath string) *exec.Cmd {
	return exec.CommandContext(
		ctx,
		"powershell.exe",
		"-NoLogo",
		"-NoProfile",
		"-NonInteractive",
		"-ExecutionPolicy",
		"Bypass",
		"-Command",
		powerShellCommandScript(command, cwdPath),
	)
}

func powerShellCommandScript(command, cwdPath string) string {
	cwdPath = powerShellSingleQuote(cwdPath)
	return strings.Join([]string{
		"$__ollama_status = 0",
		". {",
		"try {",
		command,
		"  $__ollama_success = $?",
		"  $__ollama_last_exit = $global:LASTEXITCODE",
		"  if ($__ollama_success) {",
		"    $__ollama_status = 0",
		"  } elseif ($__ollama_last_exit -is [int] -and $__ollama_last_exit -ne 0) {",
		"    $__ollama_status = $__ollama_last_exit",
		"  } else {",
		"    $__ollama_status = 1",
		"  }",
		"} catch {",
		"  Write-Error $_",
		"  $__ollama_status = 1",
		"} finally {",
		"  try { [System.IO.File]::WriteAllText(" + cwdPath + ", (Get-Location).ProviderPath, [System.Text.Encoding]::UTF8) } catch {}",
		"}",
		"} | Out-String -Stream",
		"exit $__ollama_status",
	}, "\n")
}

func powerShellSingleQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "''") + "'"
}

func runBashCommand(cmd *exec.Cmd) error {
	if err := cmd.Start(); err != nil {
		return err
	}
	if job, err := createBashJob(cmd.Process.Pid); err == nil {
		bashJobHandles.Store(cmd.Process.Pid, job)
		defer releaseBashJob(cmd.Process.Pid)
	}
	return cmd.Wait()
}

func killBashCommand(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	releaseBashJob(cmd.Process.Pid)
	_ = cmd.Process.Kill()
	return nil
}

func createBashJob(pid int) (windows.Handle, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return 0, err
	}

	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		_ = windows.CloseHandle(job)
		return 0, err
	}

	process, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		_ = windows.CloseHandle(job)
		return 0, err
	}
	defer windows.CloseHandle(process)

	if err := windows.AssignProcessToJobObject(job, process); err != nil {
		_ = windows.CloseHandle(job)
		return 0, err
	}
	return job, nil
}

func releaseBashJob(pid int) {
	value, ok := bashJobHandles.LoadAndDelete(pid)
	if !ok {
		return
	}
	if job, ok := value.(windows.Handle); ok {
		_ = windows.CloseHandle(job)
	}
}
