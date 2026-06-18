//go:build windows

package tools

import (
	"os/exec"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var bashJobHandles sync.Map

func configureBashCommand(*exec.Cmd) {}

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
