package llm

import (
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	runnerJobOnce sync.Once
	runnerJob     windows.Handle
	runnerJobErr  error
)

// AddRunnerToJob assigns a runner process to Ollama's process-lifetime Job Object.
func AddRunnerToJob(pid int) error {
	runnerJobOnce.Do(func() {
		runnerJob, runnerJobErr = createRunnerJob()
	})
	if runnerJobErr != nil {
		return runnerJobErr
	}

	return assignProcessToJob(runnerJob, pid)
}

func createRunnerJob() (windows.Handle, error) {
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

	// Keep the job open for the lifetime of ollama. Windows closes the handle
	// and terminates every assigned runner when ollama exits.
	return job, nil
}

func assignProcessToJob(job windows.Handle, pid int) error {
	process, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(process)

	return windows.AssignProcessToJobObject(job, process)
}
