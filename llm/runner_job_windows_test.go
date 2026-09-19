package llm

import (
	"os"
	"os/exec"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

const runnerJobHelperEnv = "OLLAMA_TEST_RUNNER_JOB_HELPER"

func TestRunnerJobKillsOnlyAssignedProcess(t *testing.T) {
	owned := startRunnerJobHelper(t)
	unrelated := startRunnerJobHelper(t)

	job, err := createRunnerJob()
	if err != nil {
		t.Fatalf("createRunnerJob(): %v", err)
	}
	if err := assignProcessToJob(job, owned.Process.Pid); err != nil {
		_ = windows.CloseHandle(job)
		t.Fatalf("assignProcessToJob(%d): %v", owned.Process.Pid, err)
	}
	if err := windows.CloseHandle(job); err != nil {
		t.Fatalf("close job: %v", err)
	}

	wait := make(chan error, 1)
	go func() {
		wait <- owned.Wait()
	}()
	select {
	case <-wait:
	case <-time.After(5 * time.Second):
		t.Fatal("assigned process remained alive after its job was closed")
	}

	process, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(unrelated.Process.Pid))
	if err != nil {
		t.Fatalf("open unrelated process %d: %v", unrelated.Process.Pid, err)
	}
	defer windows.CloseHandle(process)

	status, err := windows.WaitForSingleObject(process, 0)
	if err != nil {
		t.Fatalf("query unrelated process %d: %v", unrelated.Process.Pid, err)
	}
	if status != uint32(windows.WAIT_TIMEOUT) {
		t.Fatalf("unrelated process wait status = %d, want WAIT_TIMEOUT", status)
	}
}

func TestRunnerJobHelper(t *testing.T) {
	if os.Getenv(runnerJobHelperEnv) == "" {
		return
	}

	time.Sleep(time.Hour)
}

func startRunnerJobHelper(t *testing.T) *exec.Cmd {
	t.Helper()

	cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestRunnerJobHelper$")
	cmd.Env = append(os.Environ(), runnerJobHelperEnv+"=1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper process: %v", err)
	}
	t.Cleanup(func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})
	return cmd
}
