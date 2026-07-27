//go:build darwin

// Package posixspawn provides a posix_spawnp-based replacement for
// exec.Cmd.Start on macOS to work around atfork-handler hangs.
//
// On macOS, fork() in a process that has loaded libmlx.dylib or initialized
// Metal/Network.framework can dead-loop inside the atfork child handlers
// (nw_settings_child_has_forked -> _os_log_preferences_refresh).
// posix_spawnp bypasses atfork handlers entirely.
//
// See: https://github.com/ollama/ollama/issues/17057
package posixspawn

/*
#include <errno.h>
#include <spawn.h>
#include <stdlib.h>
#include <unistd.h>

static int ollamaPosixSpawnp(pid_t *pid, const char *file,
                             char *const argv[], char *const envp[],
                             int outfd, int errfd) {
	posix_spawn_file_actions_t actions;
	posix_spawnattr_t attr;
	int ret;

	posix_spawn_file_actions_init(&actions);
	posix_spawnattr_init(&attr);

	// Queue all dup2 operations first — a close before a dup2 would destroy
	// the source fd needed for a subsequent dup2 (combined-pipe case).
	if (outfd >= 0) {
		posix_spawn_file_actions_adddup2(&actions, outfd, STDOUT_FILENO);
	}
	if (errfd >= 0 && errfd != outfd) {
		posix_spawn_file_actions_adddup2(&actions, errfd, STDERR_FILENO);
	} else if (outfd >= 0) {
		posix_spawn_file_actions_adddup2(&actions, outfd, STDERR_FILENO);
	}

	// Now close the original fds — they have already been dup'd above.
	if (outfd >= 0) {
		posix_spawn_file_actions_addclose(&actions, outfd);
	}
	if (errfd >= 0 && errfd != outfd) {
		posix_spawn_file_actions_addclose(&actions, errfd);
	}

	posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
	posix_spawnattr_setpgroup(&attr, 0);

	ret = posix_spawnp(pid, file, &actions, &attr, argv, envp);

	posix_spawn_file_actions_destroy(&actions);
	posix_spawnattr_destroy(&attr);

	return ret;
}
*/
import "C"

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
	"unsafe"
)

// StartCmd spawns cmd using posix_spawnp instead of fork/exec.
//
// It handles pipe creation for cmd.Stdout and cmd.Stderr, launches the
// child via posix_spawnp, and starts goroutines that copy from the pipe
// read ends to the cmd output writers.
func StartCmd(cmd *exec.Cmd) error {
	cArgv := make([]*C.char, len(cmd.Args)+1)
	for i, s := range cmd.Args {
		cArgv[i] = C.CString(s)
	}
	cArgv[len(cmd.Args)] = nil
	defer func() {
		for i := range len(cmd.Args) {
			C.free(unsafe.Pointer(cArgv[i]))
		}
	}()

	cEnvp := make([]*C.char, len(cmd.Env)+1)
	for i, s := range cmd.Env {
		cEnvp[i] = C.CString(s)
	}
	cEnvp[len(cmd.Env)] = nil
	defer func() {
		for i := range len(cmd.Env) {
			C.free(unsafe.Pointer(cEnvp[i]))
		}
	}()

	var (
		outRead, outWrite *os.File
		errRead, errWrite *os.File
		pipeErr           error
		combined          bool
	)
	if cmd.Stdout != nil {
		outRead, outWrite, pipeErr = os.Pipe()
		if pipeErr != nil {
			return fmt.Errorf("stdout pipe: %w", pipeErr)
		}
	}

	if cmd.Stderr != nil {
		if cmd.Stderr == cmd.Stdout {
			errRead, errWrite = outRead, outWrite
			combined = true
		} else {
			errRead, errWrite, pipeErr = os.Pipe()
			if pipeErr != nil {
				if outRead != nil {
					outRead.Close()
					outWrite.Close()
				}
				return fmt.Errorf("stderr pipe: %w", pipeErr)
			}
		}
	}

	outFd := C.int(-1)
	errFd := C.int(-1)
	if outWrite != nil {
		outFd = C.int(outWrite.Fd())
	}
	if errWrite != nil && errWrite != outWrite {
		errFd = C.int(errWrite.Fd())
	}

	var pid C.pid_t
	ret := C.ollamaPosixSpawnp(&pid, cArgv[0], &cArgv[0], &cEnvp[0], outFd, errFd)
	if ret != 0 {
		if outRead != nil {
			outRead.Close()
			outWrite.Close()
		}
		if errRead != nil && errRead != outRead {
			errRead.Close()
			errWrite.Close()
		}
		return fmt.Errorf("posix_spawnp: %w", syscall.Errno(ret))
	}

	if outWrite != nil {
		outWrite.Close()
	}
	if errWrite != nil && errWrite != outWrite {
		errWrite.Close()
	}

	cmd.Process, _ = os.FindProcess(int(pid))

	if outRead != nil && cmd.Stdout != nil {
		go func() {
			io.Copy(cmd.Stdout, outRead)
			outRead.Close()
		}()
	}
	if errRead != nil && errRead != outRead && cmd.Stderr != nil {
		go func() {
			io.Copy(cmd.Stderr, errRead)
			errRead.Close()
		}()
	}

	return nil
}
