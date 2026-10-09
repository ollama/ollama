package llm

import (
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
)

type ExitStatus int

const (
	exitStatusUnknown ExitStatus = -1
	exitStatusOK      ExitStatus = 0
)

func ExitStatusFromError(err error) ExitStatus {
	if err == nil {
		return exitStatusOK
	}

	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return ExitStatus(exitErr.ExitCode())
	}

	return exitStatusUnknown
}

func (s ExitStatus) Known() bool {
	return s != exitStatusUnknown
}

func (s ExitStatus) String() string {
	return formatExitStatus(s)
}

func (s ExitStatus) LogValue() slog.Value {
	return logExitStatus(s)
}

func decimalExitStatus(s ExitStatus) string {
	switch s {
	case exitStatusOK:
		return "OK"
	case exitStatusUnknown:
		return "unknown"
	}

	return fmt.Sprintf("exit status %d", int(s))
}
