//go:build windows

package llm

import (
	"fmt"
	"log/slog"

	"golang.org/x/sys/windows"
)

const (
	ntstatusSeverityMask  = 0xc0000000
	ntstatusSeverityError = 0xc0000000
)

func formatExitStatus(s ExitStatus) string {
	if s == exitStatusOK || s == exitStatusUnknown {
		return decimalExitStatus(s)
	}

	raw := uint32(s)
	if raw&ntstatusSeverityMask != ntstatusSeverityError {
		return decimalExitStatus(s)
	}

	return fmt.Sprintf("exit status 0x%08x: %s", raw, windows.NTStatus(raw).Error())
}

func logExitStatus(s ExitStatus) slog.Value {
	if s == exitStatusOK || s == exitStatusUnknown {
		return slog.StringValue(s.String())
	}

	raw := uint32(s)
	if raw&ntstatusSeverityMask != ntstatusSeverityError {
		return slog.IntValue(int(s))
	}

	return slog.GroupValue(
		slog.Int("code", int(s)),
		slog.String("hex", fmt.Sprintf("0x%08x", raw)),
		slog.String("ntstatus", windows.NTStatus(raw).Error()),
	)
}
