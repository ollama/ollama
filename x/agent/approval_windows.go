//go:build windows

package agent

import (
	"os"

	"golang.org/x/sys/windows"
)

// flushStdin clears any buffered console input on Windows.
func flushStdin(_ int) {
	handle := windows.Handle(os.Stdin.Fd())
	_ = windows.FlushConsoleInputBuffer(handle)
}
