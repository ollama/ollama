package llm

import (
	"syscall"
)

const (
	CREATE_DEFAULT_ERROR_MODE   = 0x04000000
	ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
	CREATE_NO_WINDOW            = 0x08000000
)

var LlamaServerSysProcAttr = &syscall.SysProcAttr{
	// Wire up the default error handling logic If for some reason a DLL is
	// missing in the path this will pop up a GUI Dialog explaining the fault so
	// the user can either fix their PATH, or report a bug. Without this
	// setting, the process exits immediately with a generic exit status but no
	// way to (easily) figure out what the actual missing DLL was.
	//
	// Setting Above Normal priority class ensures when running as a "background service"
	// with "programs" given best priority, we aren't starved of cpu cycles
	CreationFlags: CREATE_DEFAULT_ERROR_MODE | ABOVE_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW,
}
