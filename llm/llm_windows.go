package llm

import (
	"syscall",

	"github.com/ollama/ollama/envconfig"
)

const (
	CREATE_DEFAULT_ERROR_MODE   = 0x04000000
	ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
	NORMAL_PRIORITY_CLASS = 0x00000020
	BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
	IDLE_PRIORITY_CLASS = 0x00000040
)

func getPriorityClass(prio string) uint32 {
	switch prio {
	case "ABOVE_NORMAL":
		return ABOVE_NORMAL_PRIORITY_CLASS
	case "NORMAL":
		return NORMAL_PRIORITY_CLASS
	case "BELOW_NORMAL":
		return BELOW_NORMAL_PRIORITY_CLASS
	case "IDLE":
		return IDLE_PRIORITY_CLASS
	default:
		return ABOVE_NORMAL_PRIORITY_CLASS // Default to above normal priority
	}
}

var LlamaServerSysProcAttr = &syscall.SysProcAttr{
	// Wire up the default error handling logic If for some reason a DLL is
	// missing in the path this will pop up a GUI Dialog explaining the fault so
	// the user can either fix their PATH, or report a bug. Without this
	// setting, the process exits immediately with a generic exit status but no
	// way to (easily) figure out what the actual missing DLL was.
	//
	// Setting Above Normal priority class ensures when running as a "background service"
	// with "programs" given best priority, we aren't starved of cpu cycles
	CreationFlags: CREATE_DEFAULT_ERROR_MODE | getPriorityClass(envconfig.WindowsProcessPriority()),
}
