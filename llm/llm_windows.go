package llm

import (
	"embed"
	"syscall"
)

// unused on windows
var libEmbed embed.FS

const CREATE_DEFAULT_ERROR_MODE = 0x04000000

var LlamaServerSysProcAttr = &syscall.SysProcAttr{
	// Wire up the default error handling logic If for some reason a DLL is
	// missing in the path this will pop up a GUI Dialog explaining the fault so
	// the user can either fix their PATH, or report a bug. Without this
	// setting, the process exits immediately with a generic exit status but no
	// way to (easily) figure out what the actual missing DLL was.
	CreationFlags: CREATE_DEFAULT_ERROR_MODE,
}
