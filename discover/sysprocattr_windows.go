//go:build windows

package discover

import "syscall"

// createNoWindow prevents a console window from appearing when a child
// process is launched. It uses the same value llm defines (0x08000000) but is
// kept local so we don't pull in llm.LlamaServerSysProcAttr's unrelated
// priority/error-mode flags.
const createNoWindow = 0x08000000

func discoverSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: createNoWindow}
}
