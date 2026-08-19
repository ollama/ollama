//go:build windows

package wintray

import "golang.org/x/sys/windows"

var (
	user32 = windows.NewLazySystemDLL("User32.dll")

	pFindWindow         = user32.NewProc("FindWindowW")
	pSendMessageTimeout = user32.NewProc("SendMessageTimeoutW")
)

const (
	sendMessageAbortIfHung = 0x0002

	wmCopyData = 0x004A
	wmUser     = 0x0400
)

type copyDataStruct struct {
	dataID    uintptr
	byteCount uint32
	data      uintptr
}
