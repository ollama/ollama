//go:build windows

package main

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	user32   = windows.NewLazySystemDLL("user32.dll")
	kernel32 = windows.NewLazySystemDLL("kernel32.dll")

	procOpenClipboard    = user32.NewProc("OpenClipboard")
	procCloseClipboard   = user32.NewProc("CloseClipboard")
	procEmptyClipboard   = user32.NewProc("EmptyClipboard")
	procSetClipboardData = user32.NewProc("SetClipboardData")
	procGlobalAlloc      = kernel32.NewProc("GlobalAlloc")
	procGlobalFree       = kernel32.NewProc("GlobalFree")
	procGlobalLock       = kernel32.NewProc("GlobalLock")
	procGlobalUnlock     = kernel32.NewProc("GlobalUnlock")
	procRtlMoveMemory    = kernel32.NewProc("RtlMoveMemory")
)

// copyToClipboard puts text on the clipboard.
func copyToClipboard(text string) error {
	const (
		cfUnicodeText = 13
		gmemMoveable  = 0x0002
	)
	data, err := windows.UTF16FromString(text)
	if err != nil {
		return err
	}
	size := uintptr(len(data)) * unsafe.Sizeof(data[0])

	if r, _, err := procOpenClipboard.Call(0); r == 0 {
		return fmt.Errorf("open clipboard: %w", err)
	}
	defer procCloseClipboard.Call()
	procEmptyClipboard.Call()

	// The clipboard owns the memory once SetClipboardData succeeds.
	memory, _, err := procGlobalAlloc.Call(gmemMoveable, size)
	if memory == 0 {
		return fmt.Errorf("allocate clipboard memory: %w", err)
	}
	locked, _, err := procGlobalLock.Call(memory)
	if locked == 0 {
		procGlobalFree.Call(memory)
		return fmt.Errorf("lock clipboard memory: %w", err)
	}
	procRtlMoveMemory.Call(locked, uintptr(unsafe.Pointer(&data[0])), size)
	procGlobalUnlock.Call(memory)
	if r, _, err := procSetClipboardData.Call(cfUnicodeText, memory); r == 0 {
		procGlobalFree.Call(memory)
		return fmt.Errorf("set clipboard text: %w", err)
	}
	return nil
}
