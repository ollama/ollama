//go:build windows

package wintray

import (
	"fmt"
	"log/slog"
	"runtime"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	focusWindowMessage = wmUser + 3 // Kept stable for compatibility with older Ollama builds.
	urlSchemeMessageID = 1
	// Bound allocation from the cross-process WM_COPYDATA payload to 1 MiB.
	maxURLSchemeBytes = 1 << 20
	// Do not let a hung existing process stall app startup for more than two seconds.
	messageTimeoutMS = 2000
)

func findExistingInstance() uintptr {
	className, err := windows.UTF16PtrFromString(ClassName)
	if err != nil {
		slog.Error("failed to encode instance window class", "error", err)
		return 0
	}
	hwnd, _, _ := pFindWindow.Call(uintptr(unsafe.Pointer(className)), 0)
	return hwnd
}

func CheckAndSendToExistingInstance(urlScheme string) bool {
	hwnd := findExistingInstance()
	if hwnd == 0 {
		return false
	}

	data := []byte(urlScheme)
	if len(data) == 0 || len(data) > maxURLSchemeBytes {
		return false
	}
	message := copyDataStruct{
		dataID:    urlSchemeMessageID,
		byteCount: uint32(len(data)),
		data:      uintptr(unsafe.Pointer(&data[0])),
	}
	result, ok := sendMessageWithTimeout(hwnd, wmCopyData, 0, uintptr(unsafe.Pointer(&message)))
	runtime.KeepAlive(data)
	return ok && result != 0
}

func CheckAndFocusExistingInstance(shouldFocus bool) bool {
	hwnd := findExistingInstance()
	if hwnd == 0 {
		return false
	}
	if !shouldFocus {
		slog.Info("existing instance found, not focusing due to hidden startup")
		return true
	}

	result, ok := sendMessageWithTimeout(hwnd, focusWindowMessage, 0, 0)
	if !ok || result == 0 {
		slog.Warn("existing Ollama instance did not accept focus request")
		return false
	}
	slog.Info("sent focus request to existing instance")
	return true
}

func sendMessageWithTimeout(hwnd, message, wParam, lParam uintptr) (uintptr, bool) {
	var result uintptr
	ok, _, err := pSendMessageTimeout.Call(
		hwnd,
		message,
		wParam,
		lParam,
		sendMessageAbortIfHung,
		messageTimeoutMS,
		uintptr(unsafe.Pointer(&result)),
	)
	if ok == 0 {
		slog.Warn("Windows instance message timed out", "message", fmt.Sprintf("0x%x", message), "error", err)
		return 0, false
	}
	return result, true
}
