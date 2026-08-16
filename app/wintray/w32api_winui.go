//go:build windows && winui

package wintray

import "golang.org/x/sys/windows"

var (
	kernel32 = windows.NewLazySystemDLL("Kernel32.dll")
	comctl32 = windows.NewLazySystemDLL("Comctl32.dll")
	dwmapi   = windows.NewLazySystemDLL("Dwmapi.dll")

	pCreateWindowEx        = user32.NewProc("CreateWindowExW")
	pDefWindowProc         = user32.NewProc("DefWindowProcW")
	pDefSubclassProc       = comctl32.NewProc("DefSubclassProc")
	pDestroyIcon           = user32.NewProc("DestroyIcon")
	pDestroyWindow         = user32.NewProc("DestroyWindow")
	pDwmGetWindowAttribute = dwmapi.NewProc("DwmGetWindowAttribute")
	pFindWindowEx          = user32.NewProc("FindWindowExW")
	pGetDpiForWindow       = user32.NewProc("GetDpiForWindow")
	pGetMonitorInfo        = user32.NewProc("GetMonitorInfoW")
	pGetModuleHandle       = kernel32.NewProc("GetModuleHandleW")
	pGetWindowRect         = user32.NewProc("GetWindowRect")
	pGetWindowThreadProcID = user32.NewProc("GetWindowThreadProcessId")
	pIsWindowVisible       = user32.NewProc("IsWindowVisible")
	pKillTimer             = user32.NewProc("KillTimer")
	pLoadImage             = user32.NewProc("LoadImageW")
	pMonitorFromRect       = user32.NewProc("MonitorFromRect")
	pRegisterClass         = user32.NewProc("RegisterClassExW")
	pRegisterWindowMessage = user32.NewProc("RegisterWindowMessageW")
	pRemoveWindowSubclass  = comctl32.NewProc("RemoveWindowSubclass")
	pRtlMoveMemory         = kernel32.NewProc("RtlMoveMemory")
	pSetTimer              = user32.NewProc("SetTimer")
	pSetWindowPos          = user32.NewProc("SetWindowPos")
	pSetWindowSubclass     = comctl32.NewProc("SetWindowSubclass")
	pUnregisterClass       = user32.NewProc("UnregisterClassW")
)

const (
	imageIcon       = 1
	loadFromFile    = 0x00000010
	loadDefaultSize = 0x00000040

	monitorDefaultToNearest  = 2
	dwmwaExtendedFrameBounds = 9

	swpNoSize     = 0x0001
	swpNoMove     = 0x0002
	swpNoZOrder   = 0x0004
	swpNoActivate = 0x0010

	wmWindowPosChanging = 0x0046
	wmClose             = 0x0010
	wmEndSession        = 0x0016

	wsOverlapped = 0x00000000
)

type monitorInfo struct {
	size    uint32
	monitor windows.Rect
	work    windows.Rect
	flags   uint32
}

type windowPos struct {
	window      uintptr
	insertAfter uintptr
	X           int32
	Y           int32
	width       int32
	height      int32
	flags       uint32
}
