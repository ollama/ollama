//go:build windows

package wintray

import (
	"runtime"

	"golang.org/x/sys/windows"
)

var (
	k32 = windows.NewLazySystemDLL("Kernel32.dll")
	u32 = windows.NewLazySystemDLL("User32.dll")
	s32 = windows.NewLazySystemDLL("Shell32.dll")

	pCreatePopupMenu       = u32.NewProc("CreatePopupMenu")
	pCreateWindowEx        = u32.NewProc("CreateWindowExW")
	pDefWindowProc         = u32.NewProc("DefWindowProcW")
	pDestroyWindow         = u32.NewProc("DestroyWindow")
	pDispatchMessage       = u32.NewProc("DispatchMessageW")
	pGetCursorPos          = u32.NewProc("GetCursorPos")
	pGetMessage            = u32.NewProc("GetMessageW")
	pGetModuleHandle       = k32.NewProc("GetModuleHandleW")
	pInsertMenuItem        = u32.NewProc("InsertMenuItemW")
	pLoadCursor            = u32.NewProc("LoadCursorW")
	pLoadIcon              = u32.NewProc("LoadIconW")
	pLoadImage             = u32.NewProc("LoadImageW")
	pPostMessage           = u32.NewProc("PostMessageW")
	pPostQuitMessage       = u32.NewProc("PostQuitMessage")
	pRegisterClass         = u32.NewProc("RegisterClassExW")
	pRegisterWindowMessage = u32.NewProc("RegisterWindowMessageW")
	pSetForegroundWindow   = u32.NewProc("SetForegroundWindow")
	pSetMenuInfo           = u32.NewProc("SetMenuInfo")
	pSetMenuItemInfo       = u32.NewProc("SetMenuItemInfoW")
	pShellNotifyIcon       = s32.NewProc("Shell_NotifyIconW")
	pShowWindow            = u32.NewProc("ShowWindow")
	pTrackPopupMenu        = u32.NewProc("TrackPopupMenu")
	pTranslateMessage      = u32.NewProc("TranslateMessage")
	pUnregisterClass       = u32.NewProc("UnregisterClassW")
	pUpdateWindow          = u32.NewProc("UpdateWindow")
)

const (
	CS_HREDRAW          = 0x0002
	CS_VREDRAW          = 0x0001
	CW_USEDEFAULT       = 0x80000000
	IDC_ARROW           = 32512 // Standard arrow
	IDI_APPLICATION     = 32512
	IMAGE_ICON          = 1          // Loads an icon
	LR_DEFAULTSIZE      = 0x00000040 // Loads default-size icon for windows(SM_CXICON x SM_CYICON) if cx, cy are set to zero
	LR_LOADFROMFILE     = 0x00000010 // Loads the stand-alone image from the file
	MF_BYCOMMAND        = 0x00000000
	MFS_DISABLED        = 0x00000003
	MFT_SEPARATOR       = 0x00000800
	MFT_STRING          = 0x00000000
	MIIM_BITMAP         = 0x00000080
	MIIM_FTYPE          = 0x00000100
	MIIM_ID             = 0x00000002
	MIIM_STATE          = 0x00000001
	MIIM_STRING         = 0x00000040
	MIIM_SUBMENU        = 0x00000004
	MIM_APPLYTOSUBMENUS = 0x80000000
	NIF_ICON            = 0x00000002
	NIF_TIP             = 0x00000004
	NIF_INFO            = 0x00000010
	NIF_MESSAGE         = 0x00000001
	SW_HIDE             = 0
	TPM_BOTTOMALIGN     = 0x0020
	TPM_LEFTALIGN       = 0x0000
	WM_CLOSE            = 0x0010
	WM_USER             = 0x0400
	WS_CAPTION          = 0x00C00000
	WS_MAXIMIZEBOX      = 0x00010000
	WS_MINIMIZEBOX      = 0x00020000
	WS_OVERLAPPED       = 0x00000000
	WS_OVERLAPPEDWINDOW = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
	WS_SYSMENU          = 0x00080000
	WS_THICKFRAME       = 0x00040000
)

// Not sure if this is actually needed on windows
func init() {
	runtime.LockOSThread()
}

// The POINT structure defines the x- and y- coordinates of a point.
// https://msdn.microsoft.com/en-us/library/windows/desktop/dd162805(v=vs.85).aspx
type point struct {
	X, Y int32
}
