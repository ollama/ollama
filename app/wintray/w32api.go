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

	pCreateWindowEx        = u32.NewProc("CreateWindowExW")
	pDefWindowProc         = u32.NewProc("DefWindowProcW")
	pDestroyWindow         = u32.NewProc("DestroyWindow")
	pDispatchMessage       = u32.NewProc("DispatchMessageW")
	pFindWindow            = u32.NewProc("FindWindowW")
	pGetCursorPos          = u32.NewProc("GetCursorPos")
	pGetMessage            = u32.NewProc("GetMessageW")
	pGetModuleHandle       = k32.NewProc("GetModuleHandleW")
	pLoadCursor            = u32.NewProc("LoadCursorW")
	pLoadIcon              = u32.NewProc("LoadIconW")
	pLoadImage             = u32.NewProc("LoadImageW")
	pPostMessage           = u32.NewProc("PostMessageW")
	pPostQuitMessage       = u32.NewProc("PostQuitMessage")
	pRegisterClass         = u32.NewProc("RegisterClassExW")
	pRegisterWindowMessage = u32.NewProc("RegisterWindowMessageW")
	pSendMessage           = u32.NewProc("SendMessageW")
	pSetForegroundWindow   = u32.NewProc("SetForegroundWindow")
	pShellNotifyIcon       = s32.NewProc("Shell_NotifyIconW")
	pShowWindow            = u32.NewProc("ShowWindow")
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
	NIF_ICON            = 0x00000002
	NIF_TIP             = 0x00000004
	NIF_INFO            = 0x00000010
	NIF_MESSAGE         = 0x00000001
	SW_HIDE             = 0
	WM_CLOSE            = 0x0010
	WM_RBUTTONUP        = 0x0205
	WM_RBUTTONDOWN      = 0x0204
	WM_LBUTTONUP        = 0x0202
	WM_ENDSESSION       = 0x0016
	WM_QUIT             = 0x0012
	WM_DESTROY          = 0x0002
	WM_MOUSEMOVE        = 0x0200
	WM_LBUTTONDOWN      = 0x0201
	WM_USER             = 0x0400
	WM_COPYDATA         = 0x004A
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

// COPYDATASTRUCT contains data to be passed to another application by WM_COPYDATA
// https://docs.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-copydatastruct
type COPYDATASTRUCT struct {
	DwData uintptr
	CbData uint32
	LpData uintptr
}
