//go:build windows

package wintray

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var quitOnce sync.Once

func (t *winTray) Run() {
	nativeLoop()
}

func nativeLoop() {
	// Main message pump.
	slog.Debug("starting event handling loop")
	m := &struct {
		WindowHandle windows.Handle
		Message      uint32
		Wparam       uintptr
		Lparam       uintptr
		Time         uint32
		Pt           point
		LPrivate     uint32
	}{}
	for {
		ret, _, err := pGetMessage.Call(uintptr(unsafe.Pointer(m)), 0, 0, 0)

		// If the function retrieves a message other than WM_QUIT, the return value is nonzero.
		// If the function retrieves the WM_QUIT message, the return value is zero.
		// If there is an error, the return value is -1
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644936(v=vs.85).aspx
		switch int32(ret) {
		case -1:
			slog.Error(fmt.Sprintf("get message failure: %v", err))
			return
		case 0:
			return
		default:
			pTranslateMessage.Call(uintptr(unsafe.Pointer(m))) //nolint:errcheck
			pDispatchMessage.Call(uintptr(unsafe.Pointer(m)))  //nolint:errcheck
		}
	}
}

// WindowProc callback function that processes messages sent to a window.
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms633573(v=vs.85).aspx
func (t *winTray) wndProc(hWnd windows.Handle, message uint32, wParam, lParam uintptr) (lResult uintptr) {
	const (
		WM_RBUTTONUP   = 0x0205
		WM_LBUTTONUP   = 0x0202
		WM_COMMAND     = 0x0111
		WM_ENDSESSION  = 0x0016
		WM_CLOSE       = 0x0010
		WM_DESTROY     = 0x0002
		WM_MOUSEMOVE   = 0x0200
		WM_LBUTTONDOWN = 0x0201
	)
	switch message {
	case WM_COMMAND:
		menuItemId := int32(wParam)
		// https://docs.microsoft.com/en-us/windows/win32/menurc/wm-command#menus
		switch menuItemId {
		case quitMenuID:
			select {
			case t.callbacks.Quit <- struct{}{}:
			// should not happen but in case not listening
			default:
				slog.Error("no listener on Quit")
			}
		case updateMenuID:
			select {
			case t.callbacks.Update <- struct{}{}:
			// should not happen but in case not listening
			default:
				slog.Error("no listener on Update")
			}
		case diagLogsMenuID:
			select {
			case t.callbacks.ShowLogs <- struct{}{}:
			// should not happen but in case not listening
			default:
				slog.Error("no listener on ShowLogs")
			}
		default:
			slog.Debug(fmt.Sprintf("Unexpected menu item id: %d", menuItemId))
		}
	case WM_CLOSE:
		boolRet, _, err := pDestroyWindow.Call(uintptr(t.window))
		if boolRet == 0 {
			slog.Error(fmt.Sprintf("failed to destroy window: %s", err))
		}
		err = t.wcex.unregister()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to unregister window %s", err))
		}
	case WM_DESTROY:
		// same as WM_ENDSESSION, but throws 0 exit code after all
		defer pPostQuitMessage.Call(uintptr(int32(0))) //nolint:errcheck
		fallthrough
	case WM_ENDSESSION:
		t.muNID.Lock()
		if t.nid != nil {
			err := t.nid.delete()
			if err != nil {
				slog.Error(fmt.Sprintf("failed to delete nid: %s", err))
			}
		}
		t.muNID.Unlock()
	case t.wmSystrayMessage:
		switch lParam {
		case WM_MOUSEMOVE, WM_LBUTTONDOWN:
			// Ignore these...
		case WM_RBUTTONUP, WM_LBUTTONUP:
			err := t.showMenu()
			if err != nil {
				slog.Error(fmt.Sprintf("failed to show menu: %s", err))
			}
		case 0x405: // TODO - how is this magic value derived for the notification left click
			if t.pendingUpdate {
				select {
				case t.callbacks.Update <- struct{}{}:
				// should not happen but in case not listening
				default:
					slog.Error("no listener on Update")
				}
			} else {
				select {
				case t.callbacks.DoFirstUse <- struct{}{}:
				// should not happen but in case not listening
				default:
					slog.Error("no listener on DoFirstUse")
				}
			}
		case 0x404: // Middle click or close notification
			// slog.Debug("doing nothing on close of first time notification")
		default:
			// 0x402 also seems common - what is it?
			slog.Debug(fmt.Sprintf("unmanaged app message, lParm: 0x%x", lParam))
		}
	case t.wmTaskbarCreated: // on explorer.exe restarts
		t.muNID.Lock()
		err := t.nid.add()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to refresh the taskbar on explorer restart: %s", err))
		}
		t.muNID.Unlock()
	default:
		// Calls the default window procedure to provide default processing for any window messages that an application does not process.
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms633572(v=vs.85).aspx
		lResult, _, _ = pDefWindowProc.Call(
			uintptr(hWnd),
			uintptr(message),
			wParam,
			lParam,
		)
	}
	return
}

func (t *winTray) Quit() {
	quitOnce.Do(quit)
}

func quit() {
	boolRet, _, err := pPostMessage.Call(
		uintptr(wt.window),
		WM_CLOSE,
		0,
		0,
	)
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to post close message on shutdown %s", err))
	}
}
