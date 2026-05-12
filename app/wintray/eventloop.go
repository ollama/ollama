//go:build windows

package wintray

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	quitOnce            sync.Once
	UI_REQUEST_MSG_ID   = WM_USER + 2
	FOCUS_WINDOW_MSG_ID = WM_USER + 3
)

func (t *winTray) TrayRun() {
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

		// Ignore WM_QUIT messages from the UI window, which shouldn't exit the main app
		if m.Message == WM_QUIT && t.app.UIRunning() {
			if t.app != nil {
				slog.Debug("converting WM_QUIT to terminate call on webview")
				t.app.UITerminate()
			}
			// Drain any other WM_QUIT messages
			for {
				ret, _, err = pGetMessage.Call(uintptr(unsafe.Pointer(m)), 0, 0, 0)
				if m.Message != WM_QUIT {
					break
				}
			}
		}

		// If the function retrieves a message other than WM_QUIT, the return value is nonzero.
		// If the function retrieves the WM_QUIT message, the return value is zero.
		// If there is an error, the return value is -1
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644936(v=vs.85).aspx
		switch int32(ret) {
		case -1:
			slog.Error(fmt.Sprintf("get message failure: %v", err))
			return
		case 0:
			// slog.Debug("XXX tray run loop exiting from handling", "message", fmt.Sprintf("0x%x", m.Message), "wParam", fmt.Sprintf("0x%x", m.Wparam), "lParam", fmt.Sprintf("0x%x", m.Lparam))
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
	// slog.Debug("XXX in winTray.wndProc", "message", fmt.Sprintf("0x%x", message), "wParam", fmt.Sprintf("0x%x", wParam), "lParam", fmt.Sprintf("0x%x", lParam))
	switch message {
	case WM_COMMAND:
		menuItemId := int32(wParam)
		// https://docs.microsoft.com/en-us/windows/win32/menurc/wm-command#menus
		switch menuItemId {
		case quitMenuID:
			t.app.Quit()
		case updateMenuID:
			t.app.DoUpdate()
		case openUIMenuID:
			// UI must be initialized on this thread so don't use the callbacks
			t.app.UIShow()
		case settingsUIMenuID:
			// UI must be initialized on this thread so don't use the callbacks
			t.app.UIRun("/settings")
		case diagLogsMenuID:
			t.showLogs()
		default:
			slog.Debug(fmt.Sprintf("Unexpected menu item id: %d", menuItemId))
			lResult, _, _ = pDefWindowProc.Call(
				uintptr(hWnd),
				uintptr(message),
				wParam,
				lParam,
			)
		}
	case WM_CLOSE:
		// TODO - does this need adjusting?
		// slog.Debug("XXX WM_CLOSE triggered")
		boolRet, _, err := pDestroyWindow.Call(uintptr(t.window))
		if boolRet == 0 {
			slog.Error(fmt.Sprintf("failed to destroy window: %s", err))
		}
		err = t.wcex.unregister()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to uregister windo %s", err))
		}
	case WM_DESTROY:
		// slog.Debug("XXX WM_DESTROY triggered")
		// TODO - does this need adjusting?
		// same as WM_ENDSESSION, but throws 0 exit code after all
		defer pPostQuitMessage.Call(uintptr(int32(0))) //nolint:errcheck
		fallthrough
	case WM_ENDSESSION:
		// slog.Debug("XXX WM_ENDSESSION triggered")
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
				// TODO - revamp how detecting an update is notified to the user
				t.app.DoUpdate()
			}
		case 0x404: // Middle click or close notification
			// slog.Debug("doing nothing on close of first time notification")
		default:
			// 0x402 also seems common - what is it?
			slog.Debug(fmt.Sprintf("unmanaged app message, lParm: 0x%x", lParam))
			lResult, _, _ = pDefWindowProc.Call(
				uintptr(hWnd),
				uintptr(message),
				wParam,
				lParam,
			)
		}
	case t.wmTaskbarCreated: // on explorer.exe restarts
		t.muNID.Lock()
		err := t.nid.add()
		if err != nil {
			slog.Error(fmt.Sprintf("failed to refresh the taskbar on explorer restart: %s", err))
		}
		t.muNID.Unlock()
	case uint32(UI_REQUEST_MSG_ID):
		// Requests for the UI must always come from the main event thread
		l := int(wParam)
		path := unsafe.String((*byte)(unsafe.Pointer(lParam)), l) //nolint:govet,gosec
		t.app.UIRun(path)
	case WM_COPYDATA:
		// Handle URL scheme requests from other instances
		if lParam != 0 {
			cds := (*COPYDATASTRUCT)(unsafe.Pointer(lParam)) //nolint:govet,gosec
			if cds.DwData == 1 {                             // Our identifier for URL scheme messages
				// Convert the data back to string
				data := make([]byte, cds.CbData)
				copy(data, (*[1 << 30]byte)(unsafe.Pointer(cds.LpData))[:cds.CbData:cds.CbData]) //nolint:govet,gosec
				urlScheme := string(data)
				handleURLSchemeRequest(urlScheme)
				lResult = 1 // Return non-zero to indicate success
			}
		}
	case uint32(FOCUS_WINDOW_MSG_ID):
		// Handle focus window request from another instance
		if t.app.UIRunning() {
			// If UI is already running, just show it
			t.app.UIShow()
		} else {
			// If UI is not running, start it
			t.app.UIRun("/")
		}
		lResult = 1 // Return non-zero to indicate success
	default:
		// Calls the default window procedure to provide default processing for any window messages that an application does not process.
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms633572(v=vs.85).aspx
		// slog.Debug("XXX passing through", "message", fmt.Sprintf("0x%x", message), "wParam", fmt.Sprintf("0x%x", wParam), "lParam", fmt.Sprintf("0x%x", lParam))
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
	// slog.Debug("XXX in winTray.Quit")
	t.quitting = true
	quitOnce.Do(quit)
}

func SendUIRequestMessage(path string) {
	boolRet, _, err := pPostMessage.Call(
		uintptr(wt.window),
		uintptr(UI_REQUEST_MSG_ID),
		uintptr(len(path)),
		uintptr(unsafe.Pointer(unsafe.StringData(path))),
	)
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to post UI request message %s", err))
	}
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

// findExistingInstance attempts to find an existing Ollama instance window
// Returns the window handle if found, 0 if not found
func findExistingInstance() uintptr {
	classNamePtr, err := windows.UTF16PtrFromString(ClassName)
	if err != nil {
		slog.Error("failed to convert class name to UTF16", "error", err)
		return 0
	}

	hwnd, _, _ := pFindWindow.Call(
		uintptr(unsafe.Pointer(classNamePtr)),
		0, // window name (null = any)
	)

	return hwnd
}

// CheckAndSendToExistingInstance attempts to send a URL scheme to an existing instance
// Returns true if successfully sent to existing instance, false if no instance found
func CheckAndSendToExistingInstance(urlScheme string) bool {
	hwnd := findExistingInstance()
	if hwnd == 0 {
		// No existing window found
		return false
	}

	data := []byte(urlScheme)
	cds := COPYDATASTRUCT{
		DwData: 1, // 1 to identify URL scheme messages
		CbData: uint32(len(data)),
		LpData: uintptr(unsafe.Pointer(&data[0])),
	}

	result, _, err := pSendMessage.Call(
		hwnd,
		uintptr(WM_COPYDATA),
		0, // wParam is handle to sending window (0 is ok)
		uintptr(unsafe.Pointer(&cds)),
	)

	// SendMessage returns the result from the window procedure
	// For WM_COPYDATA, non-zero means success
	if result == 0 {
		slog.Error("failed to send URL scheme message to existing instance", "error", err)
		return false
	}
	return true
}

// handleURLSchemeRequest processes a URL scheme request
func handleURLSchemeRequest(urlScheme string) {
	if urlScheme == "" {
		slog.Warn("empty URL scheme request")
		return
	}

	// Call the app callback to handle URL scheme requests
	// This will delegate to the main app logic
	if wt.app != nil {
		if urlHandler, ok := wt.app.(URLSchemeHandler); ok {
			urlHandler.HandleURLScheme(urlScheme)
		} else {
			slog.Warn("app does not implement URLSchemeHandler interface")
		}
	} else {
		slog.Warn("wt.app is nil")
	}
}

// CheckAndFocusExistingInstance attempts to find an existing instance and optionally focus it
// Returns true if an existing instance was found, false otherwise
func CheckAndFocusExistingInstance(shouldFocus bool) bool {
	hwnd := findExistingInstance()
	if hwnd == 0 {
		// No existing window found
		return false
	}

	if !shouldFocus {
		slog.Info("existing instance found, not focusing due to startHidden")
		return true
	}

	// Send focus message to existing instance
	result, _, err := pSendMessage.Call(
		hwnd,
		uintptr(FOCUS_WINDOW_MSG_ID),
		0, // wParam not used
		0, // lParam not used
	)

	// SendMessage returns the result from the window procedure
	// For our custom message, non-zero means success
	if result == 0 {
		slog.Error("failed to send focus message to existing instance", "error", err)
		return false
	}

	slog.Info("sent focus request to existing instance")

	return true
}
