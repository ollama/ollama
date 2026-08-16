//go:build windows && winui

package wintray

import (
	"fmt"
	"log/slog"
	"unsafe"

	"golang.org/x/sys/windows"
)

type instanceWindow struct {
	app                   AppCallbacks
	window                windows.Handle
	instance              windows.Handle
	class                 *wndClassEx
	taskbarCreatedMessage uint32
	onTaskbarCreated      func()
}

func newInstanceWindow(classNameText string, app AppCallbacks, onTaskbarCreated func()) (*instanceWindow, error) {
	w := &instanceWindow{app: app, onTaskbarCreated: onTaskbarCreated}

	messageName, err := windows.UTF16PtrFromString("TaskbarCreated")
	if err != nil {
		return nil, err
	}
	message, _, callErr := pRegisterWindowMessage.Call(uintptr(unsafe.Pointer(messageName)))
	if message == 0 {
		return nil, fmt.Errorf("register TaskbarCreated: %w", callErr)
	}
	w.taskbarCreatedMessage = uint32(message)

	instance, _, callErr := pGetModuleHandle.Call(0)
	if instance == 0 {
		return nil, fmt.Errorf("get module handle: %w", callErr)
	}
	w.instance = windows.Handle(instance)

	className, err := windows.UTF16PtrFromString(classNameText)
	if err != nil {
		return nil, err
	}
	w.class = &wndClassEx{
		WndProc:   windows.NewCallback(w.windowProc),
		Instance:  w.instance,
		ClassName: className,
	}
	if err := w.class.register(); err != nil {
		return nil, err
	}

	handle, _, callErr := pCreateWindowEx.Call(
		0,
		uintptr(unsafe.Pointer(className)),
		0,
		wsOverlapped,
		0, 0, 0, 0,
		0, 0,
		uintptr(w.instance),
		0,
	)
	if handle == 0 {
		_ = w.class.unregister()
		return nil, fmt.Errorf("create window: %w", callErr)
	}
	w.window = windows.Handle(handle)
	return w, nil
}

func (w *instanceWindow) windowProc(hwnd windows.Handle, message uint32, wParam, lParam uintptr) uintptr {
	switch message {
	case wmCopyData:
		if lParam == 0 {
			return 0
		}
		var data copyDataStruct
		pRtlMoveMemory.Call(uintptr(unsafe.Pointer(&data)), lParam, unsafe.Sizeof(data)) //nolint:errcheck
		if data.dataID != urlSchemeMessageID || data.byteCount == 0 || data.byteCount > maxURLSchemeBytes || data.data == 0 {
			return 0
		}
		bytes := make([]byte, data.byteCount)
		pRtlMoveMemory.Call(uintptr(unsafe.Pointer(&bytes[0])), data.data, uintptr(data.byteCount)) //nolint:errcheck
		urlScheme := string(bytes)
		if handler, ok := w.app.(URLSchemeHandler); ok {
			handler.HandleURLScheme(urlScheme)
			return 1
		}
		slog.Warn("app does not implement URL scheme handling")
		return 0

	case focusWindowMessage:
		if w.app.UIRunning() {
			w.app.UIShow()
		} else {
			w.app.UIRun("/")
		}
		return 1

	case wmClose:
		// WM_CLOSE from an external process acts like a graceful close request:
		// close the main UI first, then quit if it is already closed. The tray's
		// private Quit action calls app.Quit directly.
		if w.app.UIRunning() {
			w.app.UITerminate()
		} else {
			w.app.Quit()
		}
		return 0

	case wmEndSession:
		if wParam != 0 {
			w.app.Quit()
		}
		return 0

	case w.taskbarCreatedMessage:
		if w.onTaskbarCreated != nil {
			w.onTaskbarCreated()
		}
		return 0
	}

	result, _, _ := pDefWindowProc.Call(uintptr(hwnd), uintptr(message), wParam, lParam)
	return result
}

func (w *instanceWindow) destroy() error {
	if w.window != 0 {
		result, _, err := pDestroyWindow.Call(uintptr(w.window))
		if result == 0 {
			return fmt.Errorf("destroy window: %w", err)
		}
		w.window = 0
	}
	if w.class != nil {
		if err := w.class.unregister(); err != nil {
			return err
		}
		w.class = nil
	}
	return nil
}
