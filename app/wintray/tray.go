//go:build windows

package wintray

import (
	"crypto/md5"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ollama/ollama/app/assets"
	"golang.org/x/sys/windows"
)

const (
	UpdateIconName = "tray_upgrade.ico"
	IconName       = "tray.ico"
	ClassName      = "OllamaClass"
)

func NewTray(app AppCallbacks) (TrayCallbacks, error) {
	updateIcon, err := assets.GetIcon(UpdateIconName)
	if err != nil {
		return nil, fmt.Errorf("failed to load icon %s: %w", UpdateIconName, err)
	}
	icon, err := assets.GetIcon(IconName)
	if err != nil {
		return nil, fmt.Errorf("failed to load icon %s: %w", IconName, err)
	}

	return InitTray(icon, updateIcon, app)
}

type TrayCallbacks interface {
	Quit()
	TrayRun()
	UpdateAvailable(ver string) error
	GetIconHandle() windows.Handle
}

// TrayWebView is the WebView2 surface that the Windows tray flyout needs. The
// app creates it because the shared webview package links app-owned callbacks.
type TrayWebView interface {
	Bind(name string, f interface{}) error
	Destroy()
	Eval(js string)
	SetHtml(html string)
}

type AppCallbacks interface {
	UIRun(path string)
	UIShow()
	UITerminate()
	UIRunning() bool
	Quit()
	DoUpdate()
}

// TrayWebViewFactory provides the WebView2 surface used by the tray flyout.
type TrayWebViewFactory interface {
	// NewTrayWebView embeds a WebView in the nonzero Windows HWND supplied by
	// the tray UI thread. The tray owns and destroys the returned view on that
	// same thread before it destroys the host window.
	NewTrayWebView(window unsafe.Pointer) (TrayWebView, error)
}

// TrayFlyoutStyle selects the token set used by the tray flyout.
type TrayFlyoutStyle string

const (
	TrayFlyoutStyleFluent TrayFlyoutStyle = "fluent"
	TrayFlyoutStyleOllama TrayFlyoutStyle = "ollama"
)

// TrayFlyoutStyleProvider optionally selects the flyout appearance in the app
// backend. The flyout does not expose this choice as an end-user control.
type TrayFlyoutStyleProvider interface {
	TrayFlyoutStyle() TrayFlyoutStyle
}

type URLSchemeHandler interface {
	HandleURLScheme(urlScheme string)
}

// Helpful sources: https://github.com/golang/exp/blob/master/shiny/driver/internal/win32

// Contains information about loaded resources
type winTray struct {
	instance,
	icon,
	defaultIcon,
	cursor,
	window windows.Handle

	loadedImages   map[string]windows.Handle
	muLoadedImages sync.RWMutex

	nid   *notifyIconData
	muNID sync.RWMutex
	wcex  *wndClassEx

	wmSystrayMessage,
	wmTaskbarCreated uint32

	muState sync.RWMutex // guards pendingUpdate and updateNotified

	pendingUpdate  bool
	updateNotified bool // Only pop up the notification once - TODO consider daily nag?
	normalIcon     []byte
	updateIcon     []byte

	flyout             *trayFlyout
	flyoutInitializing bool // WebView2 initialization pumps messages and may reenter the tray callback.
	shuttingDown       bool

	app AppCallbacks
}

var wt winTray

func InitTray(icon, updateIcon []byte, app AppCallbacks) (*winTray, error) {
	if _, ok := app.(TrayWebViewFactory); !ok {
		return nil, errors.New("app does not provide a tray WebView2 factory")
	}
	// WebView enables DPI awareness when it creates its first top-level window.
	// The tray must do it earlier so a start-hidden flyout and main UI agree.
	if err := enableFlyoutDPIAwareness(); err != nil {
		return nil, err
	}
	wt.normalIcon = icon
	wt.updateIcon = updateIcon
	wt.app = app
	if err := wt.initInstance(); err != nil {
		return nil, fmt.Errorf("Unable to init instance: %w\n", err)
	}

	iconFilePath, err := iconBytesToFilePath(wt.normalIcon)
	if err != nil {
		return nil, fmt.Errorf("Unable to write icon data to temp file: %w", err)
	}
	if err := wt.setIcon(iconFilePath); err != nil {
		return nil, fmt.Errorf("Unable to set icon: %w", err)
	}

	h, err := wt.loadIconFrom(iconFilePath)
	if err != nil {
		return nil, fmt.Errorf("Unable to set default icon: %w", err)
	}
	wt.defaultIcon = h

	return &wt, nil
}

func (t *winTray) initInstance() error {
	const (
		windowName = ""
	)

	t.wmSystrayMessage = WM_USER + 1
	t.loadedImages = make(map[string]windows.Handle)

	taskbarEventNamePtr, _ := windows.UTF16PtrFromString("TaskbarCreated")
	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644947
	res, _, err := pRegisterWindowMessage.Call(
		uintptr(unsafe.Pointer(taskbarEventNamePtr)),
	)
	if res == 0 { // success 0xc000-0xfff
		return fmt.Errorf("failed to register window: %w", err)
	}
	t.wmTaskbarCreated = uint32(res)

	instanceHandle, _, err := pGetModuleHandle.Call(0)
	if instanceHandle == 0 {
		return err
	}
	t.instance = windows.Handle(instanceHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms648072(v=vs.85).aspx
	iconHandle, _, err := pLoadIcon.Call(0, uintptr(IDI_APPLICATION))
	if iconHandle == 0 {
		return err
	}
	t.icon = windows.Handle(iconHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms648391(v=vs.85).aspx
	cursorHandle, _, err := pLoadCursor.Call(0, uintptr(IDC_ARROW))
	if cursorHandle == 0 {
		return err
	}
	t.cursor = windows.Handle(cursorHandle)

	classNamePtr, err := windows.UTF16PtrFromString(ClassName)
	if err != nil {
		return err
	}

	windowNamePtr, err := windows.UTF16PtrFromString(windowName)
	if err != nil {
		return err
	}

	t.wcex = &wndClassEx{
		Style:      CS_HREDRAW | CS_VREDRAW,
		WndProc:    windows.NewCallback(t.wndProc),
		Instance:   t.instance,
		Icon:       t.icon,
		Cursor:     t.cursor,
		Background: windows.Handle(6), // (COLOR_WINDOW + 1)
		ClassName:  classNamePtr,
		IconSm:     t.icon,
	}
	if err := t.wcex.register(); err != nil {
		return err
	}

	windowHandle, _, err := pCreateWindowEx.Call(
		uintptr(0),
		uintptr(unsafe.Pointer(classNamePtr)),
		uintptr(unsafe.Pointer(windowNamePtr)),
		uintptr(WS_OVERLAPPEDWINDOW),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(0),
		uintptr(0),
		uintptr(t.instance),
		uintptr(0),
	)
	if windowHandle == 0 {
		return err
	}
	t.window = windows.Handle(windowHandle)

	pShowWindow.Call(uintptr(t.window), uintptr(SW_HIDE)) //nolint:errcheck

	boolRet, _, err := pUpdateWindow.Call(uintptr(t.window))
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to update window: %s", err))
	}

	t.muNID.Lock()
	defer t.muNID.Unlock()
	t.nid = &notifyIconData{
		Wnd:             t.window,
		ID:              100,
		Flags:           NIF_MESSAGE,
		CallbackMessage: t.wmSystrayMessage,
	}
	t.nid.Size = uint32(unsafe.Sizeof(*t.nid))

	return t.nid.add()
}

func (t *winTray) showFlyout() error {
	if t.shuttingDown {
		return nil
	}
	if t.flyout != nil {
		return t.flyout.show()
	}
	if t.flyoutInitializing {
		slog.Debug("ignoring reentrant tray flyout show request")
		return nil
	}

	t.flyoutInitializing = true
	defer func() { t.flyoutInitializing = false }()

	flyout, err := newTrayFlyout(t)
	if err != nil {
		return err
	}
	if t.shuttingDown {
		flyout.destroy()
		return nil
	}
	t.flyout = flyout
	return flyout.show()
}

func iconBytesToFilePath(iconBytes []byte) (string, error) {
	bh := md5.Sum(iconBytes)
	dataHash := hex.EncodeToString(bh[:])
	iconFilePath := filepath.Join(os.TempDir(), "ollama_temp_icon_"+dataHash)

	if _, err := os.Stat(iconFilePath); os.IsNotExist(err) {
		if err := os.WriteFile(iconFilePath, iconBytes, 0o644); err != nil {
			return "", err
		}
	}
	return iconFilePath, nil
}

// Loads an image from file and shows it in tray.
// Shell_NotifyIcon: https://msdn.microsoft.com/en-us/library/windows/desktop/bb762159(v=vs.85).aspx
func (t *winTray) setIcon(src string) error {
	h, err := t.loadIconFrom(src)
	if err != nil {
		return err
	}

	t.muNID.Lock()
	defer t.muNID.Unlock()
	t.nid.Icon = h
	t.nid.Flags |= NIF_ICON | NIF_TIP
	if toolTipUTF16, err := syscall.UTF16FromString("Ollama"); err == nil {
		copy(t.nid.Tip[:], toolTipUTF16)
	} else {
		return err
	}
	t.nid.Size = uint32(unsafe.Sizeof(*t.nid))

	return t.nid.modify()
}

// Loads an image from file to be shown in tray or menu item.
// LoadImage: https://msdn.microsoft.com/en-us/library/windows/desktop/ms648045(v=vs.85).aspx
func (t *winTray) loadIconFrom(src string) (windows.Handle, error) {
	// Save and reuse handles of loaded images
	t.muLoadedImages.RLock()
	h, ok := t.loadedImages[src]
	t.muLoadedImages.RUnlock()
	if !ok {
		srcPtr, err := windows.UTF16PtrFromString(src)
		if err != nil {
			return 0, err
		}
		res, _, err := pLoadImage.Call(
			0,
			uintptr(unsafe.Pointer(srcPtr)),
			IMAGE_ICON,
			0,
			0,
			LR_LOADFROMFILE|LR_DEFAULTSIZE,
		)
		if res == 0 {
			return 0, err
		}
		h = windows.Handle(res)
		t.muLoadedImages.Lock()
		t.loadedImages[src] = h
		t.muLoadedImages.Unlock()
	}
	return h, nil
}

func (t *winTray) GetIconHandle() windows.Handle {
	return t.defaultIcon
}
