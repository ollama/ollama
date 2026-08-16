//go:build windows && winui

package wintray

import (
	"crypto/sha256"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/gen2brain/iup-go/iup"
	"github.com/ollama/ollama/app/assets"
	"golang.org/x/sys/windows"
)

const (
	bootstrapDLLName  = "Microsoft.WindowsAppRuntime.Bootstrap.dll"
	menuFont          = "Segoe UI, -14"
	postMessageQuit   = "quit"
	postMessageUpdate = "update"
	normalImageName   = "OLLAMA_TRAY_ICON"
	updateImageName   = "OLLAMA_TRAY_UPDATE_ICON"
	trayPopupSubclass = 1
	// Poll briefly until WinUI materializes the native popup that IupPopup creates.
	trayPopupPollIntervalMS = 10
	// PopupWindowSiteBridge includes transparent shadow padding. Let that
	// padding overlap adjacent shell flyouts so the painted surfaces retain a
	// small native-looking gap instead of the sum of both shadow margins.
	trayPopupShadowOverlapDIP = 18
)

var (
	winUIOnce sync.Once
	winUIErr  error
	// Keep our bootstrap reference for the process lifetime. IUP loads the same
	// module by basename when it initializes the Windows App SDK.
	bootstrapDLL       *windows.DLL
	bootstrapDLLPath   string
	trayPopupProc      = windows.NewCallback(trayPopupWindowProc)
	trayPopupTimerProc = windows.NewCallback(trayPopupTimerCallback)

	trayPopupPlacementMu sync.Mutex // protects activeTrayPopup and its mutable fields
	activeTrayPopup      *trayPopupPlacement
)

type trayEdge uint8

const (
	trayEdgeUnknown trayEdge = iota
	trayEdgeBottom
	trayEdgeTop
	trayEdgeLeft
	trayEdgeRight
)

type trayPopupPlacement struct {
	anchorX int
	anchorY int
	edge    trayEdge
	timerID uintptr
	popup   windows.Handle
}

type logDirectoryHandler interface {
	ShowLogs() error
}

func init() {
	// WinUI initializes a single-threaded COM apartment and its DispatcherQueue
	// on the calling thread. Keep the primary app goroutine on that thread.
	runtime.LockOSThread()
}

type winTray struct {
	app AppCallbacks

	host         iup.Ihandle
	tray         iup.Ihandle
	menu         iup.Ihandle
	notification iup.Ihandle
	normalImage  iup.Ihandle
	updateImage  iup.Ihandle

	normalIconHandle windows.Handle
	updateIconHandle windows.Handle
	instance         *instanceWindow

	stateMu        sync.Mutex // protects host and updateNotified during shutdown
	updateNotified bool
}

// Initialize loads the Windows App Runtime and initializes IUP on the locked
// primary OS thread. Calls are process-wide and idempotent.
func Initialize() error {
	winUIOnce.Do(func() {
		bootstrapDLL, bootstrapDLLPath, winUIErr = loadBootstrapDLL()
		if winUIErr != nil {
			winUIErr = fmt.Errorf("initialize IUP WinUI: %w: %v", ErrWindowsAppRuntimeUnavailable, winUIErr)
			return
		}

		if result := iup.Open(); result != iup.NOERROR && result != iup.OPENED {
			installer := filepath.Join(filepath.Dir(bootstrapDLLPath), "WindowsAppRuntimeInstall.exe")
			if _, err := os.Stat(installer); err == nil {
				winUIErr = fmt.Errorf("initialize IUP WinUI: %w; install it with %q or reinstall Ollama", ErrWindowsAppRuntimeUnavailable, installer)
			} else {
				winUIErr = fmt.Errorf("initialize IUP WinUI: %w; reinstall Ollama", ErrWindowsAppRuntimeUnavailable)
			}
			return
		}

		if driver := iup.GetGlobal("DRIVER"); driver != "WinUI" {
			winUIErr = fmt.Errorf("initialize IUP WinUI: got driver %q; build with -tags winui", driver)
		}
	})

	// Keep the absolute-path bootstrap load rooted for the lifetime of the app;
	// IUP later resolves this module by basename.
	runtime.KeepAlive(bootstrapDLL)
	return winUIErr
}

func NewTray(app AppCallbacks) (_ TrayCallbacks, retErr error) {
	if err := Initialize(); err != nil {
		return nil, err
	}

	t := &winTray{app: app}
	defer func() {
		if retErr != nil {
			t.destroy()
		}
	}()

	iup.SetGlobal("APPID", "com.ollama.app")
	iup.SetGlobal("APPNAME", "Ollama")
	iup.SetGlobal("LOCKLOOP", "YES")

	var err error
	t.normalImage, t.normalIconHandle, err = loadIconAsset(IconName)
	if err != nil {
		return nil, err
	}
	t.updateImage, t.updateIconHandle, err = loadIconAsset(UpdateIconName)
	if err != nil {
		return nil, err
	}

	t.normalImage.SetHandle(normalImageName)
	t.updateImage.SetHandle(updateImageName)

	// A mapped dialog gives MenuFlyout a XAML anchor without placing an Ollama
	// window on screen or in the taskbar.
	t.host = iup.Dialog(iup.Label(""))
	t.host.SetAttribute("TITLE", "Ollama UI host")
	t.host.SetAttribute("HIDETASKBAR", "YES")
	t.host.SetAttribute("RESIZE", "NO")
	t.host.SetAttribute("SIZE", "1x1")
	t.host.SetCallback("CLOSE_CB", iup.CloseFunc(func(iup.Ihandle) int {
		// External WM_CLOSE is handled by the dedicated instance window. The
		// hidden XAML host must never turn one close request into a full quit.
		return iup.IGNORE
	}))
	t.host.SetCallback("POSTMESSAGE_CB", iup.PostMessageFunc(t.onPostMessage))
	if result := iup.Map(t.host); result != iup.NOERROR {
		return nil, fmt.Errorf("map the IUP WinUI menu host: result %d", result)
	}
	t.menu, err = t.buildMenu(false)
	if err != nil {
		return nil, err
	}

	t.tray = iup.Tray()
	iup.SetAttributeHandle(t.tray, "IMAGE", t.normalImage)
	iup.SetAttributeHandle(t.tray, "MENU", t.menu)
	t.tray.SetAttribute("TIP", "Ollama")
	t.tray.SetCallback("TRAYCLICK_CB", iup.TrayClickFunc(t.onTrayClick))
	t.tray.SetAttribute("VISIBLE", "YES")

	t.notification = iup.Notify()
	t.notification.SetCallback("NOTIFY_CB", iup.NotifyFunc(t.onNotification))
	t.notification.SetCallback("ERROR_CB", iup.ErrorFunc(func(_ iup.Ihandle, message string) int {
		slog.Error("failed to show Windows notification", "error", message)
		return iup.DEFAULT
	}))

	t.instance, err = newInstanceWindow(ClassName, app, t.restoreAfterExplorerRestart)
	if err != nil {
		return nil, fmt.Errorf("create the Ollama instance window: %w", err)
	}

	return t, nil
}

func loadBootstrapDLL() (*windows.DLL, string, error) {
	runtimeDir := map[string]string{
		"amd64": "x64",
		"arm64": "arm64",
	}[runtime.GOARCH]
	if runtimeDir == "" {
		return nil, "", fmt.Errorf("unsupported Windows architecture %q", runtime.GOARCH)
	}

	var paths []string
	if executable, err := os.Executable(); err == nil {
		executableDir := filepath.Dir(executable)
		paths = append(paths,
			filepath.Join(executableDir, bootstrapDLLName),
			filepath.Join(executableDir, "app-runtime", runtimeDir, bootstrapDLLName),
		)
	}
	// go run places the executable in a temporary directory. Non-trimmed
	// development builds retain this source path, which lets them find the
	// staged runtime without trusting the process working directory.
	if _, sourceFile, _, ok := runtime.Caller(0); ok && filepath.IsAbs(sourceFile) {
		sourceRoot := filepath.Clean(filepath.Join(filepath.Dir(sourceFile), "..", ".."))
		paths = append(paths, filepath.Join(sourceRoot, "dist", "app-runtime", runtimeDir, bootstrapDLLName))
	}

	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			continue
		}
		dll, err := windows.LoadDLL(path)
		if err != nil {
			return nil, "", fmt.Errorf("load Windows App Runtime bootstrap %q: %w", path, err)
		}
		return dll, path, nil
	}

	return nil, "", fmt.Errorf("Windows App Runtime bootstrap DLL not found (searched %s)", strings.Join(paths, ", "))
}

func (t *winTray) buildMenu(includeUpdate bool) (iup.Ihandle, error) {
	item := func(title string, action func()) iup.Ihandle {
		menuItem := iup.MenuItem(title)
		menuItem.SetCallback("ACTION", iup.ActionFunc(func(iup.Ihandle) int {
			slog.Debug("Windows tray menu action", "item", title)
			action()
			return iup.DEFAULT
		}))
		return menuItem
	}

	children := []iup.Ihandle{
		item(openUIMenuTitle, func() { t.app.UIRun("/") }),
		item(settingsUIMenuTitle, func() { t.app.UIRun("/settings") }),
	}

	if includeUpdate {
		updateLabel := iup.MenuItem(updateAvailableMenuTitle)
		updateLabel.SetAttribute("ACTIVE", "NO")
		children = append(children,
			iup.MenuSeparator(),
			updateLabel,
			item(updateMenuTitle, t.app.DoUpdate),
			iup.MenuSeparator(),
		)
	}

	children = append(children,
		item(diagLogsMenuTitle, func() {
			handler, ok := t.app.(logDirectoryHandler)
			if !ok {
				slog.Error("app does not implement log directory handling")
				return
			}
			if err := handler.ShowLogs(); err != nil {
				slog.Error("failed to open log directory", "error", err)
			}
		}),
		iup.MenuSeparator(),
		item(quitMenuTitle, t.app.Quit),
	)

	menu := iup.Menu(children...)
	// WinUI font sizes are device-independent pixels. A negative IUP font size
	// keeps IUP from applying monitor DPI before XAML applies it itself.
	menu.SetAttribute("FONT", menuFont)
	if result := iup.Map(menu); result != iup.NOERROR {
		menu.Destroy()
		return 0, fmt.Errorf("map the IUP WinUI tray menu: result %d", result)
	}
	return menu, nil
}

func (t *winTray) onTrayClick(_ iup.Ihandle, button, pressed, doubleClick int) int {
	// IUP normally anchors left-click popups at the cursor. Anchor at the edge
	// of the tray surface instead so WinUI can place the whole menu above it.
	if (button == 1 && pressed == 0 && doubleClick == 0) || (button == 3 && pressed == 1 && doubleClick == 0) {
		x, y, edge := trayMenuPosition()
		placement := startTrayPopupPlacement(x, y, edge)
		slog.Debug("opening Windows tray menu", "x", x, "y", y)
		iup.Popup(t.menu, x, y)
		stopTrayPopupPlacement(placement)
		// IUP opens a second popup after a right-click callback unless ignored.
		return iup.IGNORE
	}
	return iup.DEFAULT
}

func startTrayPopupPlacement(anchorX, anchorY int, edge trayEdge) *trayPopupPlacement {
	placement := &trayPopupPlacement{anchorX: anchorX, anchorY: anchorY, edge: edge}

	trayPopupPlacementMu.Lock()
	activeTrayPopup = placement
	trayPopupPlacementMu.Unlock()

	timerID, _, callErr := pSetTimer.Call(0, 0, trayPopupPollIntervalMS, trayPopupTimerProc)
	if timerID == 0 {
		slog.Warn("failed to schedule Windows tray menu placement", "error", callErr)
		return placement
	}

	trayPopupPlacementMu.Lock()
	placement.timerID = timerID
	trayPopupPlacementMu.Unlock()
	return placement
}

func stopTrayPopupPlacement(placement *trayPopupPlacement) {
	trayPopupPlacementMu.Lock()
	if activeTrayPopup == placement {
		activeTrayPopup = nil
	}
	timerID := placement.timerID
	popup := placement.popup
	placement.timerID = 0
	placement.popup = 0
	trayPopupPlacementMu.Unlock()

	if timerID != 0 {
		pKillTimer.Call(0, timerID) //nolint:errcheck
	}
	if popup != 0 {
		pRemoveWindowSubclass.Call(uintptr(popup), trayPopupProc, trayPopupSubclass) //nolint:errcheck
	}
}

func trayPopupTimerCallback(_ windows.Handle, _ uint32, timerID, _ uintptr) uintptr {
	trayPopupPlacementMu.Lock()
	placement := activeTrayPopup
	if placement == nil || placement.timerID != timerID {
		trayPopupPlacementMu.Unlock()
		return 0
	}
	anchorX, anchorY, edge := placement.anchorX, placement.anchorY, placement.edge
	trayPopupPlacementMu.Unlock()

	popup, rect, ok := nearestVisiblePopup(anchorX, anchorY)
	if !ok || rect.Right-rect.Left <= 1 || rect.Bottom-rect.Top <= 1 {
		return 0
	}

	left, top, moved := trayPopupPosition(popup, rect, edge)
	if !moved {
		finishTrayPopupTimer(placement, timerID, 0)
		return 0
	}

	position := packTrayPopupPosition(left, top)
	result, _, callErr := pSetWindowSubclass.Call(
		uintptr(popup),
		trayPopupProc,
		trayPopupSubclass,
		position,
	)
	if result == 0 {
		slog.Warn("failed to stabilize Windows tray menu position", "error", callErr)
		finishTrayPopupTimer(placement, timerID, 0)
		return 0
	}

	result, _, callErr = pSetWindowPos.Call(
		uintptr(popup),
		0,
		uintptr(int64(left)),
		uintptr(int64(top)),
		0,
		0,
		swpNoSize|swpNoZOrder|swpNoActivate,
	)
	if result == 0 {
		pRemoveWindowSubclass.Call(uintptr(popup), trayPopupProc, trayPopupSubclass) //nolint:errcheck
		slog.Warn("failed to position Windows tray menu", "error", callErr)
		finishTrayPopupTimer(placement, timerID, 0)
		return 0
	}

	finishTrayPopupTimer(placement, timerID, popup)
	slog.Debug("positioned Windows tray menu", "left", left, "top", top)
	return 0
}

func finishTrayPopupTimer(placement *trayPopupPlacement, timerID uintptr, popup windows.Handle) {
	trayPopupPlacementMu.Lock()
	if activeTrayPopup == placement && placement.timerID == timerID {
		placement.timerID = 0
		placement.popup = popup
	}
	trayPopupPlacementMu.Unlock()
	pKillTimer.Call(0, timerID) //nolint:errcheck
}

func trayMenuPosition() (int, int, trayEdge) {
	count, x, y := iup.GetInt2(0, "CURSORPOS")
	if count != 2 {
		return iup.MOUSEPOS, iup.MOUSEPOS, trayEdgeUnknown
	}

	cursorRect := windows.Rect{Left: int32(x), Top: int32(y), Right: int32(x) + 1, Bottom: int32(y) + 1}
	monitor, _, _ := pMonitorFromRect.Call(uintptr(unsafe.Pointer(&cursorRect)), monitorDefaultToNearest)
	if monitor == 0 {
		return x, y, trayEdgeUnknown
	}

	info := monitorInfo{size: uint32(unsafe.Sizeof(monitorInfo{}))}
	if ok, _, _ := pGetMonitorInfo.Call(monitor, uintptr(unsafe.Pointer(&info))); ok == 0 {
		return x, y, trayEdgeUnknown
	}

	switch {
	case info.work.Bottom < info.monitor.Bottom:
		return x, int(info.work.Bottom), trayEdgeBottom
	case info.work.Top > info.monitor.Top:
		return x, int(info.work.Top), trayEdgeTop
	case info.work.Left > info.monitor.Left:
		return int(info.work.Left), y, trayEdgeLeft
	case info.work.Right < info.monitor.Right:
		return int(info.work.Right), y, trayEdgeRight
	}
	return x, y, trayEdgeUnknown
}

func nearestVisiblePopup(anchorX, anchorY int) (windows.Handle, windows.Rect, bool) {
	className, err := windows.UTF16PtrFromString("Microsoft.UI.Content.PopupWindowSiteBridge")
	if err != nil {
		return 0, windows.Rect{}, false
	}

	pid := uint32(os.Getpid())
	var nearest windows.Handle
	var nearestRect windows.Rect
	var nearestDistance int64 = 1<<63 - 1
	var after uintptr
	for {
		hwnd, _, _ := pFindWindowEx.Call(0, after, uintptr(unsafe.Pointer(className)), 0)
		if hwnd == 0 {
			break
		}
		after = hwnd

		var windowPID uint32
		pGetWindowThreadProcID.Call(hwnd, uintptr(unsafe.Pointer(&windowPID))) //nolint:errcheck
		visible, _, _ := pIsWindowVisible.Call(hwnd)
		var rect windows.Rect
		if windowPID != pid || visible == 0 || !getWindowRect(hwnd, &rect) {
			continue
		}

		distance := pointToRectDistanceSquared(int32(anchorX), int32(anchorY), rect)
		if distance < nearestDistance {
			nearest = windows.Handle(hwnd)
			nearestRect = rect
			nearestDistance = distance
		}
	}
	return nearest, nearestRect, nearest != 0
}

func trayPopupPosition(popup windows.Handle, windowRect windows.Rect, edge trayEdge) (int32, int32, bool) {
	left, top := windowRect.Left, windowRect.Top
	visibleRect := visibleWindowRect(uintptr(popup), windowRect)
	shadowOverlap := trayPopupShadowOverlap(popup)
	moved := false

	for _, class := range []string{
		"Shell_TrayWnd",
		"Shell_SecondaryTrayWnd",
		"TopLevelWindowForOverflowXamlIsland",
		"NotifyIconOverflowWindow",
	} {
		className, err := windows.UTF16PtrFromString(class)
		if err != nil {
			continue
		}

		var after uintptr
		for {
			surface, _, _ := pFindWindowEx.Call(0, after, uintptr(unsafe.Pointer(className)), 0)
			if surface == 0 {
				break
			}
			after = surface

			var surfaceWindowRect windows.Rect
			visible, _, _ := pIsWindowVisible.Call(surface)
			if visible == 0 || !getWindowRect(surface, &surfaceWindowRect) {
				continue
			}
			surfaceRect := visibleWindowRect(surface, surfaceWindowRect)
			dx, dy, ok := trayPopupOffset(visibleRect, surfaceRect, edge, shadowOverlap)
			if !ok {
				continue
			}
			left += dx
			top += dy
			visibleRect.Left += dx
			visibleRect.Right += dx
			visibleRect.Top += dy
			visibleRect.Bottom += dy
			moved = true
		}
	}
	return left, top, moved
}

func trayPopupOffset(popupRect, surfaceRect windows.Rect, edge trayEdge, shadowOverlap int32) (int32, int32, bool) {
	if !rectsOverlap(popupRect, surfaceRect) {
		return 0, 0, false
	}

	switch edge {
	case trayEdgeBottom:
		return 0, surfaceRect.Top - popupRect.Bottom + shadowOverlap, true
	case trayEdgeTop:
		return 0, surfaceRect.Bottom - popupRect.Top - shadowOverlap, true
	case trayEdgeLeft:
		return surfaceRect.Right - popupRect.Left - shadowOverlap, 0, true
	case trayEdgeRight:
		return surfaceRect.Left - popupRect.Right + shadowOverlap, 0, true
	default:
		return 0, 0, false
	}
}

func trayPopupShadowOverlap(popup windows.Handle) int32 {
	dpi, _, _ := pGetDpiForWindow.Call(uintptr(popup))
	if dpi == 0 {
		dpi = 96
	}
	return int32((dpi*trayPopupShadowOverlapDIP + 48) / 96)
}

func getWindowRect(hwnd uintptr, rect *windows.Rect) bool {
	result, _, _ := pGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(rect)))
	return result != 0
}

func visibleWindowRect(hwnd uintptr, fallback windows.Rect) windows.Rect {
	var rect windows.Rect
	result, _, _ := pDwmGetWindowAttribute.Call(
		hwnd,
		dwmwaExtendedFrameBounds,
		uintptr(unsafe.Pointer(&rect)),
		unsafe.Sizeof(rect),
	)
	if result != 0 || rect.Right <= rect.Left || rect.Bottom <= rect.Top {
		return fallback
	}
	return rect
}

func rectsOverlap(a, b windows.Rect) bool {
	return a.Left < b.Right && a.Right > b.Left && a.Top < b.Bottom && a.Bottom > b.Top
}

func pointToRectDistanceSquared(x, y int32, rect windows.Rect) int64 {
	var dx, dy int64
	if x < rect.Left {
		dx = int64(rect.Left - x)
	} else if x > rect.Right {
		dx = int64(x - rect.Right)
	}
	if y < rect.Top {
		dy = int64(rect.Top - y)
	} else if y > rect.Bottom {
		dy = int64(y - rect.Bottom)
	}
	return dx*dx + dy*dy
}

func packTrayPopupPosition(left, top int32) uintptr {
	return uintptr(uint64(uint32(left)) | uint64(uint32(top))<<32)
}

func trayPopupWindowProc(hwnd windows.Handle, message uint32, wParam, lParam, _ uintptr, position uintptr) uintptr {
	if message == wmWindowPosChanging && lParam != 0 {
		// WM_WINDOWPOSCHANGING guarantees that lParam points to WINDOWPOS for
		// the duration of this callback.
		windowPosition := (*windowPos)(unsafe.Pointer(lParam)) //nolint:govet
		if windowPosition.flags&swpNoMove == 0 {
			windowPosition.X = int32(uint32(position))
			windowPosition.Y = int32(uint32(uint64(position) >> 32))
		}
	}
	result, _, _ := pDefSubclassProc.Call(uintptr(hwnd), uintptr(message), wParam, lParam)
	return result
}

func (t *winTray) onPostMessage(_ iup.Ihandle, message string, _ int, payload any) int {
	switch message {
	case postMessageQuit:
		iup.ExitLoop()
	case postMessageUpdate:
		version, _ := payload.(string)
		if err := t.showUpdate(version); err != nil {
			slog.Error("failed to update Windows tray UI", "error", err)
			t.stateMu.Lock()
			t.updateNotified = false
			t.stateMu.Unlock()
		}
	default:
		slog.Warn("unexpected IUP tray message", "message", message)
	}
	return iup.DEFAULT
}

func (t *winTray) onNotification(_ iup.Ihandle, actionID int) int {
	slog.Debug("Windows update notification action", "action", actionID)
	if actionID == 0 || actionID == 1 {
		t.app.DoUpdate()
	} else {
		slog.Debug("ignoring unknown notification action", "action", actionID)
	}
	return iup.DEFAULT
}

func (t *winTray) showUpdate(version string) error {
	newMenu, err := t.buildMenu(true)
	if err != nil {
		return err
	}

	oldMenu := t.menu
	t.menu = newMenu
	iup.SetAttributeHandle(t.tray, "MENU", newMenu)
	if oldMenu != 0 {
		oldMenu.Destroy()
	}

	iup.SetAttributeHandle(t.tray, "IMAGE", t.updateImage)
	t.notification.SetAttribute("TITLE", updateTitle)
	t.notification.SetAttribute("BODY", updateNotificationBody(version))
	t.notification.SetAttribute("ICON", normalImageName)
	t.notification.SetAttribute("ACTION1", updateMenuTitle)
	t.notification.SetAttribute("SHOW", "YES")
	return nil
}

func (t *winTray) UpdateAvailable(version string) error {
	t.stateMu.Lock()
	defer t.stateMu.Unlock()
	if t.host == 0 {
		return fmt.Errorf("IUP WinUI tray is not initialized")
	}
	if t.updateNotified {
		return nil
	}
	t.updateNotified = true

	slog.Debug("queueing Windows update menu and notification", "version", version)
	iup.PostMessage(t.host, postMessageUpdate, 0, version)
	return nil
}

func (t *winTray) restoreAfterExplorerRestart() {
	if t.tray == 0 {
		return
	}
	slog.Debug("restoring tray icon after Explorer restart")
	t.tray.SetAttribute("VISIBLE", "NO")
	t.tray.SetAttribute("VISIBLE", "YES")
}

func (t *winTray) Quit() {
	t.stateMu.Lock()
	defer t.stateMu.Unlock()
	if t.host != 0 {
		iup.PostMessage(t.host, postMessageQuit, 0, nil)
	}
}

func (t *winTray) TrayRun() {
	slog.Debug("starting IUP WinUI event loop")
	result := iup.MainLoop()
	if result != iup.NOERROR {
		slog.Error("IUP WinUI event loop exited with an error", "result", result)
	}
	t.destroy()
}

func (t *winTray) GetIconHandle() windows.Handle {
	return t.normalIconHandle
}

func (t *winTray) destroy() {
	slog.Debug("stopping IUP WinUI tray")
	if t.instance != nil {
		slog.Debug("destroying Ollama instance window")
		if err := t.instance.destroy(); err != nil {
			slog.Warn("failed to destroy Ollama instance window", "error", err)
		}
		t.instance = nil
	}
	if t.notification != 0 {
		slog.Debug("destroying IUP notification manager")
		t.notification.SetAttribute("CLOSE", "YES")
		t.notification.Destroy()
		t.notification = 0
	}
	if t.tray != 0 {
		slog.Debug("destroying IUP tray icon")
		t.tray.SetAttribute("VISIBLE", "NO")
		t.tray.Destroy()
		t.tray = 0
	}
	if t.menu != 0 {
		slog.Debug("destroying IUP tray menu")
		t.menu.Destroy()
		t.menu = 0
	}
	t.stateMu.Lock()
	host := t.host
	t.host = 0
	t.stateMu.Unlock()
	if host != 0 {
		slog.Debug("destroying IUP menu host")
		host.Destroy()
	}
	if t.normalImage != 0 {
		slog.Debug("destroying IUP normal tray image")
		t.normalImage.Destroy()
		t.normalImage = 0
	}
	if t.updateImage != 0 {
		slog.Debug("destroying IUP update tray image")
		t.updateImage.Destroy()
		t.updateImage = 0
	}
	if t.normalIconHandle != 0 {
		slog.Debug("destroying normal Win32 icon")
		pDestroyIcon.Call(uintptr(t.normalIconHandle)) //nolint:errcheck
		t.normalIconHandle = 0
	}
	if t.updateIconHandle != 0 {
		slog.Debug("destroying update Win32 icon")
		pDestroyIcon.Call(uintptr(t.updateIconHandle)) //nolint:errcheck
		t.updateIconHandle = 0
	}
	// The current IUP WinUI backend faults in WindowsXamlManager::Close after
	// MenuFlyout resources have been instantiated. All Ollama-owned handles are
	// released above and this loop only ends at process shutdown, so leave the
	// process-wide WinUI objects to Windows until upstream teardown is fixed.
	slog.Debug("IUP WinUI tray stopped")
}

func loadIconAsset(name string) (iup.Ihandle, windows.Handle, error) {
	data, err := assets.GetIcon(name)
	if err != nil {
		return 0, 0, fmt.Errorf("load %s: %w", name, err)
	}

	hash := fmt.Sprintf("%x", sha256.Sum256(data))
	path := filepath.Join(os.TempDir(), "ollama-tray-"+hash+".ico")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return 0, 0, fmt.Errorf("write temporary %s: %w", name, err)
	}
	defer os.Remove(path)

	// WinUI images are WinRT WriteableBitmap objects, not Win32 HICONs. Let
	// IUP decode the ICO with WIC, and load a separate HICON for the main
	// WebView2 window. Passing the HICON to ImageFromHandle is invalid for the
	// WinUI driver and can dereference it as a COM object.
	image := iup.ImageGetHandle(path)
	if image == 0 {
		return 0, 0, fmt.Errorf("decode %s as an IUP WinUI image", name)
	}

	pathPtr, err := windows.UTF16PtrFromString(path)
	if err != nil {
		image.Destroy()
		return 0, 0, fmt.Errorf("encode icon path: %w", err)
	}
	handle, _, callErr := pLoadImage.Call(
		0,
		uintptr(unsafe.Pointer(pathPtr)),
		imageIcon,
		0,
		0,
		loadFromFile|loadDefaultSize,
	)
	if handle == 0 {
		image.Destroy()
		return 0, 0, fmt.Errorf("load %s as a Windows icon: %w", name, callErr)
	}
	return image, windows.Handle(handle), nil
}

func updateNotificationBody(version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		return "A new version of Ollama is ready to install"
	}
	return fmt.Sprintf(updateMessage, version)
}
