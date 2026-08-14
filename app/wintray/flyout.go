//go:build windows

package wintray

import (
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	flyoutClassName       = "OllamaTrayFlyout"
	flyoutWidth           = int32(304)
	flyoutEdgeGap         = int32(8)
	flyoutDefaultDPI      = int32(96)
	flyoutMinHeightPixels = int32(100)  // Reject incomplete bootstrap measurements.
	flyoutMaxHeightPixels = int32(8192) // Bound asynchronous input well above any monitor work area.
)

var (
	flyoutDWM = windows.NewLazySystemDLL("Dwmapi.dll")
	flyoutOLE = windows.NewLazySystemDLL("Ole32.dll")

	pCoInitializeEx         = flyoutOLE.NewProc("CoInitializeEx")
	pCoUninitialize         = flyoutOLE.NewProc("CoUninitialize")
	pDwmSetWindowAttribute  = flyoutDWM.NewProc("DwmSetWindowAttribute")
	pGetClientRect          = u32.NewProc("GetClientRect")
	pGetDpiForWindow        = u32.NewProc("GetDpiForWindow")
	pGetMonitorInfo         = u32.NewProc("GetMonitorInfoW")
	pGetWindow              = u32.NewProc("GetWindow")
	pIsWindowVisible        = u32.NewProc("IsWindowVisible")
	pMonitorFromRect        = u32.NewProc("MonitorFromRect")
	pMoveWindow             = u32.NewProc("MoveWindow")
	pSetFocus               = u32.NewProc("SetFocus")
	pSetProcessDPIAwareness = u32.NewProc("SetProcessDpiAwarenessContext")
	pSetWindowPos           = u32.NewProc("SetWindowPos")
	pShellNotifyIconGetRect = s32.NewProc("Shell_NotifyIconGetRect")
)

var dpiAwarenessContextPerMonitorV2 = ^uintptr(3) // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (-4)

const (
	CS_DROPSHADOW            = 0x00020000
	COINIT_APARTMENTTHREADED = 0x00000002
	GW_CHILD                 = 5
	MONITOR_DEFAULTTONEAREST = 2
	SWP_NOSIZE               = 0x0001
	SWP_NOACTIVATE           = 0x0010
	SWP_NOOWNERZORDER        = 0x0200
	SWP_NOZORDER             = 0x0004
	SWP_SHOWWINDOW           = 0x0040
	WA_INACTIVE              = 0
	WM_ACTIVATE              = 0x0006
	WM_DPICHANGED            = 0x02E0
	WM_SIZE                  = 0x0005
	WS_CLIPCHILDREN          = 0x02000000
	WS_CLIPSIBLINGS          = 0x04000000
	WS_EX_TOOLWINDOW         = 0x00000080
	WS_EX_TOPMOST            = 0x00000008
	WS_POPUP                 = 0x80000000
)

// trayFlyoutHTML uses one semantic layout for both visual styles. The
// data-style token sets in the document are the only platform-look layer.
//
//go:embed flyout.html
var trayFlyoutHTML string

type trayFlyoutState struct {
	PendingUpdate bool            `json:"pendingUpdate"`
	Style         TrayFlyoutStyle `json:"style"`
	Labels        flyoutLabels    `json:"labels"`
}

type flyoutLabels struct {
	Open            string `json:"open"`
	Settings        string `json:"settings"`
	UpdateAvailable string `json:"updateAvailable"`
	Update          string `json:"update"`
	Logs            string `json:"logs"`
	Quit            string `json:"quit"`
}

type flyoutMetrics struct {
	DevicePixelRatio float64 `json:"devicePixelRatio"`
	InnerWidth       int     `json:"innerWidth"`
	InnerHeight      int     `json:"innerHeight"`
	BodyFontSize     string  `json:"bodyFontSize"`
	ContentHeight    int     `json:"contentHeight"`
	PhysicalHeight   int     `json:"physicalHeight"`
}

type trayFlyout struct {
	tray *winTray

	window              windows.Handle
	widget              windows.Handle
	wcex                *wndClassEx
	view                TrayWebView
	style               TrayFlyoutStyle
	clickGate           trayClickGate
	contentHeightPixels int32
	showPending         bool
	hiding              bool
	classRegistered     bool
	comInitialized      bool
}

// trayClickGate makes the tray icon behave as a true toggle regardless of
// whether Windows deactivates the popup before or after the mouse-down callback.
type trayClickGate struct {
	deactivatedOverIcon bool
	suppressMouseUp     bool
}

type trayFlyoutAction uintptr

const (
	trayFlyoutActionOpen trayFlyoutAction = iota + 1
	trayFlyoutActionSettings
	trayFlyoutActionLogs
	trayFlyoutActionUpdate
	trayFlyoutActionQuit
)

func (g *trayClickGate) deactivate(overIcon bool) {
	g.deactivatedOverIcon = overIcon
}

func (g *trayClickGate) mouseDown(visible bool) bool {
	g.suppressMouseUp = visible || g.deactivatedOverIcon
	g.deactivatedOverIcon = false
	return g.suppressMouseUp
}

func (g *trayClickGate) mouseUp() bool {
	suppress := g.suppressMouseUp
	g.suppressMouseUp = false
	return suppress
}

type notifyIconIdentifier struct {
	Size     uint32
	Wnd      windows.Handle
	ID       uint32
	GuidItem windows.GUID
}

type rect struct {
	Left, Top, Right, Bottom int32
}

type monitorInfo struct {
	Size    uint32
	Monitor rect
	Work    rect
	Flags   uint32
}

func enableFlyoutDPIAwareness() error {
	result, _, callErr := pSetProcessDPIAwareness.Call(dpiAwarenessContextPerMonitorV2)
	if result != 0 || errors.Is(callErr, windows.ERROR_ACCESS_DENIED) {
		return nil
	}
	return fmt.Errorf("enable per-monitor DPI awareness: %w", callErr)
}

func newTrayFlyout(t *winTray) (_ *trayFlyout, err error) {
	style := TrayFlyoutStyleFluent
	if provider, ok := t.app.(TrayFlyoutStyleProvider); ok {
		style = provider.TrayFlyoutStyle()
	}
	if err := validateTrayFlyoutStyle(style); err != nil {
		return nil, err
	}
	slog.Debug("initializing tray flyout", "style", style)
	f := &trayFlyout{
		tray:  t,
		style: style,
	}
	factory, ok := t.app.(TrayWebViewFactory)
	if !ok {
		return nil, errors.New("app does not provide a tray WebView2 factory")
	}
	defer func() {
		if err != nil {
			f.destroy()
		}
	}()

	if err := f.initializeCOM(); err != nil {
		return nil, err
	}
	if err := f.createWindow(); err != nil {
		return nil, err
	}
	if err := f.prepareWindow(); err != nil {
		return nil, err
	}

	// The WebView wrapper accepts the HWND value in a void pointer on Windows.
	f.view, err = factory.NewTrayWebView(unsafe.Pointer(uintptr(f.window))) //nolint:govet
	if err != nil {
		return nil, fmt.Errorf("create tray WebView2 controller: %w", err)
	}

	f.widget = f.firstChild()
	if f.widget == 0 {
		return nil, errors.New("unable to find tray WebView2 child window")
	}
	if err := f.bind(); err != nil {
		return nil, err
	}
	f.view.SetHtml(trayFlyoutHTML)
	if err := f.resizeWidget(); err != nil {
		return nil, err
	}
	slog.Debug("tray flyout initialized", "window", f.window, "widget", f.widget)

	return f, nil
}

func (f *trayFlyout) initializeCOM() error {
	hr, _, _ := pCoInitializeEx.Call(0, COINIT_APARTMENTTHREADED)
	if hr != 0 && hr != 1 {
		return fmt.Errorf("initialize COM for tray flyout: HRESULT 0x%08x", uint32(hr))
	}
	f.comInitialized = true
	return nil
}

func (f *trayFlyout) createWindow() error {
	className, err := windows.UTF16PtrFromString(flyoutClassName)
	if err != nil {
		return err
	}

	f.wcex = &wndClassEx{
		Style:      CS_HREDRAW | CS_VREDRAW | CS_DROPSHADOW,
		WndProc:    windows.NewCallback(f.wndProc),
		Instance:   f.tray.instance,
		Icon:       f.tray.icon,
		Cursor:     f.tray.cursor,
		Background: windows.Handle(6), // COLOR_WINDOW + 1
		ClassName:  className,
		IconSm:     f.tray.icon,
	}
	if err := f.wcex.register(); err != nil {
		return fmt.Errorf("register tray flyout window: %w", err)
	}
	f.classRegistered = true

	window, _, callErr := pCreateWindowEx.Call(
		WS_EX_TOOLWINDOW|WS_EX_TOPMOST,
		uintptr(unsafe.Pointer(className)),
		0,
		WS_POPUP|WS_CLIPCHILDREN|WS_CLIPSIBLINGS,
		0,
		0,
		uintptr(flyoutWidth),
		uintptr(flyoutMinHeightPixels),
		uintptr(f.tray.window),
		0,
		uintptr(f.tray.instance),
		0,
	)
	if window == 0 {
		return fmt.Errorf("create tray flyout window: %w", callErr)
	}
	f.window = windows.Handle(window)
	f.roundCorners()
	return nil
}

func (f *trayFlyout) prepareWindow() error {
	bounds, err := f.bounds(flyoutMinHeightPixels)
	if err != nil {
		return err
	}
	result, _, callErr := pSetWindowPos.Call(
		uintptr(f.window),
		0,
		uintptr(bounds.Left),
		uintptr(bounds.Top),
		uintptr(bounds.Right-bounds.Left),
		uintptr(bounds.Bottom-bounds.Top),
		SWP_NOACTIVATE|SWP_NOOWNERZORDER|SWP_NOZORDER,
	)
	if result == 0 {
		return fmt.Errorf("prepare tray flyout window: %w", callErr)
	}
	return nil
}

func (f *trayFlyout) bind() error {
	if err := f.view.Bind("trayGetState", f.state); err != nil {
		return fmt.Errorf("bind tray state: %w", err)
	}
	if err := f.view.Bind("trayReportMetrics", f.reportMetrics); err != nil {
		return fmt.Errorf("bind tray metrics: %w", err)
	}
	if err := f.view.Bind("trayRequestHeight", f.requestHeight); err != nil {
		return fmt.Errorf("bind tray height: %w", err)
	}
	if err := f.view.Bind("trayAction", f.action); err != nil {
		return fmt.Errorf("bind tray action: %w", err)
	}
	return nil
}

func (f *trayFlyout) reportMetrics(metrics flyoutMetrics) {
	slog.Debug(
		"tray flyout web metrics",
		"device_pixel_ratio", metrics.DevicePixelRatio,
		"inner_width", metrics.InnerWidth,
		"inner_height", metrics.InnerHeight,
		"body_font_size", metrics.BodyFontSize,
		"content_height", metrics.ContentHeight,
		"physical_height", metrics.PhysicalHeight,
	)
}

func (f *trayFlyout) requestHeight(heightPixels int) error {
	if heightPixels < int(flyoutMinHeightPixels) || heightPixels > int(flyoutMaxHeightPixels) {
		return fmt.Errorf("tray flyout physical height %d is outside the supported range", heightPixels)
	}
	result, _, callErr := pPostMessage.Call(
		uintptr(f.tray.window),
		trayFlyoutResizeMessage,
		uintptr(heightPixels),
		uintptr(f.window),
	)
	if result == 0 {
		return fmt.Errorf("post tray flyout resize: %w", callErr)
	}
	return nil
}

func (f *trayFlyout) state() trayFlyoutState {
	return trayFlyoutState{
		PendingUpdate: f.tray.hasPendingUpdate(),
		Style:         f.style,
		Labels: flyoutLabels{
			Open:            openUIMenuTitle,
			Settings:        settingsUIMenuTitle,
			UpdateAvailable: updateAvailableMenuTitle,
			Update:          updateMenuTitle,
			Logs:            diagLogsMenuTitle,
			Quit:            quitMenuTitle,
		},
	}
}

func validateTrayFlyoutStyle(style TrayFlyoutStyle) error {
	switch style {
	case TrayFlyoutStyleFluent, TrayFlyoutStyleOllama:
		return nil
	default:
		return fmt.Errorf("unknown tray style %q", style)
	}
}

func (f *trayFlyout) action(name string) error {
	if name == "dismiss" {
		slog.Debug("dismissing tray flyout")
		f.hide()
		return nil
	}

	action, err := f.resolveAction(name)
	if err != nil {
		return err
	}
	f.hide()
	slog.Debug("queueing tray flyout action", "action", name)
	return f.tray.postFlyoutAction(action)
}

func (f *trayFlyout) resolveAction(name string) (trayFlyoutAction, error) {
	switch name {
	case "open":
		return trayFlyoutActionOpen, nil
	case "settings":
		return trayFlyoutActionSettings, nil
	case "logs":
		return trayFlyoutActionLogs, nil
	case "update":
		if !f.tray.hasPendingUpdate() {
			return 0, errors.New("no update is pending")
		}
		return trayFlyoutActionUpdate, nil
	case "quit":
		return trayFlyoutActionQuit, nil
	default:
		return 0, fmt.Errorf("unknown tray action %q", name)
	}
}

func (t *winTray) postFlyoutAction(action trayFlyoutAction) error {
	result, _, callErr := pPostMessage.Call(
		uintptr(t.window),
		trayFlyoutActionMessage,
		uintptr(action),
		0,
	)
	if result == 0 {
		return fmt.Errorf("post tray flyout action: %w", callErr)
	}
	return nil
}

func (t *winTray) handleFlyoutAction(action trayFlyoutAction) error {
	slog.Debug("handling queued tray flyout action", "action", action)
	switch action {
	case trayFlyoutActionOpen:
		t.app.UIShow()
	case trayFlyoutActionSettings:
		t.app.UIRun("/settings")
	case trayFlyoutActionLogs:
		return t.showLogs()
	case trayFlyoutActionUpdate:
		if !t.hasPendingUpdate() {
			return errors.New("no update is pending")
		}
		t.app.DoUpdate()
	case trayFlyoutActionQuit:
		t.app.Quit()
	default:
		return fmt.Errorf("unknown queued tray action %d", action)
	}
	return nil
}

func (f *trayFlyout) show() error {
	visible := f.isVisible()
	slog.Debug("tray flyout show requested", "visible", visible)
	if visible {
		f.hide()
		return nil
	}
	if f.contentHeightPixels == 0 {
		f.showPending = true
		f.refresh()
		slog.Debug("waiting for tray flyout content measurement")
		return nil
	}
	return f.showAtHeight(f.contentHeightPixels)
}

func (f *trayFlyout) showAtHeight(heightPixels int32) error {
	bounds, err := f.bounds(heightPixels)
	if err != nil {
		return err
	}
	f.refresh()

	result, _, callErr := pSetWindowPos.Call(
		uintptr(f.window),
		^uintptr(0), // HWND_TOPMOST
		uintptr(bounds.Left),
		uintptr(bounds.Top),
		uintptr(bounds.Right-bounds.Left),
		uintptr(bounds.Bottom-bounds.Top),
		SWP_NOOWNERZORDER|SWP_SHOWWINDOW,
	)
	if result == 0 {
		return fmt.Errorf("position tray flyout: %w", callErr)
	}
	slog.Debug("tray flyout shown", "left", bounds.Left, "top", bounds.Top, "width", bounds.Right-bounds.Left, "height", bounds.Bottom-bounds.Top)
	if result, _, callErr := pSetForegroundWindow.Call(uintptr(f.window)); result == 0 {
		slog.Debug("unable to activate tray flyout", "error", callErr)
	}
	pSetFocus.Call(uintptr(f.widget)) //nolint:errcheck
	return nil
}

func (f *trayFlyout) resizeToContent(heightPixels int32) error {
	if heightPixels < flyoutMinHeightPixels || heightPixels > flyoutMaxHeightPixels {
		return fmt.Errorf("tray flyout physical height %d is outside the supported range", heightPixels)
	}
	if heightPixels == f.contentHeightPixels {
		return nil
	}
	if f.showPending {
		if err := f.showAtHeight(heightPixels); err != nil {
			return err
		}
		f.showPending = false
		f.contentHeightPixels = heightPixels
		slog.Debug("tray flyout content measured", "physical_height", heightPixels)
		return nil
	}
	if !f.isVisible() {
		f.contentHeightPixels = heightPixels
		slog.Debug("tray flyout content measured", "physical_height", heightPixels)
		return nil
	}

	bounds, err := f.bounds(heightPixels)
	if err != nil {
		return err
	}
	result, _, callErr := pSetWindowPos.Call(
		uintptr(f.window),
		^uintptr(0), // HWND_TOPMOST
		uintptr(bounds.Left),
		uintptr(bounds.Top),
		uintptr(bounds.Right-bounds.Left),
		uintptr(bounds.Bottom-bounds.Top),
		SWP_NOOWNERZORDER,
	)
	if result == 0 {
		return fmt.Errorf("resize tray flyout to content: %w", callErr)
	}
	f.contentHeightPixels = heightPixels
	slog.Debug("tray flyout content measured", "physical_height", heightPixels)
	slog.Debug("tray flyout resized to content", "left", bounds.Left, "top", bounds.Top, "width", bounds.Right-bounds.Left, "height", bounds.Bottom-bounds.Top)
	return nil
}

func (f *trayFlyout) hide() {
	if f.hiding || !f.isVisible() {
		return
	}
	f.hiding = true
	defer func() { f.hiding = false }()
	slog.Debug("hiding tray flyout")
	pShowWindow.Call(uintptr(f.window), SW_HIDE) //nolint:errcheck
}

func (f *trayFlyout) deactivate(nextWindow windows.Handle) {
	if f.hiding || !f.isVisible() {
		return
	}
	overIcon := f.cursorOverTrayIcon()
	slog.Debug("tray flyout deactivated", "next_window", nextWindow, "cursor_over_tray_icon", overIcon)
	f.clickGate.deactivate(overIcon)
	f.hide()
}

func (f *trayFlyout) trayMouseDown() {
	visible := f.isVisible()
	deactivatedOverIcon := f.clickGate.deactivatedOverIcon
	suppress := f.clickGate.mouseDown(visible)
	slog.Debug("tray icon mouse down", "flyout_visible", visible, "deactivated_over_icon", deactivatedOverIcon, "suppress_mouse_up", suppress)
	if suppress {
		f.hide()
	}
}

func (f *trayFlyout) trayMouseUp() bool {
	suppress := f.clickGate.mouseUp()
	slog.Debug("tray icon mouse up", "suppress", suppress)
	return suppress
}

func (f *trayFlyout) isVisible() bool {
	if f.window == 0 {
		return false
	}
	visible, _, _ := pIsWindowVisible.Call(uintptr(f.window))
	return visible != 0
}

func (f *trayFlyout) cursorOverTrayIcon() bool {
	icon, err := f.tray.iconRect()
	if err != nil {
		return false
	}
	cursor := point{}
	if result, _, _ := pGetCursorPos.Call(uintptr(unsafe.Pointer(&cursor))); result == 0 {
		return false
	}
	return pointInRect(cursor, icon)
}

func (f *trayFlyout) refresh() {
	if f.view != nil {
		f.view.Eval("window.refreshTrayState && window.refreshTrayState()")
	}
}

func (f *trayFlyout) bounds(contentHeightPixels int32) (rect, error) {
	if f.window == 0 {
		return rect{}, errors.New("tray flyout window is unavailable")
	}
	anchor, err := f.tray.iconRect()
	if err != nil {
		slog.Debug("unable to locate tray icon; positioning flyout from cursor", "error", err)
		p := point{}
		if result, _, callErr := pGetCursorPos.Call(uintptr(unsafe.Pointer(&p))); result == 0 {
			return rect{}, fmt.Errorf("locate tray flyout: %w", callErr)
		}
		anchor = rect{Left: p.X, Top: p.Y, Right: p.X + 1, Bottom: p.Y + 1}
	}

	monitor, _, callErr := pMonitorFromRect.Call(
		uintptr(unsafe.Pointer(&anchor)),
		MONITOR_DEFAULTTONEAREST,
	)
	if monitor == 0 {
		return rect{}, fmt.Errorf("find tray monitor: %w", callErr)
	}
	info := monitorInfo{Size: uint32(unsafe.Sizeof(monitorInfo{}))}
	if result, _, callErr := pGetMonitorInfo.Call(monitor, uintptr(unsafe.Pointer(&info))); result == 0 {
		return rect{}, fmt.Errorf("read tray monitor: %w", callErr)
	}

	if !f.isVisible() {
		// Move the hidden host onto the target monitor before reading its DPI.
		// Preserve its size so WebView2 never observes a transient 1x1 viewport.
		if result, _, callErr := pSetWindowPos.Call(
			uintptr(f.window),
			0,
			uintptr(anchor.Left),
			uintptr(anchor.Top),
			0,
			0,
			SWP_NOSIZE|SWP_NOACTIVATE|SWP_NOOWNERZORDER|SWP_NOZORDER,
		); result == 0 {
			return rect{}, fmt.Errorf("move tray flyout to target monitor: %w", callErr)
		}
	}
	dpi, _, _ := pGetDpiForWindow.Call(uintptr(f.window))
	if dpi == 0 {
		return rect{}, errors.New("tray flyout window returned zero DPI")
	}
	width := scaleForDPI(flyoutWidth, uint32(dpi))
	// WebView2 can apply Windows text scaling on top of the HWND's monitor DPI.
	// JavaScript therefore reports the measured height in physical pixels; do
	// not scale it again using GetDpiForWindow.
	height := contentHeightPixels
	gap := scaleForDPI(flyoutEdgeGap, uint32(dpi))
	if available := info.Work.Bottom - info.Work.Top - 2*gap; height > available {
		height = available
	}

	return placeFlyout(anchor, info.Work, width, height, gap), nil
}

func (t *winTray) iconRect() (rect, error) {
	t.muNID.RLock()
	if t.nid == nil {
		t.muNID.RUnlock()
		return rect{}, errors.New("tray icon is not initialized")
	}
	identifier := notifyIconIdentifier{
		Wnd:      t.nid.Wnd,
		ID:       t.nid.ID,
		GuidItem: t.nid.GuidItem,
	}
	t.muNID.RUnlock()
	identifier.Size = uint32(unsafe.Sizeof(identifier))

	result := rect{}
	hr, _, _ := pShellNotifyIconGetRect.Call(
		uintptr(unsafe.Pointer(&identifier)),
		uintptr(unsafe.Pointer(&result)),
	)
	if hr != 0 {
		return rect{}, fmt.Errorf("locate tray icon: HRESULT 0x%08x", uint32(hr))
	}
	return result, nil
}

func placeFlyout(anchor, work rect, width, height, gap int32) rect {
	x := anchor.Right - width
	y := anchor.Top - gap - height

	switch {
	case anchor.Top >= work.Bottom: // Bottom taskbar.
	case anchor.Bottom <= work.Top: // Top taskbar.
		y = anchor.Bottom + gap
	case anchor.Left >= work.Right: // Right taskbar.
		x = anchor.Left - gap - width
		y = anchor.Bottom - height
	case anchor.Right <= work.Left: // Left taskbar.
		x = anchor.Right + gap
		y = anchor.Bottom - height
	case work.Bottom-anchor.Bottom > anchor.Top-work.Top:
		y = anchor.Bottom + gap
	}

	x = clampFlyoutCoordinate(x, work.Left, work.Right-width)
	y = clampFlyoutCoordinate(y, work.Top, work.Bottom-height)
	return rect{Left: x, Top: y, Right: x + width, Bottom: y + height}
}

func clampFlyoutCoordinate(value, low, high int32) int32 {
	if high < low {
		return low
	}
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func pointInRect(p point, r rect) bool {
	return p.X >= r.Left && p.X < r.Right && p.Y >= r.Top && p.Y < r.Bottom
}

func scaleForDPI(value int32, dpi uint32) int32 {
	return int32((int64(value)*int64(dpi) + int64(flyoutDefaultDPI)/2) / int64(flyoutDefaultDPI))
}

func (f *trayFlyout) firstChild() windows.Handle {
	child, _, _ := pGetWindow.Call(uintptr(f.window), GW_CHILD)
	return windows.Handle(child)
}

func (f *trayFlyout) resizeWidget() error {
	if f.widget == 0 {
		f.widget = f.firstChild()
	}
	if f.widget == 0 {
		return errors.New("tray WebView2 child window is unavailable")
	}
	client := rect{}
	if result, _, callErr := pGetClientRect.Call(uintptr(f.window), uintptr(unsafe.Pointer(&client))); result == 0 {
		return fmt.Errorf("read tray flyout client rectangle: %w", callErr)
	}
	if result, _, callErr := pMoveWindow.Call(
		uintptr(f.widget),
		0,
		0,
		uintptr(client.Right-client.Left),
		uintptr(client.Bottom-client.Top),
		1,
	); result == 0 {
		return fmt.Errorf("resize tray WebView2 child window: %w", callErr)
	}
	return nil
}

func (f *trayFlyout) roundCorners() {
	const (
		dwmWindowCornerPreference = 33
		dwmRoundPreference        = uint32(2)
	)
	preference := dwmRoundPreference
	// Older Windows versions ignore this attribute and keep square corners.
	pDwmSetWindowAttribute.Call( //nolint:errcheck
		uintptr(f.window),
		dwmWindowCornerPreference,
		uintptr(unsafe.Pointer(&preference)),
		unsafe.Sizeof(preference),
	)
}

func (f *trayFlyout) wndProc(hWnd windows.Handle, message uint32, wParam, lParam uintptr) uintptr {
	switch message {
	case WM_SIZE:
		if f.widget != 0 {
			if err := f.resizeWidget(); err != nil {
				slog.Debug("unable to resize tray WebView2 child window", "error", err)
			}
		}
		return 0
	case WM_ACTIVATE:
		if uint16(wParam) == WA_INACTIVE {
			f.deactivate(windows.Handle(lParam))
		}
		return 0
	case WM_DPICHANGED:
		if lParam != 0 {
			// Windows supplies lParam as a pointer to the suggested window rectangle.
			suggested := (*rect)(unsafe.Pointer(lParam)) //nolint:govet
			if result, _, callErr := pSetWindowPos.Call(
				uintptr(hWnd),
				0,
				uintptr(suggested.Left),
				uintptr(suggested.Top),
				uintptr(suggested.Right-suggested.Left),
				uintptr(suggested.Bottom-suggested.Top),
				SWP_NOACTIVATE|SWP_NOOWNERZORDER|SWP_NOZORDER,
			); result == 0 {
				slog.Debug("unable to resize tray flyout after DPI change", "error", callErr)
			}
		}
		return 0
	case WM_CLOSE:
		f.hide()
		return 0
	case WM_DESTROY:
		f.window = 0
		return 0
	default:
		result, _, _ := pDefWindowProc.Call(uintptr(hWnd), uintptr(message), wParam, lParam)
		return result
	}
}

func (f *trayFlyout) destroy() {
	if f.view != nil {
		f.view.Destroy()
		f.view = nil
	}
	if f.window != 0 {
		if result, _, err := pDestroyWindow.Call(uintptr(f.window)); result == 0 {
			slog.Debug("unable to destroy tray flyout window", "error", err)
		}
		f.window = 0
	}
	if f.classRegistered {
		if err := f.wcex.unregister(); err != nil {
			slog.Debug("unable to unregister tray flyout window", "error", err)
		}
		f.classRegistered = false
	}
	if f.comInitialized {
		pCoUninitialize.Call() //nolint:errcheck
		f.comInitialized = false
	}
}
