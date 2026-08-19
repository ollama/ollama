//go:build windows

package wintray

import (
	"strings"
	"testing"
	"unsafe"
)

type recordingAppCallbacks struct {
	path    string
	showed  bool
	updated bool
	quit    bool
}

type appCallbacksWithoutWebView struct{}

func (*appCallbacksWithoutWebView) UIRun(string)    {}
func (*appCallbacksWithoutWebView) UIShow()         {}
func (*appCallbacksWithoutWebView) UITerminate()    {}
func (*appCallbacksWithoutWebView) UIRunning() bool { return false }
func (*appCallbacksWithoutWebView) Quit()           {}
func (*appCallbacksWithoutWebView) DoUpdate()       {}

func (a *recordingAppCallbacks) UIRun(path string) { a.path = path }
func (a *recordingAppCallbacks) UIShow()           { a.showed = true }
func (*recordingAppCallbacks) UITerminate()        {}
func (*recordingAppCallbacks) UIRunning() bool     { return false }
func (a *recordingAppCallbacks) Quit()             { a.quit = true }
func (a *recordingAppCallbacks) DoUpdate()         { a.updated = true }
func (*recordingAppCallbacks) NewTrayWebView(unsafe.Pointer) (TrayWebView, error) {
	return nil, nil
}
func (*recordingAppCallbacks) TrayFlyoutStyle() TrayFlyoutStyle { return TrayFlyoutStyleFluent }

func TestInitTrayRequiresWebViewFactory(t *testing.T) {
	if _, err := InitTray(nil, nil, &appCallbacksWithoutWebView{}); err == nil {
		t.Fatal("InitTray accepted an app without WebView2 tray support")
	}
}

func TestFlyoutStateUsesMenuLabels(t *testing.T) {
	flyout := trayFlyout{tray: &winTray{}, style: TrayFlyoutStyleFluent}
	got := flyout.state().Labels
	want := flyoutLabels{
		Open:            openUIMenuTitle,
		Settings:        settingsUIMenuTitle,
		UpdateAvailable: updateAvailableMenuTitle,
		Update:          updateMenuTitle,
		Logs:            diagLogsMenuTitle,
		Quit:            quitMenuTitle,
	}
	if got != want {
		t.Fatalf("flyout labels = %#v, want %#v", got, want)
	}
}

func TestFlyoutOmitsStylePickerAndChevrons(t *testing.T) {
	for _, marker := range []string{"style-picker", "data-style-choice", "chevron", "traySetStyle"} {
		t.Run(marker, func(t *testing.T) {
			if strings.Contains(trayFlyoutHTML, marker) {
				t.Errorf("flyout HTML still exposes %q", marker)
			}
		})
	}
}

func TestFlyoutMeasuresContentHeight(t *testing.T) {
	for _, marker := range []string{`id="flyout-content"`, "ResizeObserver", "trayRequestHeight", "physicalHeight"} {
		t.Run(marker, func(t *testing.T) {
			if !strings.Contains(trayFlyoutHTML, marker) {
				t.Errorf("flyout HTML is missing dynamic-size contract %q", marker)
			}
		})
	}
	if strings.Contains(trayFlyoutHTML, "overflow-y: auto") {
		t.Error("flyout HTML enables a scrollbar that can feed back into content measurement")
	}
}

func TestFlyoutKeyboardNavigation(t *testing.T) {
	for _, marker := range []string{`"ArrowDown"`, `"ArrowUp"`, `"Home"`, `"End"`} {
		t.Run(marker, func(t *testing.T) {
			if !strings.Contains(trayFlyoutHTML, marker) {
				t.Errorf("flyout HTML is missing keyboard navigation contract %s", marker)
			}
		})
	}
	if !strings.Contains(trayFlyoutHTML, `aria-labelledby="update-available-label"`) {
		t.Error("flyout update region does not reuse its visible label as its accessible name")
	}
}

func TestFlyoutDisablesBrowserContextMenu(t *testing.T) {
	if !strings.Contains(trayFlyoutHTML, `document.addEventListener("contextmenu"`) {
		t.Fatal("flyout HTML does not suppress the browser context menu")
	}
}

func TestShowFlyoutIgnoresLifecycleTransitions(t *testing.T) {
	for _, tt := range []struct {
		name               string
		flyoutInitializing bool
		shuttingDown       bool
	}{
		{name: "initializing", flyoutInitializing: true},
		{name: "shutting down", shuttingDown: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tray := winTray{
				flyoutInitializing: tt.flyoutInitializing,
				shuttingDown:       tt.shuttingDown,
			}
			if err := tray.showFlyout(); err != nil {
				t.Fatalf("showFlyout() returned %v", err)
			}
			if tray.flyout != nil {
				t.Fatal("showFlyout() created a flyout during a lifecycle transition")
			}
		})
	}
}

func TestResizeToContentKeepsStateOnError(t *testing.T) {
	const previousHeight = int32(200)
	flyout := trayFlyout{
		tray:                &winTray{},
		contentHeightPixels: previousHeight,
		showPending:         true,
	}
	if err := flyout.resizeToContent(300); err == nil {
		t.Fatal("resizeToContent succeeded without a flyout window")
	}
	if flyout.contentHeightPixels != previousHeight || !flyout.showPending {
		t.Fatalf("flyout state = (%d, %v), want (%d, true)", flyout.contentHeightPixels, flyout.showPending, previousHeight)
	}
}

func TestFlyoutMenuOrder(t *testing.T) {
	items := []string{
		`data-action="open"`,
		`data-action="settings"`,
		`id="update-card"`,
		`data-action="logs"`,
		`data-action="quit"`,
	}
	previous := -1
	for _, item := range items {
		t.Run(item, func(t *testing.T) {
			index := strings.Index(trayFlyoutHTML, item)
			if index < 0 {
				t.Fatalf("flyout HTML is missing %s", item)
			}
			if index <= previous {
				t.Fatalf("%s occurs out of menu order", item)
			}
			previous = index
		})
	}
}

func TestFlyoutActions(t *testing.T) {
	tests := []struct {
		name    string
		action  string
		want    trayFlyoutAction
		pending bool
		check   func(*testing.T, *recordingAppCallbacks)
	}{
		{
			name:   "open",
			action: "open",
			want:   trayFlyoutActionOpen,
			check: func(t *testing.T, app *recordingAppCallbacks) {
				if !app.showed {
					t.Fatal("open action did not show the UI")
				}
			},
		},
		{
			name:   "settings",
			action: "settings",
			want:   trayFlyoutActionSettings,
			check: func(t *testing.T, app *recordingAppCallbacks) {
				if app.path != "/settings" {
					t.Fatalf("settings path = %q, want %q", app.path, "/settings")
				}
			},
		},
		{
			name:   "logs",
			action: "logs",
			want:   trayFlyoutActionLogs,
			check:  func(*testing.T, *recordingAppCallbacks) {},
		},
		{
			name:    "update",
			action:  "update",
			want:    trayFlyoutActionUpdate,
			pending: true,
			check: func(t *testing.T, app *recordingAppCallbacks) {
				if !app.updated {
					t.Fatal("update action did not start the update")
				}
			},
		},
		{
			name:   "quit",
			action: "quit",
			want:   trayFlyoutActionQuit,
			check: func(t *testing.T, app *recordingAppCallbacks) {
				if !app.quit {
					t.Fatal("quit action did not quit the app")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			app := &recordingAppCallbacks{}
			tray := &winTray{app: app, pendingUpdate: tt.pending}
			flyout := trayFlyout{tray: tray}
			action, err := flyout.resolveAction(tt.action)
			if err != nil {
				t.Fatalf("resolveAction(%q) returned %v", tt.action, err)
			}
			if action != tt.want {
				t.Fatalf("resolveAction(%q) = %d, want %d", tt.action, action, tt.want)
			}
			// showLogs launches Explorer, so its mapping is verified without executing it.
			if action != trayFlyoutActionLogs {
				if err := tray.handleFlyoutAction(action); err != nil {
					t.Fatalf("handleFlyoutAction(%d) returned %v", action, err)
				}
			}
			tt.check(t, app)
		})
	}

	flyout := trayFlyout{tray: &winTray{app: &recordingAppCallbacks{}}}
	if _, err := flyout.resolveAction("update"); err == nil {
		t.Fatal("update action succeeded without a pending update")
	}
	if _, err := flyout.resolveAction("unknown"); err == nil {
		t.Fatal("unknown action succeeded")
	}
	if err := flyout.tray.handleFlyoutAction(0); err == nil {
		t.Fatal("unknown queued action succeeded")
	}
}

func TestPlaceFlyout(t *testing.T) {
	const contentHeight = int32(304)
	tests := []struct {
		name   string
		anchor rect
		work   rect
		want   rect
	}{
		{
			name:   "bottom taskbar",
			anchor: rect{Left: 1800, Top: 1040, Right: 1840, Bottom: 1080},
			work:   rect{Left: 0, Top: 0, Right: 1920, Bottom: 1040},
			want:   rect{Left: 1536, Top: 728, Right: 1840, Bottom: 1032},
		},
		{
			name:   "top taskbar",
			anchor: rect{Left: 1800, Top: 0, Right: 1840, Bottom: 40},
			work:   rect{Left: 0, Top: 40, Right: 1920, Bottom: 1080},
			want:   rect{Left: 1536, Top: 48, Right: 1840, Bottom: 352},
		},
		{
			name:   "right taskbar",
			anchor: rect{Left: 1880, Top: 1000, Right: 1920, Bottom: 1040},
			work:   rect{Left: 0, Top: 0, Right: 1880, Bottom: 1080},
			want:   rect{Left: 1568, Top: 736, Right: 1872, Bottom: 1040},
		},
		{
			name:   "left taskbar",
			anchor: rect{Left: 0, Top: 1000, Right: 40, Bottom: 1040},
			work:   rect{Left: 40, Top: 0, Right: 1920, Bottom: 1080},
			want:   rect{Left: 48, Top: 736, Right: 352, Bottom: 1040},
		},
		{
			name:   "cursor fallback clamps to work area",
			anchor: rect{Left: 4, Top: 4, Right: 5, Bottom: 5},
			work:   rect{Left: 0, Top: 0, Right: 800, Bottom: 600},
			want:   rect{Left: 0, Top: 13, Right: 304, Bottom: 317},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := placeFlyout(tt.anchor, tt.work, flyoutWidth, contentHeight, flyoutEdgeGap)
			if got != tt.want {
				t.Fatalf("placeFlyout() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestValidateTrayFlyoutStyle(t *testing.T) {
	for _, style := range []TrayFlyoutStyle{TrayFlyoutStyleFluent, TrayFlyoutStyleOllama} {
		t.Run(string(style), func(t *testing.T) {
			if err := validateTrayFlyoutStyle(style); err != nil {
				t.Fatalf("validateTrayFlyoutStyle(%q) returned %v", style, err)
			}
			if !strings.Contains(trayFlyoutHTML, `data-style="`+string(style)+`"`) {
				t.Fatalf("flyout HTML has no token set for %q", style)
			}
		})
	}
	if err := validateTrayFlyoutStyle("future"); err == nil {
		t.Fatal("validateTrayFlyoutStyle accepted an unknown style")
	}
}

func TestTrayClickGate(t *testing.T) {
	t.Run("deactivation arrives before tray mouse down", func(t *testing.T) {
		gate := trayClickGate{}
		gate.deactivate(true)
		if !gate.mouseDown(false) {
			t.Fatal("mouse down did not remember the tray-icon deactivation")
		}
		if !gate.mouseUp() {
			t.Fatal("mouse up reopened the popup")
		}
		if gate.mouseUp() {
			t.Fatal("dismissal leaked into the next click")
		}
	})

	t.Run("tray mouse down arrives before deactivation", func(t *testing.T) {
		gate := trayClickGate{}
		if !gate.mouseDown(true) {
			t.Fatal("mouse down did not dismiss the visible popup")
		}
		if !gate.mouseUp() {
			t.Fatal("mouse up reopened the popup")
		}
	})

	t.Run("hidden popup opens normally", func(t *testing.T) {
		gate := trayClickGate{}
		gate.deactivate(false)
		if gate.mouseDown(false) || gate.mouseUp() {
			t.Fatal("an unrelated deactivation suppressed the tray click")
		}
	})
}

func TestPointInRect(t *testing.T) {
	r := rect{Left: 10, Top: 20, Right: 30, Bottom: 40}
	for _, tt := range []struct {
		name string
		p    point
		want bool
	}{
		{name: "inside", p: point{X: 20, Y: 30}, want: true},
		{name: "top left", p: point{X: 10, Y: 20}, want: true},
		{name: "right edge", p: point{X: 30, Y: 30}, want: false},
		{name: "bottom edge", p: point{X: 20, Y: 40}, want: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := pointInRect(tt.p, r); got != tt.want {
				t.Fatalf("pointInRect(%#v, %#v) = %t, want %t", tt.p, r, got, tt.want)
			}
		})
	}
}

func TestScaleForDPI(t *testing.T) {
	const dpi = 144
	want := flyoutWidth * 3 / 2
	if got := scaleForDPI(flyoutWidth, dpi); got != want {
		t.Fatalf("scaleForDPI(%d, %d) = %d, want %d", flyoutWidth, dpi, got, want)
	}
}
