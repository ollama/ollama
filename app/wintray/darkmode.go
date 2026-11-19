//go:build windows

package wintray

import (
	"syscall"

	"golang.org/x/sys/windows"
)

// Dark-mode support relies on undocumented uxtheme.dll exports (referenced by
// ordinal) that Windows itself uses to theme its controls and context menus.
// They were added in Windows 10 1809 (build 17763); on older builds the same
// ordinals map to unrelated internal functions, so they are only resolved when
// the running OS build is new enough.
//
// Getting the tray's TrackPopupMenu to render dark takes three steps, and on
// Windows 10 (unlike Windows 11) all three are required:
//   - opt the process into the system app theme (SetPreferredAppMode, or
//     AllowDarkModeForApp before build 18362),
//   - opt the menu's owner window in (AllowDarkModeForWindow) — Windows 10 does
//     not theme the popup from the process app mode alone, and
//   - flush the cached menu themes (FlushMenuThemes).
const (
	// uxtheme.dll export ordinals.
	ordRefreshImmersiveColorPolicyState = 104
	ordAllowDarkModeForWindow           = 133
	ordSetPreferredAppMode              = 135 // AllowDarkModeForApp on builds 17763..18361
	ordFlushMenuThemes                  = 136

	// PreferredAppMode::AllowDark — follow the system "app mode" setting rather
	// than forcing a fixed theme. Also a valid TRUE for AllowDarkModeForApp.
	appModeAllowDark = 1

	// Windows 10 1809 — first build with the dark-mode menu exports.
	buildDarkModeAvailable = 17763
)

var (
	uxtheme         = windows.NewLazySystemDLL("uxtheme.dll")
	pGetProcAddress = k32.NewProc("GetProcAddress")

	pRefreshImmersiveColorPolicyState uintptr
	pAllowDarkModeForWindow           uintptr
	pSetPreferredAppMode              uintptr
	pFlushMenuThemes                  uintptr
)

func init() {
	if windows.RtlGetVersion().BuildNumber < buildDarkModeAvailable {
		return
	}
	if err := uxtheme.Load(); err != nil {
		return
	}
	// GetProcAddress accepts an ordinal in place of a name when the high word is
	// zero (MAKEINTRESOURCE), which is how these unnamed exports are resolved.
	resolve := func(ordinal uintptr) uintptr {
		addr, _, _ := pGetProcAddress.Call(uxtheme.Handle(), ordinal)
		return addr
	}
	pSetPreferredAppMode = resolve(ordSetPreferredAppMode)
	pAllowDarkModeForWindow = resolve(ordAllowDarkModeForWindow)
	pFlushMenuThemes = resolve(ordFlushMenuThemes)
	pRefreshImmersiveColorPolicyState = resolve(ordRefreshImmersiveColorPolicyState)
}

// enableDarkModeMenus opts the process into following the Windows app theme.
// Call once before creating windows. Best-effort: a no-op on Windows builds
// without dark-mode support.
func enableDarkModeMenus() {
	if pSetPreferredAppMode == 0 {
		return
	}
	// On builds 17763..18361 ordinal 135 is AllowDarkModeForApp(BOOL); on 18362+
	// it is SetPreferredAppMode(enum). The argument 1 means TRUE / AllowDark for
	// both, so the same call works without branching on the build number.
	syscall.SyscallN(pSetPreferredAppMode, appModeAllowDark) //nolint:errcheck
	refreshImmersiveColorPolicyState()
}

// applyDarkModeToWindow opts a specific window into dark mode so menus it owns
// (the tray's TrackPopupMenu) render dark on Windows 10, which—unlike Windows
// 11—does not theme the popup from the process app mode alone. Call after the
// window is created and after enableDarkModeMenus.
func applyDarkModeToWindow(hwnd windows.Handle) {
	if pAllowDarkModeForWindow == 0 {
		return
	}
	syscall.SyscallN(pAllowDarkModeForWindow, uintptr(hwnd), 1) //nolint:errcheck // 1 == allow
	refreshImmersiveColorPolicyState()
	flushMenuThemes()
}

// onThemeChanged re-applies menu theming after a live light/dark switch so the
// next time the tray menu opens it uses the current colors.
func onThemeChanged() {
	refreshImmersiveColorPolicyState()
	flushMenuThemes()
}

func refreshImmersiveColorPolicyState() {
	if pRefreshImmersiveColorPolicyState == 0 {
		return
	}
	syscall.SyscallN(pRefreshImmersiveColorPolicyState) //nolint:errcheck
}

func flushMenuThemes() {
	if pFlushMenuThemes == 0 {
		return
	}
	syscall.SyscallN(pFlushMenuThemes) //nolint:errcheck
}
