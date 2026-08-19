//go:build windows

package wintray

import (
	"errors"

	"golang.org/x/sys/windows"
)

// ErrWindowsAppRuntimeUnavailable reports that the WinUI bootstrap or runtime
// required by the Windows desktop app could not be loaded.
var ErrWindowsAppRuntimeUnavailable = errors.New("Windows App Runtime 1.8 is unavailable")

const (
	UpdateIconName = "tray_upgrade.ico"
	IconName       = "tray.ico"
	ClassName      = "OllamaClass"
)

type TrayCallbacks interface {
	Quit()
	TrayRun()
	UpdateAvailable(ver string) error
	GetIconHandle() windows.Handle
}

type AppCallbacks interface {
	UIRun(path string)
	UIShow()
	UITerminate()
	UIRunning() bool
	Quit()
	DoUpdate()
}

type URLSchemeHandler interface {
	HandleURLScheme(urlScheme string)
}
