//go:build windows && !winui

package wintray

import "fmt"

// iup-go only exposes its WinUI backend when the consuming build supplies the
// winui tag. Go packages cannot enable build tags for their dependencies, so
// this fallback keeps untagged builds compilable and reports how to opt in.
func Initialize() error {
	return fmt.Errorf("the Windows app must be built with the winui build tag; use: go run -tags winui .\\app\\cmd\\app\\")
}

func NewTray(AppCallbacks) (TrayCallbacks, error) {
	return nil, Initialize()
}
