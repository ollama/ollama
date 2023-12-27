package tray

import "github.com/getlantern/systray"

var (
	Title   = "Ollama"
	ToolTip = "Ollama"

	UpdateIconName = "iconUpdateTemplate@2x"
	IconName       = "iconTemplate@2x"
)

type OllamaTray struct {
	// TODO
	UpdateAvailable bool
	availUpdateMI   *systray.MenuItem
	installUpdateMI *systray.MenuItem
	icon            []byte
	updateIcon      []byte
	upgradeCB       func()
}
