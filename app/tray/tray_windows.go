package tray

import (
	"github.com/uppercaveman/ollama-server/app/tray/commontray"
	"github.com/uppercaveman/ollama-server/app/tray/wintray"
)

func InitPlatformTray(icon, updateIcon []byte) (commontray.OllamaTray, error) {
	return wintray.InitTray(icon, updateIcon)
}
