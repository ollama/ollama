package tray

import (
	"ollama.com/app/tray/commontray"
	"ollama.com/app/tray/wintray"
)

func InitPlatformTray(icon, updateIcon []byte) (commontray.OllamaTray, error) {
	return wintray.InitTray(icon, updateIcon)
}
