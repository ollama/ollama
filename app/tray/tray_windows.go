package tray

import (
	"github.com/ollama/ollama/app/tray/commontray"
	"github.com/ollama/ollama/app/tray/wintray"
)

func InitPlatformTray(icon, updateIcon []byte) (commontray.OllamaTray, error) {
	return wintray.InitTray(icon, updateIcon)
}
