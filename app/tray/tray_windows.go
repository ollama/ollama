package tray

import (
	"github.com/jmorganca/ollama/app/tray/commontray"
	"github.com/jmorganca/ollama/app/tray/wintray"
)

func InitPlatformTray(icon, updateIcon []byte) (commontray.OllamaTray, error) {
	return wintray.InitTray(icon, updateIcon)
}
