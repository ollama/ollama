package tray

import (
	"github.com/zhuangjie1125/ollama/app/tray/commontray"
	"github.com/zhuangjie1125/ollama/app/tray/wintray"
)

func InitPlatformTray(icon, updateIcon []byte) (commontray.OllamaTray, error) {
	return wintray.InitTray(icon, updateIcon)
}
