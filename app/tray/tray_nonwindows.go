//go:build !windows

package tray

import (
	"errors"

	"github.com/zhuangjie1125/ollama/app/tray/commontray"
)

func InitPlatformTray(icon, updateIcon []byte) (commontray.OllamaTray, error) {
	return nil, errors.New("not implemented")
}
