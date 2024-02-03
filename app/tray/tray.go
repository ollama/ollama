package tray

import (
	"fmt"
	"runtime"

	"github.com/jmorganca/ollama/app/assets"
	"github.com/jmorganca/ollama/app/tray/commontray"
)

func NewTray(upgradeCB func() error) (commontray.OllamaTray, error) {
	extension := ".png"
	if runtime.GOOS == "windows" {
		extension = ".ico"
	}
	iconName := commontray.UpdateIconName + extension
	updateIcon, err := assets.GetIcon(iconName)
	if err != nil {
		return nil, fmt.Errorf("Failed to load icon %s: %w", iconName, err)
	}
	iconName = commontray.IconName + extension
	icon, err := assets.GetIcon(iconName)
	if err != nil {
		return nil, fmt.Errorf("Failed to load icon %s: %w", iconName, err)
	}

	tray, err := InitPlatformTray(icon, updateIcon)
	if err != nil {
		return nil, err
	}

	return tray, nil
}
