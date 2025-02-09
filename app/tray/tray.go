package tray

import (
    "fmt"
    "runtime"

    "github.com/ollama/ollama/app/assets"
    "github.com/ollama/ollama/app/tray/commontray"
)

var getIcon = assets.GetIcon
var initTray = InitPlatformTray

func NewTray() (commontray.OllamaTray, error) {
    extension := ".png"
    if runtime.GOOS == "windows" {
        extension = ".ico"
    }
    iconName := commontray.UpdateIconName + extension
    updateIcon, err := getIcon(iconName)
    if err != nil {
        return nil, fmt.Errorf("failed to load icon %s: %w", iconName, err)
    }
    iconName = commontray.IconName + extension
    icon, err := getIcon(iconName)
    if err != nil {
        return nil, fmt.Errorf("failed to load icon %s: %w", iconName, err)
    }

    return initTray(icon, updateIcon)
}
