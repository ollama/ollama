package tray

import (
	"fmt"
	"log"
	"runtime"

	"github.com/getlantern/systray"
	"github.com/jmorganca/ollama/desktop/assets"
)

func NewTray(upgradeCB func() error) (*OllamaTray, error) {
	extension := ".png"
	if runtime.GOOS == "windows" {
		extension = ".ico"
	}
	iconName := UpdateIconName + extension
	updateIcon, err := assets.GetIcon(iconName)
	if err != nil {
		return nil, fmt.Errorf("Failed to load icon %s: %w", iconName, err)
	}
	iconName = IconName + extension
	icon, err := assets.GetIcon(iconName)
	if err != nil {
		return nil, fmt.Errorf("Failed to load icon %s: %w", iconName, err)
	}

	return &OllamaTray{
		UpdateAvailable: false,
		updateIcon:      updateIcon,
		icon:            icon,
		upgradeCB:       upgradeCB,
	}, nil
}

func (tray *OllamaTray) SetUpdateAvailable(updateAvailable bool) {
	log.Printf("OllamaTray.SetUpdateAvailable called witih %v", updateAvailable)
	if tray.UpdateAvailable == updateAvailable {
		return
	}
	tray.UpdateAvailable = updateAvailable
	if tray.UpdateAvailable {
		tray.availUpdateMI.Show()
		tray.installUpdateMI.Show()
		systray.SetTemplateIcon(tray.updateIcon, tray.updateIcon)
	} else {
		tray.availUpdateMI.Hide()
		tray.installUpdateMI.Hide()
		systray.SetTemplateIcon(tray.icon, tray.icon)
	}
}

func (tray *OllamaTray) Run() {
	systray.Run(tray.onReady, tray.onExit)
}

func (tray *OllamaTray) onExit() {
	fmt.Println("XXX exiting")
}

func (tray *OllamaTray) onReady() {
	log.Printf("XXX tray onReady called")
	if tray.UpdateAvailable {
		systray.SetTemplateIcon(tray.updateIcon, tray.updateIcon)
	} else {
		systray.SetTemplateIcon(tray.icon, tray.icon)
	}

	tray.availUpdateMI = systray.AddMenuItem("An update is available", "An update is available")
	tray.availUpdateMI.Disable()
	tray.installUpdateMI = systray.AddMenuItem("Restart to update", "Restart to update")
	go func() {
		<-tray.installUpdateMI.ClickedCh
		log.Printf("XXX triggering restart to update")
		tray.upgradeCB()
	}()
	systray.AddSeparator()
	if !tray.UpdateAvailable {
		tray.availUpdateMI.Hide()
		tray.installUpdateMI.Hide()
	}
	// systray.SetTitle(Title)
	systray.SetTooltip(ToolTip)
	myQuitOrig := systray.AddMenuItem("Quit Ollama", "Quit Ollama")
	go func() {
		<-myQuitOrig.ClickedCh
		log.Println("XXX Tray Quitting")
		systray.Quit()
		log.Println("XXX Tray Finished quit") // notreached
	}()
	log.Printf("XXX tray Finished onReady")
}
