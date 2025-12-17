//go:build windows

package wintray

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	_ = iota
	openUIMenuID
	settingsUIMenuID
	updateSeparatorMenuID
	updateAvailableMenuID
	updateMenuID
	separatorMenuID
	diagLogsMenuID
	diagSeparatorMenuID
	quitMenuID
)

func (t *winTray) initMenus() error {
	if err := t.addOrUpdateMenuItem(openUIMenuID, 0, openUIMenuTitle, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w", err)
	}
	if err := t.addOrUpdateMenuItem(settingsUIMenuID, 0, settingsUIMenuTitle, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w", err)
	}
	if err := t.addOrUpdateMenuItem(diagLogsMenuID, 0, diagLogsMenuTitle, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}
	if err := t.addSeparatorMenuItem(diagSeparatorMenuID, 0); err != nil {
		return fmt.Errorf("unable to create menu entries %w", err)
	}

	if err := t.addOrUpdateMenuItem(quitMenuID, 0, quitMenuTitle, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w", err)
	}
	return nil
}

func (t *winTray) ClearUpdateAvailable() error {
	if t.updateNotified {
		slog.Debug("clearing update notification and menu items")
		if err := t.removeMenuItem(updateSeparatorMenuID, 0); err != nil {
			return fmt.Errorf("unable to remove menu entries %w", err)
		}
		if err := t.removeMenuItem(updateAvailableMenuID, 0); err != nil {
			return fmt.Errorf("unable to remove menu entries %w", err)
		}
		if err := t.removeMenuItem(updateMenuID, 0); err != nil {
			return fmt.Errorf("unable to remove menu entries %w", err)
		}
		if err := t.removeMenuItem(separatorMenuID, 0); err != nil {
			return fmt.Errorf("unable to remove menu entries %w", err)
		}
		iconFilePath, err := iconBytesToFilePath(wt.normalIcon)
		if err != nil {
			return fmt.Errorf("unable to write icon data to temp file: %w", err)
		}
		if err := t.setIcon(iconFilePath); err != nil {
			return fmt.Errorf("unable to set icon: %w", err)
		}
		t.updateNotified = false
		t.pendingUpdate = false
	}
	return nil
}

func (t *winTray) UpdateAvailable(ver string) error {
	if !t.updateNotified {
		slog.Debug("updating menu and sending notification for new update")
		if err := t.addSeparatorMenuItem(updateSeparatorMenuID, 0); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		if err := t.addOrUpdateMenuItem(updateAvailableMenuID, 0, updateAvailableMenuTitle, true); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		if err := t.addOrUpdateMenuItem(updateMenuID, 0, updateMenuTitle, false); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		if err := t.addSeparatorMenuItem(separatorMenuID, 0); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		iconFilePath, err := iconBytesToFilePath(wt.updateIcon)
		if err != nil {
			return fmt.Errorf("unable to write icon data to temp file: %w", err)
		}
		if err := wt.setIcon(iconFilePath); err != nil {
			return fmt.Errorf("unable to set icon: %w", err)
		}
		t.updateNotified = true

		t.pendingUpdate = true
		// Now pop up the notification
		t.muNID.Lock()
		defer t.muNID.Unlock()
		copy(t.nid.InfoTitle[:], windows.StringToUTF16(updateTitle))
		copy(t.nid.Info[:], windows.StringToUTF16(fmt.Sprintf(updateMessage, ver)))
		t.nid.Flags |= NIF_INFO
		t.nid.Timeout = 10
		t.nid.Size = uint32(unsafe.Sizeof(*wt.nid))
		err = t.nid.modify()
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *winTray) showLogs() error {
	localAppData := os.Getenv("LOCALAPPDATA")
	AppDataDir := filepath.Join(localAppData, "Ollama")
	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	slog.Debug(fmt.Sprintf("viewing logs with start %s", AppDataDir))
	cmd := exec.Command(cmd_path, "/c", "start", AppDataDir)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: false, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to open log dir: %s", err))
	}
	return nil
}
