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

func (t *winTray) UpdateAvailable(ver string) error {
	t.muState.Lock()
	defer t.muState.Unlock()

	t.pendingUpdate = true
	if !t.updateNotified {
		slog.Debug("updating tray icon and sending notification for new update")
		iconFilePath, err := iconBytesToFilePath(t.updateIcon)
		if err != nil {
			return fmt.Errorf("unable to write icon data to temp file: %w", err)
		}
		if err := t.setIcon(iconFilePath); err != nil {
			return fmt.Errorf("unable to set icon: %w", err)
		}
		// Now pop up the notification
		t.muNID.Lock()
		defer t.muNID.Unlock()
		copy(t.nid.InfoTitle[:], windows.StringToUTF16(updateTitle))
		copy(t.nid.Info[:], windows.StringToUTF16(fmt.Sprintf(updateMessage, ver)))
		t.nid.Flags |= NIF_INFO
		t.nid.Timeout = 10
		t.nid.Size = uint32(unsafe.Sizeof(*t.nid))
		err = t.nid.modify()
		if err != nil {
			return err
		}
		t.updateNotified = true
	}
	return nil
}

func (t *winTray) hasPendingUpdate() bool {
	t.muState.RLock()
	defer t.muState.RUnlock()
	return t.pendingUpdate
}

func (t *winTray) showLogs() error {
	appDataDir := filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama")
	cmdPath := "c:\\Windows\\system32\\cmd.exe"
	slog.Debug("viewing logs", "path", appDataDir)
	cmd := exec.Command(cmdPath, "/c", "start", appDataDir)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: false, CreationFlags: 0x08000000}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("open log directory: %w", err)
	}
	return nil
}
