//go:build windows

package main

import (
	"errors"
	"os"
	"path/filepath"

	"golang.org/x/sys/windows/registry"
)

// loginItemStatus is whether Ollama starts when people sign in to Windows.
// Ollama starts from a shortcut in the Startup folder, which the installer
// creates and Settings adds or removes.
type loginItemStatus int

const (
	loginItemOff loginItemStatus = iota
	loginItemOn
	// loginItemOffInSettings means the shortcut exists, but the Startup
	// apps page of Windows Settings or Task Manager turned it off.
	loginItemOffInSettings
	// loginItemUnavailable means there is no shortcut to start from, as in
	// development builds.
	loginItemUnavailable
)

// startupApprovedKey records which Startup folder items people turned off.
const startupApprovedKey = `Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\StartupFolder`

func getLoginItemStatus() loginItemStatus {
	if _, err := os.Stat(startupShortcut); err != nil {
		if _, err := os.Stat(installedShortcut()); err != nil {
			return loginItemUnavailable
		}
		return loginItemOff
	}
	if startupTurnedOffInSettings() {
		return loginItemOffInSettings
	}
	return loginItemOn
}

// setLoginItemEnabled adds or removes the Startup folder shortcut. Turning
// it on also undoes turning it off in Windows Settings.
func setLoginItemEnabled(enabled bool) error {
	if !enabled {
		if err := os.Remove(startupShortcut); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	if err := createLoginShortcut(); err != nil {
		return err
	}
	k, err := registry.OpenKey(registry.CURRENT_USER, startupApprovedKey, registry.SET_VALUE)
	if errors.Is(err, registry.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	defer k.Close()
	// Startup folder items without a value are on.
	if err := k.DeleteValue(filepath.Base(startupShortcut)); err != nil && !errors.Is(err, registry.ErrNotExist) {
		return err
	}
	return nil
}

func startupTurnedOffInSettings() bool {
	k, err := registry.OpenKey(registry.CURRENT_USER, startupApprovedKey, registry.QUERY_VALUE)
	if err != nil {
		return false
	}
	defer k.Close()
	value, _, err := k.GetBinaryValue(filepath.Base(startupShortcut))
	// The low bit of the first byte is set while the item is turned off.
	return err == nil && len(value) > 0 && value[0]&1 != 0
}
