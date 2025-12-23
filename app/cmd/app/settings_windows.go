//go:build windows

package main

import "github.com/ollama/ollama/app/store"

// SetSettingsStore sets the store reference for settings callbacks (stub for Windows)
func SetSettingsStore(s *store.Store) {
	// TODO: Implement Windows native settings
}

// SetRestartCallback sets the function to call when settings change requires a restart (stub for Windows)
func SetRestartCallback(cb func()) {
	// TODO: Implement Windows native settings
}

