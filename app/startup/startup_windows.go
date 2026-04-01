//go:build windows

package startup

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

type pathConfig struct {
	shortcutOrigin  string
	startupShortcut string
}

type windowsRegistrar struct {
	pathConfig pathConfig
}

func NewRegistrar(options ...func(*pathConfig)) Registrar {
	pathConfig := &pathConfig{
		shortcutOrigin:  filepath.Join(os.Getenv("LOCALAPPDATA"), "Programs", "Ollama", "lib", "Ollama.lnk"),
		startupShortcut: filepath.Join(os.Getenv("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "Ollama.lnk"),
	}
	for _, option := range options {
		option(pathConfig)
	}
	return &windowsRegistrar{
		pathConfig: *pathConfig,
	}
}

// GetState implements [Registrar].
func (w *windowsRegistrar) GetState() (RegistrationState, error) {
	// Check if the origin shortcut actually exists. It may be missing from a portable install or if it was deleted at some point.
	if _, err := os.Stat(w.pathConfig.shortcutOrigin); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			slog.Info("Startup shortcut origin does not exist, autostart registration is not supported", "shortcutOrigin", w.pathConfig.shortcutOrigin)
			return RegistrationState{Supported: false, Registered: false}, nil
		}
		return RegistrationState{Supported: false, Registered: false}, fmt.Errorf("failed to read the startup shortcut origin %s : %w", w.pathConfig.shortcutOrigin, err)
	}

	_, err := os.Stat(w.pathConfig.startupShortcut)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return RegistrationState{Supported: true, Registered: false}, nil
		}
		return RegistrationState{Supported: true, Registered: false}, fmt.Errorf("failed to read the startup shortcut %s : %w", w.pathConfig.startupShortcut, err)
	}
	return RegistrationState{Supported: true, Registered: true}, nil
}

// Register implements [Registrar].
func (w *windowsRegistrar) Register() error {
	// The installer lays down a shortcut for us so we can copy it without
	// having to resort to calling COM APIs to establish the shortcut
	_, err := os.Stat(w.pathConfig.startupShortcut)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			in, err := os.Open(w.pathConfig.shortcutOrigin)
			if err != nil {
				return fmt.Errorf("unable to open shortcut %s : %w", w.pathConfig.shortcutOrigin, err)
			}
			defer in.Close()
			out, err := os.Create(w.pathConfig.startupShortcut)
			if err != nil {
				return fmt.Errorf("unable to open startup link %s : %w", w.pathConfig.startupShortcut, err)
			}
			defer out.Close()
			_, err = io.Copy(out, in)
			if err != nil {
				return fmt.Errorf("unable to copy shortcut %s : %w", w.pathConfig.startupShortcut, err)
			}
			err = out.Sync()
			if err != nil {
				return fmt.Errorf("unable to sync shortcut %s : %w", w.pathConfig.startupShortcut, err)
			}
			slog.Info("Created Startup shortcut", "shortcut", w.pathConfig.startupShortcut)
		} else {
			slog.Warn("unexpected error looking up Startup shortcut", "error", err)
		}
	} else {
		slog.Debug("Startup link already exists", "shortcut", w.pathConfig.startupShortcut)
	}
	return nil
}

// Deregister implements [Registrar].
func (w *windowsRegistrar) Deregister() error {
	state, err := w.GetState()
	if err != nil {
		return fmt.Errorf("failed to determine if startup app is registered: %w", err)
	}
	if !state.Registered {
		slog.Debug("Attempted to disable autostart but it was already disabled")
		return nil
	}

	err = os.Remove(w.pathConfig.startupShortcut)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("unable to remove startup shortcut %s : %w", w.pathConfig.startupShortcut, err)
	}
	return nil
}
