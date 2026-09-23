// Package onboarding stores the welcome completion shared by the app and CLI.
// It is local to the OS user and independent of account authentication.
package onboarding

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const CurrentVersion = 1

// State uses ~/.ollama by default. Dir isolates custom app stores and tests.
type State struct {
	Dir string
}

func (s State) path() (string, error) {
	dir := s.Dir
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, ".ollama")
	}
	return filepath.Join(dir, fmt.Sprintf("onboarding-v%d.completed", CurrentVersion)), nil
}

func (s State) Completed() (bool, error) {
	path, err := s.path()
	if err != nil {
		return false, err
	}
	info, err := os.Stat(path)
	if err == nil {
		if !info.Mode().IsRegular() {
			return false, fmt.Errorf("onboarding completion is not a regular file: %s", path)
		}
		return true, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return false, err
	}

	return false, nil
}

// Complete creates an immutable marker. Concurrent app and CLI completion
// cannot overwrite each other's model settings or undo completed onboarding.
func (s State) Complete() error {
	path, err := s.path()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("onboarding completion is not a regular file: %s", path)
		}
		return nil
	}
	if err != nil {
		return err
	}
	return file.Close()
}
