//go:build !windows && !darwin

// startup_default.go contains the default implementation of the startup package
// for platforms that do not support startup registration.
package startup

import (
	"log/slog"
	"runtime"
)

func NewRegistrar() Registrar {
	return &defaultRegistrar{}
}

type defaultRegistrar struct{}

// State implements [Registrar].
func (d *defaultRegistrar) GetState() (RegistrationState, error) {
	return RegistrationState{Supported: false, Registered: false}, nil
}

// Register implements [Registrar].
func (d *defaultRegistrar) Register() error {
	slog.Debug("Attempted to register app for startup on an unsupported OS", "os", runtime.GOOS)
	return nil
}

// Deregister implements [Registrar].
func (d *defaultRegistrar) Deregister() error {
	slog.Debug("Attempted to deregister app for startup on an unsupported OS", "os", runtime.GOOS)
	return nil
}
