//go:build windows && updater_unsigned

package updater

import "log/slog"

func init() {
	verifyInstallScriptSignature = func(filename string) error {
		slog.Warn("install.ps1 signature verification disabled by updater_unsigned build tag", "script", filename)
		return nil
	}
}
