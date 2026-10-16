//go:build (windows || darwin) && !updater_localtest

package updater

func updateCheckURLBase() string {
	return configuredUpdateCheckURLBase()
}
