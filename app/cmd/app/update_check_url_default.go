//go:build (windows || darwin) && !updater_localtest

package main

func maybeConfigureLocalUpdateCheckURL(args []string, index int) (bool, int) {
	return false, 0
}
