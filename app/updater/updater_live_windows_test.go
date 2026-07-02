//go:build windows && updater_live

package updater

func liveExpectedFilename() string {
	return installScriptName
}

func liveStagedUpdate() string {
	return getStagedInstallScript()
}

func wrapLiveUpdateVerification(called *bool) func() {
	oldVerifyInstallScriptSignature := verifyInstallScriptSignature
	verifyInstallScriptSignature = func(filename string) error {
		*called = true
		return oldVerifyInstallScriptSignature(filename)
	}
	return func() {
		verifyInstallScriptSignature = oldVerifyInstallScriptSignature
	}
}
