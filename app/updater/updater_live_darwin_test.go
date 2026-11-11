//go:build darwin && updater_live

package updater

func liveExpectedFilename() string {
	return "Ollama-darwin.zip"
}

func liveStagedUpdate() string {
	return getStagedUpdate()
}

func wrapLiveUpdateVerification(called *bool) func() {
	oldVerifyDownload := VerifyDownload
	VerifyDownload = func() error {
		*called = true
		return verifyDownload()
	}
	return func() {
		VerifyDownload = oldVerifyDownload
	}
}
