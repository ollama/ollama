//go:build !darwin

package launch

func restoreCodexAppProfileFromDesktop() error {
	return restoreCodexAppProfile()
}
