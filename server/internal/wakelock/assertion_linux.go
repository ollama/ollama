//go:build linux

package wakelock

// Linux wake-lock is currently a stub. The intended implementation invokes
// `systemd-inhibit --what=idle --who=ollama --why=<reason> --mode=block sleep infinity`
// as a subprocess, mirroring the macOS caffeinate approach, and kills it on
// release. Equivalently the assertion can be obtained without a subprocess by
// calling org.freedesktop.login1.Manager.Inhibit on the system D-Bus and
// holding the returned file descriptor open for the duration of the
// assertion.
//
// On systems without systemd / logind (some embedded distributions) there is
// no portable way to inhibit suspend from a user process, so callers should
// treat acquisition failure as non-fatal.
//
// References:
//   https://www.freedesktop.org/wiki/Software/systemd/inhibit/
//   https://github.com/ollama/ollama/issues/4072

func init() {
	newAssertion = func(reason string) (assertion, error) {
		return noopAssertion{}, nil
	}
}

type noopAssertion struct{}

func (noopAssertion) release() {}
