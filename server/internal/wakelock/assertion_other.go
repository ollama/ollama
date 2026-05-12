//go:build !darwin && !linux && !windows

package wakelock

func init() {
	newAssertion = func(reason string) (assertion, error) {
		return noopAssertion{}, nil
	}
}

type noopAssertion struct{}

func (noopAssertion) release() {}
