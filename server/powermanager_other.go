//go:build !darwin && !windows && !linux

package server

// platformAcquirePowerAssertion is a no-op on platforms without a sleep
// inhibition implementation. Inference proceeds normally (issue #4072).
var platformAcquirePowerAssertion = func() (func(), bool) {
	return nil, false
}
