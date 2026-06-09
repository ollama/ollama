//go:build !linux && !windows

package discover

import "errors"

// PowerLimitManager is a stub for unsupported platforms
type PowerLimitManager struct{}

// NewPowerLimitManager returns an error on unsupported platforms
func NewPowerLimitManager() (*PowerLimitManager, error) {
	return nil, errors.New("GPU power limit management is not supported on this platform")
}

// Close is a no-op stub
func (pm *PowerLimitManager) Close() error {
	return nil
}

// GetDeviceCount returns an error on unsupported platforms
func (pm *PowerLimitManager) GetDeviceCount() (uint32, error) {
	return 0, errors.New("GPU power limit management is not supported on this platform")
}

// SetPowerLimit returns an error on unsupported platforms
func (pm *PowerLimitManager) SetPowerLimit(deviceIndex uint32, powerLimitMW uint32) error {
	return errors.New("GPU power limit management is not supported on this platform")
}

// GetPowerLimit returns an error on unsupported platforms
func (pm *PowerLimitManager) GetPowerLimit(deviceIndex uint32) (uint32, error) {
	return 0, errors.New("GPU power limit management is not supported on this platform")
}

// GetPowerLimitConstraints returns an error on unsupported platforms
func (pm *PowerLimitManager) GetPowerLimitConstraints(deviceIndex uint32) (minLimit, maxLimit uint32, err error) {
	return 0, 0, errors.New("GPU power limit management is not supported on this platform")
}

// SetPowerLimitForAll returns an error on unsupported platforms
func (pm *PowerLimitManager) SetPowerLimitForAll(powerLimitWatts int) error {
	return errors.New("GPU power limit management is not supported on this platform")
}

// GetPowerLimitsForAll returns an error on unsupported platforms
func (pm *PowerLimitManager) GetPowerLimitsForAll() ([]uint32, error) {
	return nil, errors.New("GPU power limit management is not supported on this platform")
}
