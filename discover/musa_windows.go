package discover

import (
	"errors"
)

// Only called once during bootstrap
func MUSAGetGPUInfo() ([]MusaGPUInfo, error) {
	return []MusaGPUInfo{}, errors.New("unsupported platform")
}

func (gpus MusaGPUInfoList) RefreshFreeMemory() error {
	return errors.New("unsupported platform")
}
