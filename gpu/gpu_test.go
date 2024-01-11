package gpu

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBasicGetGPUInfo(t *testing.T) {
	info := GetGPUInfo()
	assert.Contains(t, "cuda rocm cpu default", info.Library)

	switch runtime.GOOS {
	case "darwin":
		// TODO - remove this once MacOS returns some size for CPU
		return
	case "linux", "windows":
		assert.Greater(t, info.TotalMemory, uint64(0))
		assert.Greater(t, info.FreeMemory, uint64(0))
		assert.Greater(t, info.DeviceCount, uint32(0))
	default:
		return
	}
}

func TestCPUMemInfo(t *testing.T) {
	info, err := getCPUMem()
	assert.NoError(t, err)
	switch runtime.GOOS {
	case "darwin":
		t.Skip("CPU memory not populated on darwin")
	case "linux", "windows":
		assert.Greater(t, info.TotalMemory, uint64(0))
		assert.Greater(t, info.FreeMemory, uint64(0))
	default:
		return
	}
}

// TODO - add some logic to figure out card type through other means and actually verify we got back what we expected
