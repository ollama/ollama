package gpu

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBasicGetGPUInfo(t *testing.T) {
	info := GetGPUInfo()
	assert.Greater(t, len(info), 0)
	assert.Contains(t, "cuda rocm cpu metal", info[0].Library)
	if info[0].Library != "cpu" {
		assert.Greater(t, info[0].TotalMemory, uint64(0))
		assert.Greater(t, info[0].FreeMemory, uint64(0))
	}
}

func TestCPUMemInfo(t *testing.T) {
	info, err := GetCPUMem()
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
