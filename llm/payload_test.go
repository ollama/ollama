package llm

import (
	"testing"

	"github.com/ollama/ollama/gpu"
	"github.com/stretchr/testify/assert"
)

func TestGetDynLibs(t *testing.T) {
	availableDynLibs = map[string]string{
		"cpu": "X_cpu",
	}
	assert.Equal(t, false, rocmDynLibPresent())
	res := getDynLibs(gpu.GpuInfo{Library: "cpu"})
	assert.Len(t, res, 1)
	assert.Equal(t, availableDynLibs["cpu"], res[0])

	variant := gpu.GetCPUVariant()
	if variant != "" {
		variant = "_" + variant
	}
	availableDynLibs = map[string]string{
		"rocm_v5":       "X_rocm_v5",
		"rocm_v6":       "X_rocm_v6",
		"cpu" + variant: "X_cpu",
	}
	assert.Equal(t, true, rocmDynLibPresent())
	res = getDynLibs(gpu.GpuInfo{Library: "rocm"})
	assert.Len(t, res, 3)
	assert.Equal(t, availableDynLibs["rocm_v5"], res[0])
	assert.Equal(t, availableDynLibs["rocm_v6"], res[1])
	assert.Equal(t, availableDynLibs["cpu"+variant], res[2])

	res = getDynLibs(gpu.GpuInfo{Library: "rocm", Variant: "v6"})
	assert.Len(t, res, 3)
	assert.Equal(t, availableDynLibs["rocm_v6"], res[0])
	assert.Equal(t, availableDynLibs["rocm_v5"], res[1])
	assert.Equal(t, availableDynLibs["cpu"+variant], res[2])

	res = getDynLibs(gpu.GpuInfo{Library: "cuda"})
	assert.Len(t, res, 1)
	assert.Equal(t, availableDynLibs["cpu"+variant], res[0])

	res = getDynLibs(gpu.GpuInfo{Library: "default"})
	assert.Len(t, res, 1)
	assert.Equal(t, "default", res[0])

	availableDynLibs = map[string]string{
		"rocm":          "X_rocm_v5",
		"cpu" + variant: "X_cpu",
	}
	assert.Equal(t, true, rocmDynLibPresent())
	res = getDynLibs(gpu.GpuInfo{Library: "rocm", Variant: "v6"})
	assert.Len(t, res, 2)
	assert.Equal(t, availableDynLibs["rocm"], res[0])
	assert.Equal(t, availableDynLibs["cpu"+variant], res[1])
}
