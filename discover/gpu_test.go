package discover

import (
	"runtime"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBasicGetGPUInfo(t *testing.T) {
	info := GetGPUInfo()
	assert.NotEmpty(t, len(info))
	assert.Contains(t, "cuda rocm cpu metal", info[0].Library)
	if info[0].Library != "cpu" {
		assert.Greater(t, info[0].TotalMemory, uint64(0))
		assert.Greater(t, info[0].FreeMemory, uint64(0))
	}
}

func TestCPUMemInfo(t *testing.T) {
	info, err := GetCPUMem()
	require.NoError(t, err)
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

func TestByLibrary(t *testing.T) {
	type testCase struct {
		input  []GpuInfo
		expect int
	}

	testCases := map[string]*testCase{
		"empty":                    {input: []GpuInfo{}, expect: 0},
		"cpu":                      {input: []GpuInfo{{Library: "cpu"}}, expect: 1},
		"cpu + GPU":                {input: []GpuInfo{{Library: "cpu"}, {Library: "cuda"}}, expect: 2},
		"cpu + 2 GPU no variant":   {input: []GpuInfo{{Library: "cpu"}, {Library: "cuda"}, {Library: "cuda"}}, expect: 2},
		"cpu + 2 GPU same variant": {input: []GpuInfo{{Library: "cpu"}, {Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v11"}}, expect: 2},
		"cpu + 2 GPU diff variant": {input: []GpuInfo{{Library: "cpu"}, {Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v12"}}, expect: 2},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			resp := (GpuInfoList)(v.input).ByLibrary()
			if len(resp) != v.expect {
				t.Fatalf("expected length %d, got %d => %+v", v.expect, len(resp), resp)
			}
		})
	}
}

func TestByVariant(t *testing.T) {
	type testCase struct {
		input  []GpuInfo
		expect []GpuInfo
	}

	testCases := map[string]*testCase{
		"empty":                {input: []GpuInfo{}, expect: []GpuInfo{}},
		"one item, no variant": {input: []GpuInfo{{Library: "cpu"}}, expect: []GpuInfo{{Library: "cpu"}}},
		"both v11":             {input: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v11"}}, expect: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v11"}}},
		"v11, v12":             {input: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v12"}}, expect: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v12"}}},
		"v12, v11":             {input: []GpuInfo{{Library: "cuda", Variant: "v12"}, {Library: "cuda", Variant: "v11"}}, expect: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v12"}}},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			resp := append(make([]GpuInfo, 0, len(v.input)), v.input...)
			sort.Sort(ByVariant(resp))
			if len(resp) != len(v.expect) {
				t.Fatalf("expected length %d, got %d => %+v", len(v.expect), len(resp), resp)
			}
			for i := range resp {
				if resp[i].Variant != v.expect[i].Variant || resp[i].Library != v.expect[i].Library {
					t.Fatalf("expected index %d, got %v wanted %+v", i, resp[i], v.expect[i])
				}
			}
		})
	}
}

func TestBestRunnerName(t *testing.T) {
	type testCase struct {
		input  GpuInfoList
		expect string
	}

	testCases := map[string]*testCase{
		"empty":                {input: []GpuInfo{}, expect: ""},
		"one item, no variant": {input: []GpuInfo{{Library: "cpu"}}, expect: "cpu"},
		"both v11":             {input: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v11"}}, expect: "cuda_v11"},
		"v11, v12":             {input: []GpuInfo{{Library: "cuda", Variant: "v11"}, {Library: "cuda", Variant: "v12"}}, expect: "cuda_v11"},
		"v12, v11":             {input: []GpuInfo{{Library: "cuda", Variant: "v12"}, {Library: "cuda", Variant: "v11"}}, expect: "cuda_v11"},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			resp := v.input.BestRunnerName()
			if resp != v.expect {
				t.Fatalf("got %v wanted %+v", resp, v.expect)
			}
		})
	}
}

// TODO - add some logic to figure out card type through other means and actually verify we got back what we expected
