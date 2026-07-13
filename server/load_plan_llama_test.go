package server

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestLlamaServerLoadPlanMemoryAssessment(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	scenario := newScenarioRequestWithContext(t, ctx, "memory-assessment-model", 1*format.GigaByte, nil, nil, 131072)
	scenario.req.opts.NumCtx = 32768

	proposal := loadProposal{
		systemInfo: ml.SystemInfo{
			TotalMemory: 64 * format.GibiByte,
			FreeMemory:  64 * format.GibiByte,
		},
		gpus: []ml.DeviceInfo{{
			DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
			TotalMemory: 24 * format.GibiByte,
			FreeMemory:  20 * format.GibiByte,
		}},
		numParallel: 1,
		completion:  true,
	}
	plan, err := newLlamaServerLoadPlan(scenario.req, proposal)
	require.NoError(t, err)

	predictedCtx := effectiveLlamaServerContext(scenario.req.opts.NumCtx, plan.model, proposal.numParallel)
	predictedModel := llm.PredictServerVRAM(scenario.req.model.ModelPath, plan.model, predictedCtx)
	available, _, _ := availableMemoryForPlacement(proposal.systemInfo, plan.gpus, plan.launchOpts)
	require.Equal(t, predictedModel, plan.memory.predictedModel)
	require.Equal(t, predictedModel+generationBatchSurchargeForCompletion(true, plan.launchOpts.NumBatch), plan.memory.predictedLoad)
	require.Equal(t, available, plan.memory.available)
	require.NotZero(t, plan.memory.predictedLoad)
	require.NotZero(t, plan.memory.available)
}

func TestLlamaServerLoadPlanCarriesFitTargetFromGPUOverhead(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	t.Setenv("OLLAMA_GPU_OVERHEAD", "2147483648")
	t.Setenv(llm.LlamaServerFitTargetEnv, "")

	scenario := newScenarioRequestWithContext(t, ctx, "fit-target-model", 1*format.GigaByte, nil, nil, 32768)
	plan, err := newLlamaServerLoadPlan(scenario.req, loadProposal{
		systemInfo: ml.SystemInfo{FreeMemory: 64 * format.GibiByte},
		gpus: []ml.DeviceInfo{{
			DeviceID:   ml.DeviceID{ID: "0", Library: "CUDA"},
			FreeMemory: 24 * format.GibiByte,
		}},
		numParallel: 1,
		completion:  true,
	})
	require.NoError(t, err)
	require.Equal(t, []uint64{2048}, plan.config.FitTargetsMiB)
}

func TestAvailableMemoryForLoadUsesWorstSharedMemoryMeasurement(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "")
	t.Setenv(llm.LlamaServerFitTargetEnv, "")

	metalReserve := llamaServerReserveForTest("Metal")
	vulkanReserve := llamaServerReserveForTest("Vulkan")
	cudaReserve := llamaServerReserveForTest("CUDA")

	tests := []struct {
		name              string
		systemFree        uint64
		gpus              []ml.DeviceInfo
		wantAvailable     uint64
		wantGPUFree       uint64
		wantSystemLimited bool
	}{
		{
			name:       "integrated metal uses lower system free",
			systemFree: 80 * format.GigaByte,
			gpus: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{Library: "Metal"},
				Integrated: true,
				FreeMemory: 300 * format.GigaByte,
			}},
			wantAvailable:     80*format.GigaByte - metalReserve,
			wantGPUFree:       300 * format.GigaByte,
			wantSystemLimited: true,
		},
		{
			name:       "integrated gpu uses lower system free",
			systemFree: 6 * format.GigaByte,
			gpus: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{Library: "Vulkan"},
				Integrated: true,
				FreeMemory: 12 * format.GigaByte,
			}},
			wantAvailable:     6*format.GigaByte - vulkanReserve,
			wantGPUFree:       12 * format.GigaByte,
			wantSystemLimited: true,
		},
		{
			name:       "discrete metal ignores lower system free",
			systemFree: 6 * format.GigaByte,
			gpus: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{Library: "Metal"},
				FreeMemory: 12 * format.GigaByte,
			}},
			wantAvailable: 12*format.GigaByte - metalReserve,
			wantGPUFree:   12 * format.GigaByte,
		},
		{
			name:       "discrete gpu ignores lower system free",
			systemFree: 6 * format.GigaByte,
			gpus: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{Library: "CUDA"},
				FreeMemory: 12 * format.GigaByte,
			}},
			wantAvailable: 12*format.GigaByte - cudaReserve,
			wantGPUFree:   12 * format.GigaByte,
		},
		{
			name:       "mixed gpus only clamp integrated contribution",
			systemFree: 6 * format.GigaByte,
			gpus: []ml.DeviceInfo{
				{
					DeviceID:   ml.DeviceID{Library: "CUDA"},
					FreeMemory: 12 * format.GigaByte,
				},
				{
					DeviceID:   ml.DeviceID{Library: "Vulkan"},
					Integrated: true,
					FreeMemory: 10 * format.GigaByte,
				},
			},
			wantAvailable:     18*format.GigaByte - cudaReserve - vulkanReserve,
			wantGPUFree:       22 * format.GigaByte,
			wantSystemLimited: true,
		},
		{
			name:       "shared gpu keeps lower adjusted gpu baseline",
			systemFree: 20 * format.GigaByte,
			gpus: []ml.DeviceInfo{{
				DeviceID:   ml.DeviceID{Library: "Metal"},
				Integrated: true,
				FreeMemory: 12 * format.GigaByte,
			}},
			wantAvailable: 12*format.GigaByte - metalReserve,
			wantGPUFree:   12 * format.GigaByte,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			available, gpuFree, systemLimited := availableMemoryForLoad(ml.SystemInfo{FreeMemory: tt.systemFree}, tt.gpus)
			require.Equal(t, tt.wantAvailable, available)
			require.Equal(t, tt.wantGPUFree, gpuFree)
			require.Equal(t, tt.wantSystemLimited, systemLimited)
		})
	}
}

// llamaServerReserveForTest computes the production device reserve; callers
// must clear OLLAMA_GPU_OVERHEAD and LLAMA_ARG_FIT_TARGET first.
func llamaServerReserveForTest(library string) uint64 {
	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: library}}
	return llamaServerDeviceReserve(gpu, 0)
}

func TestSelectLlamaServerPlacement(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "")
	t.Setenv(llm.LlamaServerFitTargetEnv, "")

	systemInfo := ml.SystemInfo{FreeMemory: 14 * format.GigaByte}

	tests := []struct {
		name             string
		gpus             []ml.DeviceInfo
		predictedVRAM    uint64
		opts             api.Options
		schedSpread      string
		wantLibrary      string
		wantMainGPU      *int
		wantSelectedGPUs int
		wantGPUID        string
	}{
		{
			name:          "selects largest same-backend GPU",
			predictedVRAM: 8 * format.GigaByte,
			gpus: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "small", FreeMemory: 10 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, Name: "large", FreeMemory: 20 * format.GigaByte},
			},
			opts:             api.DefaultOptions(),
			wantLibrary:      "CUDA",
			wantMainGPU:      new(0),
			wantSelectedGPUs: 1,
			wantGPUID:        "1",
		},
		{
			name:          "explicit main gpu selects matching backend group",
			predictedVRAM: 8 * format.GigaByte,
			gpus: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 10 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, FreeMemory: 20 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, FreeMemory: 24 * format.GigaByte},
			},
			opts: api.Options{
				Runner: api.Runner{MainGPU: new(1), NumGPU: -1},
			},
			wantLibrary:      "ROCm",
			wantMainGPU:      new(0),
			wantSelectedGPUs: 1,
			wantGPUID:        "1",
		},
		{
			name:          "integrated GPU is capped by system free memory",
			predictedVRAM: 12 * format.GigaByte,
			gpus: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}, Integrated: true, FreeMemory: 32 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "1", Library: "Metal"}, FreeMemory: 16 * format.GigaByte},
			},
			opts:             api.DefaultOptions(),
			wantLibrary:      "Metal",
			wantMainGPU:      new(0),
			wantSelectedGPUs: 1,
			wantGPUID:        "1",
		},
		{
			name:          "prefers discrete GPU over integrated GPU with more available memory",
			predictedVRAM: 8 * format.GigaByte,
			gpus: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}, Name: "integrated", Integrated: true, FreeMemory: 32 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "1", Library: "Vulkan"}, Name: "discrete", FreeMemory: 10 * format.GigaByte},
			},
			opts:             api.DefaultOptions(),
			wantLibrary:      "Vulkan",
			wantMainGPU:      new(0),
			wantSelectedGPUs: 1,
			wantGPUID:        "1",
		},
		{
			name:          "spread disables automatic single GPU selection",
			predictedVRAM: 8 * format.GigaByte,
			schedSpread:   "1",
			gpus: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 10 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, FreeMemory: 20 * format.GigaByte},
			},
			opts:             api.DefaultOptions(),
			wantLibrary:      "CUDA",
			wantSelectedGPUs: 2,
		},
		{
			name:          "no single fit chooses best backend group for llama-server split",
			predictedVRAM: 30 * format.GigaByte,
			gpus: []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 10 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, FreeMemory: 18 * format.GigaByte},
				{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, FreeMemory: 12 * format.GigaByte},
			},
			opts:             api.DefaultOptions(),
			wantLibrary:      "CUDA",
			wantSelectedGPUs: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_SCHED_SPREAD", tt.schedSpread)

			selected, launchOpts := selectLlamaServerPlacement(systemInfo, tt.gpus, tt.predictedVRAM, tt.opts)
			require.Len(t, selected, tt.wantSelectedGPUs)
			require.Equal(t, tt.wantLibrary, selected[0].Library)
			if tt.wantGPUID != "" {
				require.Equal(t, tt.wantGPUID, selected[0].ID)
			}
			if tt.wantMainGPU == nil {
				require.Nil(t, launchOpts.MainGPU)
			} else {
				require.NotNil(t, launchOpts.MainGPU)
				require.Equal(t, *tt.wantMainGPU, *launchOpts.MainGPU)
			}
		})
	}
}

func TestSelectLlamaServerPlacementReserveEnv(t *testing.T) {
	check := func(name string, gpus []ml.DeviceInfo, overhead, fitTarget string, want int) {
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_GPU_OVERHEAD", overhead)
			t.Setenv(llm.LlamaServerFitTargetEnv, fitTarget)

			selected, _ := selectLlamaServerPlacement(ml.SystemInfo{}, gpus, 20307*format.MebiByte, api.DefaultOptions())
			require.Len(t, selected, want)
		})
	}

	issue16599GPUs := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 23336 * format.MebiByte},
		{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, FreeMemory: 7106 * format.MebiByte},
	}
	check("issue 16599 selects single GPU with default reserve", issue16599GPUs, "", "", 1)
	check("gpu overhead prevents single GPU selection", issue16599GPUs, "3221225472", "", 2)
	check("fit target prevents single GPU selection", issue16599GPUs, "", "4096", 2)

	reversedGPUs := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 7106 * format.MebiByte},
		{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, FreeMemory: 23336 * format.MebiByte},
	}
	check("comma fit target uses selected visible gpu index", reversedGPUs, "", "1024,4096", 1)

	t.Run("comma fit target uses selected backend visible order", func(t *testing.T) {
		t.Setenv("OLLAMA_GPU_OVERHEAD", "")
		t.Setenv(llm.LlamaServerFitTargetEnv, "1024,4096,4096")

		mixedVendorGPUs := []ml.DeviceInfo{
			{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 7106 * format.MebiByte},
			{DeviceID: ml.DeviceID{ID: "0", Library: "ROCm"}, FreeMemory: 7106 * format.MebiByte},
			{DeviceID: ml.DeviceID{ID: "1", Library: "ROCm"}, FreeMemory: 23336 * format.MebiByte},
		}
		selected, launchOpts := selectLlamaServerPlacement(ml.SystemInfo{}, mixedVendorGPUs, 20307*format.MebiByte, api.DefaultOptions())

		require.Len(t, selected, 1)
		require.Equal(t, "ROCm", selected[0].Library)
		require.Equal(t, "1", selected[0].ID)
		require.NotNil(t, launchOpts.MainGPU)
		require.Equal(t, 0, *launchOpts.MainGPU)
	})
}

func TestLlamaServerFitTargetsForRunner(t *testing.T) {
	cudaGPU := func(id string, fitTarget ...string) ml.DeviceInfo {
		gpu := ml.DeviceInfo{
			DeviceID: ml.DeviceID{ID: id, Library: "CUDA"},
		}
		if len(fitTarget) > 0 {
			gpu.RunnerEnvOverrides = map[string]string{
				llm.LlamaServerFitTargetEnv: fitTarget[0],
			}
		}
		return gpu
	}

	rocmGPU := func(id string, fitTarget string) ml.DeviceInfo {
		return ml.DeviceInfo{
			DeviceID: ml.DeviceID{ID: id, Library: "ROCm"},
			RunnerEnvOverrides: map[string]string{
				llm.LlamaServerFitTargetEnv: fitTarget,
			},
		}
	}

	check := func(name, overhead, fitTarget string, gpus []ml.DeviceInfo, want []uint64) {
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_GPU_OVERHEAD", overhead)
			t.Setenv(llm.LlamaServerFitTargetEnv, fitTarget)

			require.Equal(t, want, llamaServerFitTargetsForRunner(gpus))
		})
	}

	oneGPU := []ml.DeviceInfo{cudaGPU("0")}
	check("overhead below default keeps llama.cpp default", "536870912", "", oneGPU, nil)
	check("overhead above default becomes the margin", "2147483648", "", oneGPU, []uint64{2048})
	check("overhead rounds up", "1073741825", "", oneGPU, []uint64{1025})
	check("explicit fit target wins", "2147483648", "512", oneGPU, nil)
	check("no devices", "2147483648", "", nil, nil)
	check("per-device override is preserved", "", "",
		[]ml.DeviceInfo{rocmGPU("0", "61440")}, []uint64{61440})
	check("per-device override composes with overhead by visible order", "2147483648", "",
		[]ml.DeviceInfo{cudaGPU("0"), rocmGPU("1", "61440")}, []uint64{2048, 61440})
	check("larger overhead wins over smaller per-device override", "2147483648", "",
		[]ml.DeviceInfo{rocmGPU("0", "512")}, []uint64{2048})
	check("small per-device override is normalized to llama.cpp default", "", "",
		[]ml.DeviceInfo{rocmGPU("0", "512")}, []uint64{1024})
	check("explicit fit target wins over per-device override", "2147483648", "512",
		[]ml.DeviceInfo{rocmGPU("0", "61440")}, nil)
	check("invalid per-device override is normalized to llama.cpp default", "", "",
		[]ml.DeviceInfo{rocmGPU("0", "not-a-number")}, []uint64{1024})
	check("overhead applies per device", "2147483648", "",
		[]ml.DeviceInfo{cudaGPU("0"), cudaGPU("1")}, []uint64{2048, 2048})
	check("all-default devices stay unset", "", "",
		[]ml.DeviceInfo{cudaGPU("0"), cudaGPU("1")}, nil)
}

func TestLlamaServerLoadPlanEvictionHeadroom(t *testing.T) {
	tests := []struct {
		name        string
		plan        llamaServerLoadPlan
		requireFull bool
		loadedCount int
		want        loadedRunnerFit
	}{
		{
			name: "does not evict when no runners are loaded",
			plan: llamaServerLoadPlan{
				gpus:   []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}},
				memory: loadMemoryAssessment{predictedLoad: 17 * format.GigaByte, available: 20 * format.GigaByte},
			},
			requireFull: true,
		},
		{
			name: "does not evict when partial offload is allowed",
			plan: llamaServerLoadPlan{
				gpus:   []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}},
				memory: loadMemoryAssessment{predictedLoad: 17 * format.GigaByte, available: 20 * format.GigaByte},
			},
			loadedCount: 1,
		},
		{
			name: "does not evict at headroom boundary",
			plan: llamaServerLoadPlan{
				gpus:   []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}},
				memory: loadMemoryAssessment{predictedLoad: 16 * format.GigaByte, available: 20 * format.GigaByte},
			},
			requireFull: true,
			loadedCount: 1,
			want:        loadedRunnerFits,
		},
		{
			name: "evicts above headroom boundary",
			plan: llamaServerLoadPlan{
				gpus:   []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}},
				memory: loadMemoryAssessment{predictedLoad: 17 * format.GigaByte, available: 20 * format.GigaByte},
			},
			requireFull: true,
			loadedCount: 1,
			want:        loadedRunnerNeedsEviction,
		},
		{
			name: "evicts when available memory is unknown",
			plan: llamaServerLoadPlan{
				gpus:   []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}},
				memory: loadMemoryAssessment{predictedLoad: 1 * format.GigaByte},
			},
			requireFull: true,
			loadedCount: 1,
			want:        loadedRunnerNeedsEviction,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, tt.plan.assessLoadedRunnerFit(tt.requireFull, tt.loadedCount))
		})
	}
}

func TestLlamaServerFitTargetBytes(t *testing.T) {
	check := func(name, value string, visibleIndex int, wantMiB uint64) {
		t.Run(name, func(t *testing.T) {
			t.Setenv(llm.LlamaServerFitTargetEnv, value)
			require.Equal(t, wantMiB*format.MebiByte, llamaServerFitTargetBytes(visibleIndex))
		})
	}

	check("unset uses llama.cpp default", "", 0, llm.LlamaServerDefaultFitTargetMiB)
	check("single value broadcasts to all devices", "512", 3, 512)
	check("comma separated selects device", "512,2048", 1, 2048)
	check("slash separated selects device", "512/2048", 1, 2048)
	check("device beyond list keeps default", "512,2048", 2, llm.LlamaServerDefaultFitTargetMiB)
	check("invalid value keeps default", "not-a-number", 0, llm.LlamaServerDefaultFitTargetMiB)
}

func TestLlamaServerDisableMmapReason(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()

	scenario := newScenarioRequestWithContext(t, ctx, "mmap-default-model", 1*format.GigaByte, nil, nil, 8192)
	f, err := llm.LoadModel(scenario.req.model.ModelPath, 1024)
	require.NoError(t, err)
	fullOffloadLayers := int(f.KV().BlockCount()) + 1

	useMmap := true
	cudaGPU := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, TotalMemory: 100 * format.GigaByte, FreeMemory: 80 * format.GigaByte}}
	metalGPU := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}}}
	integratedGPU := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, Integrated: true, TotalMemory: 100 * format.GigaByte, FreeMemory: 80 * format.GigaByte}}
	pressureSystem := ml.SystemInfo{TotalMemory: 100 * format.GigaByte, FreeMemory: 50 * format.GigaByte}
	pressureMemory := loadMemoryAssessment{predictedModel: 30 * format.GigaByte, available: 80 * format.GigaByte}

	tests := []struct {
		name           string
		goos           string
		opts           api.Options
		gpus           []ml.DeviceInfo
		memory         loadMemoryAssessment
		systemInfo     ml.SystemInfo
		modelSize      uint64
		loadedMmapSize uint64
		want           string
	}{
		{
			name: "explicit use_mmap true wins",
			goos: "windows",
			opts: api.Options{Runner: api.Runner{NumGPU: -1, UseMMap: &useMmap}},
			gpus: cudaGPU,
		},
		{
			name: "cpu-only request disables mmap",
			goos: "linux",
			opts: api.Options{Runner: api.Runner{NumGPU: 0}},
			gpus: cudaGPU,
			want: "cpu",
		},
		{
			name: "no GPU devices disables mmap",
			goos: "linux",
			opts: api.Options{Runner: api.Runner{NumGPU: -1}},
			want: "cpu",
		},
		{
			name: "windows cuda disables mmap",
			goos: "windows",
			opts: api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus: cudaGPU,
			want: "windows_cuda",
		},
		{
			name: "metal partial offload disables mmap",
			goos: "darwin",
			opts: api.Options{Runner: api.Runner{NumGPU: fullOffloadLayers - 1}},
			gpus: metalGPU,
			want: "metal_partial_offload",
		},
		{
			name: "metal full offload keeps default",
			goos: "darwin",
			opts: api.Options{Runner: api.Runner{NumGPU: fullOffloadLayers}},
			gpus: metalGPU,
		},
		{
			name:   "metal auto partial offload disables mmap",
			goos:   "darwin",
			opts:   api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus:   metalGPU,
			memory: loadMemoryAssessment{predictedModel: 30 * format.GigaByte, available: 20 * format.GigaByte},
			want:   "metal_partial_offload",
		},
		{
			name:   "metal auto full offload keeps default",
			goos:   "darwin",
			opts:   api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus:   metalGPU,
			memory: loadMemoryAssessment{predictedModel: 10 * format.GigaByte, available: 20 * format.GigaByte},
		},
		{
			name: "linux cuda keeps default",
			goos: "linux",
			opts: api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus: cudaGPU,
		},
		{
			name:           "linux host memory pressure disables mmap",
			goos:           "linux",
			opts:           api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus:           cudaGPU,
			memory:         pressureMemory,
			systemInfo:     pressureSystem,
			modelSize:      20 * format.GigaByte,
			loadedMmapSize: 25 * format.GigaByte,
			want:           "host_memory_pressure",
		},
		{
			name:           "only the Linux pressure heuristic applies",
			goos:           "darwin",
			opts:           api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus:           cudaGPU,
			memory:         pressureMemory,
			systemInfo:     pressureSystem,
			modelSize:      20 * format.GigaByte,
			loadedMmapSize: 25 * format.GigaByte,
		},
		{
			name:           "shared-memory GPU keeps the normal mmap path",
			goos:           "linux",
			opts:           api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus:           integratedGPU,
			memory:         pressureMemory,
			systemInfo:     pressureSystem,
			modelSize:      20 * format.GigaByte,
			loadedMmapSize: 25 * format.GigaByte,
		},
		{
			name:           "tight VRAM keeps mmap so partial offload stays file-backed",
			goos:           "linux",
			opts:           api.Options{Runner: api.Runner{NumGPU: -1}},
			gpus:           cudaGPU,
			memory:         loadMemoryAssessment{predictedModel: 70 * format.GigaByte, available: 80 * format.GigaByte},
			systemInfo:     pressureSystem,
			modelSize:      20 * format.GigaByte,
			loadedMmapSize: 25 * format.GigaByte,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan := llamaServerLoadPlan{
				model:       f,
				gpus:        tt.gpus,
				requestOpts: tt.opts,
				launchOpts:  tt.opts,
				memory:      tt.memory,
			}
			require.Equal(t, tt.want, llamaServerDisableMmapReason(tt.goos, plan, tt.systemInfo, tt.modelSize, tt.loadedMmapSize))
		})
	}
}
