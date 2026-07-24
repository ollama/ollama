package server

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestSchedLlamaServerConfigCarriesMemoryAssessment(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")

	s := InitScheduler(ctx)
	scenario := newScenarioRequestWithContext(t, ctx, "vision-regression-model", 1*format.GigaByte, nil, nil, 131072)
	scenario.req.opts.NumCtx = 32768

	systemInfo := ml.SystemInfo{
		TotalMemory: 64 * format.GibiByte,
		FreeMemory:  64 * format.GibiByte,
	}
	gpus := []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
		TotalMemory: 24 * format.GibiByte,
		FreeMemory:  20 * format.GibiByte,
	}}

	requestedCtx := scenario.req.opts.NumCtx
	var gotConfig llm.LlamaServerConfig
	var gotOpts api.Options
	var gotGpus []ml.DeviceInfo
	var gotModel *ggml.GGML
	s.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int, config llm.LlamaServerConfig) (llm.LlamaServer, error) {
		gotConfig = config
		gotOpts = opts
		gotGpus = append([]ml.DeviceInfo(nil), gpus...)
		gotModel = f
		scenario.srv.modelPath = model
		return scenario.srv, nil
	}

	require.False(t, s.load(scenario.req, systemInfo, gpus, false))
	select {
	case err := <-scenario.req.errCh:
		require.NoError(t, err)
	case <-scenario.req.successCh:
	case <-ctx.Done():
		t.Fatal("timed out waiting for load")
	}

	require.NotNil(t, gotModel)
	require.NotEmpty(t, gotGpus)
	predictedCtx := effectiveLlamaServerContext(requestedCtx, gotModel, 1)
	predictedModel := llm.PredictServerVRAM(scenario.req.model.ModelPath, gotModel, predictedCtx)
	available, _, _ := availableMemoryForPlacement(systemInfo, gotGpus, gotOpts)
	require.Equal(t, predictedModel+generationBatchSurchargeForCompletion(true, gotOpts.NumBatch), gotConfig.PredictedVRAM)
	require.Equal(t, available, gotConfig.AvailableVRAM)
	require.NotZero(t, gotConfig.PredictedVRAM)
	require.NotZero(t, gotConfig.AvailableVRAM)
}

func TestLlamaServerLoadPlanCarriesFitTargetFromGPUOverhead(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer done()
	t.Setenv("OLLAMA_GPU_OVERHEAD", "2147483648")
	t.Setenv(llamaServerFitTargetEnv, "")

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
	require.Equal(t, uint64(2048), plan.config.FitTargetMiB)
}

func TestAvailableMemoryForLoadUsesWorstSharedMemoryMeasurement(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "")
	t.Setenv(llamaServerFitTargetEnv, "")

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

func llamaServerReserveForTest(library string) uint64 {
	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: library}}
	return gpu.MinimumMemory() + llamaServerDefaultFitTargetMiB*format.MebiByte
}

func TestSelectLlamaServerPlacement(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "")
	t.Setenv(llamaServerFitTargetEnv, "")

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
			wantMainGPU:      testIntPtr(0),
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
				Runner: api.Runner{MainGPU: testIntPtr(1), NumGPU: -1},
			},
			wantLibrary:      "ROCm",
			wantMainGPU:      testIntPtr(0),
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
			wantMainGPU:      testIntPtr(0),
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
			wantMainGPU:      testIntPtr(0),
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
			t.Setenv(llamaServerFitTargetEnv, fitTarget)

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
		t.Setenv(llamaServerFitTargetEnv, "1024,4096,4096")

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

func TestLlamaServerFitTargetForRunner(t *testing.T) {
	check := func(name, overhead, fitTarget string, want uint64) {
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_GPU_OVERHEAD", overhead)
			t.Setenv(llamaServerFitTargetEnv, fitTarget)

			require.Equal(t, want, llamaServerFitTargetForRunner())
		})
	}

	check("below default", "536870912", "", 0)
	check("above default", "2147483648", "", 2048)
	check("rounds up", "1073741825", "", 1025)
	check("explicit fit target wins", "2147483648", "512", 0)
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

func testIntPtr(v int) *int {
	return &v
}
