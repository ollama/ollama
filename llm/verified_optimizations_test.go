package llm

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
)

// TestCheckpointMemoryOptimization verifies the checkpoint memory optimization
// Corresponds to formal proof in proof/VERIFIED_final.v: checkpoint_saves_memory
func TestCheckpointMemoryOptimization(t *testing.T) {
	optimizer := NewVerifiedMemoryOptimizer(true, false)

	testCases := []struct {
		layers   int
		expected uint64
		name     string
	}{
		{4, 3, "minimal_layers_4"},    // sqrt(4) + 1 = 3
		{16, 5, "medium_layers_16"},   // sqrt(16) + 1 = 5
		{64, 9, "large_layers_64"},    // sqrt(64) + 1 = 9
		{100, 11, "very_large_100"},   // sqrt(100) + 1 = 11
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := optimizer.CheckpointMemoryEstimate(tc.layers)
			if result != tc.expected {
				t.Errorf("Checkpoint memory estimate failed: layers=%d, expected=%d, got=%d",
					tc.layers, tc.expected, result)
			}

			// Verify formal proof property: checkpoint < standard (for layers >= 4)
			if tc.layers >= 4 {
				standard := uint64(tc.layers)
				if result >= standard {
					t.Errorf("Checkpoint optimization failed to reduce memory: checkpoint=%d >= standard=%d",
						result, standard)
				}
			}
		})
	}
}

// TestMLACompression verifies the MLA compression optimization
// Corresponds to formal proof in proof/VERIFIED_final.v: mla_saves_memory
func TestMLACompression(t *testing.T) {
	optimizer := NewVerifiedMemoryOptimizer(false, true)

	testCases := []struct {
		kvSize   uint64
		expected uint64
		name     string
	}{
		{28, 1, "minimal_kv_28"},     // 28/28 = 1
		{280, 10, "medium_kv_280"},   // 280/28 = 10
		{1500, 53, "large_kv_1500"}, // 1500/28 = 53
		{2800, 100, "huge_kv_2800"}, // 2800/28 = 100
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := optimizer.MLACompressionEstimate(tc.kvSize)
			if result != tc.expected {
				t.Errorf("MLA compression failed: kvSize=%d, expected=%d, got=%d",
					tc.kvSize, tc.expected, result)
			}

			// Verify formal proof property: compressed < original (for kvSize >= 28)
			if tc.kvSize >= 28 {
				if result >= tc.kvSize {
					t.Errorf("MLA compression failed to reduce size: compressed=%d >= original=%d",
						result, tc.kvSize)
				}
			}
		})
	}
}

// TestGPUDeviceScore verifies the device scoring algorithm
// Corresponds to formal proof in proof/verified_gpu_backend.v: device_score
func TestGPUDeviceScore(t *testing.T) {
	testCases := []struct {
		name string
		spec *GPUDeviceSpec
		expectedScore int
	}{
		{
			name: "intel_arc_b580",
			spec: &GPUDeviceSpec{
				MemorySizeGB:        12,
				PeakTFLOPSFP32:      17,
				SupportsTensorCores: true,
			},
			expectedScore: 12*10 + 17 + 50, // 187
		},
		{
			name: "rtx_3070",
			spec: &GPUDeviceSpec{
				MemorySizeGB:        8,
				PeakTFLOPSFP32:      20,
				SupportsTensorCores: true,
			},
			expectedScore: 8*10 + 20 + 50, // 150
		},
		{
			name: "no_tensor_cores",
			spec: &GPUDeviceSpec{
				MemorySizeGB:        16,
				PeakTFLOPSFP32:      15,
				SupportsTensorCores: false,
			},
			expectedScore: 16*10 + 15, // 175
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			score := tc.spec.DeviceScore()
			if score != tc.expectedScore {
				t.Errorf("Device score mismatch: expected=%d, got=%d", tc.expectedScore, score)
			}
		})
	}
}

// TestDeviceSelection verifies multi-vendor GPU selection
// Corresponds to formal proof in proof/verified_gpu_backend.v: device_selection_sound
func TestDeviceSelection(t *testing.T) {
	selector := NewVerifiedDeviceSelector()

	// Create mock GPU list
	gpus := discover.GpuInfoList{
		{
			Library: "cuda",
		},
		{
			Library: "oneapi",
		},
		{
			Library: "rocm",
		},
	}

	// Set memory values and names directly (embedded fields)
	gpus[0].Name = "NVIDIA GeForce RTX 3070"
	gpus[0].TotalMemory = 8 * 1024 * 1024 * 1024   // 8GB
	gpus[0].FreeMemory = 6 * 1024 * 1024 * 1024    // 6GB free
	gpus[1].Name = "Intel Arc B580"
	gpus[1].TotalMemory = 12 * 1024 * 1024 * 1024  // 12GB
	gpus[1].FreeMemory = 10 * 1024 * 1024 * 1024   // 10GB free
	gpus[2].Name = "AMD Radeon RX 6700 XT"
	gpus[2].TotalMemory = 12 * 1024 * 1024 * 1024  // 12GB
	gpus[2].FreeMemory = 8 * 1024 * 1024 * 1024    // 8GB free

	testCases := []struct {
		name           string
		memoryRequired int
		expectedDevice string
	}{
		{
			name:           "low_memory_requirement",
			memoryRequired: 4,
			expectedDevice: "Intel Arc B580", // Higher score due to more memory + tensor cores
		},
		{
			name:           "high_memory_requirement",
			memoryRequired: 8,
			expectedDevice: "Intel Arc B580", // Only one with enough free memory
		},
		{
			name:           "impossible_requirement",
			memoryRequired: 15,
			expectedDevice: "", // No device has enough memory
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			selectedGPU := selector.SelectBestDevice(gpus, tc.memoryRequired)

			if tc.expectedDevice == "" {
				if selectedGPU != nil {
					t.Errorf("Expected no device selected, but got: %s", selectedGPU.Name)
				}
			} else {
				if selectedGPU == nil {
					t.Errorf("Expected device %s, but got nil", tc.expectedDevice)
				} else if selectedGPU.Name != tc.expectedDevice {
					t.Errorf("Expected device %s, but got %s", tc.expectedDevice, selectedGPU.Name)
				}

				// Verify formal proof property: selected device has enough memory
				freeGB := int(selectedGPU.FreeMemory / (1024 * 1024 * 1024))
				if freeGB < tc.memoryRequired {
					t.Errorf("Selected device violates memory constraint: required=%dGB, available=%dGB",
						tc.memoryRequired, freeGB)
				}
			}
		})
	}
}

// TestCombinedOptimizations verifies the complete optimization system
// Corresponds to formal proof in proof/complete_test_suite.v: full_system_optimization_verified
func TestCombinedOptimizations(t *testing.T) {
	optimizer := NewVerifiedMemoryOptimizer(true, true)

	// Test the optimization statistics without requiring GGML
	stats := optimizer.GetOptimizationStats(64, 1500)

	// Verify optimizations provide savings
	checkpointSavings := stats["checkpoint_savings"].(uint64)
	mlaSavings := stats["mla_savings"].(uint64)

	expectedCheckpointLayers := uint64(9) // sqrt(64) + 1 = 9
	expectedCheckpointSavings := uint64(64) - expectedCheckpointLayers

	if checkpointSavings != expectedCheckpointSavings {
		t.Errorf("Checkpoint savings incorrect: expected=%d, got=%d",
			expectedCheckpointSavings, checkpointSavings)
	}

	expectedMLASavings := uint64(1500) - uint64(1500/28)
	if mlaSavings != expectedMLASavings {
		t.Errorf("MLA savings incorrect: expected=%d, got=%d",
			expectedMLASavings, mlaSavings)
	}

	efficiency := stats["memory_efficiency"].(float64)
	if efficiency <= 0 || efficiency >= 1 {
		t.Errorf("Invalid memory efficiency: %f", efficiency)
	}

	t.Logf("Optimization successful: checkpoint_savings=%d, mla_savings=%d, efficiency=%.2f%%",
		checkpointSavings, mlaSavings, efficiency*100)
}

// TestMultiVendorBackend verifies multi-vendor GPU backend functionality
func TestMultiVendorBackend(t *testing.T) {
	testCases := []struct {
		name         string
		gpuName      string
		expectedVendor GPUVendor
		expectedBackend GPUBackend
	}{
		{
			name:            "nvidia_rtx",
			gpuName:         "NVIDIA GeForce RTX 3070",
			expectedVendor:  NVIDIA,
			expectedBackend: CUDA_BACKEND,
		},
		{
			name:            "intel_arc",
			gpuName:         "Intel Arc B580",
			expectedVendor:  INTEL,
			expectedBackend: ONEAPI_BACKEND,
		},
		{
			name:            "amd_radeon",
			gpuName:         "AMD Radeon RX 6700 XT",
			expectedVendor:  AMD,
			expectedBackend: ROCM_BACKEND,
		},
		{
			name:            "apple_m_series",
			gpuName:         "Apple M1 Pro",
			expectedVendor:  APPLE,
			expectedBackend: METAL_BACKEND,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockGPU := &discover.GpuInfo{
				Library: "test",
			}
			mockGPU.Name = tc.gpuName
			mockGPU.TotalMemory = 8 * 1024 * 1024 * 1024

			backend := NewUnifiedGPUBackend(mockGPU)

			if backend.vendor != tc.expectedVendor {
				t.Errorf("Vendor detection failed: expected=%d, got=%d", tc.expectedVendor, backend.vendor)
			}

			if backend.backend != tc.expectedBackend {
				t.Errorf("Backend detection failed: expected=%d, got=%d", tc.expectedBackend, backend.backend)
			}

			// Test operation estimation
			gemmOp := Operation{
				Type:  UNIFIED_GEMM,
				M:     1024,
				N:     1024,
				K:     1024,
				Dtype: "fp16",
			}

			time := backend.EstimateOperationTime(gemmOp)
			if time <= 0 {
				t.Errorf("Invalid operation time estimate: %f", time)
			}

			t.Logf("GPU %s: vendor=%d, backend=%d, GEMM time=%.2fms",
				tc.gpuName, backend.vendor, backend.backend, time)
		})
	}
}

// TestIntegratedOptimizer verifies the complete optimization pipeline
func TestIntegratedOptimizer(t *testing.T) {
	optimizer := NewIntegratedOptimizer(true, true)

	// Mock GPU list
	gpus := discover.GpuInfoList{
		{
			Library: "oneapi",
		},
		{
			Library: "cuda",
		},
	}
	gpus[0].Name = "Intel Arc B580"
	gpus[0].TotalMemory = 12 * 1024 * 1024 * 1024
	gpus[0].FreeMemory = 10 * 1024 * 1024 * 1024
	gpus[1].Name = "NVIDIA GeForce RTX 3070"
	gpus[1].TotalMemory = 8 * 1024 * 1024 * 1024
	gpus[1].FreeMemory = 6 * 1024 * 1024 * 1024

	ctx := context.Background()
	opts := api.Options{}
	opts.NumCtx = 2048

	selectedGPU, estimate, err := optimizer.OptimizeForModel(ctx, "test-model", gpus, opts)
	if err != nil {
		t.Fatalf("Integration test failed: %v", err)
	}

	if selectedGPU == nil {
		t.Fatal("No GPU selected")
	}

	// Verify a GPU was selected
	if selectedGPU.Name == "" {
		t.Error("Selected GPU has no name")
	}

	// Verify memory estimate is reasonable
	if estimate.VRAMSize == 0 {
		t.Error("Memory estimate is zero")
	}

	if estimate.Layers == 0 {
		t.Error("No layers estimated")
	}

	t.Logf("Integration test successful: selected %s, estimated %d layers, %d bytes VRAM",
		selectedGPU.Name, estimate.Layers, estimate.VRAMSize)
}

// TestOptimizationStats verifies statistics reporting
func TestOptimizationStats(t *testing.T) {
	optimizer := NewVerifiedMemoryOptimizer(true, true)

	stats := optimizer.GetOptimizationStats(64, 1500)

	// Verify expected keys exist
	expectedKeys := []string{
		"checkpoint_enabled", "mla_enabled",
		"original_layers", "checkpoint_layers", "checkpoint_savings",
		"original_kv_cache", "compressed_kv_cache", "mla_savings",
		"total_memory_saved", "memory_efficiency",
	}

	for _, key := range expectedKeys {
		if _, exists := stats[key]; !exists {
			t.Errorf("Missing stats key: %s", key)
		}
	}

	// Verify memory savings are positive
	if stats["checkpoint_savings"].(uint64) == 0 {
		t.Error("No checkpoint savings reported")
	}

	if stats["mla_savings"].(uint64) == 0 {
		t.Error("No MLA savings reported")
	}

	efficiency := stats["memory_efficiency"].(float64)
	if efficiency <= 0 || efficiency >= 1 {
		t.Errorf("Invalid memory efficiency: %f", efficiency)
	}

	t.Logf("Optimization stats: checkpoint_savings=%d, mla_savings=%d, efficiency=%.2f%%",
		stats["checkpoint_savings"], stats["mla_savings"], efficiency*100)
}

// Benchmark tests for performance verification
func BenchmarkCheckpointOptimization(b *testing.B) {
	optimizer := NewVerifiedMemoryOptimizer(true, false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = optimizer.CheckpointMemoryEstimate(64)
	}
}

func BenchmarkMLACompression(b *testing.B) {
	optimizer := NewVerifiedMemoryOptimizer(false, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = optimizer.MLACompressionEstimate(1500)
	}
}

func BenchmarkDeviceSelection(b *testing.B) {
	selector := NewVerifiedDeviceSelector()
	gpus := discover.GpuInfoList{
		{},
		{},
	}
	gpus[0].Name = "Intel Arc B580"
	gpus[0].TotalMemory = 12 * 1024 * 1024 * 1024
	gpus[0].FreeMemory = 10 * 1024 * 1024 * 1024
	gpus[1].Name = "NVIDIA GeForce RTX 3070"
	gpus[1].TotalMemory = 8 * 1024 * 1024 * 1024
	gpus[1].FreeMemory = 6 * 1024 * 1024 * 1024

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = selector.SelectBestDevice(gpus, 4)
	}
}