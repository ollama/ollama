package llm

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/discover"
)

// QuickCheck-style property tests using our generators

func TestQuickCheckCheckpointMemoryReduction(t *testing.T) {
	runner := NewPropertyTestRunner()
	layerGen := NewLayerCountGenerator()

	property := func(layers int) error {
		optimizer := NewVerifiedMemoryOptimizer(true, false)

		checkpoint := optimizer.CheckpointMemoryEstimate(layers)
		standard := uint64(layers)

		// Property 1: For layers >= 4, checkpoint < standard
		if layers >= 4 && checkpoint >= standard {
			return fmt.Errorf("checkpoint optimization failed: %d >= %d for %d layers",
				checkpoint, standard, layers)
		}

		// Property 2: Checkpoint should never be zero
		if checkpoint == 0 {
			return fmt.Errorf("checkpoint memory should never be zero for %d layers", layers)
		}

		// Property 3: Checkpoint should be roughly sqrt(layers) + 1
		expectedApprox := uint64(approximateSqrt(layers) + 1)
		if layers >= 4 && checkpoint != expectedApprox {
			return fmt.Errorf("checkpoint memory not as expected: got %d, expected ~%d for %d layers",
				checkpoint, expectedApprox, layers)
		}

		return nil
	}

	if err := runner.RunIntProperty(layerGen, property); err != nil {
		t.Errorf("QuickCheck checkpoint property failed: %v", err)
	}
}

func TestQuickCheckMLACompressionRatio(t *testing.T) {
	runner := NewPropertyTestRunner()
	kvGen := NewKVCacheSizeGenerator()

	property := func(kvSize uint64) error {
		optimizer := NewVerifiedMemoryOptimizer(false, true)

		compressed := optimizer.MLACompressionEstimate(kvSize)

		// Property 1: For kvSize >= 28, compressed < original
		if kvSize >= 28 && compressed >= kvSize {
			return fmt.Errorf("MLA compression failed: %d >= %d for input %d",
				compressed, kvSize, kvSize)
		}

		// Property 2: Compression ratio should be approximately 1/28
		if kvSize >= 280 { // Avoid division by small numbers
			ratio := float64(compressed) / float64(kvSize)
			expectedRatio := 1.0 / 28.0

			if ratio > expectedRatio*1.1 || ratio < expectedRatio*0.9 {
				return fmt.Errorf("compression ratio out of range: %.4f vs expected %.4f for input %d",
					ratio, expectedRatio, kvSize)
			}
		}

		// Property 3: Monotonicity - larger input should yield larger or equal output
		if kvSize > 28 {
			smallerCompressed := optimizer.MLACompressionEstimate(kvSize - 1)
			if compressed < smallerCompressed {
				return fmt.Errorf("MLA compression not monotonic: %d < %d for %d vs %d",
					compressed, smallerCompressed, kvSize, kvSize-1)
			}
		}

		return nil
	}

	if err := runner.RunUintProperty(kvGen, property); err != nil {
		t.Errorf("QuickCheck MLA property failed: %v", err)
	}
}

func TestQuickCheckGPUDeviceScoring(t *testing.T) {
	runner := NewPropertyTestRunner()
	gpuGen := NewGPUSpecGenerator()

	property := func(gpu *GPUDeviceSpec) error {
		score := gpu.DeviceScore()

		// Property 1: Score should be deterministic
		score2 := gpu.DeviceScore()
		if score != score2 {
			return fmt.Errorf("GPU scoring not deterministic: %d != %d", score, score2)
		}

		// Property 2: Score should be positive
		if score <= 0 {
			return fmt.Errorf("GPU score should be positive: %d", score)
		}

		// Property 3: Memory should dominate the score
		expectedMinScore := gpu.MemorySizeGB * 10
		if score < expectedMinScore {
			return fmt.Errorf("score too low: %d < %d (memory component)", score, expectedMinScore)
		}

		// Property 4: Tensor cores should add exactly 50 points
		expectedScore := gpu.MemorySizeGB*10 + gpu.PeakTFLOPSFP32
		if gpu.SupportsTensorCores {
			expectedScore += 50
		}
		if score != expectedScore {
			return fmt.Errorf("score calculation incorrect: got %d, expected %d", score, expectedScore)
		}

		// Property 5: Higher memory should yield higher score (all else equal)
		if gpu.MemorySizeGB > 4 {
			lowerMemGPU := *gpu
			lowerMemGPU.MemorySizeGB = gpu.MemorySizeGB - 1
			lowerScore := lowerMemGPU.DeviceScore()

			if score <= lowerScore {
				return fmt.Errorf("higher memory should yield higher score: %d <= %d", score, lowerScore)
			}
		}

		return nil
	}

	if err := runner.RunGPUSpecProperty(gpuGen, property); err != nil {
		t.Errorf("QuickCheck GPU scoring property failed: %v", err)
	}
}

func TestQuickCheckCombinedOptimizations(t *testing.T) {
	runner := NewPropertyTestRunner()
	configGen := NewModelConfigGenerator()

	property := func(config ModelConfig) error {
		// Test all optimization combinations
		optimizers := []struct {
			checkpoint, mla bool
			name           string
		}{
			{false, false, "none"},
			{true, false, "checkpoint"},
			{false, true, "mla"},
			{true, true, "both"},
		}

		originalTotal := uint64(config.Layers) + config.KVSize

		for _, opt := range optimizers {
			optimizer := NewVerifiedMemoryOptimizer(opt.checkpoint, opt.mla)

			checkpointMem := optimizer.CheckpointMemoryEstimate(config.Layers)
			compressedKV := optimizer.MLACompressionEstimate(config.KVSize)
			optimizedTotal := checkpointMem + compressedKV

			// Property 1: Optimization should never increase memory
			if optimizedTotal > originalTotal {
				return fmt.Errorf("optimization %s increased memory: %d > %d (config: %+v)",
					opt.name, optimizedTotal, originalTotal, config)
			}

			// Property 2: With both optimizations on reasonable inputs, should save significantly
			if opt.checkpoint && opt.mla && config.Layers >= 16 && config.KVSize >= 280 {
				savingsRatio := float64(originalTotal-optimizedTotal) / float64(originalTotal)
				if savingsRatio < 0.5 { // Should save at least 50%
					return fmt.Errorf("insufficient savings with both optimizations: %.2f%% for config %+v",
						savingsRatio*100, config)
				}
			}

			// Property 3: Statistics should be self-consistent
			stats := optimizer.GetOptimizationStats(config.Layers, uint64(config.KVSize))
			totalSaved := stats["total_memory_saved"].(uint64)
			efficiency := stats["memory_efficiency"].(float64)

			expectedSaved := originalTotal - optimizedTotal
			if totalSaved != expectedSaved {
				return fmt.Errorf("stats inconsistent: total_saved=%d != expected=%d",
					totalSaved, expectedSaved)
			}

			expectedEfficiency := float64(totalSaved) / float64(originalTotal)
			if efficiency != expectedEfficiency {
				return fmt.Errorf("efficiency inconsistent: got=%.4f != expected=%.4f",
					efficiency, expectedEfficiency)
			}
		}

		return nil
	}

	if err := runner.RunModelConfigProperty(configGen, property); err != nil {
		t.Errorf("QuickCheck combined optimizations property failed: %v", err)
	}
}

func TestQuickCheckMultiVendorConsistency(t *testing.T) {
	runner := NewPropertyTestRunner()
	gpuGen := NewGPUSpecGenerator()

	property := func(gpuSpec *GPUDeviceSpec) error {
		// Create a mock GpuInfo based on the spec
		mockGPU := &discover.GpuInfo{
			Library: getLibraryForVendor(gpuSpec.Vendor),
		}
		mockGPU.Name = gpuSpec.DeviceName
		mockGPU.TotalMemory = uint64(gpuSpec.MemorySizeGB) * 1024 * 1024 * 1024

		backend := NewUnifiedGPUBackend(mockGPU)

		// Property 1: Vendor should be detected correctly
		if backend.vendor != gpuSpec.Vendor {
			return fmt.Errorf("vendor detection failed: got %d, expected %d for %s",
				backend.vendor, gpuSpec.Vendor, gpuSpec.DeviceName)
		}

		// Property 2: Operation time estimation should be deterministic
		testOp := Operation{
			Type:  UNIFIED_GEMM,
			M:     1024,
			N:     1024,
			K:     1024,
			Dtype: "fp16",
		}

		time1 := backend.EstimateOperationTime(testOp)
		time2 := backend.EstimateOperationTime(testOp)

		if time1 != time2 {
			return fmt.Errorf("operation time not deterministic: %f != %f", time1, time2)
		}

		// Property 3: Time should be positive and reasonable
		if time1 <= 0 {
			return fmt.Errorf("operation time should be positive: %f", time1)
		}

		if time1 > 1000 { // More than 1 second is probably unreasonable
			return fmt.Errorf("operation time unreasonably high: %f ms", time1)
		}

		// Property 4: Tensor cores should improve performance
		if gpuSpec.SupportsTensorCores {
			noTensorOp := testOp
			noTensorOp.Dtype = "fp32" // Typically no tensor core acceleration

			fp16Time := backend.EstimateOperationTime(testOp)
			fp32Time := backend.EstimateOperationTime(noTensorOp)

			// FP16 with tensor cores should be faster than FP32
			if fp16Time >= fp32Time {
				return fmt.Errorf("tensor cores not providing speedup: fp16=%f >= fp32=%f",
					fp16Time, fp32Time)
			}
		}

		return nil
	}

	if err := runner.RunGPUSpecProperty(gpuGen, property); err != nil {
		t.Errorf("QuickCheck multi-vendor property failed: %v", err)
	}
}

func TestQuickCheckWorkloadAnalysisOptimality(t *testing.T) {
	runner := NewPropertyTestRunner()
	workloadGen := NewWorkloadGenerator()

	property := func(operations []Operation) error {
		// Create a set of test GPUs with different characteristics
		testGPUs := []discover.GpuInfo{
			{Library: "cuda"},
			{Library: "oneapi"},
			{Library: "rocm"},
		}

		// Set names and memory
		testGPUs[0].Name = "NVIDIA Test GPU"
		testGPUs[0].TotalMemory = 16 * 1024 * 1024 * 1024
		testGPUs[1].Name = "Intel Test GPU"
		testGPUs[1].TotalMemory = 12 * 1024 * 1024 * 1024
		testGPUs[2].Name = "AMD Test GPU"
		testGPUs[2].TotalMemory = 8 * 1024 * 1024 * 1024

		analyzer := NewWorkloadAnalyzer()

		// Add operations to analyzer
		for _, op := range operations {
			analyzer.AddOperation(op)
		}

		selected := analyzer.AnalyzeWorkload(testGPUs)

		// Property 1: Should select a GPU when GPUs are available
		if selected == nil {
			return fmt.Errorf("no GPU selected despite %d available GPUs", len(testGPUs))
		}

		// Property 2: Selected GPU should be in the provided list
		found := false
		for i := range testGPUs {
			if &testGPUs[i] == selected {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("selected GPU not in provided list")
		}

		// Property 3: Selection should be deterministic
		selected2 := analyzer.AnalyzeWorkload(testGPUs)
		if selected != selected2 {
			return fmt.Errorf("workload analysis not deterministic")
		}

		// Property 4: With empty workload, should still select a GPU
		emptyAnalyzer := NewWorkloadAnalyzer()
		emptySelected := emptyAnalyzer.AnalyzeWorkload(testGPUs)
		if emptySelected == nil {
			return fmt.Errorf("no GPU selected for empty workload")
		}

		return nil
	}

	if err := runner.RunWorkloadProperty(workloadGen, property); err != nil {
		t.Errorf("QuickCheck workload analysis property failed: %v", err)
	}
}

// Helper functions

func approximateSqrt(n int) int {
	if n <= 0 {
		return 0
	}
	if n == 1 {
		return 1
	}

	// Simple integer square root approximation
	x := n
	for {
		y := (x + n/x) / 2
		if y >= x {
			return x
		}
		x = y
	}
}

func getLibraryForVendor(vendor GPUVendor) string {
	switch vendor {
	case NVIDIA:
		return "cuda"
	case AMD:
		return "rocm"
	case INTEL:
		return "oneapi"
	case APPLE:
		return "metal"
	default:
		return "unknown"
	}
}

// Integration test combining QuickCheck with SMT verification
func TestQuickCheckSMTIntegration(t *testing.T) {
	runner := NewPropertyTestRunner()
	runner.MaxTests = 100 // Fewer tests for integration testing

	configGen := NewModelConfigGenerator()

	property := func(config ModelConfig) error {
		// Test that our implementation matches SMT specifications
		optimizer := NewVerifiedMemoryOptimizer(true, true)

		// Get implementation results
		checkpointMem := optimizer.CheckpointMemoryEstimate(config.Layers)
		compressedKV := optimizer.MLACompressionEstimate(config.KVSize)

		// Verify against SMT concrete values where possible
		if config.Layers == 64 && config.KVSize == 1500 {
			// These are the values verified in our SMT proofs
			if checkpointMem != 9 {
				return fmt.Errorf("SMT verification failed: checkpoint for 64 layers should be 9, got %d",
					checkpointMem)
			}
			if compressedKV != 53 {
				return fmt.Errorf("SMT verification failed: MLA for 1500 should be 53, got %d",
					compressedKV)
			}
		}

		// General SMT-verified properties
		if config.Layers >= 4 && checkpointMem >= uint64(config.Layers) {
			return fmt.Errorf("SMT property violated: checkpoint_memory(%d) = %d >= %d",
				config.Layers, checkpointMem, config.Layers)
		}

		if config.KVSize >= 28 && compressedKV >= config.KVSize {
			return fmt.Errorf("SMT property violated: mla_compression(%d) = %d >= %d",
				config.KVSize, compressedKV, config.KVSize)
		}

		return nil
	}

	if err := runner.RunModelConfigProperty(configGen, property); err != nil {
		t.Errorf("QuickCheck SMT integration failed: %v", err)
	}
}

// Benchmark QuickCheck property tests
func BenchmarkQuickCheckCheckpointProperty(b *testing.B) {
	runner := NewPropertyTestRunner()
	runner.MaxTests = b.N
	layerGen := NewLayerCountGenerator()

	property := func(layers int) error {
		optimizer := NewVerifiedMemoryOptimizer(true, false)
		_ = optimizer.CheckpointMemoryEstimate(layers)
		return nil
	}

	b.ResetTimer()
	if err := runner.RunIntProperty(layerGen, property); err != nil {
		b.Errorf("Benchmark property failed: %v", err)
	}
}

func BenchmarkQuickCheckMLAProperty(b *testing.B) {
	runner := NewPropertyTestRunner()
	runner.MaxTests = b.N
	kvGen := NewKVCacheSizeGenerator()

	property := func(kvSize uint64) error {
		optimizer := NewVerifiedMemoryOptimizer(false, true)
		_ = optimizer.MLACompressionEstimate(kvSize)
		return nil
	}

	b.ResetTimer()
	if err := runner.RunUintProperty(kvGen, property); err != nil {
		b.Errorf("Benchmark property failed: %v", err)
	}
}