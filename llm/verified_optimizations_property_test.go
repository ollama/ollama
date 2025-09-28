package llm

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/ollama/ollama/discover"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Property-based test generators
type PropertyTestGenerator struct {
	rand *rand.Rand
}

func NewPropertyTestGenerator() *PropertyTestGenerator {
	return &PropertyTestGenerator{
		rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Generate random layer counts in realistic ranges
func (g *PropertyTestGenerator) GenLayers() int {
	// Generate layer counts from 1 to 200 (realistic transformer sizes)
	return g.rand.Intn(200) + 1
}

// Generate random KV cache sizes
func (g *PropertyTestGenerator) GenKVSize() uint64 {
	// Generate sizes from 1 to 100,000
	return uint64(g.rand.Intn(100000) + 1)
}

// Generate GPU configurations
func (g *PropertyTestGenerator) GenGPUSpec() *GPUDeviceSpec {
	vendors := []GPUVendor{NVIDIA, AMD, INTEL, APPLE}

	vendor := vendors[g.rand.Intn(len(vendors))]
	memoryGB := g.rand.Intn(32) + 4  // 4-36 GB
	computeUnits := g.rand.Intn(80) + 10  // 10-90 units
	tflops := g.rand.Intn(50) + 5  // 5-55 TFLOPS
	tensorCores := g.rand.Float32() > 0.5  // 50% chance

	return &GPUDeviceSpec{
		Vendor:              vendor,
		MemorySizeGB:        memoryGB,
		ComputeUnits:        computeUnits,
		PeakTFLOPSFP32:      tflops,
		SupportsTensorCores: tensorCores,
		SupportsHalfPrec:    true,
		MemoryBandwidthGBps: g.rand.Intn(1000) + 100,
		DeviceName:          "Generated GPU",
	}
}

// Generate GPU info for testing
func (g *PropertyTestGenerator) GenGPUInfo() discover.GpuInfo {
	vendors := []string{"cuda", "rocm", "oneapi", "metal"}
	names := []string{"Test GPU A", "Test GPU B", "Test GPU C"}

	gpu := discover.GpuInfo{
		Library: vendors[g.rand.Intn(len(vendors))],
	}

	gpu.Name = names[g.rand.Intn(len(names))]
	gpu.TotalMemory = uint64(g.rand.Intn(32)+4) * 1024 * 1024 * 1024  // 4-36 GB
	gpu.FreeMemory = gpu.TotalMemory - uint64(g.rand.Intn(int(gpu.TotalMemory/1024/1024/1024/2))) * 1024 * 1024 * 1024

	return gpu
}

// Property: Checkpoint memory should always be less than standard memory for layers >= 4
func TestPropertyCheckpointReduction(t *testing.T) {
	gen := NewPropertyTestGenerator()
	optimizer := NewVerifiedMemoryOptimizer(true, false)

	const numTests = 1000

	for i := 0; i < numTests; i++ {
		layers := gen.GenLayers()

		checkpointMem := optimizer.CheckpointMemoryEstimate(layers)
		standardMem := uint64(layers)

		if layers >= 4 {
			if checkpointMem >= standardMem {
				t.Errorf("Property violation: checkpoint_memory(%d) = %d >= standard_memory(%d) = %d",
					layers, checkpointMem, layers, standardMem)
			}
		}

		// Additional properties
		if checkpointMem == 0 {
			t.Errorf("Checkpoint memory should never be zero for layers=%d", layers)
		}

		if checkpointMem > standardMem {
			t.Errorf("Checkpoint memory should never exceed standard memory: %d > %d for layers=%d",
				checkpointMem, standardMem, layers)
		}
	}
}

// Property: MLA compression should always reduce size for inputs >= 28
func TestPropertyMLACompression(t *testing.T) {
	gen := NewPropertyTestGenerator()
	optimizer := NewVerifiedMemoryOptimizer(false, true)

	const numTests = 1000

	for i := 0; i < numTests; i++ {
		kvSize := gen.GenKVSize()

		compressed := optimizer.MLACompressionEstimate(kvSize)

		if kvSize >= 28 {
			if compressed >= kvSize {
				t.Errorf("Property violation: MLA compression failed for kvSize=%d, compressed=%d",
					kvSize, compressed)
			}
		}

		// Compression should be deterministic
		compressed2 := optimizer.MLACompressionEstimate(kvSize)
		if compressed != compressed2 {
			t.Errorf("MLA compression not deterministic: %d != %d for kvSize=%d",
				compressed, compressed2, kvSize)
		}

		// Compression should be monotonic (larger input -> larger or equal output)
		if kvSize > 28 {
			smallerCompressed := optimizer.MLACompressionEstimate(kvSize - 1)
			if compressed < smallerCompressed {
				t.Errorf("MLA compression not monotonic: %d < %d for kvSize=%d vs %d",
					compressed, smallerCompressed, kvSize, kvSize-1)
			}
		}
	}
}

// Property: GPU device scoring should be consistent and deterministic
func TestPropertyGPUScoring(t *testing.T) {
	gen := NewPropertyTestGenerator()

	const numTests = 500

	for i := 0; i < numTests; i++ {
		spec := gen.GenGPUSpec()

		score1 := spec.DeviceScore()
		score2 := spec.DeviceScore()

		// Scoring should be deterministic
		if score1 != score2 {
			t.Errorf("GPU scoring not deterministic: %d != %d for %+v", score1, score2, spec)
		}

		// Score should be positive
		if score1 <= 0 {
			t.Errorf("GPU score should be positive: %d for %+v", score1, spec)
		}

		// Memory component should dominate for high memory
		expectedMinScore := spec.MemorySizeGB * 10
		if score1 < expectedMinScore {
			t.Errorf("Score too low: %d < %d (memory component) for %+v",
				score1, expectedMinScore, spec)
		}

		// Tensor core bonus should be applied correctly
		tensorBonus := 0
		if spec.SupportsTensorCores {
			tensorBonus = 50
		}
		expectedScore := spec.MemorySizeGB*10 + spec.PeakTFLOPSFP32 + tensorBonus
		if score1 != expectedScore {
			t.Errorf("Score calculation incorrect: got %d, expected %d for %+v",
				score1, expectedScore, spec)
		}
	}
}

// Property: Device selection should respect memory constraints
func TestPropertyDeviceSelection(t *testing.T) {
	gen := NewPropertyTestGenerator()
	selector := NewVerifiedDeviceSelector()

	const numTests = 200

	for i := 0; i < numTests; i++ {
		// Generate a list of GPUs
		numGPUs := gen.rand.Intn(5) + 1  // 1-5 GPUs
		gpus := make(discover.GpuInfoList, numGPUs)

		for j := 0; j < numGPUs; j++ {
			gpus[j] = gen.GenGPUInfo()
		}

		// Test different memory requirements
		memRequirements := []int{2, 4, 8, 12, 16, 24, 32}

		for _, memReq := range memRequirements {
			selected := selector.SelectBestDevice(gpus, memReq)

			if selected != nil {
				// Selected device must satisfy memory constraint
				selectedMemGB := int(selected.TotalMemory / (1024 * 1024 * 1024))
				if selectedMemGB < memReq {
					t.Errorf("Selected device violates memory constraint: %dGB < %dGB required",
						selectedMemGB, memReq)
				}

				// Check if there's a better device that also satisfies constraints
				for k := range gpus {
					other := &gpus[k]
					otherMemGB := int(other.TotalMemory / (1024 * 1024 * 1024))

					if otherMemGB >= memReq && other != selected {
						// Both satisfy constraint, selected should have >= score
						selectedScore := getKnownDeviceScore(selected.Name)
						otherScore := getKnownDeviceScore(other.Name)

						// If both are unknown devices, can't compare scores
						if selectedScore >= 0 && otherScore >= 0 {
							if selectedScore < otherScore {
								t.Errorf("Suboptimal device selected: %s (score %d) vs %s (score %d)",
									selected.Name, selectedScore, other.Name, otherScore)
							}
						}
					}
				}
			} else {
				// If no device selected, verify no device can satisfy constraint
				canSatisfy := false
				for k := range gpus {
					if int(gpus[k].TotalMemory/(1024*1024*1024)) >= memReq {
						canSatisfy = true
						break
					}
				}
				if canSatisfy {
					t.Errorf("No device selected despite having devices that satisfy %dGB requirement", memReq)
				}
			}
		}
	}
}

// Helper function to get scores for known devices
func getKnownDeviceScore(name string) int {
	switch name {
	case "NVIDIA GeForce RTX 3070":
		return 150
	case "Intel Arc B580":
		return 187
	case "AMD Radeon RX 6700 XT":
		return 145
	default:
		return -1  // Unknown device
	}
}

// Property: Combined optimizations should never increase memory usage
func TestPropertyCombinedOptimizationsNeverIncrease(t *testing.T) {
	gen := NewPropertyTestGenerator()

	const numTests = 500

	for i := 0; i < numTests; i++ {
		layers := gen.GenLayers()
		kvSize := gen.GenKVSize()

		// Test all combinations of optimizations
		configs := []struct {
			useCheckpoint, useMLA bool
			name                  string
		}{
			{false, false, "none"},
			{true, false, "checkpoint_only"},
			{false, true, "mla_only"},
			{true, true, "both"},
		}

		originalTotal := uint64(layers) + kvSize

		for _, config := range configs {
			optimizer := NewVerifiedMemoryOptimizer(config.useCheckpoint, config.useMLA)

			checkpointMem := optimizer.CheckpointMemoryEstimate(layers)
			compressedKV := optimizer.MLACompressionEstimate(kvSize)
			optimizedTotal := checkpointMem + compressedKV

			// Optimized should never exceed original
			if optimizedTotal > originalTotal {
				t.Errorf("Optimization increased memory: %d > %d (layers=%d, kv=%d, config=%s)",
					optimizedTotal, originalTotal, layers, kvSize, config.name)
			}

			// With optimizations enabled, should save memory for reasonable sizes
			if config.useCheckpoint && config.useMLA && layers >= 16 && kvSize >= 280 {
				if optimizedTotal >= originalTotal {
					t.Errorf("Combined optimizations failed to save memory: %d >= %d (layers=%d, kv=%d)",
						optimizedTotal, originalTotal, layers, kvSize)
				}
			}
		}
	}
}

// Property: Optimization statistics should be self-consistent
func TestPropertyOptimizationStatisticsConsistency(t *testing.T) {
	gen := NewPropertyTestGenerator()

	const numTests = 300

	for i := 0; i < numTests; i++ {
		layers := gen.GenLayers()
		kvSize := gen.GenKVSize()

		optimizer := NewVerifiedMemoryOptimizer(true, true)
		stats := optimizer.GetOptimizationStats(layers, kvSize)

		// Extract values
		checkpointEnabled := stats["checkpoint_enabled"].(bool)
		mlaEnabled := stats["mla_enabled"].(bool)
		originalLayers := stats["original_layers"].(int)
		checkpointLayers := stats["checkpoint_layers"].(uint64)
		checkpointSavings := stats["checkpoint_savings"].(uint64)
		originalKV := stats["original_kv_cache"].(uint64)
		compressedKV := stats["compressed_kv_cache"].(uint64)
		mlaSavings := stats["mla_savings"].(uint64)
		totalSaved := stats["total_memory_saved"].(uint64)
		efficiency := stats["memory_efficiency"].(float64)

		// Verify consistency
		if originalLayers != layers {
			t.Errorf("Original layers mismatch: %d != %d", originalLayers, layers)
		}

		if originalKV != kvSize {
			t.Errorf("Original KV mismatch: %d != %d", originalKV, kvSize)
		}

		if checkpointEnabled {
			expectedCheckpointSavings := uint64(layers) - checkpointLayers
			if checkpointSavings != expectedCheckpointSavings {
				t.Errorf("Checkpoint savings inconsistent: %d != %d",
					checkpointSavings, expectedCheckpointSavings)
			}
		}

		if mlaEnabled {
			expectedMLASavings := kvSize - compressedKV
			if mlaSavings != expectedMLASavings {
				t.Errorf("MLA savings inconsistent: %d != %d", mlaSavings, expectedMLASavings)
			}
		}

		expectedTotalSaved := checkpointSavings + mlaSavings
		if totalSaved != expectedTotalSaved {
			t.Errorf("Total savings inconsistent: %d != %d", totalSaved, expectedTotalSaved)
		}

		// Efficiency should be in valid range
		if efficiency < 0 || efficiency > 1 {
			t.Errorf("Efficiency out of range: %f", efficiency)
		}

		// For non-zero totals, efficiency calculation should be consistent
		if originalLayers+int(originalKV) > 0 {
			expectedEfficiency := float64(totalSaved) / float64(uint64(originalLayers)+originalKV)
			if math.Abs(efficiency-expectedEfficiency) > 1e-10 {
				t.Errorf("Efficiency calculation inconsistent: %f != %f", efficiency, expectedEfficiency)
			}
		}
	}
}

// Property: Multi-vendor backend should handle all vendors consistently
func TestPropertyMultiVendorConsistency(t *testing.T) {
	gen := NewPropertyTestGenerator()

	vendors := []struct {
		name     string
		expected GPUVendor
	}{
		{"NVIDIA GeForce RTX 3070", NVIDIA},
		{"Intel Arc B580", INTEL},
		{"AMD Radeon RX 6700 XT", AMD},
		{"Apple M1 Pro", APPLE},
	}

	for _, vendor := range vendors {
		gpu := gen.GenGPUInfo()
		gpu.Name = vendor.name

		backend := NewUnifiedGPUBackend(&gpu)

		if backend.vendor != vendor.expected {
			t.Errorf("Vendor detection failed: %s -> %d, expected %d",
				vendor.name, backend.vendor, vendor.expected)
		}

		// Test operation estimation consistency
		for opType := UNIFIED_GEMM; opType <= COPY; opType++ {
			op := Operation{
				Type: opType,
				M: 1024, N: 1024, K: 1024,
				Size:  1024,
				Dtype: "fp16",
			}

			time1 := backend.EstimateOperationTime(op)
			time2 := backend.EstimateOperationTime(op)

			if time1 != time2 {
				t.Errorf("Operation time estimation not deterministic: %f != %f for %s",
					time1, time2, vendor.name)
			}

			if time1 < 0 {
				t.Errorf("Operation time should be non-negative: %f for %s", time1, vendor.name)
			}
		}
	}
}

// Property: Workload analyzer should be consistent and optimal
func TestPropertyWorkloadAnalysis(t *testing.T) {
	gen := NewPropertyTestGenerator()

	const numTests = 100

	for i := 0; i < numTests; i++ {
		analyzer := NewWorkloadAnalyzer()

		// Add random operations
		numOps := gen.rand.Intn(10) + 1
		for j := 0; j < numOps; j++ {
			op := Operation{
				Type:  OperationType(gen.rand.Intn(5)), // 0-4
				M:     gen.rand.Intn(2048) + 256,
				N:     gen.rand.Intn(2048) + 256,
				K:     gen.rand.Intn(2048) + 256,
				Size:  gen.rand.Intn(10000) + 100,
				Dtype: "fp16",
			}
			analyzer.AddOperation(op)
		}

		// Generate GPUs
		numGPUs := gen.rand.Intn(4) + 2  // 2-5 GPUs
		gpus := make(discover.GpuInfoList, numGPUs)
		for j := 0; j < numGPUs; j++ {
			gpus[j] = gen.GenGPUInfo()
		}

		selected := analyzer.AnalyzeWorkload(gpus)

		if len(gpus) > 0 && selected == nil {
			t.Errorf("No GPU selected despite having %d GPUs available", len(gpus))
		}

		if selected != nil {
			// Verify selected GPU is in the list
			found := false
			for j := range gpus {
				if &gpus[j] == selected {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Selected GPU not in provided list")
			}
		}

		// Test determinism - same workload should yield same result
		selected2 := analyzer.AnalyzeWorkload(gpus)
		if selected != selected2 {
			t.Errorf("Workload analysis not deterministic")
		}
	}
}

// Benchmark property-based tests
func BenchmarkPropertyCheckpointReduction(b *testing.B) {
	gen := NewPropertyTestGenerator()
	optimizer := NewVerifiedMemoryOptimizer(true, false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layers := gen.GenLayers()
		_ = optimizer.CheckpointMemoryEstimate(layers)
	}
}

func BenchmarkPropertyMLACompression(b *testing.B) {
	gen := NewPropertyTestGenerator()
	optimizer := NewVerifiedMemoryOptimizer(false, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kvSize := gen.GenKVSize()
		_ = optimizer.MLACompressionEstimate(kvSize)
	}
}

func BenchmarkPropertyGPUScoring(b *testing.B) {
	gen := NewPropertyTestGenerator()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spec := gen.GenGPUSpec()
		_ = spec.DeviceScore()
	}
}