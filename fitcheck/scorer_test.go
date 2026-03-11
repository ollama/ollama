package fitcheck_test

import (
	"testing"

	"github.com/ollama/ollama/fitcheck"
	"github.com/ollama/ollama/ml"
)

const testGB = uint64(1024 * 1024 * 1024)

// metalHW returns a hardware profile simulating an Apple Silicon machine.
func metalHW(freeVRAMGB, totalVRAMGB, freeRAMGB, totalRAMGB, diskGB uint64) fitcheck.HardwareProfile {
	return fitcheck.HardwareProfile{
		BestGPU: &ml.DeviceInfo{
			DeviceID:    ml.DeviceID{Library: "Metal"},
			Name:        "Apple M-series",
			FreeMemory:  freeVRAMGB * testGB,
			TotalMemory: totalVRAMGB * testGB,
		},
		RAMAvailableBytes:   freeRAMGB * testGB,
		RAMTotalBytes:       totalRAMGB * testGB,
		DiskModelAvailBytes: diskGB * testGB,
	}
}

// cudaHW returns a hardware profile simulating an NVIDIA CUDA machine.
func cudaHW(computeMajor int, freeVRAMGB, totalVRAMGB, freeRAMGB, totalRAMGB, diskGB uint64) fitcheck.HardwareProfile {
	return fitcheck.HardwareProfile{
		BestGPU: &ml.DeviceInfo{
			DeviceID:     ml.DeviceID{Library: "CUDA"},
			Name:         "NVIDIA GPU",
			FreeMemory:   freeVRAMGB * testGB,
			TotalMemory:  totalVRAMGB * testGB,
			ComputeMajor: computeMajor,
		},
		RAMAvailableBytes:   freeRAMGB * testGB,
		RAMTotalBytes:       totalRAMGB * testGB,
		DiskModelAvailBytes: diskGB * testGB,
	}
}

// cpuHW returns a hardware profile with no GPU.
func cpuHW(freeRAMGB, totalRAMGB, diskGB uint64) fitcheck.HardwareProfile {
	return fitcheck.HardwareProfile{
		BestGPU:             nil,
		RAMAvailableBytes:   freeRAMGB * testGB,
		RAMTotalBytes:       totalRAMGB * testGB,
		DiskModelAvailBytes: diskGB * testGB,
	}
}

// assertSorted verifies candidates are in non-decreasing tier order.
func assertSorted(t *testing.T, candidates []fitcheck.ModelCandidate) {
	t.Helper()
	for i := 1; i < len(candidates); i++ {
		if candidates[i-1].Tier > candidates[i].Tier {
			t.Errorf("sort violation at index %d: %s > %s",
				i, candidates[i-1].Tier, candidates[i].Tier)
		}
	}
}

// TestScore_IdealMetalMachine verifies that small models score Ideal on a
// capable Apple Silicon machine and that results are sorted.
func TestScore_IdealMetalMachine(t *testing.T) {
	hw := metalHW(18, 18, 24, 36, 200)
	candidates := fitcheck.Score(hw, fitcheck.Requirements)
	if len(candidates) == 0 {
		t.Fatal("expected candidates")
	}
	if candidates[0].Tier != fitcheck.TierIdeal {
		t.Errorf("expected first candidate to be TierIdeal, got %s (model: %s)",
			candidates[0].Tier, candidates[0].Req.Name)
	}
	assertSorted(t, candidates)
}

// TestScore_NoGPU_SmallModel verifies CPU-only machines report RunMode "CPU".
func TestScore_NoGPU_SmallModel(t *testing.T) {
	hw := cpuHW(16, 16, 200)
	req := fitcheck.ModelRequirement{
		Name: "test:small", VRAMMinMB: 0, RAMMinMB: 3000, DiskSizeMB: 800,
	}
	candidates := fitcheck.Score(hw, []fitcheck.ModelRequirement{req})
	if len(candidates) == 0 {
		t.Fatal("expected candidates")
	}
	if candidates[0].RunMode != "CPU" {
		t.Errorf("expected RunMode=CPU, got %s", candidates[0].RunMode)
	}
}

// TestScore_TooLarge verifies a model that exceeds both RAM and disk is TierTooLarge.
func TestScore_TooLarge(t *testing.T) {
	hw := cpuHW(4, 8, 10) // only 10 GB disk; model needs 40 GB
	req := fitcheck.ModelRequirement{
		Name: "test:huge", VRAMMinMB: 0, RAMMinMB: 64000, DiskSizeMB: 40000,
	}
	candidates := fitcheck.Score(hw, []fitcheck.ModelRequirement{req})
	if len(candidates) == 0 {
		t.Fatal("expected candidates")
	}
	if candidates[0].Tier != fitcheck.TierTooLarge {
		t.Errorf("expected TierTooLarge, got %s", candidates[0].Tier)
	}
}

// TestScore_GPUPlusOffload verifies partial-VRAM models land in GPU+CPU mode.
func TestScore_GPUPlusOffload(t *testing.T) {
	// GPU has 4 GB free; model needs 10 GB VRAM; RAM has 32 GB — offload is possible.
	hw := cudaHW(8, 4, 10, 32, 64, 200)
	req := fitcheck.ModelRequirement{
		Name: "test:offload", VRAMMinMB: 10000, RAMMinMB: 16000, DiskSizeMB: 9000,
	}
	candidates := fitcheck.Score(hw, []fitcheck.ModelRequirement{req})
	if len(candidates) == 0 {
		t.Fatal("expected candidates")
	}
	c := candidates[0]
	if c.RunMode != "GPU+CPU" {
		t.Errorf("expected RunMode=GPU+CPU, got %s", c.RunMode)
	}
	if c.Tier == fitcheck.TierIdeal {
		t.Errorf("partial-offload model should not be TierIdeal")
	}
}

// TestScore_ExactVRAMBoundary verifies a model that exactly fits in VRAM scores GPU.
func TestScore_ExactVRAMBoundary(t *testing.T) {
	const vramMB = uint64(5000)
	hw := cudaHW(8, vramMB/1024+1, vramMB/1024+2, 16, 32, 200)
	req := fitcheck.ModelRequirement{
		Name: "test:exact", VRAMMinMB: vramMB, RAMMinMB: 8000, DiskSizeMB: 4700,
	}
	candidates := fitcheck.Score(hw, []fitcheck.ModelRequirement{req})
	if len(candidates) == 0 {
		t.Fatal("expected candidates")
	}
	if candidates[0].RunMode != "GPU" {
		t.Errorf("expected RunMode=GPU when VRAM just fits, got %s", candidates[0].RunMode)
	}
}

// TestScore_NoDisk verifies that insufficient disk → TierTooLarge regardless of RAM/VRAM.
func TestScore_NoDisk(t *testing.T) {
	hw := metalHW(18, 18, 24, 36, 1) // only 1 GB disk
	req := fitcheck.ModelRequirement{
		Name: "test:nodisk", VRAMMinMB: 2000, RAMMinMB: 4000, DiskSizeMB: 5000,
	}
	candidates := fitcheck.Score(hw, []fitcheck.ModelRequirement{req})
	if len(candidates) == 0 {
		t.Fatal("expected candidates")
	}
	if candidates[0].Tier != fitcheck.TierTooLarge {
		t.Errorf("no-disk model should be TierTooLarge, got %s", candidates[0].Tier)
	}
}

// TestScore_SortStability verifies that within the same tier, higher score comes first.
func TestScore_SortStability(t *testing.T) {
	hw := metalHW(18, 18, 24, 36, 200)
	// Two small models that both score Ideal; the smaller one should score higher.
	reqs := []fitcheck.ModelRequirement{
		{Name: "big:7b", VRAMMinMB: 5000, RAMMinMB: 8000, DiskSizeMB: 4700},
		{Name: "small:1b", VRAMMinMB: 1100, RAMMinMB: 2000, DiskSizeMB: 800},
	}
	candidates := fitcheck.Score(hw, reqs)
	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}
	if candidates[0].Tier != fitcheck.TierIdeal || candidates[1].Tier != fitcheck.TierIdeal {
		t.Skip("both models need to be Ideal for this test to be meaningful")
	}
	if candidates[0].Score < candidates[1].Score {
		t.Errorf("within TierIdeal, higher score should come first: got %.3f before %.3f",
			candidates[0].Score, candidates[1].Score)
	}
}

// TestScore_CUDA_ComputeGenerations verifies that newer CUDA generations yield higher EstTPS.
func TestScore_CUDA_ComputeGenerations(t *testing.T) {
	req := fitcheck.ModelRequirement{
		Name: "test:cuda", VRAMMinMB: 5000, RAMMinMB: 8000, DiskSizeMB: 4700,
	}
	sm7 := fitcheck.Score(cudaHW(7, 12, 12, 32, 64, 200), []fitcheck.ModelRequirement{req})
	sm9 := fitcheck.Score(cudaHW(9, 12, 12, 32, 64, 200), []fitcheck.ModelRequirement{req})

	if len(sm7) == 0 || len(sm9) == 0 {
		t.Fatal("expected candidates")
	}
	if sm9[0].EstTPS <= sm7[0].EstTPS {
		t.Errorf("SM9 should be faster than SM7: got %d vs %d tok/s",
			sm9[0].EstTPS, sm7[0].EstTPS)
	}
}

// TestScore_FullCatalogueNonempty verifies the built-in catalogue produces results.
func TestScore_FullCatalogueNonempty(t *testing.T) {
	hw := cpuHW(8, 16, 500)
	candidates := fitcheck.Score(hw, fitcheck.Requirements)
	if len(candidates) != len(fitcheck.Requirements) {
		t.Errorf("expected %d candidates, got %d", len(fitcheck.Requirements), len(candidates))
	}
	assertSorted(t, candidates)
}

// TestScore_TierStringRoundtrip verifies all tiers have non-empty string representations.
func TestScore_TierStringRoundtrip(t *testing.T) {
	tiers := []fitcheck.CompatibilityTier{
		fitcheck.TierIdeal,
		fitcheck.TierGood,
		fitcheck.TierMarginal,
		fitcheck.TierPossible,
		fitcheck.TierTooLarge,
	}
	for _, tier := range tiers {
		s := tier.String()
		if s == "" || s == "unknown" {
			t.Errorf("tier %d has bad string: %q", tier, s)
		}
	}
}
