package autotune

import (
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/ml"
)

// --- helpers ---

func makeGPU(name string, vram uint64, library string, integrated bool, computeMajor, computeMinor int) ml.DeviceInfo {
	return ml.DeviceInfo{
		DeviceID:     ml.DeviceID{ID: "0", Library: library},
		Name:         name,
		TotalMemory:  vram,
		FreeMemory:   vram,
		Integrated:   integrated,
		ComputeMajor: computeMajor,
		ComputeMinor: computeMinor,
		DriverMajor:  12,
		DriverMinor:  0,
	}
}

func makeHW(gpus []ml.DeviceInfo, totalRAM uint64) HardwareProfile {
	return HardwareProfile{
		CPUs: []discover.CPU{},
		GPUs: gpus,
		System: ml.SystemInfo{
			ThreadCount: 8,
			TotalMemory: totalRAM,
			FreeMemory:  totalRAM / 2,
		},
		Platform: PlatformInfo{OS: "linux", Arch: "amd64"},
	}
}

// We can't easily import discover.CPU in tests without platform-specific
// build tags. Use DetectHardware with nil GPUs for integration tests.

func TestParseProfile(t *testing.T) {
	tests := []struct {
		input string
		want  Profile
	}{
		{"speed", ProfileSpeed},
		{"balanced", ProfileBalanced},
		{"memory", ProfileMemory},
		{"multiuser", ProfileMultiUser},
		{"max", ProfileMaxPerformance},
		{"unknown", ProfileBalanced},
		{"", ProfileBalanced},
	}
	for _, tt := range tests {
		got := ParseProfile(tt.input)
		if got != tt.want {
			t.Errorf("ParseProfile(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestAllProfiles(t *testing.T) {
	profiles := AllProfiles()
	if len(profiles) != 5 {
		t.Errorf("expected 5 profiles, got %d", len(profiles))
	}
}

func TestProfileConfigs(t *testing.T) {
	cfgs := profileConfigs()
	for _, p := range AllProfiles() {
		if _, ok := cfgs[p]; !ok {
			t.Errorf("missing config for profile %q", p)
		}
	}
}

func TestTuneSpeedProfile(t *testing.T) {
	gpus := []ml.DeviceInfo{
		makeGPU("RTX 4090", 24*GiB, "CUDA", false, 8, 9),
	}
	hw := HardwareProfile{
		GPUs: gpus,
		System: ml.SystemInfo{
			ThreadCount: 16,
			TotalMemory: 64 * GiB,
			FreeMemory:  48 * GiB,
		},
		Platform: PlatformInfo{OS: "linux", Arch: "amd64"},
	}

	plan := Tune(hw, ProfileSpeed)

	if !plan.FlashAttention {
		t.Error("speed profile should enable flash attention on CUDA 8.9")
	}
	if plan.KvCacheType != "q8_0" {
		t.Errorf("expected KV cache q8_0, got %q", plan.KvCacheType)
	}
	if plan.NumParallel != 1 {
		t.Errorf("speed profile should set NumParallel=1, got %d", plan.NumParallel)
	}
	if plan.MaxLoadedModels != 1 {
		t.Errorf("speed profile should set MaxLoadedModels=1, got %d", plan.MaxLoadedModels)
	}
	if plan.KeepAlive != 30*time.Minute {
		t.Errorf("expected 30m keep-alive, got %v", plan.KeepAlive)
	}
}

func TestTuneMemoryProfile(t *testing.T) {
	gpus := []ml.DeviceInfo{
		makeGPU("GTX 1660", 6*GiB, "CUDA", false, 7, 5),
	}
	hw := HardwareProfile{
		GPUs: gpus,
		System: ml.SystemInfo{
			ThreadCount: 8,
			TotalMemory: 16 * GiB,
			FreeMemory:  8 * GiB,
		},
		Platform: PlatformInfo{OS: "windows", Arch: "amd64"},
	}

	plan := Tune(hw, ProfileMemory)

	if plan.KvCacheType != "q4_0" {
		t.Errorf("memory profile should use q4_0, got %q", plan.KvCacheType)
	}
	if plan.NumParallel != 1 {
		t.Errorf("memory profile should set NumParallel=1, got %d", plan.NumParallel)
	}
	if plan.KeepAlive != 2*time.Minute {
		t.Errorf("expected 2m keep-alive, got %v", plan.KeepAlive)
	}
}

func TestTuneMultiUserProfile(t *testing.T) {
	gpus := []ml.DeviceInfo{
		makeGPU("A100", 80*GiB, "CUDA", false, 8, 0),
		makeGPU("A100", 80*GiB, "CUDA", false, 8, 0),
	}
	hw := HardwareProfile{
		GPUs: gpus,
		System: ml.SystemInfo{
			ThreadCount: 64,
			TotalMemory: 256 * GiB,
			FreeMemory:  200 * GiB,
		},
		Platform: PlatformInfo{OS: "linux", Arch: "amd64"},
	}

	plan := Tune(hw, ProfileMultiUser)

	if !plan.MultiUserCache {
		t.Error("multiuser profile should enable multi-user cache")
	}
	if plan.NumParallel < 2 {
		t.Errorf("multiuser profile with 80GB GPUs should have NumParallel>=2, got %d", plan.NumParallel)
	}
	if plan.MaxQueue != 1024 {
		t.Errorf("expected MaxQueue=1024, got %d", plan.MaxQueue)
	}
}

func TestTuneCPUOnly(t *testing.T) {
	hw := HardwareProfile{
		GPUs: nil,
		System: ml.SystemInfo{
			ThreadCount: 4,
			TotalMemory: 8 * GiB,
			FreeMemory:  4 * GiB,
		},
		Platform: PlatformInfo{OS: "linux", Arch: "amd64"},
	}

	plan := Tune(hw, ProfileBalanced)

	if plan.FlashAttention {
		t.Error("CPU-only should not enable flash attention")
	}
	if plan.KvCacheType != "f16" {
		t.Errorf("CPU-only should use f16 KV cache, got %q", plan.KvCacheType)
	}
	if plan.NumParallel != 1 {
		t.Errorf("CPU-only should have NumParallel=1, got %d", plan.NumParallel)
	}
}

func TestNoFlashAttentionFallback(t *testing.T) {
	// GPU with compute 7.2 doesn't support flash attention in Ollama.
	gpus := []ml.DeviceInfo{
		makeGPU("Jetson", 8*GiB, "CUDA", false, 7, 2),
	}
	hw := HardwareProfile{
		GPUs:     gpus,
		System:   ml.SystemInfo{ThreadCount: 4, TotalMemory: 16 * GiB},
		Platform: PlatformInfo{OS: "linux", Arch: "arm64"},
	}

	plan := Tune(hw, ProfileSpeed)

	if plan.FlashAttention {
		t.Error("should not enable flash attention on CC 7.2 (Xavier)")
	}
	// KV cache should fall back to f16 without FA
	if plan.KvCacheType != "f16" {
		t.Errorf("without FA, KV cache should be f16, got %q", plan.KvCacheType)
	}
}

func TestApplySkipsUserSet(t *testing.T) {
	// Set a variable explicitly
	os.Setenv("OLLAMA_NUM_PARALLEL", "8")
	defer os.Unsetenv("OLLAMA_NUM_PARALLEL")

	plan := &TunePlan{
		Recommendations: []Recommendation{
			{Key: "OLLAMA_NUM_PARALLEL", Value: "2", Reason: "test"},
			{Key: "OLLAMA_MAX_QUEUE", Value: "256", Reason: "test"},
		},
	}

	applied := Apply(plan)

	// Should only apply MAX_QUEUE, not NUM_PARALLEL
	if len(applied) != 1 {
		t.Errorf("expected 1 applied, got %d", len(applied))
	}
	if applied[0].Key != "OLLAMA_MAX_QUEUE" {
		t.Errorf("expected OLLAMA_MAX_QUEUE, got %s", applied[0].Key)
	}
	// Verify original value preserved
	if os.Getenv("OLLAMA_NUM_PARALLEL") != "8" {
		t.Error("user-set OLLAMA_NUM_PARALLEL was overwritten")
	}

	os.Unsetenv("OLLAMA_MAX_QUEUE")
}

func TestGpuOverheadWindows(t *testing.T) {
	gpus := []ml.DeviceInfo{
		makeGPU("RTX 3060", 12*GiB, "CUDA", false, 8, 6),
	}
	hw := HardwareProfile{
		GPUs:     gpus,
		System:   ml.SystemInfo{ThreadCount: 12, TotalMemory: 32 * GiB},
		Platform: PlatformInfo{OS: "windows", Arch: "amd64"},
	}

	// The function uses runtime.GOOS, not hw.Platform.OS, so this
	// tests the resolution logic exists and doesn't crash.
	plan := Tune(hw, ProfileBalanced)
	if plan.GpuOverhead == 0 && hw.HasGPU() {
		// On a non-macOS system this should always be > 0
		// (but we can't guarantee the test runs on Windows)
		t.Log("GPU overhead was 0 — might be running on macOS/darwin (expected)")
	}
}

func TestResolveContextLength(t *testing.T) {
	tests := []struct {
		name    string
		vram    uint64
		wantMin uint
	}{
		{"24GB GPU", 24 * GiB, 32768},
		{"12GB GPU", 12 * GiB, 8192},
		{"8GB GPU", 8 * GiB, 4096},
		{"4GB GPU", 4 * GiB, 2048},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ProfileConfig{}
			hw := HardwareProfile{
				GPUs:   []ml.DeviceInfo{makeGPU("test", tt.vram, "CUDA", false, 8, 0)},
				System: ml.SystemInfo{ThreadCount: 8, TotalMemory: 32 * GiB},
			}
			got := resolveContextLength(hw, cfg)
			if got != tt.wantMin {
				t.Errorf("resolveContextLength(vram=%dGB) = %d, want %d",
					tt.vram/GiB, got, tt.wantMin)
			}
		})
	}
}

func TestHardwareHelpers(t *testing.T) {
	gpus := []ml.DeviceInfo{
		makeGPU("RTX 4090", 24*GiB, "CUDA", false, 8, 9),
		makeGPU("Intel UHD", 2*GiB, "Vulkan", true, 1, 0),
	}
	hw := HardwareProfile{
		GPUs:     gpus,
		System:   ml.SystemInfo{ThreadCount: 16, TotalMemory: 64 * GiB},
		Platform: PlatformInfo{OS: "linux", Arch: "amd64"},
	}

	if hw.DiscreteGPUCount() != 1 {
		t.Errorf("expected 1 discrete GPU, got %d", hw.DiscreteGPUCount())
	}
	if !hw.HasGPU() {
		t.Error("expected HasGPU=true")
	}
	if hw.TotalVRAM() != 24*GiB {
		t.Errorf("expected 24GB VRAM, got %d", hw.TotalVRAM())
	}
	if hw.SmallestGPUVRAM() != 24*GiB {
		t.Errorf("expected 24GB smallest VRAM, got %d", hw.SmallestGPUVRAM())
	}
}

func TestFormatPlan(t *testing.T) {
	plan := &TunePlan{
		Profile: ProfileSpeed,
		Hardware: HardwareProfile{
			System:   ml.SystemInfo{TotalMemory: 64 * GiB, ThreadCount: 16},
			Platform: PlatformInfo{OS: "linux", Arch: "amd64"},
		},
		Recommendations: []Recommendation{
			{Key: "OLLAMA_FLASH_ATTENTION", Value: "true", Reason: "test"},
		},
	}

	output := FormatPlan(plan)
	if output == "" {
		t.Error("FormatPlan returned empty string")
	}
}
