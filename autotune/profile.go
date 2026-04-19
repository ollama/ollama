// Package autotune provides automatic hardware detection and performance
// optimization for Ollama. It analyzes the available CPU, GPU, and memory
// resources and generates optimal configuration values for each machine.
package autotune

import "time"

// Profile represents a performance tuning profile.
type Profile string

const (
	// ProfileSpeed maximizes token generation speed at the cost of more
	// VRAM/RAM usage. Best for single-user local setups.
	ProfileSpeed Profile = "speed"

	// ProfileBalanced provides a good tradeoff between speed and resource
	// usage. This is the default profile.
	ProfileBalanced Profile = "balanced"

	// ProfileMemory minimizes memory usage so larger models can be loaded
	// on constrained hardware.
	ProfileMemory Profile = "memory"

	// ProfileMultiUser optimizes for concurrent users with shared cache
	// and multiple parallel slots.
	ProfileMultiUser Profile = "multiuser"

	// ProfileMaxPerformance squeezes every bit of performance: aggressive
	// KV cache quantization, flash attention, full GPU offload, large
	// batch sizes.
	ProfileMaxPerformance Profile = "max"
)

// AllProfiles returns every recognised profile.
func AllProfiles() []Profile {
	return []Profile{
		ProfileSpeed,
		ProfileBalanced,
		ProfileMemory,
		ProfileMultiUser,
		ProfileMaxPerformance,
	}
}

// ParseProfile converts a string to a Profile. Returns ProfileBalanced for
// unknown values.
func ParseProfile(s string) Profile {
	switch Profile(s) {
	case ProfileSpeed, ProfileBalanced, ProfileMemory, ProfileMultiUser, ProfileMaxPerformance:
		return Profile(s)
	default:
		return ProfileBalanced
	}
}

// ProfileConfig holds the tuning knobs for a profile. Zero/empty values mean
// "keep the auto-detected default".
type ProfileConfig struct {
	// FlashAttention preference: nil = auto, true = force on, false = force off.
	FlashAttention *bool

	// KvCacheType override (e.g. "f16", "q8_0", "q4_0"). Empty = auto.
	KvCacheType string

	// NumParallel override. 0 = auto.
	NumParallel uint

	// MaxLoadedModels override. 0 = auto.
	MaxLoadedModels uint

	// KeepAlive override. 0 = auto.
	KeepAlive time.Duration

	// GpuOverhead bytes to reserve per GPU. 0 = auto-detect.
	GpuOverhead uint64

	// MaxQueue depth. 0 = default.
	MaxQueue uint

	// MultiUserCache enables shared prompt caching.
	MultiUserCache *bool

	// BatchSize override. 0 = auto.
	BatchSize int

	// ContextLength override. 0 = auto.
	ContextLength uint

	// SchedSpread forces multi-GPU spreading. nil = auto.
	SchedSpread *bool
}

func boolPtr(b bool) *bool { return &b }

// profileConfigs returns the static profile settings. Dynamic values are
// computed at runtime by the Tuner based on actual hardware.
func profileConfigs() map[Profile]ProfileConfig {
	return map[Profile]ProfileConfig{
		ProfileSpeed: {
			FlashAttention:  boolPtr(true),
			KvCacheType:     "q8_0",
			NumParallel:     1,
			MaxLoadedModels: 1,
			KeepAlive:       30 * time.Minute,
			MaxQueue:        256,
			MultiUserCache:  boolPtr(false),
		},
		ProfileBalanced: {
			FlashAttention:  boolPtr(true),
			KvCacheType:     "q8_0",
			NumParallel:     0, // auto
			MaxLoadedModels: 0,
			KeepAlive:       5 * time.Minute,
			MaxQueue:        512,
		},
		ProfileMemory: {
			FlashAttention:  boolPtr(true),
			KvCacheType:     "q4_0",
			NumParallel:     1,
			MaxLoadedModels: 1,
			KeepAlive:       2 * time.Minute,
			MaxQueue:        128,
			MultiUserCache:  boolPtr(false),
			SchedSpread:     boolPtr(false),
		},
		ProfileMultiUser: {
			FlashAttention:  boolPtr(true),
			KvCacheType:     "q8_0",
			NumParallel:     0, // auto: VRAM-based
			MaxLoadedModels: 0,
			KeepAlive:       10 * time.Minute,
			MaxQueue:        1024,
			MultiUserCache:  boolPtr(true),
		},
		ProfileMaxPerformance: {
			FlashAttention:  boolPtr(true),
			KvCacheType:     "q4_0",
			NumParallel:     1,
			MaxLoadedModels: 1,
			KeepAlive:       60 * time.Minute,
			MaxQueue:        256,
			MultiUserCache:  boolPtr(false),
		},
	}
}
