package turboquant

import (
	"fmt"
	"os"
)

// BlockVersion is the on-the-wire Block format version produced by the CPU
// reference encoder and consumed by the dump-and-replay tooling. The GPU
// kernels do not serialise blocks; they read/write packed tensors directly.
// Bump this any time the Block struct (block.go) or its MarshalBinary /
// UnmarshalBinary wire layout changes, and update the decoder's accepted
// versions accordingly.
const BlockVersion = 1

type Preset struct {
	ID             uint8
	Name           string
	RotationSeed   uint64
	KeyPrimaryBits int
	ValueBits      int
	OutlierBits    int
	OutlierCount   int
	// AsymmetricPrimary, when true, centres each per-block rotated vector by
	// its mean before scalar quantization and stores that mean in Block.Zero.
	// Decoding unconditionally adds Zero back. Targets models whose learned
	// K bias produces a non-zero-mean rotated distribution (Qwen 2 family).
	AsymmetricPrimary bool
}

// Ship presets — the only TurboQuant configurations exposed via
// OLLAMA_KV_CACHE_TYPE. Each enables:
//
//   - Walsh-Hadamard rotation (per-layer, deterministic by RotationSeed)
//   - Centred-asymmetric Lloyd-Max scalar quantization (mean per block,
//     stored in Block.Zero; recovers families with learned K bias such as
//     Qwen 2.x)
//   - 32-channel outlier split (heavy-tail channels quantized at OutlierBits)
//
// Ablation env vars (consumed by ApplyEnvOverrides at the PresetByName /
// PresetFromDType boundary):
//
//	OLLAMA_TQ_DISABLE_OUTLIERS=1   — clears OutlierCount/OutlierBits
//	OLLAMA_TQ_DISABLE_ASYMMETRIC=1 — clears AsymmetricPrimary
//
// Setting both produces a symmetric, no-outlier baseline of the same bit
// width — useful for ablation against the full configuration.
var (
	// tq2 — 2-bit K + 2-bit V. Highest compression tier.
	PresetTQ2 = newAsymmetricPreset(1, "tq2", 2, 2, 0x25c0ffee, 3, 32)

	// tq3 — 3-bit K + 3-bit V. Default balanced tier.
	PresetTQ3 = newAsymmetricPreset(2, "tq3", 3, 3, 0x35c0ffee, 4, 32)

	// tq3k — 3-bit K only, V at f16. ValueBits=0 signals K-only mode.
	PresetTQ3K = newAsymmetricPreset(3, "tq3k", 3, 0, 0x35c0ffee, 4, 32)

	// tq2k — 2-bit K only, V at f16.
	PresetTQ2K = newAsymmetricPreset(4, "tq2k", 2, 0, 0x25c0ffee, 3, 32)

	// tq4 — 4-bit K + 4-bit V. Highest fidelity tier; 16-entry codebook.
	PresetTQ4 = newAsymmetricPreset(25, "tq4", 4, 4, 0x45c0ffee, 5, 32)

	// tq4k — 4-bit K only, V at f16.
	PresetTQ4K = newAsymmetricPreset(26, "tq4k", 4, 0, 0x45c0ffee, 5, 32)

	// V-only presets: K stays as raw f16, V is TQ-compressed. For models
	// whose K distribution doesn't quantize well (e.g. unusually wide
	// channel statistics) but whose V is well-behaved. KeyPrimaryBits=0
	// signals V-only mode. Outlier knobs are reused for V's own
	// outlier-split path; ValueBits is the primary V bit width.

	// tq2v — 2-bit V only, K at f16.
	PresetTQ2V = newAsymmetricPreset(27, "tq2v", 0, 2, 0x25c0ffee, 3, 32)

	// tq3v — 3-bit V only, K at f16.
	PresetTQ3V = newAsymmetricPreset(28, "tq3v", 0, 3, 0x35c0ffee, 4, 32)

	// tq4v — 4-bit V only, K at f16.
	PresetTQ4V = newAsymmetricPreset(29, "tq4v", 0, 4, 0x45c0ffee, 5, 32)
)

func newPreset(id uint8, name string, keyBits int, valueBits int, seed uint64, outlierBits int, outlierCount int) Preset {
	return Preset{
		ID:             id,
		Name:           name,
		RotationSeed:   seed,
		KeyPrimaryBits: keyBits,
		ValueBits:      valueBits,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}
}

// newAsymmetricPreset constructs a preset with the same knobs as newPreset
// plus AsymmetricPrimary=true.
func newAsymmetricPreset(id uint8, name string, keyBits int, valueBits int, seed uint64, outlierBits int, outlierCount int) Preset {
	p := newPreset(id, name, keyBits, valueBits, seed, outlierBits, outlierCount)
	p.AsymmetricPrimary = true
	return p
}

func (p Preset) HasOutlierSplit() bool {
	return p.OutlierBits > 0 && p.OutlierCount > 0
}

// HasAsymmetricPrimary reports whether this preset uses centred-asymmetric
// primary quantization (mean offset per block, stored in Block.Zero) rather
// than the default symmetric path.
func (p Preset) HasAsymmetricPrimary() bool {
	return p.AsymmetricPrimary
}

// ApplyEnvOverrides applies ablation env vars to a preset. Used by
// PresetByName and the kvcache runtime path; tests that need a specific
// configuration should construct Preset{} directly to bypass any env
// state.
func ApplyEnvOverrides(p Preset) Preset {
	if os.Getenv("OLLAMA_TQ_DISABLE_OUTLIERS") == "1" {
		p.OutlierBits = 0
		p.OutlierCount = 0
	}
	if os.Getenv("OLLAMA_TQ_DISABLE_ASYMMETRIC") == "1" {
		p.AsymmetricPrimary = false
	}
	return p
}

// PresetByName resolves a user-facing preset string (the values an end
// user can pass via OLLAMA_KV_CACHE_TYPE). Six TurboQuant tiers
// (tq2/tq3/tq4 plus K-only siblings tq2k/tq3k/tq4k).
//
// Returns are post-env-override (see ApplyEnvOverrides). Tests that need
// a specific configuration should construct Preset{} directly so env
// state can't perturb them.
func PresetByName(name string) (Preset, error) {
	switch name {
	case "tq2":
		return ApplyEnvOverrides(PresetTQ2), nil
	case "tq3":
		return ApplyEnvOverrides(PresetTQ3), nil
	case "tq3k":
		return ApplyEnvOverrides(PresetTQ3K), nil
	case "tq2k":
		return ApplyEnvOverrides(PresetTQ2K), nil
	case "tq4":
		return ApplyEnvOverrides(PresetTQ4), nil
	case "tq4k":
		return ApplyEnvOverrides(PresetTQ4K), nil
	case "tq2v":
		return ApplyEnvOverrides(PresetTQ2V), nil
	case "tq3v":
		return ApplyEnvOverrides(PresetTQ3V), nil
	case "tq4v":
		return ApplyEnvOverrides(PresetTQ4V), nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset %q", name)
	}
}

// PresetByID resolves a Preset by its ID byte (used in serialised blocks
// and the DType enum). Returns the as-defined preset without env-var
// overrides — IDs identify the wire-encoded format, which doesn't shift
// with runtime knobs.
func PresetByID(id uint8) (Preset, error) {
	switch id {
	case PresetTQ2.ID:
		return PresetTQ2, nil
	case PresetTQ3.ID:
		return PresetTQ3, nil
	case PresetTQ3K.ID:
		return PresetTQ3K, nil
	case PresetTQ2K.ID:
		return PresetTQ2K, nil
	case PresetTQ4.ID:
		return PresetTQ4, nil
	case PresetTQ4K.ID:
		return PresetTQ4K, nil
	case PresetTQ2V.ID:
		return PresetTQ2V, nil
	case PresetTQ3V.ID:
		return PresetTQ3V, nil
	case PresetTQ4V.ID:
		return PresetTQ4V, nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset id %d", id)
	}
}
