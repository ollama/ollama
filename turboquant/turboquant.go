package turboquant

import (
	"fmt"
	"os"
)

// BlockVersion 7 introduces two layout changes to the v6 Block format:
//
//  1. ChannelIndices []uint16 → ChannelBitmap []byte. One bit per channel
//     of the full (pre-split) vector, set iff that channel belongs to this
//     sub-block. At headDim=128 with a 32-channel outlier split, 16 bytes
//     of bitmap per sub-block replaces 192 + 64 = 256 bytes of indices —
//     ~240 bytes saved per encoded vector pair, only on the CPU /
//     offline-serialised path used by unit tests and dump-and-replay
//     tooling. The GPU kernel layout amortises outlier indices across
//     cells per-head independently of this format and is unaffected.
//
//  2. New Zero float32 field per Block, supporting centred-asymmetric
//     primary quantization (Preset.AsymmetricPrimary). Symmetric blocks
//     write 0; decoding is unconditional, so symmetric blocks decode
//     bit-identically to v6.
//
// Both changes ship under the same v7 bump because the bitmap rework was
// itself unreleased — there is no intermediate v7-without-Zero in the
// wild that would need back-compat handling.
const BlockVersion = 7

type vectorRole uint8

const (
	roleGeneric vectorRole = iota
	roleKey
	roleValue
)

type vectorObjective uint8

const (
	objectiveMSE vectorObjective = iota + 1
	objectiveProduct
)

type Preset struct {
	ID             uint8
	Name           string
	RotationSeed   uint64
	KeyPrimaryBits int
	ValueBits      int
	QJLRowsDivisor int
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
//   - Householder QR rotation (per-layer, deterministic by RotationSeed)
//   - Centred-asymmetric Lloyd-Max scalar quantization (mean per block,
//     stored in Block.Zero; recovers Qwen 2.x family with learned K bias)
//   - 32-channel outlier split (heavy-tail channels quantized at OutlierBits)
//
// QJL is intentionally not used. Empirically PPL-negative on every model
// tested (llama 3.x, Qwen 2.5/3, gemma 3/4) and the RaBitQ critique
// (arXiv:2604.19528) shows QJL's per-layer cosine ρ ≈ 0.85 decays
// geometrically with depth — fine on 32-layer models but breaks past 60.
// The QJL implementation remains in encoders, kernels, and reference Go
// code so research / regression tests can continue to exercise it via
// directly-constructed Preset{} values; it is not user-reachable.
//
// Subtractive ablation env vars (consumed by ApplyEnvOverrides at the
// PresetByName / PresetFromDType boundary):
//
//	OLLAMA_TQ_DISABLE_OUTLIERS=1  — clears OutlierCount/OutlierBits
//	OLLAMA_TQ_DISABLE_ASYMMETRIC=1 — clears AsymmetricPrimary
//
// Setting both reproduces the pre-refactor "tq3" baseline (symmetric,
// no outliers, no QJL) without needing a separate preset name.
var (
	// tq2 — 2-bit K + 2-bit V. Highest compression tier.
	PresetTQ2 = newAsymmetricPreset(1, "tq2", 2, 2, 0, 0x25c0ffee, 3, 32)

	// tq3 — 3-bit K + 3-bit V. Default balanced tier.
	PresetTQ3 = newAsymmetricPreset(2, "tq3", 3, 3, 0, 0x35c0ffee, 4, 32)

	// tq3k — 3-bit K only, V at f16. ValueBits=0 signals K-only mode.
	PresetTQ3K = newAsymmetricPreset(3, "tq3k", 3, 0, 0, 0x35c0ffee, 4, 32)

	// tq2k — 2-bit K only, V at f16.
	PresetTQ2K = newAsymmetricPreset(4, "tq2k", 2, 0, 0, 0x25c0ffee, 3, 32)

	// tq4 — 4-bit K + 4-bit V. Highest fidelity tier; 16-entry codebook.
	PresetTQ4 = newAsymmetricPreset(25, "tq4", 4, 4, 0, 0x45c0ffee, 5, 32)

	// tq4k — 4-bit K only, V at f16.
	PresetTQ4K = newAsymmetricPreset(26, "tq4k", 4, 0, 0, 0x45c0ffee, 5, 32)
)

func newPreset(id uint8, name string, keyBits int, valueBits int, qjlRowsDivisor int, seed uint64, outlierBits int, outlierCount int) Preset {
	return Preset{
		ID:             id,
		Name:           name,
		RotationSeed:   seed,
		KeyPrimaryBits: keyBits,
		ValueBits:      valueBits,
		QJLRowsDivisor: qjlRowsDivisor,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}
}

// newAsymmetricPreset constructs a preset with the same knobs as newPreset
// plus AsymmetricPrimary=true.
func newAsymmetricPreset(id uint8, name string, keyBits int, valueBits int, qjlRowsDivisor int, seed uint64, outlierBits int, outlierCount int) Preset {
	p := newPreset(id, name, keyBits, valueBits, qjlRowsDivisor, seed, outlierBits, outlierCount)
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

// ApplyEnvOverrides applies subtractive ablation env vars to a preset.
// Used by PresetByName and the kvcache runtime path; tests that need a
// specific configuration should construct Preset{} directly to bypass any
// env state.
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
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset id %d", id)
	}
}

func (p Preset) KeyQJLRows(dim int) int {
	if dim <= 0 {
		return 0
	}
	if p.QJLRowsDivisor <= 0 {
		return 0
	}
	rows := dim / p.QJLRowsDivisor
	if rows < 1 {
		return 1
	}
	return rows
}
