package turboquant

import (
	"fmt"
)

const BlockVersion = 6

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
}

var (
	// All four tq* presets ship with OutlierCount=0 (pure uniform Lloyd-Max
	// after Householder QR rotation, i.e. the core of TurboQuant Algorithm 1
	// §3.1 without the optional outlier split from §4.3 or the QJL residual
	// sketch from Algorithm 2). The uniform defaults were chosen after
	// measuring that on the models this fork ships against (llama, gemma3,
	// qwen3-coder), outlier split hurts both decode throughput and PPL — the
	// paper's split targets heavy-tailed rotated K distributions, which these
	// models don't exhibit, and the extra metadata (92 vs 52 bytes/head/cell
	// at oc=32) translates to ~25% decode regression on 3B-class models at
	// short context. Keeping the defaults symmetric across tq2 / tq3 / tq2k /
	// tq3k means the digit in the preset name maps directly to effective
	// bits/elem: "tq3" is exactly 3 bits, not the 3.25 bits you'd get under
	// outlier split with oc=32.
	//
	// The outlier-split kernel path remains in the code (encode/dequant
	// dispatchers check op_params[2..3] and route to the outlier variant when
	// both are non-zero). A future dynamic-dispatch PR can enable it per
	// model / per env var — e.g. for the qwen2 family once Phase 2A
	// asymmetric quantization lands, or for models with larger headDim where
	// the metadata overhead amortizes better. See project_tq_backlog.md for
	// the planned dynamic-oc dispatch work.

	// tq2: 2-bit K + 2-bit V, both rotated and Lloyd-Max quantized. Highest
	// compression tier — effective 2 bits/elem both sides.
	PresetTQ2 = newPreset(1, "tq2", 2, 2, 1, 0x25c0ffee, 3, 0)

	// tq3: 3-bit K + 3-bit V, both rotated and Lloyd-Max quantized. Default
	// "balanced" tier — effective 3 bits/elem both sides.
	PresetTQ3 = newPreset(2, "tq3", 3, 3, 1, 0x35c0ffee, 4, 0)

	// tq3k: 3-bit K only, V stays as f16. ~40% KV VRAM savings with near-f16
	// decode (no V dequant at all). ValueBits=0 signals K-only mode to the
	// kvcache layer.
	PresetTQ3K = newPreset(3, "tq3k", 3, 0, 1, 0x35c0ffee, 4, 0)

	// tq2k: 2-bit K only, V stays as f16. Maximum K compression with f16 V;
	// smallest K footprint before PPL degrades too much. ValueBits=0 signals
	// K-only mode to the kvcache layer.
	PresetTQ2K = newPreset(4, "tq2k", 2, 0, 1, 0x25c0ffee, 3, 0)
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

func (p Preset) HasOutlierSplit() bool {
	return p.OutlierBits > 0 && p.OutlierCount > 0
}

func PresetByName(name string) (Preset, error) {
	switch name {
	case "tq2":
		return PresetTQ2, nil
	case "tq3":
		return PresetTQ3, nil
	case "tq3k":
		return PresetTQ3K, nil
	case "tq2k":
		return PresetTQ2K, nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset %q", name)
	}
}

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

