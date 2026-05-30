package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

// Rotation seed modifiers for outlier and regular sub-blocks.
// Using ASCII mnemonics: "OUTL1ER\0" and "REGULAR\0".
const (
	outlierSeedXOR = uint64(0x4f55544c31455200)
	regularSeedXOR = uint64(0x524547554c415200)
)

type EncodedVector struct {
	Version uint8
	Preset  Preset
	Dim     int
	Blocks  []Block
}

// EncodeVector / EncodeKeyVector / EncodeValueVector are thin wrappers that
// dispatch on bit-width: K uses preset.KeyPrimaryBits, V/Generic use
// preset.ValueBits. The three names survive for backward source-API
// compatibility with existing tests; they all flow through encodeVector now
// that the role/objective metadata they used to tag is gone (see Block doc
// comment for the removal history).
func EncodeVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, preset.ValueBits)
}

func EncodeKeyVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, preset.KeyPrimaryBits)
}

func EncodeValueVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, preset.ValueBits)
}

func encodeVector(values []float32, preset Preset, bits int) (EncodedVector, error) {
	dim := len(values)
	if dim <= 0 {
		return EncodedVector{}, fmt.Errorf("invalid turboquant vector dim %d", dim)
	}
	if bits <= 0 || bits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid turboquant bit width %d", bits)
	}

	if preset.HasOutlierSplit() && dim > preset.OutlierCount {
		return encodeVectorWithOutliers(values, preset, bits)
	}

	block, err := encodeSubBlock(values, nil, preset, bits, preset.RotationSeed)
	if err != nil {
		return EncodedVector{}, err
	}
	return EncodedVector{
		Version: BlockVersion,
		Preset:  preset,
		Dim:     dim,
		Blocks:  []Block{block},
	}, nil
}

// encodeVectorWithOutliers splits the vector into outlier and regular channel
// sub-blocks, each encoded independently with its own rotation. The outlier
// block is stored first in Blocks.
func encodeVectorWithOutliers(values []float32, preset Preset, regularBits int) (EncodedVector, error) {
	split := SplitOutlierChannels(values, preset.OutlierCount)

	outlierBits := preset.OutlierBits
	if outlierBits <= 0 || outlierBits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid outlier bit width %d", outlierBits)
	}
	if regularBits <= 0 || regularBits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid regular bit width %d", regularBits)
	}

	outlierSeed := preset.RotationSeed ^ outlierSeedXOR
	regularSeed := preset.RotationSeed ^ regularSeedXOR

	fullDim := len(values)
	outlierBlock, err := encodeSubBlock(split.OutlierValues, channelBitmapFromIndices(fullDim, split.OutlierIndices), preset, outlierBits, outlierSeed)
	if err != nil {
		return EncodedVector{}, fmt.Errorf("outlier block: %w", err)
	}

	regularBlock, err := encodeSubBlock(split.RegularValues, channelBitmapFromIndices(fullDim, split.RegularIndices), preset, regularBits, regularSeed)
	if err != nil {
		return EncodedVector{}, fmt.Errorf("regular block: %w", err)
	}

	return EncodedVector{
		Version: BlockVersion,
		Preset:  preset,
		Dim:     len(values),
		Blocks:  []Block{outlierBlock, regularBlock},
	}, nil
}

// encodeSubBlock encodes a sub-vector as a single Block. If channelBitmap is
// nil the block covers all channels of the encoded vector (single-block
// path); otherwise the set bits identify which channels of the full vector
// this sub-block's data maps to.
func encodeSubBlock(values []float32, channelBitmap []byte, preset Preset, bits int, rotationSeed uint64) (Block, error) {
	dim := len(values)
	if dim <= 0 {
		return Block{}, fmt.Errorf("empty sub-block")
	}

	codebook, boundaries := scalarCodebook(dim, bits)
	rotation := BuildRotation(dim, rotationSeed)
	rotated := ApplyRotation(values, rotation)

	// Dispatch on preset.HasAsymmetricPrimary():
	//   symmetric: zero = 0,            scale = RMS(rotated)
	//   asymmetric: zero = mean(rotated), scale = RMS(rotated - zero)
	// Downstream quantization normalises by (value - zero) / scale in both
	// cases; for the symmetric path zero=0 so this collapses to the original
	// value / scale formula.
	var (
		zero  float32
		scale float32
	)
	if preset.HasAsymmetricPrimary() {
		zero, scale = asymmetricBlockStats(rotated)
	} else {
		scale = blockScale(rotated)
	}

	primaryCodes := make([]uint8, dim)
	if scale == 0 {
		for i := range primaryCodes {
			primaryCodes[i] = quantizeScalarByBoundary(0, codebook, boundaries)
		}
	} else {
		for i, value := range rotated {
			normalized := (value - zero) / scale
			primaryCodes[i] = quantizeScalarByBoundary(normalized, codebook, boundaries)
		}
		// EDEN biased scale refinement — uses centered values (rotated[i] - zero).
		scale = edenRefineScale(rotated, primaryCodes, codebook, boundaries, scale, zero)
	}

	block := Block{
		Version:        BlockVersion,
		PresetID:       preset.ID,
		OriginalDim:    uint16(dim),
		BlockDim:       uint16(dim),
		RegularBits:    uint8(bits),
		RotationSeed:   rotationSeed,
		ChannelBitmap:  channelBitmap,
		Scale:          scale,
		Zero:           zero,
		RegularIndices: packBits(primaryCodes, bits),
	}
	return block, nil
}

// EncodeKeyPerHead quantizes a single attention head's key vector using
// rotation + Lloyd-Max without outlier split. Returns
// packed N-bit indices and an RMS scale. The quantized values are in
// rotated space — the caller must rotate Q to match at attention time.
//
// This produces a GPU-friendly flat representation: just packed bits + scale.
func EncodeKeyPerHead(values []float32, preset Preset) (packedIndices []byte, scale float32, err error) {
	dim := len(values)
	if dim <= 0 {
		return nil, 0, fmt.Errorf("empty head vector")
	}
	bits := preset.KeyPrimaryBits
	if bits <= 0 || bits >= 8 {
		return nil, 0, fmt.Errorf("invalid bit width %d", bits)
	}

	codebook, boundaries := scalarCodebook(dim, bits)
	rotation := BuildRotation(dim, preset.RotationSeed)
	rotated := ApplyRotation(values, rotation)
	scale = blockScale(rotated)

	codes := make([]uint8, dim)
	if scale > 0 {
		for i, v := range rotated {
			codes[i] = quantizeScalarByBoundary(v/scale, codebook, boundaries)
		}
	}

	// EDEN biased scale refinement (Option B: two-pass).
	// Given assignment {codes[i]}, the MSE-optimal scale is
	// S* = Σ(v[i]·c[codes[i]]) / Σ(c[codes[i]]²).
	scale = edenRefineScale(rotated, codes, codebook, boundaries, scale, 0)

	return packBits(codes, bits), scale, nil
}

// DequantKeyPerHead reconstructs f32 values from packed indices + scale in
// rotated space (no inverse rotation). Used for CPU-side testing/fallback.
func DequantKeyPerHead(packedIndices []byte, scale float32, headDim, bits int) []float32 {
	codebook, _ := scalarCodebook(headDim, bits)
	indices := unpackBits(packedIndices, bits, headDim)
	out := make([]float32, headDim)
	for i, idx := range indices {
		out[i] = dequantizeScalar(idx, codebook) * scale
	}
	return out
}

// OutlierPerHead is the CPU reference for TurboQuant paper Algorithm 1
// Sec 4.3's outlier-split K encoding. Mirrors the GPU kernel
// tq_encode_kernel_outlier / tq_dequant_multihead_kernel_outlier in
// ml/backend/ggml/ggml/src/ggml-cuda/tq-*.cu:
//
//  1. Rotate K by Householder QR of a random Gaussian matrix (shared
//     rotation, same as EncodeKeyPerHead).
//  2. Select the top-K channels by absolute rotated magnitude as
//     outliers.
//  3. Compute independent RMS scales for regular and outlier sub-blocks.
//  4. Quantize each sub-block with its own Lloyd-Max codebook at the
//     preset's primary bits (regular) and outlier bits (outlier).
//  5. Return packed regular + packed outlier streams, per-sub-block
//     scales, and the channel index list.
//
// Unlike EncodeKeyVector / encodeVectorWithOutliers, the split happens
// in ROTATED space (single rotation matmul) because that is what the
// GPU can do cheaply and what the paper's symmetric-rotation formulation
// assumes. The Go reference exists to validate the GPU kernel
// bit-exactly, not to reproduce the block-protocol outlier path.
type OutlierPerHead struct {
	RegularPacked  []byte
	RegularScale   float32
	RegularZero    float32 // per-sub-block mean (asymmetric primary); 0 in symmetric mode
	OutlierPacked  []byte
	OutlierScale   float32
	OutlierZero    float32 // per-sub-block mean (asymmetric primary); 0 in symmetric mode
	OutlierIndices []int
}

func EncodeKeyPerHeadOutlier(values []float32, preset Preset) (OutlierPerHead, error) {
	dim := len(values)
	if dim <= 0 {
		return OutlierPerHead{}, fmt.Errorf("empty head vector")
	}
	bits := preset.KeyPrimaryBits
	if bits <= 0 || bits >= 8 {
		return OutlierPerHead{}, fmt.Errorf("invalid regular bit width %d", bits)
	}
	outlierBits := preset.OutlierBits
	outlierCount := preset.OutlierCount
	// Reject outlierCount==dim here even though SplitOutlierChannels handles
	// the all-outlier case, because this per-head GPU-reference encoder also
	// emits a regular sub-block (regularCount = dim - outlierCount). At
	// regularCount == 0 the regular packed buffer would have zero bytes and
	// the matching GPU dequant kernel asserts on a non-zero regular stride.
	// The block-protocol path (SplitOutlierChannels) doesn't have that
	// constraint — it stores everything as outlier indices.
	if outlierBits <= 0 || outlierBits >= 8 || outlierCount <= 0 || outlierCount >= dim {
		return OutlierPerHead{}, fmt.Errorf("invalid outlier split %d@%dbits over dim=%d", outlierCount, outlierBits, dim)
	}
	regularCount := dim - outlierCount

	rotation := BuildRotation(dim, preset.RotationSeed)
	rotated := ApplyRotation(values, rotation)

	// Step 2: top-K outlier select by abs(rotated). Mirrors the serial
	// thread-0 selection in tq_encode_kernel_outlier step 4. We use the
	// same "mark-and-scan" algorithm so the outlier set order matches
	// the GPU kernel exactly for a given input.
	isOutlier := make([]bool, dim)
	outlierPos := make([]int, 0, outlierCount)
	outlierVal := make([]float32, 0, outlierCount)
	for range outlierCount {
		bestVal := float32(-1.0)
		bestIdx := 0
		for i := range dim {
			if isOutlier[i] {
				continue
			}
			a := rotated[i]
			if a < 0 {
				a = -a
			}
			if a > bestVal {
				bestVal = a
				bestIdx = i
			}
		}
		isOutlier[bestIdx] = true
		outlierPos = append(outlierPos, bestIdx)
		outlierVal = append(outlierVal, rotated[bestIdx])
	}

	// Build the regular channel position list in ascending order. The
	// GPU kernel walks i=0..dim-1 and appends non-outlier positions to
	// s_reg_pos in order; the CPU reference does the same so packed
	// regular slot r maps to the same original channel on both sides.
	regularRotated := make([]float32, 0, regularCount)
	for i := range dim {
		if !isOutlier[i] {
			regularRotated = append(regularRotated, rotated[i])
		}
	}

	// Per-sub-block stats: mean (asymmetric only) + scale.
	// Mirrors tq-encode.cu Step 5: regular sub-block first, then outlier
	// sub-block. Symmetric path uses RMS scale and zero mean.
	var regularZero, outlierZero float32
	var regularScale, outlierScale float32
	if preset.HasAsymmetricPrimary() {
		regularZero, regularScale = asymmetricBlockStats(regularRotated)
		outlierZero, outlierScale = asymmetricBlockStats(outlierVal)
	} else {
		regularScale = blockScale(regularRotated)
		outlierScale = blockScale(outlierVal)
	}

	regularCodebook, regularBoundaries := scalarCodebook(dim, bits)
	outlierCodebook, outlierBoundaries := scalarCodebook(dim, outlierBits)

	regularCodes := make([]uint8, regularCount)
	if regularScale > 0 {
		for r, v := range regularRotated {
			regularCodes[r] = quantizeScalarByBoundary((v-regularZero)/regularScale, regularCodebook, regularBoundaries)
		}
		// Path B (adaptive RMS-vs-EDEN): try EDEN refinement, keep whichever
		// of RMS-only or EDEN-refined has lower per-cell reconstruction MSE.
		// Matches tq-encode.cu's Step 5b/8 — provably non-worse than RMS,
		// and necessary for the GPU↔CPU comparison to converge on inputs
		// where EDEN happens to overfit the codebook.
		regularScale = pathBRefineScale(regularRotated, regularCodes, regularCodebook, regularBoundaries, regularScale, regularZero)
	}
	outlierCodes := make([]uint8, outlierCount)
	if outlierScale > 0 {
		for r, v := range outlierVal {
			outlierCodes[r] = quantizeScalarByBoundary((v-outlierZero)/outlierScale, outlierCodebook, outlierBoundaries)
		}
		outlierScale = pathBRefineScale(outlierVal, outlierCodes, outlierCodebook, outlierBoundaries, outlierScale, outlierZero)
	}

	return OutlierPerHead{
		RegularPacked:  packBits(regularCodes, bits),
		RegularScale:   regularScale,
		RegularZero:    regularZero,
		OutlierPacked:  packBits(outlierCodes, outlierBits),
		OutlierScale:   outlierScale,
		OutlierZero:    outlierZero,
		OutlierIndices: outlierPos,
	}, nil
}

// DequantKeyPerHeadOutlier is the CPU reference for the outlier-split
// dequant. Returns the reconstructed rotated-space vector that the
// caller can compare against ApplyRotation(original, rotation).
func DequantKeyPerHeadOutlier(enc OutlierPerHead, preset Preset, headDim int) []float32 {
	outlierCount := len(enc.OutlierIndices)
	regularCount := headDim - outlierCount
	bits := preset.KeyPrimaryBits
	outlierBits := preset.OutlierBits

	regularCodebook, _ := scalarCodebook(headDim, bits)
	outlierCodebook, _ := scalarCodebook(headDim, outlierBits)

	regularIdx := unpackBits(enc.RegularPacked, bits, regularCount)
	outlierIdx := unpackBits(enc.OutlierPacked, outlierBits, outlierCount)

	// Build position-to-slot lookups. outlier_slot_for[i] is the index
	// in OutlierIndices that equals i (or -1 if i is regular).
	outlierSlotFor := make([]int, headDim)
	for i := range outlierSlotFor {
		outlierSlotFor[i] = -1
	}
	for slot, pos := range enc.OutlierIndices {
		outlierSlotFor[pos] = slot
	}

	out := make([]float32, headDim)
	regPos := 0
	for i := range headDim {
		if outlierSlotFor[i] >= 0 {
			out[i] = dequantizeScalar(outlierIdx[outlierSlotFor[i]], outlierCodebook)*enc.OutlierScale + enc.OutlierZero
		} else {
			out[i] = dequantizeScalar(regularIdx[regPos], regularCodebook)*enc.RegularScale + enc.RegularZero
			regPos++
		}
	}
	return out
}

// edenRefineScale applies two-pass EDEN biased scale refinement (Option B).
// Given current assignment codes[], the MSE-optimal scale is
// S* = Σ((v[i]-zero)·c[codes[i]]) / Σ(c[codes[i]]²).
// Pass 1: compute S1, re-quantize with S1.  Pass 2: compute S2, return S2.
// Falls back to the initial scale if S* is non-positive or denominator is tiny.
func edenRefineScale(rotated []float32, codes []uint8, codebook []float32, boundaries []float32, scale, zero float32) float32 {
	for range 2 {
		var num, den float64
		for i, code := range codes {
			ci := float64(codebook[code])
			vi := float64(rotated[i]) - float64(zero)
			num += vi * ci
			den += ci * ci
		}
		if den < 1e-12 || num <= 0 {
			break
		}
		scale = float32(num / den)
		// Re-quantize with refined scale for the next pass.
		for i, v := range rotated {
			codes[i] = quantizeScalarByBoundary((v-zero)/scale, codebook, boundaries)
		}
	}
	return scale
}

// pathBRefineScale runs Path B (adaptive RMS-vs-EDEN). Mirrors tq-encode.cu's
// Step 5b: compute both the RMS-only and EDEN-refined (scale, codes) pairs,
// then keep whichever has lower per-cell reconstruction MSE. The codes slice
// is mutated in place to reflect the chosen pair.
//
// rmsCodes is the result of the initial quantization at scaleRMS (callers pass
// codes already populated by the RMS quantize step). zero is the per-block
// mean for asymmetric mode, 0 for symmetric. Returns the chosen scale.
//
// Provably non-worse than RMS-only: if EDEN's refined codes have higher MSE
// than the RMS pair, the function restores the RMS codes and scale.
func pathBRefineScale(rotated []float32, codes []uint8, codebook []float32, boundaries []float32, scaleRMS, zero float32) float32 {
	if len(codes) == 0 {
		return scaleRMS
	}
	// Save RMS codes before EDEN overwrites them.
	rmsCodes := append([]uint8(nil), codes...)

	scaleEDEN := edenRefineScale(rotated, codes, codebook, boundaries, scaleRMS, zero)

	// Per-cell MSE for EDEN-refined (codes + scaleEDEN).
	var errEDEN float64
	for i, code := range codes {
		predicted := float64(codebook[code]) * float64(scaleEDEN)
		actual := float64(rotated[i]) - float64(zero)
		d := actual - predicted
		errEDEN += d * d
	}

	// Per-cell MSE for RMS-only (rmsCodes + scaleRMS).
	var errRMS float64
	for i, code := range rmsCodes {
		predicted := float64(codebook[code]) * float64(scaleRMS)
		actual := float64(rotated[i]) - float64(zero)
		d := actual - predicted
		errRMS += d * d
	}

	if errRMS < errEDEN {
		copy(codes, rmsCodes)
		return scaleRMS
	}
	return scaleEDEN
}

func blockScale(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	var sumSquares float64
	for _, value := range values {
		sumSquares += float64(value * value)
	}
	if sumSquares < 1e-12 {
		return 0
	}
	return float32(math.Sqrt(sumSquares / float64(len(values))))
}

// asymmetricBlockStats returns (mean, RMS-of-centred) for use by the
// centred-asymmetric-primary path. Symmetric presets still call blockScale
// directly; this helper is a sibling, not a replacement.
//
// The returned scale is the RMS of `values - mean`, so downstream quantisation
// normalises by (value - mean) / scale just like the symmetric path normalises
// by value / scale — the Lloyd-Max codebook stays the same, the only change
// is that we centre the distribution first.
func asymmetricBlockStats(values []float32) (zero, scale float32) {
	n := len(values)
	if n == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range values {
		sum += float64(v)
	}
	mean := sum / float64(n)
	var sumSquares float64
	for _, v := range values {
		d := float64(v) - mean
		sumSquares += d * d
	}
	if sumSquares < 1e-12 {
		return float32(mean), 0
	}
	return float32(mean), float32(math.Sqrt(sumSquares / float64(n)))
}

func (e EncodedVector) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	header := []any{
		e.Version,
		e.Preset.ID,
		uint32(e.Dim),
		uint32(len(e.Blocks)),
	}
	for _, field := range header {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			return nil, err
		}
	}
	for _, block := range e.Blocks {
		blockData, err := block.MarshalBinary()
		if err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(blockData))); err != nil {
			return nil, err
		}
		buf.Write(blockData)
	}
	return buf.Bytes(), nil
}
