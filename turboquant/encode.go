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

func EncodeVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, roleGeneric, objectiveMSE, preset.ValueBits)
}

func EncodeKeyVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, roleKey, objectiveProduct, preset.KeyPrimaryBits)
}

func EncodeValueVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, roleValue, objectiveMSE, preset.ValueBits)
}

func encodeVector(values []float32, preset Preset, role vectorRole, objective vectorObjective, bits int) (EncodedVector, error) {
	dim := len(values)
	if dim <= 0 {
		return EncodedVector{}, fmt.Errorf("invalid turboquant vector dim %d", dim)
	}
	if bits <= 0 || bits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid turboquant bit width %d", bits)
	}

	if preset.HasOutlierSplit() && dim > preset.OutlierCount {
		return encodeVectorWithOutliers(values, preset, role, objective, bits)
	}

	block, err := encodeSubBlock(values, nil, preset, role, objective, bits, preset.RotationSeed)
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
func encodeVectorWithOutliers(values []float32, preset Preset, role vectorRole, objective vectorObjective, regularBits int) (EncodedVector, error) {
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

	outlierBlock, err := encodeSubBlock(split.OutlierValues, split.OutlierIndices, preset, role, objective, outlierBits, outlierSeed)
	if err != nil {
		return EncodedVector{}, fmt.Errorf("outlier block: %w", err)
	}

	// Regular block always uses MSE (no QJL sketch) regardless of the key/value
	// role. QJL is only applied to the outlier block, which concentrates the
	// residual correction budget on the highest-magnitude channels.
	regularBlock, err := encodeSubBlock(split.RegularValues, split.RegularIndices, preset, role, objectiveMSE, regularBits, regularSeed)
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

// encodeSubBlock encodes a sub-vector (identified by channelIndices into the
// original full-dim vector) as a single Block. If channelIndices is nil, the
// block covers all channels (single-block legacy path).
func encodeSubBlock(values []float32, channelIndices []uint16, preset Preset, role vectorRole, objective vectorObjective, bits int, rotationSeed uint64) (Block, error) {
	dim := len(values)
	if dim <= 0 {
		return Block{}, fmt.Errorf("empty sub-block")
	}

	codebook, boundaries := scalarCodebook(dim, bits)
	rotation := BuildRotation(dim, rotationSeed)
	rotated := ApplyRotation(values, rotation)
	scale := blockScale(rotated)
	primaryCodes := make([]uint8, dim)
	reconRotated := make([]float32, dim)
	if scale == 0 {
		for i := range primaryCodes {
			primaryCodes[i] = quantizeScalarByBoundary(0, codebook, boundaries)
			reconRotated[i] = 0
		}
	} else {
		for i, value := range rotated {
			normalized := value / scale
			idx := quantizeScalarByBoundary(normalized, codebook, boundaries)
			primaryCodes[i] = idx
			reconRotated[i] = dequantizeScalar(idx, codebook) * scale
		}
	}

	qjlRows := 0
	if objective == objectiveProduct {
		qjlRows = preset.KeyQJLRows(dim)
	}

	block := Block{
		Version:        BlockVersion,
		PresetID:       preset.ID,
		Role:           uint8(role),
		Objective:      uint8(objective),
		OriginalDim:    uint16(dim),
		PaddedDim:      uint16(dim),
		BlockDim:       uint16(dim),
		RegularBits:    uint8(bits),
		RotationSeed:   rotationSeed,
		CodebookID:     uint16(bits),
		QJLRows:        uint16(qjlRows),
		AuxLayoutID:    1,
		ChannelIndices: channelIndices,
		Scale:          scale,
		RegularIndices: packBits(primaryCodes, bits),
		Residual:       encodeResidual(rotated, reconRotated, qjlRows, rotationSeed^0x9e3779b97f4a7c15),
	}
	return block, nil
}

// EncodeKeyPerHead quantizes a single attention head's key vector using
// rotation + Lloyd-Max without outlier split or QJL residual. Returns
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
	OutlierPacked  []byte
	OutlierScale   float32
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
	for r := 0; r < outlierCount; r++ {
		bestVal := float32(-1.0)
		bestIdx := 0
		for i := 0; i < dim; i++ {
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
	for i := 0; i < dim; i++ {
		if !isOutlier[i] {
			regularRotated = append(regularRotated, rotated[i])
		}
	}

	regularScale := blockScale(regularRotated)
	outlierScale := blockScale(outlierVal)

	regularCodebook, regularBoundaries := scalarCodebook(dim, bits)
	outlierCodebook, outlierBoundaries := scalarCodebook(dim, outlierBits)

	regularCodes := make([]uint8, regularCount)
	if regularScale > 0 {
		for r, v := range regularRotated {
			regularCodes[r] = quantizeScalarByBoundary(v/regularScale, regularCodebook, regularBoundaries)
		}
	}
	outlierCodes := make([]uint8, outlierCount)
	if outlierScale > 0 {
		for r, v := range outlierVal {
			outlierCodes[r] = quantizeScalarByBoundary(v/outlierScale, outlierCodebook, outlierBoundaries)
		}
	}

	return OutlierPerHead{
		RegularPacked:  packBits(regularCodes, bits),
		RegularScale:   regularScale,
		OutlierPacked:  packBits(outlierCodes, outlierBits),
		OutlierScale:   outlierScale,
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
	for i := 0; i < headDim; i++ {
		if outlierSlotFor[i] >= 0 {
			out[i] = dequantizeScalar(outlierIdx[outlierSlotFor[i]], outlierCodebook) * enc.OutlierScale
		} else {
			out[i] = dequantizeScalar(regularIdx[regPos], regularCodebook) * enc.RegularScale
			regPos++
		}
	}
	return out
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
