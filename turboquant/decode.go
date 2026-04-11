package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

// DecodeVector fully dequantizes an encoded vector back to float32 in
// original space. This applies inverse rotation and, for product-mode
// blocks, adds the reconstructed QJL residual.
func DecodeVector(data []byte) ([]float32, Preset, error) {
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		return nil, Preset{}, err
	}

	decoded := make([]float32, ev.Dim)
	offset := 0
	for _, block := range ev.Blocks {
		blockDim := int(block.OriginalDim)
		codebook, _ := scalarCodebook(blockDim, int(block.RegularBits))
		indices := unpackBits(block.RegularIndices, int(block.RegularBits), blockDim)
		rotated := make([]float32, blockDim)
		for i, idx := range indices {
			rotated[i] = dequantizeScalar(idx, codebook) * block.Scale
		}
		if vectorObjective(block.Objective) == objectiveProduct {
			residual := reconstructResidual(blockDim, block.Residual)
			for i := range rotated {
				rotated[i] += residual[i]
			}
		}
		original := ApplyInverseRotation(rotated, BuildRotation(blockDim, block.RotationSeed))

		if len(block.ChannelIndices) == blockDim {
			// Scatter to original channel positions.
			for i, chIdx := range block.ChannelIndices {
				decoded[chIdx] = original[i]
			}
		} else {
			// Legacy single-block: fill contiguous range.
			copy(decoded[offset:], original)
			offset += blockDim
		}
	}

	return decoded, ev.Preset, nil
}

// UnmarshalEncodedVector deserializes an EncodedVector from its binary form.
func UnmarshalEncodedVector(data []byte) (EncodedVector, error) {
	r := bytes.NewReader(data)
	var version uint8
	var presetID uint8
	var dim uint32
	var blockCount uint32
	for _, field := range []any{&version, &presetID, &dim, &blockCount} {
		if err := binary.Read(r, binary.LittleEndian, field); err != nil {
			return EncodedVector{}, err
		}
	}
	if version != BlockVersion {
		return EncodedVector{}, fmt.Errorf("unsupported encoded vector version %d", version)
	}

	preset, err := PresetByID(presetID)
	if err != nil {
		return EncodedVector{}, err
	}

	blocks := make([]Block, 0, blockCount)
	totalDim := 0
	for i := 0; i < int(blockCount); i++ {
		var blockLen uint32
		if err := binary.Read(r, binary.LittleEndian, &blockLen); err != nil {
			return EncodedVector{}, err
		}
		blockData := make([]byte, blockLen)
		if _, err := io.ReadFull(r, blockData); err != nil {
			return EncodedVector{}, err
		}
		var block Block
		if err := block.UnmarshalBinary(blockData); err != nil {
			return EncodedVector{}, err
		}
		if block.PresetID != preset.ID {
			return EncodedVector{}, fmt.Errorf("block preset id %d does not match encoded preset %d", block.PresetID, preset.ID)
		}
		if len(block.RegularIndices) != expectedPackedBytes(int(block.OriginalDim), int(block.RegularBits)) {
			return EncodedVector{}, fmt.Errorf("invalid primary index length %d for dim %d and bits %d", len(block.RegularIndices), block.OriginalDim, block.RegularBits)
		}
		if block.Residual.SketchDim != block.QJLRows {
			return EncodedVector{}, fmt.Errorf("residual sketch dim %d does not match qjl rows %d", block.Residual.SketchDim, block.QJLRows)
		}
		totalDim += int(block.OriginalDim)
		blocks = append(blocks, block)
	}
	if r.Len() != 0 {
		return EncodedVector{}, fmt.Errorf("unexpected trailing bytes in encoded vector: %d", r.Len())
	}
	if totalDim != int(dim) {
		return EncodedVector{}, fmt.Errorf("encoded vector dim mismatch: header=%d blocks=%d", dim, totalDim)
	}

	return EncodedVector{
		Version: version,
		Preset:  preset,
		Dim:     int(dim),
		Blocks:  blocks,
	}, nil
}
