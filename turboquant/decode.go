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
			// block.Zero is 0 for symmetric presets, so this term is a no-op
			// on the legacy path; for asymmetric-primary presets it re-adds
			// the per-block mean the encoder subtracted before quantization.
			rotated[i] = dequantizeScalar(idx, codebook)*block.Scale + block.Zero
		}
		if vectorObjective(block.Objective) == objectiveProduct {
			residual := reconstructResidual(blockDim, block.Residual)
			for i := range rotated {
				rotated[i] += residual[i]
			}
		}
		original := ApplyInverseRotation(rotated, BuildRotation(blockDim, block.RotationSeed))

		if len(block.ChannelBitmap) > 0 {
			// Scatter to original channel positions via the bitmap: the i'th
			// set bit is the original-vector index of the i'th value in this
			// sub-block.
			i := 0
			iterateBitmap(block.ChannelBitmap, ev.Dim, func(chIdx int) {
				if i < blockDim {
					decoded[chIdx] = original[i]
				}
				i++
			})
		} else {
			// Single-block path: fill contiguous range.
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
	for range int(blockCount) {
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
		// If the block carries a channel bitmap, validate it covers the
		// encoded-vector dim and has the right popcount. An empty bitmap
		// indicates the single-block legacy path.
		if len(block.ChannelBitmap) > 0 {
			wantBytes := bitmapBytes(int(dim))
			if len(block.ChannelBitmap) != wantBytes {
				return EncodedVector{}, fmt.Errorf("channel bitmap %d bytes, expected %d for dim %d", len(block.ChannelBitmap), wantBytes, dim)
			}
			if pc := channelBitmapPopcount(block.ChannelBitmap, int(dim)); pc != int(block.OriginalDim) {
				return EncodedVector{}, fmt.Errorf("channel bitmap popcount %d does not match block dim %d", pc, block.OriginalDim)
			}
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
