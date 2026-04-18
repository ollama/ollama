package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

type Block struct {
	Version        uint8
	PresetID       uint8
	Role           uint8
	Objective      uint8
	OriginalDim    uint16
	PaddedDim      uint16
	BlockDim       uint16
	RegularBits    uint8
	RotationSeed   uint64
	CodebookID     uint16
	QJLRows        uint16
	AuxLayoutID    uint8
	ChannelIndices []uint16
	Scale          float32
	RegularIndices []byte
	Residual       ResidualSketch
}

func (b Block) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	fields := []any{
		b.Version,
		b.PresetID,
		b.Role,
		b.Objective,
		b.OriginalDim,
		b.PaddedDim,
		b.BlockDim,
		b.RegularBits,
		b.RotationSeed,
		b.CodebookID,
		b.QJLRows,
		b.AuxLayoutID,
		uint16(len(b.ChannelIndices)),
		b.Scale,
		uint32(len(b.RegularIndices)),
		b.Residual.Seed,
		b.Residual.Scale,
		b.Residual.SketchDim,
		uint32(len(b.Residual.Signs)),
	}
	for _, field := range fields {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			return nil, err
		}
	}
	for _, idx := range b.ChannelIndices {
		if err := binary.Write(&buf, binary.LittleEndian, idx); err != nil {
			return nil, err
		}
	}
	buf.Write(b.RegularIndices)
	buf.Write(b.Residual.Signs)
	return buf.Bytes(), nil
}

func (b *Block) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)
	var channelCount uint16
	var regularLen, residualLen uint32
	fields := []any{
		&b.Version,
		&b.PresetID,
		&b.Role,
		&b.Objective,
		&b.OriginalDim,
		&b.PaddedDim,
		&b.BlockDim,
		&b.RegularBits,
		&b.RotationSeed,
		&b.CodebookID,
		&b.QJLRows,
		&b.AuxLayoutID,
		&channelCount,
		&b.Scale,
		&regularLen,
		&b.Residual.Seed,
		&b.Residual.Scale,
		&b.Residual.SketchDim,
		&residualLen,
	}
	for _, field := range fields {
		if err := binary.Read(r, binary.LittleEndian, field); err != nil {
			return err
		}
	}
	if b.Version != BlockVersion {
		return fmt.Errorf("unsupported block version %d", b.Version)
	}
	if b.Objective != uint8(objectiveMSE) && b.Objective != uint8(objectiveProduct) {
		return fmt.Errorf("unsupported block objective %d", b.Objective)
	}
	if b.OriginalDim == 0 || b.OriginalDim > b.PaddedDim || b.BlockDim != b.PaddedDim {
		return fmt.Errorf("invalid block dims: original=%d padded=%d block=%d", b.OriginalDim, b.PaddedDim, b.BlockDim)
	}

	if channelCount > 0 {
		b.ChannelIndices = make([]uint16, channelCount)
		for i := range b.ChannelIndices {
			if err := binary.Read(r, binary.LittleEndian, &b.ChannelIndices[i]); err != nil {
				return err
			}
		}
	}

	b.RegularIndices = make([]byte, regularLen)
	b.Residual.Signs = make([]byte, residualLen)
	for _, dst := range [][]byte{b.RegularIndices, b.Residual.Signs} {
		if _, err := io.ReadFull(r, dst); err != nil {
			return err
		}
	}
	if r.Len() != 0 {
		return fmt.Errorf("unexpected trailing bytes in turboquant block: %d", r.Len())
	}
	return nil
}

func packBits(values []uint8, bitsPerValue int) []byte {
	if bitsPerValue <= 0 || len(values) == 0 {
		return nil
	}
	totalBits := len(values) * bitsPerValue
	out := make([]byte, (totalBits+7)/8)
	mask := uint8((1 << bitsPerValue) - 1)
	bitPos := 0
	for _, value := range values {
		packed := value & mask
		bytePos := bitPos / 8
		shift := bitPos % 8
		out[bytePos] |= packed << shift
		if shift+bitsPerValue > 8 {
			out[bytePos+1] |= packed >> (8 - shift)
		}
		bitPos += bitsPerValue
	}
	return out
}

func unpackBits(data []byte, bitsPerValue, count int) []uint8 {
	out := make([]uint8, count)
	if bitsPerValue <= 0 {
		return out
	}
	mask := uint8((1 << bitsPerValue) - 1)
	bitPos := 0
	for i := 0; i < count; i++ {
		bytePos := bitPos / 8
		shift := bitPos % 8
		if bytePos >= len(data) {
			return out
		}
		value := data[bytePos] >> shift
		if shift+bitsPerValue > 8 && bytePos+1 < len(data) {
			value |= data[bytePos+1] << (8 - shift)
		}
		out[i] = value & mask
		bitPos += bitsPerValue
	}
	return out
}

func expectedPackedBytes(count, bitsPerValue int) int {
	if count <= 0 || bitsPerValue <= 0 {
		return 0
	}
	return ((count * bitsPerValue) + 7) / 8
}
