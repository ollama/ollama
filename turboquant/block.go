package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

// Block is a single rotated, scalar-quantised sub-vector with optional QJL
// residual. For outlier-split encoded vectors, two Blocks partition the
// original vector's channels; the partition is recorded in ChannelBitmap
// (one bit per original-vector channel, set iff the channel is a member of
// this sub-block). For single-block vectors ChannelBitmap is empty and the
// block's data maps contiguously into the original vector.
type Block struct {
	Version       uint8
	PresetID      uint8
	Role          uint8
	Objective     uint8
	OriginalDim   uint16
	PaddedDim     uint16
	BlockDim      uint16
	RegularBits   uint8
	RotationSeed  uint64
	CodebookID    uint16
	QJLRows       uint16
	AuxLayoutID   uint8
	ChannelBitmap []byte // ceil(fullDim/8) bytes; empty for single-block vectors
	Scale         float32
	// Zero is the per-block centring offset for asymmetric-primary presets.
	// Symmetric blocks (set via OLLAMA_TQ_DISABLE_ASYMMETRIC=1) write 0.
	// Decoding is unconditional: x̂ = codebook[idx] * Scale + Zero.
	Zero           float32
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
		uint16(len(b.ChannelBitmap)),
		b.Scale,
		b.Zero,
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
	buf.Write(b.ChannelBitmap)
	buf.Write(b.RegularIndices)
	buf.Write(b.Residual.Signs)
	return buf.Bytes(), nil
}

func (b *Block) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)
	var bitmapLen uint16
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
		&bitmapLen,
		&b.Scale,
		&b.Zero,
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

	b.ChannelBitmap = make([]byte, bitmapLen)
	b.RegularIndices = make([]byte, regularLen)
	b.Residual.Signs = make([]byte, residualLen)
	for _, dst := range [][]byte{b.ChannelBitmap, b.RegularIndices, b.Residual.Signs} {
		if len(dst) == 0 {
			continue
		}
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
	for i := range count {
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

// bitmapBytes returns the byte length needed to hold fullDim bits.
func bitmapBytes(fullDim int) int {
	if fullDim <= 0 {
		return 0
	}
	return (fullDim + 7) / 8
}

// channelBitmapFromIndices returns a bitmap covering fullDim channels with a
// bit set for each index in indices. Callers pass disjoint, in-range indices.
func channelBitmapFromIndices(fullDim int, indices []uint16) []byte {
	bm := make([]byte, bitmapBytes(fullDim))
	for _, idx := range indices {
		i := int(idx)
		if i < 0 || i >= fullDim {
			continue
		}
		bm[i>>3] |= 1 << (i & 7)
	}
	return bm
}

// channelBitmapPopcount returns the number of set bits in bm, optionally
// clamped to fullDim so we don't count garbage bits in the final byte.
func channelBitmapPopcount(bm []byte, fullDim int) int {
	count := 0
	last := fullDim >> 3
	tail := fullDim & 7
	for i := 0; i < last && i < len(bm); i++ {
		v := bm[i]
		v = v - ((v >> 1) & 0x55)
		v = (v & 0x33) + ((v >> 2) & 0x33)
		v = (v + (v >> 4)) & 0x0f
		count += int(v)
	}
	if tail != 0 && last < len(bm) {
		v := bm[last] & ((1 << tail) - 1)
		v = v - ((v >> 1) & 0x55)
		v = (v & 0x33) + ((v >> 2) & 0x33)
		v = (v + (v >> 4)) & 0x0f
		count += int(v)
	}
	return count
}

// iterateBitmap calls fn for each set-bit index, in ascending order, up to
// fullDim bits total.
func iterateBitmap(bm []byte, fullDim int, fn func(idx int)) {
	for i := range fullDim {
		if int(bm[i>>3])&(1<<(i&7)) != 0 {
			fn(i)
		}
	}
}
