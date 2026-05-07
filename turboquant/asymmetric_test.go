package turboquant

import (
	"testing"
)

// TestBlockRoundTripWithZero verifies that a Block carrying a non-zero Zero
// offset round-trips through MarshalBinary / UnmarshalBinary intact. The Zero
// field was added to the v7 block format alongside the channel bitmap to
// support asymmetric-primary presets; serialising it correctly is a
// prerequisite for everything the asymmetric path does downstream.
func TestBlockRoundTripWithZero(t *testing.T) {
	block := Block{
		Version:        BlockVersion,
		PresetID:       PresetTQ3.ID,
		Role:           uint8(roleKey),
		Objective:      uint8(objectiveMSE),
		OriginalDim:    4,
		PaddedDim:      4,
		BlockDim:       4,
		RegularBits:    3,
		RotationSeed:   99,
		CodebookID:     3,
		QJLRows:        0,
		AuxLayoutID:    1,
		Scale:          1.25,
		Zero:           -0.375, // representative non-zero offset
		RegularIndices: []byte{0b10101010},
	}
	raw, err := block.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got Block
	if err := got.UnmarshalBinary(raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Scale != block.Scale {
		t.Errorf("Scale: got %f want %f", got.Scale, block.Scale)
	}
	if got.Zero != block.Zero {
		t.Errorf("Zero: got %f want %f", got.Zero, block.Zero)
	}
}
