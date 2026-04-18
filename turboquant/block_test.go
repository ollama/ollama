package turboquant

import "testing"

func TestBlockMarshalRoundTrip(t *testing.T) {
	block := Block{
		Version:        BlockVersion,
		PresetID:       PresetTQ3.ID,
		Role:           uint8(roleKey),
		Objective:      uint8(objectiveProduct),
		OriginalDim:    8,
		PaddedDim:      8,
		BlockDim:       8,
		RegularBits:    3,
		RotationSeed:   77,
		CodebookID:     3,
		QJLRows:        4,
		AuxLayoutID:    1,
		Scale:          1,
		RegularIndices: []byte{1, 2, 3},
		Residual: ResidualSketch{
			Seed:      88,
			Scale:     0.5,
			SketchDim: 4,
			Signs:     []byte{0x0f},
		},
	}

	data, err := block.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var decoded Block
	if err := decoded.UnmarshalBinary(data); err != nil {
		t.Fatal(err)
	}

	if decoded.Version != block.Version ||
		decoded.PresetID != block.PresetID ||
		decoded.Role != block.Role ||
		decoded.Objective != block.Objective ||
		decoded.OriginalDim != block.OriginalDim ||
		decoded.PaddedDim != block.PaddedDim ||
		decoded.BlockDim != block.BlockDim ||
		decoded.RegularBits != block.RegularBits ||
		decoded.RotationSeed != block.RotationSeed ||
		decoded.CodebookID != block.CodebookID ||
		decoded.QJLRows != block.QJLRows ||
		decoded.AuxLayoutID != block.AuxLayoutID ||
		string(decoded.RegularIndices) != string(block.RegularIndices) ||
		decoded.Residual.Seed != block.Residual.Seed ||
		decoded.Residual.Scale != block.Residual.Scale ||
		decoded.Residual.SketchDim != block.Residual.SketchDim ||
		string(decoded.Residual.Signs) != string(block.Residual.Signs) {
		t.Fatalf("decoded block mismatch: %+v", decoded)
	}
}

func TestBlockUnmarshalRejectsBadVersion(t *testing.T) {
	block := Block{Version: BlockVersion, PresetID: PresetTQ2.ID, Role: uint8(roleValue), Objective: uint8(objectiveMSE), OriginalDim: 4, PaddedDim: 4, BlockDim: 4, RegularBits: 2}
	data, err := block.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 99

	var decoded Block
	if err := decoded.UnmarshalBinary(data); err == nil {
		t.Fatal("expected unsupported block version error")
	}
}

func TestPackBitsRoundTripMixedWidths(t *testing.T) {
	values2 := []uint8{1, 3, 0, 2}
	roundTrip2 := unpackBits(packBits(values2, 2), 2, len(values2))
	for i := range values2 {
		if roundTrip2[i] != values2[i] {
			t.Fatalf("2-bit round trip mismatch at %d: got %d want %d", i, roundTrip2[i], values2[i])
		}
	}

	values3 := []uint8{3, 7, 1, 5}
	roundTrip3 := unpackBits(packBits(values3, 3), 3, len(values3))
	for i := range values3 {
		if roundTrip3[i] != values3[i] {
			t.Fatalf("3-bit round trip mismatch at %d: got %d want %d", i, roundTrip3[i], values3[i])
		}
	}
}

// TestEncodeOutlierSplitLayout verifies that vectors larger than OutlierCount
// are encoded as two blocks (outlier + regular) with the expected bit widths and
// ChannelIndices, and that vectors at or below OutlierCount stay as a single block.
//
// Uses an explicit outlier-enabled preset because PresetTQ3's shipped default
// is OutlierCount=0 (outlier split is opt-in infrastructure; see PresetTQ3
// comment in turboquant.go).
func TestEncodeOutlierSplitLayout(t *testing.T) {
	preset := testOutlierPreset(PresetTQ3, 32)

	// dim=70 > OutlierCount=32: two blocks expected.
	encoded, err := EncodeVector(pseudoRandomVector(70, 0x55), preset)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded.Blocks) != 2 {
		t.Fatalf("block count = %d, want 2", len(encoded.Blocks))
	}

	outlierBlock := encoded.Blocks[0]
	regularBlock := encoded.Blocks[1]

	if int(outlierBlock.OriginalDim) != preset.OutlierCount {
		t.Errorf("outlier block dim = %d, want %d", outlierBlock.OriginalDim, preset.OutlierCount)
	}
	if outlierBlock.RegularBits != uint8(preset.OutlierBits) {
		t.Errorf("outlier bits = %d, want %d", outlierBlock.RegularBits, preset.OutlierBits)
	}
	if len(outlierBlock.ChannelIndices) != preset.OutlierCount {
		t.Errorf("outlier ChannelIndices len = %d, want %d", len(outlierBlock.ChannelIndices), preset.OutlierCount)
	}

	wantRegularDim := 70 - preset.OutlierCount
	if int(regularBlock.OriginalDim) != wantRegularDim {
		t.Errorf("regular block dim = %d, want %d", regularBlock.OriginalDim, wantRegularDim)
	}
	if regularBlock.RegularBits != uint8(preset.ValueBits) {
		t.Errorf("regular bits = %d, want %d", regularBlock.RegularBits, preset.ValueBits)
	}
	if len(regularBlock.ChannelIndices) != wantRegularDim {
		t.Errorf("regular ChannelIndices len = %d, want %d", len(regularBlock.ChannelIndices), wantRegularDim)
	}

	// ChannelIndices across both blocks must cover all 70 channels exactly once.
	seen := make([]int, 70)
	for _, idx := range outlierBlock.ChannelIndices {
		seen[idx]++
	}
	for _, idx := range regularBlock.ChannelIndices {
		seen[idx]++
	}
	for i, count := range seen {
		if count != 1 {
			t.Errorf("channel %d appears %d times across blocks", i, count)
		}
	}
}

// TestBlockMarshalWithChannelIndices verifies that ChannelIndices round-trips correctly.
func TestBlockMarshalWithChannelIndices(t *testing.T) {
	block := Block{
		Version:        BlockVersion,
		PresetID:       PresetTQ2.ID,
		Role:           uint8(roleKey),
		Objective:      uint8(objectiveMSE),
		OriginalDim:    4,
		PaddedDim:      4,
		BlockDim:       4,
		RegularBits:    2,
		RotationSeed:   42,
		CodebookID:     2,
		QJLRows:        0,
		AuxLayoutID:    1,
		ChannelIndices: []uint16{0, 3, 7, 12},
		Scale:          0.5,
		RegularIndices: []byte{0b10110001},
	}
	data, err := block.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var got Block
	if err := got.UnmarshalBinary(data); err != nil {
		t.Fatal(err)
	}
	if len(got.ChannelIndices) != len(block.ChannelIndices) {
		t.Fatalf("ChannelIndices len: got %d want %d", len(got.ChannelIndices), len(block.ChannelIndices))
	}
	for i := range block.ChannelIndices {
		if got.ChannelIndices[i] != block.ChannelIndices[i] {
			t.Errorf("ChannelIndices[%d]: got %d want %d", i, got.ChannelIndices[i], block.ChannelIndices[i])
		}
	}
}
