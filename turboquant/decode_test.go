package turboquant

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestUnmarshalEncodedVectorRejectsBadHeader(t *testing.T) {
	if _, err := UnmarshalEncodedVector([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected malformed header error")
	}
}

func TestUnmarshalEncodedVectorRejectsWrongVersion(t *testing.T) {
	encoded, err := EncodeVector([]float32{1, 2, 3, 4}, PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 99

	if _, err := UnmarshalEncodedVector(data); err == nil {
		t.Fatal("expected unsupported version error")
	}
}

func TestUnmarshalEncodedVectorRejectsBadPresetID(t *testing.T) {
	encoded, err := EncodeVector([]float32{1, 2, 3, 4}, PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	data[1] = 99

	if _, err := UnmarshalEncodedVector(data); err == nil {
		t.Fatal("expected bad preset id error")
	}
}

func TestUnmarshalEncodedVectorRejectsTruncatedBlockPayload(t *testing.T) {
	encoded, err := EncodeVector(pseudoRandomVector(16, 0x99), PresetTQ2)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	truncated := data[:len(data)-1]
	if _, err := UnmarshalEncodedVector(truncated); err == nil {
		t.Fatal("expected truncated block payload error")
	}
}

func TestDecodeVectorRejectsInvalidIndexLengths(t *testing.T) {
	encoded, err := EncodeVector(pseudoRandomVector(16, 0x77), PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		t.Fatal(err)
	}

	ev.Blocks[0].RegularIndices = append(ev.Blocks[0].RegularIndices, 0)
	corrupt, err := marshalTestEncodedVector(ev)
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err := DecodeVector(corrupt); err == nil {
		t.Fatal("expected invalid primary index length error")
	}
}

func TestDecodeVectorPreservesOriginalLength(t *testing.T) {
	values := pseudoRandomVector(130, 0x66)
	encoded, err := EncodeVector(values, PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := DecodeVector(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(values) {
		t.Fatalf("decoded len = %d, want %d", len(decoded), len(values))
	}
}

func marshalTestEncodedVector(ev EncodedVector) ([]byte, error) {
	var buf bytes.Buffer
	for _, field := range []any{ev.Version, ev.Preset.ID, uint32(ev.Dim), uint32(len(ev.Blocks))} {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			return nil, err
		}
	}

	for _, block := range ev.Blocks {
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

func dotFloat32(a, b []float32) float32 {
	var out float32
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}
