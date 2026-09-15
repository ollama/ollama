package create

import (
	"io"
	"slices"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

// TestConcatAxis1 verifies that concatAxis1 correctly fuses two 3-D tensors
// along axis 1 using a small concrete example.
//
// Lo tensor: shape [2, 2, 2], dtype U8 (1 byte/element).
//
//	expert0: row0=[0,1], row1=[2,3]
//	expert1: row0=[8,9], row1=[10,11]
//
// Hi tensor: shape [2, 2, 2]:
//
//	expert0: row0=[4,5], row1=[6,7]
//	expert1: row0=[12,13], row1=[14,15]
//
// Concatenating along axis 1 produces [2, 4, 2]:
//
//	expert0: [0,1,2,3,4,5,6,7]
//	expert1: [8,9,10,11,12,13,14,15]
func TestConcatAxis1(t *testing.T) {
	loRaw := []byte{0, 1, 2, 3, 8, 9, 10, 11}
	hiRaw := []byte{4, 5, 6, 7, 12, 13, 14, 15}
	loSrc := st.NewTensorDataFromBytes("gate", "U8", []int32{2, 2, 2}, loRaw)
	hiSrc := st.NewTensorDataFromBytes("up", "U8", []int32{2, 2, 2}, hiRaw)
	outShape := []int32{2, 4, 2}

	out, err := concatAxis1("gate_up", outShape, loSrc, hiSrc)
	if err != nil {
		t.Fatalf("concatAxis1: %v", err)
	}

	gotBytes, _ := io.ReadAll(out.Reader())
	wantBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

	if !slices.Equal(gotBytes, wantBytes) {
		t.Errorf("bytes = %v, want %v", gotBytes, wantBytes)
	}
	if out.Name != "gate_up" {
		t.Errorf("name = %q, want gate_up", out.Name)
	}
	if !slices.Equal(out.Shape, outShape) {
		t.Errorf("shape = %v, want %v", out.Shape, outShape)
	}
	if out.Dtype != "U8" {
		t.Errorf("dtype = %q, want U8", out.Dtype)
	}
}

// TestConcatAxis1RoundTrip verifies that concatAxis1 is the inverse of the
// split operation: concatenating two half-slabs reconstructs the original.
func TestConcatAxis1RoundTrip(t *testing.T) {
	const A, B, C = 3, 4, 5
	// Build original [A, 2*B, C] data.
	orig := make([]byte, A*2*B*C)
	for i := range orig {
		orig[i] = byte(i % 251)
	}
	// Split into lo and hi halves manually (first and second B rows of each slab).
	halfSize := B * C
	loRaw := make([]byte, A*halfSize)
	hiRaw := make([]byte, A*halfSize)
	for a := range A {
		copy(loRaw[a*halfSize:], orig[a*2*halfSize:a*2*halfSize+halfSize])
		copy(hiRaw[a*halfSize:], orig[a*2*halfSize+halfSize:(a+1)*2*halfSize])
	}

	loSrc := st.NewTensorDataFromBytes("gate", "U8", []int32{A, B, C}, loRaw)
	hiSrc := st.NewTensorDataFromBytes("up", "U8", []int32{A, B, C}, hiRaw)
	outShape := []int32{A, 2 * B, C}

	out, err := concatAxis1("gate_up", outShape, loSrc, hiSrc)
	if err != nil {
		t.Fatalf("concatAxis1: %v", err)
	}
	gotBytes, _ := io.ReadAll(out.Reader())
	if !slices.Equal(gotBytes, orig) {
		t.Errorf("round-trip failed: got %v, want %v", gotBytes, orig)
	}
}
