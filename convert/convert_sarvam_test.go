package convert

import (
	"bytes"
	"encoding/binary"
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSarvamMoETensors_QKVSplit(t *testing.T) {
	// Model with 2 heads, 1 KV head, head_dim=4
	// Q size = 2*4 = 8, K size = 1*4 = 4, V size = 1*4 = 4
	// Total fused QKV dim0 = 8+4+4 = 16
	p := sarvamMoEModel{
		HiddenLayers:     1,
		NumAttentionHeads: 2,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}

	qSize := uint64(p.NumAttentionHeads) * uint64(p.HeadDim) // 8
	kSize := uint64(p.NumKeyValueHeads) * uint64(p.HeadDim)  // 4
	vSize := uint64(p.NumKeyValueHeads) * uint64(p.HeadDim)  // 4
	totalSize := qSize + kSize + vSize                        // 16

	hiddenSize := uint64(6)
	numElements := int(totalSize * hiddenSize)
	data := make([]float32, numElements)
	for i := range data {
		data[i] = float32(i)
	}

	input := &fakeTensor{
		name:  "blk.0.attn_qkv.weight",
		shape: []uint64{totalSize, hiddenSize},
		data:  data,
	}

	next, stop := iter.Pull(splitDim(input, 0,
		split{
			Replacer: strings.NewReplacer("attn_qkv", "attn_q"),
			dim:      int(qSize),
		},
		split{
			Replacer: strings.NewReplacer("attn_qkv", "attn_k"),
			dim:      int(kSize),
		},
		split{
			Replacer: strings.NewReplacer("attn_qkv", "attn_v"),
			dim:      int(vSize),
		},
	))
	defer stop()

	// Q tensor
	{
		tt, ok := next()
		if !ok {
			t.Fatal("expected Q tensor")
		}

		if tt.Name != "blk.0.attn_q.weight" {
			t.Fatalf("expected name 'blk.0.attn_q.weight', got %q", tt.Name)
		}

		if diff := cmp.Diff(tt.Shape, []uint64{qSize, hiddenSize}); diff != "" {
			t.Errorf("Q shape mismatch (-want +got):\n%s", diff)
		}

		var b bytes.Buffer
		if _, err := tt.WriteTo(&b); err != nil {
			t.Fatal(err)
		}

		f32s := make([]float32, mul(tt.Shape))
		if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
			t.Fatal(err)
		}

		// Q should contain the first qSize rows
		want := make([]float32, qSize*hiddenSize)
		for row := uint64(0); row < qSize; row++ {
			for col := uint64(0); col < hiddenSize; col++ {
				want[row*hiddenSize+col] = float32(row*hiddenSize + col)
			}
		}
		if diff := cmp.Diff(f32s, want); diff != "" {
			t.Errorf("Q data mismatch (-want +got):\n%s", diff)
		}
	}

	// K tensor
	{
		tt, ok := next()
		if !ok {
			t.Fatal("expected K tensor")
		}

		if tt.Name != "blk.0.attn_k.weight" {
			t.Fatalf("expected name 'blk.0.attn_k.weight', got %q", tt.Name)
		}

		if diff := cmp.Diff(tt.Shape, []uint64{kSize, hiddenSize}); diff != "" {
			t.Errorf("K shape mismatch (-want +got):\n%s", diff)
		}

		var b bytes.Buffer
		if _, err := tt.WriteTo(&b); err != nil {
			t.Fatal(err)
		}

		f32s := make([]float32, mul(tt.Shape))
		if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
			t.Fatal(err)
		}

		// K should contain rows [qSize, qSize+kSize)
		want := make([]float32, kSize*hiddenSize)
		for row := uint64(0); row < kSize; row++ {
			for col := uint64(0); col < hiddenSize; col++ {
				want[row*hiddenSize+col] = float32((qSize+row)*hiddenSize + col)
			}
		}
		if diff := cmp.Diff(f32s, want); diff != "" {
			t.Errorf("K data mismatch (-want +got):\n%s", diff)
		}
	}

	// V tensor
	{
		tt, ok := next()
		if !ok {
			t.Fatal("expected V tensor")
		}

		if tt.Name != "blk.0.attn_v.weight" {
			t.Fatalf("expected name 'blk.0.attn_v.weight', got %q", tt.Name)
		}

		if diff := cmp.Diff(tt.Shape, []uint64{vSize, hiddenSize}); diff != "" {
			t.Errorf("V shape mismatch (-want +got):\n%s", diff)
		}

		var b bytes.Buffer
		if _, err := tt.WriteTo(&b); err != nil {
			t.Fatal(err)
		}

		f32s := make([]float32, mul(tt.Shape))
		if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
			t.Fatal(err)
		}

		// V should contain rows [qSize+kSize, qSize+kSize+vSize)
		want := make([]float32, vSize*hiddenSize)
		for row := uint64(0); row < vSize; row++ {
			for col := uint64(0); col < hiddenSize; col++ {
				want[row*hiddenSize+col] = float32((qSize+kSize+row)*hiddenSize + col)
			}
		}
		if diff := cmp.Diff(f32s, want); diff != "" {
			t.Errorf("V data mismatch (-want +got):\n%s", diff)
		}
	}

	// No more tensors
	{
		_, ok := next()
		if ok {
			t.Fatal("expected no more tensors after Q, K, V")
		}
	}
}
