package create

import (
	"context"
	"encoding/binary"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

func TestStackExpertTensorsPreservesSourceOrder(t *testing.T) {
	dir := t.TempDir()
	parts := []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes("expert.0", "U8", []int32{2}, []byte{1, 2}),
		safetensors.NewTensorDataFromBytes("expert.1", "U8", []int32{2}, []byte{3, 4}),
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), parts)
	spec := BlobSpec{Name: "experts", Tensors: []TensorSpec{{
		Name: "experts", Transform: TransformStackExperts, OutDtype: "U8", OutShape: []int32{2, 2},
		Sources: []SourceTensor{
			{Name: "expert.0", Dtype: "U8", Shape: []int32{2}, File: "model.safetensors"},
			{Name: "expert.1", Dtype: "U8", Shape: []int32{2}, File: "model.safetensors"},
		},
	}}}
	store := newCaptureStore()
	if _, err := WriteBlobs(context.Background(), []BlobSpec{spec}, dir, store); err != nil {
		t.Fatal(err)
	}
	blob := store.blobs["experts"]
	data := blob[8+binary.LittleEndian.Uint64(blob[:8]):]
	if !slices.Equal(data, []byte{1, 2, 3, 4}) || !slices.Equal(blobHeader(t, blob)["experts"].Shape, []int32{2, 2}) {
		t.Fatalf("stacked experts = %v, want ordered bytes and shape [2 2]", data)
	}
}
