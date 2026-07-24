package create

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"slices"
	"sort"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

type captureStore struct{ blobs map[string][]byte }

func newCaptureStore() *captureStore { return &captureStore{blobs: make(map[string][]byte)} }

func (c *captureStore) WriteBlob(r io.Reader, mediaType, name string) (LayerInfo, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return LayerInfo{}, err
	}
	c.blobs[name] = data
	return LayerInfo{Name: name, MediaType: mediaType, Digest: "sha256:" + name, Size: int64(len(data))}, nil
}

func (c *captureStore) names() []string {
	out := make([]string, 0, len(c.blobs))
	for k := range c.blobs {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

type headerEntry struct {
	Dtype string  `json:"dtype"`
	Shape []int32 `json:"shape"`
}

func blobHeader(t *testing.T, data []byte) map[string]headerEntry {
	t.Helper()
	if len(data) < 8 {
		t.Fatalf("blob too small: %d bytes", len(data))
	}
	n := binary.LittleEndian.Uint64(data[:8])
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+n], &raw); err != nil {
		t.Fatalf("parse header: %v", err)
	}
	out := make(map[string]headerEntry)
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(v, &e); err != nil {
			t.Fatalf("parse header entry %q: %v", k, err)
		}
		out[k] = e
	}
	return out
}

func f32le(v float32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	return b
}

func TestWriteBlobsCompressedNVFP4(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"],"compression_config":{"format":"nvfp4-pack-quantized"}}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight_packed", "U8", []int32{16, 8}, make([]byte, 16*8)),
		st.NewTensorDataFromBytes("linear.weight_scale", "F8_E4M3", []int32{16, 1}, make([]byte, 16)),
		st.NewTensorDataFromBytes("linear.weight_global_scale", "F32", []int32{}, f32le(4.0)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{16}, make([]byte, 32)),
	})

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatalf("ReadInventory() error = %v", err)
	}
	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	store := newCaptureStore()
	if _, err := WriteBlobs(specs, dir, store); err != nil {
		t.Fatalf("WriteBlobs() error = %v", err)
	}

	fused, ok := store.blobs["linear.weight"]
	if !ok {
		t.Fatalf("missing fused blob; got %v", store.names())
	}
	hdr := blobHeader(t, fused)

	if w := hdr["linear.weight"]; w.Dtype != "U32" || !slices.Equal(w.Shape, []int32{16, 2}) {
		t.Errorf("fused weight = %+v, want U32 [16 2] (repacked)", w)
	}
	if s := hdr["linear.weight.scale"]; s.Dtype != "U8" {
		t.Errorf("fused scale dtype = %q, want U8 (relabeled from F8_E4M3)", s.Dtype)
	}
	if g, ok := hdr["linear.weight.global_scale"]; !ok || g.Dtype != "F32" {
		t.Errorf("fused global_scale = %+v ok=%v, want F32", g, ok)
	}
	// compressed-tensors stores the global scale inverted.
	gs := readPackedTensorRaw(t, fused, "linear.weight.global_scale")
	if got := math.Float32frombits(binary.LittleEndian.Uint32(gs)); got != 0.25 {
		t.Errorf("global_scale = %v, want 0.25 (reciprocal of 4.0)", got)
	}

	// the scale companion is folded in, not its own blob.
	if _, leaked := store.blobs["linear.weight_scale"]; leaked {
		t.Error("scale companion leaked as its own blob")
	}

	// the norm passes through unchanged as its own blob.
	norm, ok := store.blobs["norm.weight"]
	if !ok {
		t.Fatalf("missing norm blob; got %v", store.names())
	}
	if nh := blobHeader(t, norm)["norm.weight"]; nh.Dtype != "BF16" || !slices.Equal(nh.Shape, []int32{16}) {
		t.Errorf("norm = %+v, want BF16 [16]", nh)
	}
}

func TestWriteBlobsQuantizeFloat(t *testing.T) {
	if !QuantizeSupported() {
		t.Skip("MLX unavailable")
	}
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.layers.0.self_attn.q_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{128}, make([]byte, 128*2)),
	})

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatalf("ReadInventory() error = %v", err)
	}
	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "int4"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	store := newCaptureStore()
	if _, err := WriteBlobs(specs, dir, store); err != nil {
		t.Fatalf("WriteBlobs() error = %v", err)
	}

	q, ok := store.blobs["model.layers.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatalf("missing q_proj blob; got %v", store.names())
	}
	hdr := blobHeader(t, q)
	if w := hdr["model.layers.0.self_attn.q_proj.weight"]; w.Dtype != "U32" {
		t.Errorf("quantized weight dtype = %q, want U32 (packed int4)", w.Dtype)
	}
	if _, ok := hdr["model.layers.0.self_attn.q_proj.weight.scale"]; !ok {
		t.Error("quantized blob missing scale")
	}

	norm, ok := store.blobs["model.norm.weight"]
	if !ok {
		t.Fatalf("missing norm blob; got %v", store.names())
	}
	if nh := blobHeader(t, norm)["model.norm.weight"]; nh.Dtype != "BF16" {
		t.Errorf("norm dtype = %q, want BF16 (kept, not quantized)", nh.Dtype)
	}
}

func TestWriteBlobsBlockFP8Decode(t *testing.T) {
	if !QuantizeSupported() {
		t.Skip("MLX unavailable")
	}
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.layers.0.mlp.down_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.down_proj.weight_scale_inv", "F32", []int32{1, 1}, f32le(1.0)),
	})

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatalf("ReadInventory() error = %v", err)
	}
	specs, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	store := newCaptureStore()
	if _, err := WriteBlobs(specs, dir, store); err != nil {
		t.Fatalf("WriteBlobs() error = %v", err)
	}

	b, ok := store.blobs["model.layers.0.mlp.down_proj.weight"]
	if !ok {
		t.Fatalf("missing decoded blob; got %v", store.names())
	}
	hdr := blobHeader(t, b)
	if w := hdr["model.layers.0.mlp.down_proj.weight"]; w.Dtype != "U32" {
		t.Errorf("decoded+quantized weight dtype = %q, want U32 (packed mxfp8)", w.Dtype)
	}
	if _, ok := hdr["model.layers.0.mlp.down_proj.weight.scale"]; !ok {
		t.Error("mxfp8 blob missing scale")
	}
	if _, leaked := store.blobs["model.layers.0.mlp.down_proj.weight_scale_inv"]; leaked {
		t.Error("fp8 scale companion leaked as its own blob")
	}
}
