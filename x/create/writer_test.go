package create

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"slices"
	"sort"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
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

func u32le(values ...uint32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], value)
	}
	return out
}

func i64le(values ...int64) []byte {
	out := make([]byte, len(values)*8)
	for i, value := range values {
		binary.LittleEndian.PutUint64(out[i*8:], uint64(value))
	}
	return out
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

func TestWriteBlobsCompressedInt4Experts(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{
		"architectures":["BailingMoeV3ForCausalLM"],
		"quantization_config":{
			"quant_method":"compressed-tensors",
			"format":"pack-quantized",
			"config_groups":{"group_0":{"weights":{
				"num_bits":4,"type":"int","symmetric":true,
				"strategy":"group","group_size":32
			}}}
		}
	}`)

	const group = "model.layers.1.mlp.experts"
	packed0 := u32le(0x01234567, 0x89abcdef, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666)
	packed1 := u32le(0x76543210, 0xfedcba98, 0x77777777, 0x88888888, 0x99999999, 0xaaaaaaaa, 0xbbbbbbbb, 0xcccccccc)
	scale0, err := EncodeFloatTensor("BF16", []float32{0.25, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	scale1, err := EncodeFloatTensor("BF16", []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes(group+".0.gate_proj.weight_packed", "I32", []int32{2, 4}, packed0),
		st.NewTensorDataFromBytes(group+".0.gate_proj.weight_scale", "BF16", []int32{2, 1}, scale0),
		st.NewTensorDataFromBytes(group+".0.gate_proj.weight_shape", "I64", []int32{2}, i64le(2, 32)),
		st.NewTensorDataFromBytes(group+".1.gate_proj.weight_packed", "I32", []int32{2, 4}, packed1),
		st.NewTensorDataFromBytes(group+".1.gate_proj.weight_scale", "BF16", []int32{2, 1}, scale1),
		st.NewTensorDataFromBytes(group+".1.gate_proj.weight_shape", "I64", []int32{2}, i64le(2, 32)),
	})

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatalf("ReadInventory() error = %v", err)
	}
	class, err := Classify(inv, "")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	specs, err := Plan(inv, class, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	store := newCaptureStore()
	if _, err := WriteBlobs(specs, dir, store); err != nil {
		t.Fatalf("WriteBlobs() error = %v", err)
	}

	blob, ok := store.blobs[group]
	if !ok {
		t.Fatalf("missing stacked expert blob; got %v", store.names())
	}
	name := group + ".gate_proj.weight"
	hdr := blobHeader(t, blob)
	if w := hdr[name]; w.Dtype != "U32" || !slices.Equal(w.Shape, []int32{2, 2, 4}) {
		t.Errorf("stacked weight = %+v, want U32 [2 2 4]", w)
	}
	if scale := hdr[name+".scale"]; scale.Dtype != "BF16" || !slices.Equal(scale.Shape, []int32{2, 2, 1}) {
		t.Errorf("stacked scale = %+v, want BF16 [2 2 1]", scale)
	}
	if bias := hdr[name+".bias"]; bias.Dtype != "BF16" || !slices.Equal(bias.Shape, []int32{2, 2, 1}) {
		t.Errorf("stacked bias = %+v, want BF16 [2 2 1]", bias)
	}

	wantPacked := append(append([]byte(nil), packed0...), packed1...)
	if got := readPackedTensorRaw(t, blob, name); !slices.Equal(got, wantPacked) {
		t.Error("packed INT4 words changed during stacking")
	}
	biasValues, err := DecodeFloatTensor("BF16", readPackedTensorRaw(t, blob, name+".bias"))
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{-2, -4, -8, -16}; !slices.Equal(biasValues, want) {
		t.Errorf("qbias = %v, want %v", biasValues, want)
	}

	headerSize := binary.LittleEndian.Uint64(blob[:8])
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(blob[8:8+headerSize], &rawHeader); err != nil {
		t.Fatal(err)
	}
	var metadata map[string]string
	if err := json.Unmarshal(rawHeader["__metadata__"], &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata["quant_type"] != "int4" || metadata["group_size"] != "32" {
		t.Errorf("metadata = %v, want int4 group_size=32", metadata)
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

func TestDecodeSourceFP8E8M0Scale(t *testing.T) {
	if !QuantizeSupported() {
		t.Skip("MLX unavailable")
	}

	thread, err := mlxthread.Start("decode-e8m0-test", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX unavailable: %v", err)
	}
	defer func() {
		if err := thread.Stop(context.Background(), func() {
			mlx.Sweep()
			mlx.ClearCache()
		}); err != nil {
			t.Fatal(err)
		}
	}()

	values, err := mlxthread.Call(context.Background(), thread, func() ([]float32, error) {
		// E4M3 byte 0x38 is 1.0 and E8M0 byte 128 is 2^(128-127) = 2.
		rawWeight := make([]uint8, 128*128)
		for i := range rawWeight {
			rawWeight[i] = 0x38
		}
		weight := mlx.FromValues(rawWeight, 128, 128)
		scale := mlx.FromValues([]uint8{128}, 1, 1)
		decoded, err := decodeSourceFP8Tensor(weight, scale)
		if err != nil {
			return nil, err
		}
		decoded = decoded.AsType(mlx.DTypeFloat32)
		mlx.Eval(decoded)
		return append([]float32(nil), decoded.Floats()...), nil
	})
	if err != nil {
		t.Fatalf("decodeSourceFP8Tensor() error = %v", err)
	}
	for _, index := range []int{0, len(values) / 2, len(values) - 1} {
		if values[index] != 2 {
			t.Fatalf("decoded[%d] = %v, want 2", index, values[index])
		}
	}
}
