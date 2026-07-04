package create

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

func writeConfigJSON(t *testing.T, dir, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestReadInventory(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed.weight", "BF16", []int32{4, 8}, make([]byte, 4*8*2)),
		st.NewTensorDataFromBytes("model.layers.0.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatalf("ReadInventory() error = %v", err)
	}
	if len(inv.Tensors) != 2 {
		t.Fatalf("got %d tensors, want 2", len(inv.Tensors))
	}
	embed, ok := inv.Tensors["model.embed.weight"]
	if !ok {
		t.Fatal("missing model.embed.weight")
	}
	if embed.Dtype != "BF16" || !slices.Equal(embed.Shape, []int32{4, 8}) || embed.File != "model.safetensors" {
		t.Errorf("embed tensor = %+v", embed)
	}
	if !inv.Has("model.layers.0.weight") {
		t.Error("Has(model.layers.0.weight) = false, want true")
	}
}

func TestReadInventoryIncomplete(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.a.weight", "BF16", []int32{4, 4}, make([]byte, 4*4*2)),
	})
	// The index references a second shard that is not present on disk.
	index := `{"weight_map":{"model.a.weight":"model-00001-of-00002.safetensors","model.b.weight":"model-00002-of-00002.safetensors"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadInventory(dir)
	if err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("ReadInventory() error = %v, want substring %q", err, "incomplete")
	}
}

func TestReadInventorySkipsConsolidated(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	// Standard HF weights — imported.
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.layers.0.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})
	// Mistral consolidated weights shipped in the same repo — must be ignored so
	// they can't shadow or pollute the model tensors.
	createTestSafetensors(t, filepath.Join(dir, "consolidated-00001-of-00001.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("layers.0.attention.wq.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatalf("ReadInventory() error = %v", err)
	}
	if len(inv.Tensors) != 1 || !inv.Has("model.layers.0.weight") {
		t.Errorf("tensors = %v, want only model.layers.0.weight", inv.Tensors)
	}
	if inv.Has("layers.0.attention.wq.weight") {
		t.Error("consolidated tensor was imported; consolidated-*.safetensors must be skipped")
	}
}

func TestReadInventoryRejectsMonolithicPlusShards(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.a.weight", "BF16", []int32{4, 4}, make([]byte, 4*4*2)),
	})
	createTestSafetensors(t, filepath.Join(dir, "model-00001-of-00001.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.a.weight", "BF16", []int32{4, 4}, make([]byte, 4*4*2)),
	})

	_, err := ReadInventory(dir)
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ReadInventory() error = %v, want an 'ambiguous source' error", err)
	}
}

func TestReadInventoryRejectsDuplicateTensorAcrossShards(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.a.weight", "BF16", []int32{4, 4}, make([]byte, 4*4*2)),
	})
	createTestSafetensors(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.a.weight", "BF16", []int32{4, 4}, make([]byte, 4*4*2)),
	})

	_, err := ReadInventory(dir)
	if err == nil || !strings.Contains(err.Error(), "duplicate tensor") {
		t.Fatalf("ReadInventory() error = %v, want a 'duplicate tensor' error", err)
	}
}

func TestReadInventoryNoModelWeights(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	// Only consolidated weights present — nothing importable.
	createTestSafetensors(t, filepath.Join(dir, "consolidated.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("layers.0.attention.wq.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})

	_, err := ReadInventory(dir)
	if err == nil || !strings.Contains(err.Error(), "no model") {
		t.Fatalf("ReadInventory() error = %v, want a 'no model ... weights' error", err)
	}
}
