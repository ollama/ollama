package create

import (
	"encoding/json"
	"io"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

func TestDeepSeekV41ExpandE8M0(t *testing.T) {
	source := safetensors.NewTensorDataFromBytes("layer.scale", "F8_E8M0", []int32{2, 2}, []byte{1, 2, 3, 4})
	spec := TensorSpec{Name: "layer.weight.scale", Transform: TransformExpandE8M0, OutDtype: "U8", OutShape: []int32{64, 2}}
	got, err := applyByteTransform(spec, []*safetensors.TensorData{source})
	if err != nil {
		t.Fatal(err)
	}
	if got.Dtype != "U8" || !slices.Equal(got.Shape, []int32{64, 2}) {
		t.Fatalf("expanded scale header = %s %v", got.Dtype, got.Shape)
	}
	buf := make([]byte, got.Size)
	if _, err := io.ReadFull(got.Reader(), buf); err != nil {
		t.Fatal(err)
	}
	for row := range 64 {
		want := []byte{1, 2}
		if row >= 32 {
			want = []byte{3, 4}
		}
		if !slices.Equal(buf[row*2:row*2+2], want) {
			t.Fatalf("row %d = %v, want %v", row, buf[row*2:row*2+2], want)
		}
	}
}

func deepSeekV41TestInventory() Inventory {
	config := []byte(`{"quantization_config":{"quant_method":"fp8","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"},"text_config":{"num_hidden_layers":1,"n_routed_experts":2,"num_nextn_predict_layers":1,"dspark_n_routed_experts":1}}`)
	inv := Inventory{
		Config:    sourceModelConfig{Architectures: []string{"DeepseekV41ForCausalLM"}},
		RawConfig: json.RawMessage(config),
		Tensors:   make(map[string]SourceTensor),
	}
	add := func(name, dtype string, shape ...int32) {
		inv.Tensors[name] = SourceTensor{Name: name, Dtype: dtype, Shape: shape}
	}
	add("embed.weight", "BF16", 128, 64)
	add("layers.0.attn.wq_a.weight", "F8_E4M3", 64, 64)
	add("layers.0.attn.wq_a.scale", "F8_E8M0", 2, 2)
	add("layers.0.engram.embed.weight", "F8_E4M3", 65, 256)
	add("layers.0.engram.embed.scale", "F8_E8M0", 65, 8)
	for _, proj := range []string{"w1", "w2", "w3"} {
		for idx := range 2 {
			base := "layers.0.ffn.experts." + string(rune('0'+idx)) + "." + proj
			add(base+".weight", "I8", 32, 32)
			add(base+".scale", "F8_E8M0", 32, 2)
		}
	}
	for _, proj := range []string{"w1", "w2", "w3"} {
		base := "mtp.0.ffn.experts.0." + proj
		add(base+".weight", "I8", 32, 32)
		add(base+".scale", "F8_E8M0", 32, 2)
	}
	return inv
}

func TestPlanDeepSeekV41PreservesPublishedQuantization(t *testing.T) {
	inv := deepSeekV41TestInventory()
	class, err := Classify(inv, "")
	if err != nil || class.Kind != SourcePrequantized || class.Quantize != "" {
		t.Fatalf("Classify() = %+v, %v", class, err)
	}
	for _, requested := range []string{"int4", "nvfp4", "mxfp4", "mxfp8"} {
		if _, err := Classify(inv, requested); err == nil {
			t.Fatalf("requantization request %q was accepted", requested)
		}
	}
	specs, err := Plan(inv, class, defaultQuantPolicy{})
	if err != nil {
		t.Fatal(err)
	}
	block, ok := specByName(specs, "layers.0.attn.wq_a.weight")
	if !ok || block.Metadata["quant_type"] != "mxfp8" || block.Tensors[1].Transform != TransformExpandE8M0 ||
		!slices.Equal(block.Tensors[1].OutShape, []int32{64, 2}) {
		t.Fatalf("block FP8 = %+v", block)
	}
	row, ok := specByName(specs, "layers.0.engram.embed.weight")
	if !ok || row.Tensors[1].Transform != TransformRelabelU8 || !slices.Equal(row.Tensors[0].OutShape, []int32{65, 64}) {
		t.Fatalf("row FP8 = %+v", row)
	}
	bank, ok := specByName(specs, "layers.0.ffn.experts")
	if !ok || bank.Metadata["quant_type"] != "mxfp4" || len(bank.Tensors) != 6 ||
		!slices.Equal(bank.Tensors[0].OutShape, []int32{2, 32, 8}) ||
		!slices.Equal(bank.Tensors[1].OutShape, []int32{2, 32, 2}) ||
		!strings.Contains(bank.Tensors[0].Sources[1].Name, ".experts.1.") {
		t.Fatalf("expert bank = %+v", bank)
	}
	mtp, ok := specByName(specs, "mtp.0.ffn.experts")
	if !ok || len(mtp.Tensors) != 6 || !slices.Equal(mtp.Tensors[0].OutShape, []int32{1, 32, 8}) {
		t.Fatalf("MTP bank = %+v", mtp)
	}
}

func TestPlanDeepSeekV41RejectsBadLayouts(t *testing.T) {
	tests := []struct {
		name string
		edit func(Inventory)
	}{
		{"missing FP8 scale", func(inv Inventory) { delete(inv.Tensors, "layers.0.attn.wq_a.scale") }},
		{"unpaired scale", func(inv Inventory) {
			inv.Tensors["other.scale"] = SourceTensor{Name: "other.scale", Dtype: "F8_E8M0", Shape: []int32{1, 2}}
		}},
		{"wrong FP4 scale shape", func(inv Inventory) {
			s := inv.Tensors["layers.0.ffn.experts.0.w1.scale"]
			s.Shape = []int32{32, 1}
			inv.Tensors[s.Name] = s
		}},
		{"wrong engram width", func(inv Inventory) {
			w := inv.Tensors["layers.0.engram.embed.weight"]
			w.Shape[1] = 128
			inv.Tensors[w.Name] = w
			s := inv.Tensors["layers.0.engram.embed.scale"]
			s.Shape[1] = 4
			inv.Tensors[s.Name] = s
		}},
		{"missing expert", func(inv Inventory) {
			delete(inv.Tensors, "layers.0.ffn.experts.1.w3.weight")
			delete(inv.Tensors, "layers.0.ffn.experts.1.w3.scale")
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inv := deepSeekV41TestInventory()
			tt.edit(inv)
			if _, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{}); err == nil {
				t.Fatal("invalid source accepted")
			}
		})
	}
}
