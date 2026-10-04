package create

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestKolibri1ImportPolicy(t *testing.T) {
	p, err := newKolibri1ImportTransform(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		shape []int32
		want  string
	}{
		{"model.layers.0.mlp.experts.gate_proj.weight", []int32{384, 512, 2560}, "nvfp4"},
		{"model.layers.0.mlp.experts.down_proj.weight", []int32{384, 2560, 512}, "nvfp4"},
		{"model.layers.0.mlp.gate.weight", []int32{384, 2560}, ""},
		{"model.layers.0.moe.router.expert_bias", []int32{384}, ""},
		{"model.layers.0.post_ffn_norm.weight", []int32{2560}, ""},
		{"model.layers.0.mlp.shared_experts.down_proj.weight", []int32{2560, 512}, "mxfp8"},
		{"model.layers.0.self_attn.q_proj.weight", []int32{6144, 2560}, "mxfp8"},
		{"lm_head.weight", []int32{128000, 2560}, "mxfp8"},
		{"model.embed_tokens.weight", []int32{128000, 2560}, "mxfp8"},
	} {
		if got := p.quantizationType(tc.name, tc.shape, "nvfp4"); got != tc.want {
			t.Errorf("%s: %s want %s", tc.name, got, tc.want)
		}
		if got := p.quantizationType(tc.name, tc.shape, ""); got != "" {
			t.Errorf("unrequested quantization of %s: %s", tc.name, got)
		}
	}
}

func TestKolibri1BF16Import(t *testing.T) {
	mlxtest.SkipIfUnavailable(t)
	dir := t.TempDir()
	config := `{"architectures":["Kolibri1ForCausalLM"],"model_type":"kolibri1"}`
	writeConfigJSON(t, dir, config)
	var tensors []*st.TensorData
	for e := range 2 {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			name := fmt.Sprintf("model.layers.0.mlp.experts.%d.%s.weight", e, proj)
			tensors = append(tensors, st.NewTensorDataFromBytes(name, "BF16", []int32{128, 128}, bytes.Repeat([]byte{0x80, 0x3a}, 128*128)))
		}
	}
	tensors = append(tensors, st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{128, 128}, bytes.Repeat([]byte{0x80, 0x3f}, 128*128)))
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	class, err := Classify(inv, "nvfp4")
	if err != nil {
		t.Fatal(err)
	}
	if class.Kind != SourceFloat {
		t.Fatalf("source kind %v, want %v", class.Kind, SourceFloat)
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := Plan(inv, class, policy)
	if err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	if _, err := WriteBlobs(context.Background(), specs, dir, store); err != nil {
		t.Fatal(err)
	}
	for _, spec := range specs {
		blob := store.blobs[spec.Name]
		header, meta := blobHeader(t, blob), blobMetadata(t, blob)
		for _, ts := range spec.Tensors {
			if header[ts.Name].Dtype != "U32" {
				t.Fatalf("%s not packed", ts.Name)
			}
			if ts.Name == "lm_head.weight" {
				if meta["quant_type"] != "mxfp8" {
					t.Fatalf("head metadata %v", meta)
				}
			} else {
				if header[ts.Name+".global_scale"].Dtype != "F32" {
					t.Fatalf("%s missing NVFP4 global scale", ts.Name)
				}
				if meta["quant_type"] != "nvfp4" || meta["group_size"] != "16" {
					t.Fatalf("expert metadata %v", meta)
				}
				if !slices.Equal(header[ts.Name].Shape, []int32{2, 128, 16}) {
					t.Fatalf("expert shape %v", header[ts.Name].Shape)
				}
			}
		}
	}
	cfg := inv.Config
	parser, err := parserNameForConfig(dir, cfg, "")
	if err != nil || parser != "kolibri1" {
		t.Fatalf("parser %q: %v", parser, err)
	}
	renderer, err := rendererNameForConfig(dir, cfg, "")
	if err != nil || renderer != "kolibri1" {
		t.Fatalf("renderer %q: %v", renderer, err)
	}
	caps := inferSafetensorsCapabilitiesFromConfig(cfg, "", parser)
	for _, cap := range []string{"completion", "thinking", "tools"} {
		if !slices.Contains(caps, cap) {
			t.Errorf("missing %s", cap)
		}
	}
}
