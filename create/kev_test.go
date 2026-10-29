package create

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/types/model"
)

func TestKevPipelineFileType(t *testing.T) {
	for _, tc := range []struct {
		name, config, tensor, dtype, want string
	}{
		{"prefixed BF16", `{"architectures":["KevForDecision"]}`, "model.embed_tokens.weight", "BF16", "BF16"},
		{"bare BF16", `{"architectures":["KevForDecision"]}`, "embed_tokens.weight", "BF16", "BF16"},
		{"F16", `{"architectures":["KevForDecision"]}`, "embed_tokens.weight", "F16", "F16"},
		{"F32", `{"architectures":["KevForDecision"]}`, "embed_tokens.weight", "F32", "F32"},
		{"config disagrees with weights", `{"architectures":["KevForDecision"],"dtype":"float16"}`, "embed_tokens.weight", "BF16", "BF16"},
		{"missing text embedding", `{"architectures":["KevForDecision"]}`, "model.norm.weight", "BF16", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeConfigJSON(t, dir, tc.config)
			bytesPerElement := 2
			if tc.dtype == "F32" {
				bytesPerElement = 4
			}
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes(tc.tensor, tc.dtype, []int32{8, 8}, make([]byte, 8*8*bytesPerElement)),
			})
			store := newCaptureStore()
			var got model.ConfigV2
			writeManifest := func(_ context.Context, _ string, info ManifestInfo) error {
				got = info.ModelConfig
				return nil
			}
			err := Create(t.Context(), "kev-test", dir, testPipelineOptions(), store, writeManifest, func(string) {})
			if tc.want == "" {
				if err == nil || !strings.Contains(err.Error(), "Kev source has no text embedding tensor") {
					t.Fatalf("Create() error = %v, want missing text embedding", err)
				}
				if len(store.blobs) != 0 {
					t.Fatalf("Create() wrote blobs for invalid dtype: %v", store.names())
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.FileType != tc.want {
				t.Errorf("file type = %q, want %q", got.FileType, tc.want)
			}
		})
	}
}

func TestKevQuantizationPreservesSensitiveTensors(t *testing.T) {
	policy, err := newTensorImportTransform(Inventory{Config: sourceModelConfig{Architectures: []string{"KevForDecision"}}})
	if err != nil {
		t.Fatal(err)
	}
	modes := [...]string{"mxfp8", "nvfp4"}
	for _, tc := range []struct {
		name string
		want [2]string
	}{
		{"kev_head.q.weight", [2]string{"", ""}},
		{"kev_head.k.weight", [2]string{"", ""}},
		{"model.language_model.layers.0.linear_attn.in_proj_a.weight", [2]string{"", ""}},
		{"layers.0.linear_attn.in_proj_b.weight", [2]string{"", ""}},
		{"model.language_model.layers.0.mlp.gate_proj.weight", [2]string{"mxfp8", "nvfp4"}},
		{"model.language_model.layers.0.mlp.down_proj.weight", [2]string{"mxfp8", "mxfp8"}},
		{"layers.0.mlp.gate_proj.weight", [2]string{"mxfp8", "nvfp4"}},
		{"layers.0.mlp.down_proj.weight", [2]string{"mxfp8", "mxfp8"}},
		{"embed_tokens.weight", [2]string{"", ""}},
	} {
		for i, mode := range modes {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				if got := policy.quantizationType(tc.name, []int32{1024, 1024}, mode); got != tc.want[i] {
					t.Errorf("quantization = %q, want %q", got, tc.want[i])
				}
			})
		}
	}
}
