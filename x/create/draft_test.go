package create

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

// recordingStore captures the blobs a pipeline run produces so tests can assert
// on their names and contents.
type recordingStore struct {
	names []string
	blobs map[string][]byte
}

func (s *recordingStore) WriteBlob(r io.Reader, mediaType, name string) (LayerInfo, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return LayerInfo{}, err
	}
	if s.blobs == nil {
		s.blobs = map[string][]byte{}
	}
	s.blobs[name] = data
	s.names = append(s.names, name)
	return LayerInfo{Digest: "sha256:" + name, Size: int64(len(data)), MediaType: mediaType, Name: name}, nil
}

func TestPrefixSpecs(t *testing.T) {
	specs := []BlobSpec{
		{
			Name: "model.layers.0.mlp.experts",
			Tensors: []TensorSpec{{
				Name:     "model.layers.0.mlp.experts.gate_proj.weight",
				Sources:  []SourceTensor{{Name: "model.layers.0.mlp.experts.0.gate_proj.weight", File: "a.safetensors"}},
				Quantize: "int8",
			}},
		},
	}
	got := prefixSpecs(specs, "draft.")

	if got[0].Name != "draft.model.layers.0.mlp.experts" {
		t.Errorf("blob name = %q, want draft.-prefixed", got[0].Name)
	}
	if got[0].Tensors[0].Name != "draft.model.layers.0.mlp.experts.gate_proj.weight" {
		t.Errorf("tensor name = %q, want draft.-prefixed", got[0].Tensors[0].Name)
	}
	if got[0].Tensors[0].Quantize != "int8" {
		t.Errorf("quantize = %q, want it preserved", got[0].Tensors[0].Quantize)
	}
	// Sources point at the source files and must not be prefixed.
	if got[0].Tensors[0].Sources[0].Name != "model.layers.0.mlp.experts.0.gate_proj.weight" {
		t.Errorf("source name = %q, want unchanged", got[0].Tensors[0].Sources[0].Name)
	}
	// The input must not be mutated.
	if specs[0].Name != "model.layers.0.mlp.experts" || specs[0].Tensors[0].Name != "model.layers.0.mlp.experts.gate_proj.weight" {
		t.Errorf("prefixSpecs mutated its input: %+v", specs[0])
	}
}

func TestDraftPolicyKeepsEmbeddingsUnquantized(t *testing.T) {
	p := draftPolicy{defaultQuantPolicy{}}

	// Draft token embeddings start unquantized regardless of the request.
	if got := p.quantizationType("model.embed_tokens.weight", []int32{4096, 2048}, "int8"); got != "" {
		t.Errorf("draft embed_tokens quant = %q, want \"\"", got)
	}
	// Other eligible weights still follow the wrapped policy.
	if got := p.quantizationType("model.layers.0.mlp.down_proj.weight", []int32{2048, 2048}, "int8"); got == "" {
		t.Errorf("draft down_proj quant = \"\", want it quantized via the inner policy")
	}
}

func TestCreateDraftLayersPrefixesNamesAndConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["LlamaForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{4, 8}, make([]byte, 4*8*2)),
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 8*2)),
	})

	store := &recordingStore{}
	layers, err := CreateDraftLayers(dir, "draft.", "draft/", "", store, func(string) {})
	if err != nil {
		t.Fatalf("CreateDraftLayers: %v", err)
	}
	if len(layers) == 0 {
		t.Fatal("CreateDraftLayers returned no layers")
	}

	// Tensor blobs are namespaced under draft.; the config under draft/.
	if !slices.Contains(store.names, "draft.model.embed_tokens.weight") {
		t.Errorf("missing draft.-prefixed tensor blob; got %v", store.names)
	}
	if !slices.Contains(store.names, "draft/config.json") {
		t.Errorf("missing draft/config.json; got %v", store.names)
	}

	// The prefix must also land inside the blob, since the runtime resolves
	// draft tensors by the "draft." name prefix.
	names := readSafetensorsHeaderNames(t, store.blobs["draft.model.embed_tokens.weight"])
	if !slices.Contains(names, "draft.model.embed_tokens.weight") {
		t.Errorf("in-blob tensor name not prefixed; got %v", names)
	}
}

func TestCreateDraftLayersRejectsEmptyPrefixes(t *testing.T) {
	store := &recordingStore{}
	if _, err := CreateDraftLayers(t.TempDir(), "", "draft/", "", store, func(string) {}); err == nil {
		t.Error("expected an error for an empty tensor prefix")
	}
	if _, err := CreateDraftLayers(t.TempDir(), "draft.", "", "", store, func(string) {}); err == nil {
		t.Error("expected an error for an empty config prefix")
	}
}
