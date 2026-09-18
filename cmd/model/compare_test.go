package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
)

type fixtureTensor struct {
	name, dtype string
	shape       []int
	data        []byte
}

// Construct bytes independently of the extraction/writer code under test.
func safetensorsBytes(t *testing.T, tensors []fixtureTensor, meta map[string]string) []byte {
	t.Helper()
	header := make(map[string]any)
	if meta != nil {
		header["__metadata__"] = meta
	}
	var payload []byte
	for _, tensor := range tensors {
		shape := append([]int{}, tensor.shape...)
		header[tensor.name] = map[string]any{"dtype": tensor.dtype, "shape": shape, "data_offsets": []int{len(payload), len(payload) + len(tensor.data)}}
		payload = append(payload, tensor.data...)
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := binary.Write(&b, binary.LittleEndian, uint64(len(raw))); err != nil {
		t.Fatal(err)
	}
	b.Write(raw)
	b.Write(payload)
	return b.Bytes()
}

func writeFixture(t *testing.T, root, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func tensorFixture(t *testing.T, root, name string, tensors []fixtureTensor, meta map[string]string) string {
	t.Helper()
	return writeFixture(t, root, name, safetensorsBytes(t, tensors, meta))
}

func blobFixture(t *testing.T, store, kind, name string, data []byte) manifest.Layer {
	t.Helper()
	sum := sha256.Sum256(data)
	hexsum := hex.EncodeToString(sum[:])
	writeFixture(t, store, "blobs/sha256-"+hexsum, data)
	return manifest.Layer{Name: name, MediaType: kind, Digest: "sha256:" + hexsum, Size: int64(len(data))}
}

func manifestFixture(t *testing.T, store, name string, layers ...manifest.Layer) {
	t.Helper()
	manifestWithConfigFixture(t, store, name, []byte(`{"model_format":"safetensors"}`), layers...)
}

func manifestWithConfigFixture(t *testing.T, store, name string, configData []byte, layers ...manifest.Layer) {
	t.Helper()
	config := blobFixture(t, store, "application/vnd.docker.container.image.v1+json", "", configData)
	m := manifest.Manifest{SchemaVersion: 2, MediaType: "application/vnd.docker.distribution.manifest.v2+json", Config: config, Layers: layers}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	writeFixture(t, store, "manifests/registry.ollama.ai/library/"+name+"/latest", data)
}

func TestNormalizationAndRepack(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	tensors := []fixtureTensor{{"model.text.weight", "BF16", []int{2}, []byte{1, 2, 3, 4}}, {"model.vision.bias", "F32", nil, []byte{5, 6, 7, 8}}}
	tensorFixture(t, a, "packed.safetensors", tensors, map[string]string{"format": "pt"})
	for i, tensor := range tensors {
		tensorFixture(t, b, []string{"first.safetensors", "second.safetensors"}[i], []fixtureTensor{tensor}, map[string]string{"format": "pt"})
	}
	writeFixture(t, b, "model.safetensors.index.json", []byte(`{"metadata":{"total_size":8},"weight_map":{"model.text.weight":"first.safetensors","model.vision.bias":"second.safetensors"}}`))
	writeFixture(t, a, "config.json", []byte(`{"z":1,"n":18446744073709551615,"a":{"two":2,"one":1}}`))
	writeFixture(t, b, "config.json", []byte("{\n\"a\": {\"one\": 1.0, \"two\": 2e0}, \"n\":18446744073709551615, \"z\":1.0\n}"))
	writeFixture(t, a, "LICENSE", []byte("hello\n\nworld \t!\n"))
	writeFixture(t, b, "LICENSE", []byte(" hello world ! "))
	r, err := Compare(t.Context(), a, b, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Equal || len(r.Metadata) != 0 || r.Summary.Equal != 2 || r.Summary.BytesHashed != 16 || r.Summary.LayoutChanges != 2 {
		t.Fatalf("unexpected comparison: %+v", r)
	}
	for _, d := range r.Tensors {
		if d.Verification != "sha256" || d.Left.SHA256 != d.Right.SHA256 {
			t.Fatalf("missing payload equality: %+v", d)
		}
	}
}

func TestTrustSharedBlobsAcrossStores(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	data := safetensorsBytes(t, []fixtureTensor{{"weight", "U32", []int{1}, []byte{1, 2, 3, 4}}, {"weight.scale", "U8", []int{1}, []byte{5}}}, map[string]string{"quant_type": "int8"})
	for i, store := range []string{a, b} {
		layer := blobFixture(t, store, manifest.MediaTypeImageTensor, "packed-layer-not-a-tensor-name", data)
		config := blobFixture(t, store, "application/vnd.ollama.image.json", "config.json", []byte([]string{`{"b":2,"a":1}`, "{\n\"a\": 1, \"b\": 2}"}[i]))
		license := blobFixture(t, store, "application/vnd.ollama.image.license", "", []byte([]string{"hello\nworld", " hello   world "}[i]))
		other := blobFixture(t, store, "application/vnd.ollama.image.license", "", []byte("another license"))
		if i == 0 {
			manifestFixture(t, store, "test", layer, config, license, other)
		} else {
			manifestFixture(t, store, "test", other, license, config, layer)
		}
	}
	r, err := Compare(t.Context(), "test", "test", Options{LeftStore: a, RightStore: b})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Equal || r.Summary.BlobMatched != 2 || r.Summary.BytesHashed != 0 || r.Summary.LeftBlobs != 1 || r.Summary.RightBlobs != 1 || r.Summary.SharedBlobs != 1 {
		t.Fatalf("did not trust shared blobs: %+v", r.Summary)
	}
	for _, d := range r.Tensors {
		if d.Left.SHA256 != "" || d.Right.SHA256 != "" || d.Verification != "blob" {
			t.Fatalf("fabricated or unnecessary payload checksum: %+v", d)
		}
	}
}

func TestManifestCapabilitiesIgnoreOrder(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	data := safetensorsBytes(t, []fixtureTensor{{"weight", "BF16", []int{1}, []byte{1, 2}}}, nil)
	for i, store := range []string{a, b} {
		tensor := blobFixture(t, store, manifest.MediaTypeImageTensor, "model", data)
		capabilities := []string{
			`{"model_format":"safetensors","capabilities":["completion","tools","thinking"]}`,
			`{"capabilities":["thinking","completion","tools"],"model_format":"safetensors"}`,
		}[i]
		manifestWithConfigFixture(t, store, "test", []byte(capabilities), tensor)
	}
	r, err := Compare(t.Context(), "test", "test", Options{LeftStore: a, RightStore: b})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Equal || len(r.Metadata) != 0 {
		t.Fatalf("capability ordering produced drift: %+v", r.Metadata)
	}
}

func TestArtifactMediaTypeIsSemantic(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	data := safetensorsBytes(t, []fixtureTensor{{"weight", "U8", []int{1}, []byte{1}}}, nil)
	for i, store := range []string{a, b} {
		tensor := blobFixture(t, store, manifest.MediaTypeImageTensor, "weight", data)
		kind := []string{"application/vnd.ollama.image.json", "application/vnd.ollama.image.params"}[i]
		config := blobFixture(t, store, kind, "settings.json", []byte(`{"value":1}`))
		manifestFixture(t, store, "test", tensor, config)
	}
	r, err := Compare(t.Context(), "test", "test", Options{LeftStore: a, RightStore: b})
	if err != nil {
		t.Fatal(err)
	}
	if r.Equal || len(r.Metadata) != 1 || r.Metadata[0].Path != "/artifact_media_types/settings.json" {
		t.Fatalf("media type drift hidden: %+v", r.Metadata)
	}
}

func TestSameDTypeDifferentDataAndCompanions(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	weight := safetensorsBytes(t, []fixtureTensor{{"model.text.weight", "U32", []int{1}, []byte{1, 2, 3, 4}}}, map[string]string{"quant_type": "nvfp4", "group_size": "16"})
	for i, store := range []string{a, b} {
		w := blobFixture(t, store, manifest.MediaTypeImageTensor, "weight", weight)
		s := blobFixture(t, store, manifest.MediaTypeImageTensor, "scale", safetensorsBytes(t, []fixtureTensor{{"model.text.weight.scale", "U8", []int{1}, []byte{byte(i)}}}, nil))
		bias := blobFixture(t, store, manifest.MediaTypeImageTensor, "bias", safetensorsBytes(t, []fixtureTensor{{"ordinary.bias", "BF16", nil, []byte{0, 1}}}, nil))
		manifestFixture(t, store, "test", w, s, bias)
	}
	r, err := Compare(t.Context(), "test", "test", Options{LeftStore: a, RightStore: b, Tensor: `^model\.text\.weight$`})
	if err != nil {
		t.Fatal(err)
	}
	if r.Equal || r.Summary.Changed != 1 || r.Summary.Equal != 1 || r.Summary.Total != 2 || r.Summary.BytesHashed != 2 {
		t.Fatalf("wrong companion coverage: %+v", r.Summary)
	}
	if d := r.Tensors[0]; d.Verification != "blob" || len(d.Changes) != 0 || d.Left.ModelDType != "NVFP4" || d.Left.DType != "U32" {
		t.Fatalf("packed weight was misreported: %+v", d)
	}
	if d := r.Tensors[1]; !slices.Contains(d.Changes, "payload") || d.Left.SHA256 == d.Right.SHA256 || d.Left.ModelDType != "NVFP4" || d.Left.DType != "U8" || d.Left.CompanionOf != "model.text.weight" {
		t.Fatalf("missed changed bytes: %+v", d)
	}
}

func TestLatePayloadDifferenceAndMetadataOnly(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	payload := make([]byte, (1<<20)+3)
	p := tensorFixture(t, a, "model.safetensors", []fixtureTensor{{"weight", "U8", []int{len(payload)}, payload}}, nil)
	payload[len(payload)-1] = 1
	q := tensorFixture(t, b, "model.safetensors", []fixtureTensor{{"weight", "U8", []int{len(payload)}, payload}}, nil)
	r, err := Compare(t.Context(), p, q, Options{MetadataOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Equal || r.Summary.NotChecked != 1 || r.Summary.BytesHashed != 0 || r.Scope != "local_metadata" {
		t.Fatalf("misleading metadata-only result: %+v", r)
	}
	r, err = Compare(t.Context(), p, q, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Equal || r.Summary.Changed != 1 || r.Summary.BytesHashed != int64(2*len(payload)) {
		t.Fatalf("missed late difference: %+v", r)
	}
}

func TestTensorSchemaAndNames(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	p := tensorFixture(t, a, "model.safetensors", []fixtureTensor{{"weight", "F32", []int{2}, make([]byte, 8)}, {"old", "BF16", []int{1}, make([]byte, 2)}}, nil)
	q := tensorFixture(t, b, "model.safetensors", []fixtureTensor{{"weight", "BF16", []int{2, 2}, make([]byte, 8)}, {"new", "BF16", []int{1}, make([]byte, 2)}}, nil)
	r, err := Compare(t.Context(), p, q, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Equal || r.Summary.Changed != 1 || r.Summary.Added != 1 || r.Summary.Removed != 1 {
		t.Fatalf("schema or names lost: %+v", r)
	}
	d := r.Tensors[2]
	if !slices.Equal(d.Changes, []string{"model_dtype", "dtype", "shape"}) || d.Left.SHA256 != d.Right.SHA256 {
		t.Fatalf("schema/payload conflated: %+v", d)
	}
	if _, err := Compare(t.Context(), p, q, Options{Tensor: "missing"}); err == nil {
		t.Error("empty filter should fail")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := Compare(ctx, p, q, Options{}); err == nil {
		t.Error("cancelled comparison succeeded")
	}
}

func TestIndexCompletenessAndMutation(t *testing.T) {
	dir := t.TempDir()
	path := tensorFixture(t, dir, "model-00001-of-00002.safetensors", []fixtureTensor{{"a", "U8", []int{1}, []byte{1}}}, nil)
	index := `{"metadata":{"total_size":2},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}`
	writeFixture(t, dir, "model.safetensors.index.json", []byte(index))
	if _, err := inspect(t.Context(), dir, ""); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("missing shard ignored: %v", err)
	}
	writeFixture(t, dir, "model-00002-of-00002.safetensors", safetensorsBytes(t, []fixtureTensor{{"b", "U8", []int{1}, []byte{2}}}, nil))
	inv, err := inspect(t.Context(), dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(inv.tensors) != 2 {
		t.Fatalf("index inventory incomplete: %d", len(inv.tensors))
	}
	if err := os.WriteFile(path, []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := inv.checkFiles(); err == nil {
		t.Error("mutation was not detected")
	}
}

func TestRejectTruncatedSafetensorsPayload(t *testing.T) {
	header := `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(header)))
	b.WriteString(header)
	b.WriteByte(1)
	path := writeFixture(t, t.TempDir(), "model.safetensors", b.Bytes())
	if _, err := Compare(t.Context(), path, path, Options{}); err == nil {
		t.Fatal("truncated safetensors payload compared equal")
	}
}

func TestTextDeterministic(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	p := tensorFixture(t, a, "model.safetensors", []fixtureTensor{{"a", "U8", []int{1}, []byte{1}}, {"b", "U8", []int{1}, []byte{2}}}, nil)
	q := tensorFixture(t, b, "model.safetensors", []fixtureTensor{{"a", "U8", []int{1}, []byte{3}}, {"b", "U8", []int{1}, []byte{4}}}, nil)
	r, err := Compare(t.Context(), p, q, Options{})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Compare(t.Context(), p, q, Options{})
	if err != nil {
		t.Fatal(err)
	}
	var out, out2 bytes.Buffer
	if err := WriteText(&out, r, false, false, 1); err != nil {
		t.Fatal(err)
	}
	if err := WriteText(&out2, r2, false, false, 1); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), out2.Bytes()) {
		t.Fatal("nondeterministic text")
	}
	for _, s := range []string{"2 changes: all payload-only within U8; 0 descriptor changes; 0 dtype transitions; metadata semantic 0 / provenance 0", "Summary\n=======", "1 change omitted", "--all", "--- tensor/model/a", "- payload sha256:", "2 tensors compared by SHA-256 (4 B read)"} {
		if !strings.Contains(out.String(), s) {
			t.Errorf("text missing %q: %s", s, out.String())
		}
	}
	for _, s := range []string{"Proof", "proof=", "0 not checked", "0 unchecked", "Components (name-based)", "Inspecting left"} {
		if strings.Contains(out.String(), s) {
			t.Errorf("text contains obsolete %q: %s", s, out.String())
		}
	}
}

func TestTensorComponent(t *testing.T) {
	for _, tc := range []struct {
		role, name, want string
	}{
		{"model", "model.language_model.layers.0.weight", "text"},
		{"model", "model.vision_tower.encoder.weight", "vision"},
		{"model", "model.audio_tower.encoder.weight", "audio"},
		{"model", "draft.model.layers.0.weight", "draft"},
		{"model", "blk.0.attn_q.weight", "model"},
		{"adapter", "weight", "adapter"},
	} {
		if got := tensorComponent(tc.role, tc.name); got != tc.want {
			t.Errorf("tensorComponent(%q, %q) = %q, want %q", tc.role, tc.name, got, tc.want)
		}
	}
}

func TestMetadataUsesDiffHeaders(t *testing.T) {
	r := &Report{
		Left:  Source{Reference: "left", Format: "safetensors"},
		Right: Source{Reference: "right", Format: "safetensors"},
		Metadata: []MetadataChange{{
			Path: "/config/size", LeftPresent: true, RightPresent: true, Left: json.Number("1"), Right: json.Number("2"),
		}},
		Tensors: []TensorChange{},
	}
	r.summarize()
	if r.Summary.SemanticMetadata != 1 || r.Summary.ProvenanceMetadata != 0 {
		t.Fatalf("metadata classification = semantic %d / provenance %d", r.Summary.SemanticMetadata, r.Summary.ProvenanceMetadata)
	}
	var out bytes.Buffer
	if err := WriteText(&out, r, false, false, 40); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"Metadata\n========", "--- metadata/config/size", "+++ metadata/config/size", "- 1", "+ 2"} {
		if !strings.Contains(out.String(), value) {
			t.Errorf("text missing %q: %s", value, out.String())
		}
	}
}

func TestLicenseMetadataUsesLineDiff(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	leftLines := []string{"License heading"}
	var changedLines []int
	for _, name := range []string{"first", "second", "third", "fourth"} {
		changedLines = append(changedLines, len(leftLines))
		leftLines = append(leftLines, "old "+name)
		for i := range 5 {
			leftLines = append(leftLines, fmt.Sprintf("common after %s %d", name, i+1))
		}
	}
	leftLines = append(leftLines, "end")
	rightLines := slices.Clone(leftLines)
	for i, name := range []string{"first", "second", "third", "fourth"} {
		rightLines[changedLines[i]] = "new " + name
	}
	tensorData := safetensorsBytes(t, []fixtureTensor{{"weight", "BF16", []int{1}, []byte{1, 2}}}, nil)
	for i, store := range []string{left, right} {
		tensor := blobFixture(t, store, manifest.MediaTypeImageTensor, "model", tensorData)
		licenseText := []string{strings.Join(leftLines, "\n"), strings.Join(rightLines, "\n")}[i]
		license := blobFixture(t, store, "application/vnd.ollama.image.license", "", []byte(licenseText))
		manifestFixture(t, store, "test", tensor, license)
	}
	r, err := Compare(t.Context(), "test", "test", Options{LeftStore: left, RightStore: right})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Metadata) != 1 || r.Metadata[0].Path != "/license/0" {
		t.Fatalf("unexpected license comparison: %+v", r.Metadata)
	}

	var out bytes.Buffer
	if err := WriteText(&out, r, false, false, 40); err != nil {
		t.Fatal(err)
	}
	text := out.String()
	for _, want := range []string{"@@ -", "-old first", "+new first", "-old third", "+new third", "metadata diff truncated; use --all"} {
		if !strings.Contains(text, want) {
			t.Errorf("license diff missing %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "old fourth") || strings.Contains(text, "new fourth") {
		t.Fatalf("default license diff included more than three hunks:\n%s", text)
	}

	out.Reset()
	if err := WriteText(&out, r, true, false, 40); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "-old fourth") || !strings.Contains(out.String(), "+new fourth") || strings.Contains(out.String(), "truncated") {
		t.Fatalf("--all did not print every license hunk:\n%s", out.String())
	}
}

func TestRenameDetection(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	tensorFixture(t, left, "model.safetensors", []fixtureTensor{{"old.namespace.weight", "BF16", []int{2}, []byte{1, 2, 3, 4}}}, nil)
	tensorFixture(t, right, "model.safetensors", []fixtureTensor{{"new.namespace.weight", "BF16", []int{2}, []byte{1, 2, 3, 4}}}, nil)
	r, err := Compare(t.Context(), left, right, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Renames) != 1 || r.Renames[0].Confidence != "payload" || r.Renames[0].Left != "old.namespace.weight" || r.Renames[0].Right != "new.namespace.weight" {
		t.Fatalf("rename not recognized: %+v", r.Renames)
	}
	var out bytes.Buffer
	if err := WriteText(&out, r, false, false, 40); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "Rename candidates\n=================") || !strings.Contains(out.String(), "same descriptor and payload sha256") {
		t.Fatalf("rename missing from report: %s", out.String())
	}
}

func TestExpertFusionDetection(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	tensorFixture(t, left, "model.safetensors", []fixtureTensor{
		{"model.layers.0.mlp.experts.0.gate_proj.weight", "BF16", []int{2}, []byte{1, 2, 3, 4}},
		{"model.layers.0.mlp.experts.1.gate_proj.weight", "BF16", []int{2}, []byte{5, 6, 7, 8}},
	}, nil)
	tensorFixture(t, right, "model.safetensors", []fixtureTensor{
		{"model.layers.0.mlp.experts.gate_proj.weight", "BF16", []int{2, 2}, []byte{1, 2, 3, 4, 5, 6, 7, 8}},
	}, nil)
	r, err := Compare(t.Context(), left, right, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.ExpertFusions) != 1 || len(r.ExpertFusions[0].Left) != 2 || len(r.ExpertFusions[0].Right) != 1 {
		t.Fatalf("expert fusion not recognized: %+v", r.ExpertFusions)
	}
	var out bytes.Buffer
	if err := WriteText(&out, r, false, false, 40); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "Expert fusion\n=============") || !strings.Contains(out.String(), "mapping: recognized from complete expert indices and logical shape") {
		t.Fatalf("expert fusion missing from report: %s", out.String())
	}
}

func TestMXFP8StatsAndNMSE(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	leftWeight := bytes.Repeat([]byte{0x38}, 32) // E4M3 1.0
	rightWeight := slices.Clone(leftWeight)
	rightWeight[0] = 0x7e // E4M3 maximum finite value; marks this block saturated
	for i, root := range []string{left, right} {
		weight := []fixtureTensor{
			{"weight", "U32", []int{8}, [][]byte{leftWeight, rightWeight}[i]},
			{"weight.scale", "U8", []int{1}, []byte{127}}, // E8M0 1.0
		}
		tensorFixture(t, root, "model.safetensors", weight, map[string]string{"quant_type": "mxfp8", "group_size": "32"})
	}
	r, err := Compare(t.Context(), left, right, Options{Stats: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.Stats == nil || len(r.Stats.Comparisons) != 1 {
		t.Fatalf("missing numeric comparison: %+v", r.Stats)
	}
	if r.Stats.Left.E4M3SaturatedBlocks != 0 || r.Stats.Right.E4M3SaturatedBlocks != 1 || r.Stats.Right.E4M3PayloadBlocks != 1 {
		t.Fatalf("wrong clipping statistics: left=%+v right=%+v", r.Stats.Left, r.Stats.Right)
	}
	if r.Stats.Comparisons[0].NMSE <= 0 || r.Summary.BytesHashed != 66 {
		t.Fatalf("wrong fidelity/hash accounting: comparison=%+v summary=%+v", r.Stats.Comparisons[0], r.Summary)
	}
	var out bytes.Buffer
	if err := WriteText(&out, r, false, true, 40); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"Quantization statistics", "E4M3-max blocks 100.000% [clipping signature]", "Dequantized NMSE (right vs left)"} {
		if !strings.Contains(out.String(), value) {
			t.Errorf("statistics output missing %q: %s", value, out.String())
		}
	}
	if strings.Contains(out.String(), "model/weight:") {
		t.Fatalf("--summary included per-tensor statistics: %s", out.String())
	}
}

func TestStatsTrustSharedBlobWithoutTensorHash(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	data := safetensorsBytes(t, []fixtureTensor{
		{"weight", "U32", []int{8}, bytes.Repeat([]byte{0x38}, 32)},
		{"weight.scale", "U8", []int{1}, []byte{127}},
	}, map[string]string{"quant_type": "mxfp8", "group_size": "32"})
	for _, store := range []string{left, right} {
		layer := blobFixture(t, store, manifest.MediaTypeImageTensor, "weight", data)
		manifestFixture(t, store, "test", layer)
	}
	r, err := Compare(t.Context(), "test", "test", Options{LeftStore: left, RightStore: right, Stats: true})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Equal || r.Summary.BytesHashed != 0 || r.Summary.BlobMatched != 2 {
		t.Fatalf("shared blob was rehashed: %+v", r.Summary)
	}
	if r.Stats.BytesRead != 33 || r.Stats.ExtraBytesRead != 33 || r.Stats.Left.MXFP8Tensors != 1 || r.Stats.Right.MXFP8Tensors != 1 {
		t.Fatalf("shared statistics were not reused: %+v", r.Stats)
	}
	for _, change := range r.Tensors {
		if change.Left.SHA256 != "" || change.Right.SHA256 != "" {
			t.Fatalf("fabricated nested checksum: %+v", change)
		}
	}
}

func TestNVFP4StatsAgainstBF16(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	bf16One := []byte{0x80, 0x3f}
	tensorFixture(t, left, "model.safetensors", []fixtureTensor{{"weight", "BF16", []int{16}, bytes.Repeat(bf16One, 16)}}, nil)
	tensorFixture(t, right, "model.safetensors", []fixtureTensor{
		{"weight", "U32", []int{2}, bytes.Repeat([]byte{0x22}, 8)}, // two E2M1 1.0 values per byte
		{"weight.scale", "U8", []int{1}, []byte{0x38}},             // E4M3 1.0
		{"weight.global_scale", "F32", nil, []byte{0, 0, 0x80, 0x3f}},
	}, map[string]string{"quant_type": "nvfp4", "group_size": "16"})
	r, err := Compare(t.Context(), left, right, Options{Stats: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.Stats == nil || len(r.Stats.Comparisons) != 1 || r.Stats.Comparisons[0].NMSE != 0 || r.Stats.Right.NVFP4Tensors != 1 || r.Stats.Right.E4M3Scales != 1 {
		t.Fatalf("wrong NVFP4 fidelity: %+v", r.Stats)
	}
}

func TestMetadataClassification(t *testing.T) {
	for _, tc := range []struct {
		path, want string
	}{
		{"/manifest_config/rootfs/type", "provenance"},
		{"/manifest/mediaType", "provenance"},
		{"/LICENSE", "provenance"},
		{"/params/temperature", "semantic"},
		{"/config/hidden_size", "semantic"},
	} {
		if got := metadataClass(tc.path); got != tc.want {
			t.Errorf("metadataClass(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}
