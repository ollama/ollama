package compatmigrate

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func nimbleFixtureKV() outKV {
	return outKV{
		"general.architecture":    "qwen35",
		"general.basename":        "Bespoke-Nimble",
		"general.finetune":        "merged-current",
		"tokenizer.chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}",
	}
}

func TestNimbleMigrationDetection(t *testing.T) {
	for _, tc := range []struct {
		name, key, value string
		want             bool
	}{
		{name: "original", want: true},
		{name: "different backbone", key: "general.architecture", value: "qwen3"},
		{name: "different model", key: "general.basename", value: "Tev1"},
		{name: "different checkpoint", key: "general.finetune", value: "v3"},
		{name: "already native", key: "qwen35.decision.type", value: "nimble"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kv := nimbleFixtureKV()
			if tc.key != "" {
				kv[tc.key] = tc.value
			}
			src := fixtureSourceModel(t, kv, []*outTensor{fixtureTensor("output_norm.weight", gguf.TensorTypeBF16, []uint64{4})})
			if got := (nimbleMigrator{}).NeedsMigration(src); got != tc.want {
				t.Fatalf("NeedsMigration=%v, want %v", got, tc.want)
			}
		})
	}
}

func TestNimbleMigrationPreservesStoreAndWeights(t *testing.T) {
	for _, asList := range []bool{false, true} {
		name := "single"
		if asList {
			name = "with MLX sibling"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			name := model.ParseName("arbitrary-user-name:latest")
			writeSourceManifest(t, name, sourceManifestInput{
				config:  model.ConfigV2{ModelFormat: "gguf", ModelFamily: "qwen35", Capabilities: []string{"decision"}},
				modelKV: nimbleFixtureKV(),
				modelTensors: []*outTensor{
					fixtureTensor("output_norm.weight", gguf.TensorTypeBF16, []uint64{4}),
					fixtureTensor("token_embd.weight", gguf.TensorTypeQ8_0, []uint64{32, 2}),
				},
			})
			source, err := manifest.ParseNamedManifestForRunner(name, "")
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range []struct{ kind, body string }{
				{manifest.MediaTypeImageSystem, "\nUse {{data}} as data; café & <tags>.\n"},
				{manifest.MediaTypeImageParams, `{"num_ctx":8194}`},
				{manifest.MediaTypeImageLicense, "Source license"},
			} {
				layer, err := manifest.NewLayer(strings.NewReader(entry.body), entry.kind)
				if err != nil {
					t.Fatal(err)
				}
				source.Layers = append(source.Layers, layer)
			}
			source.Runner = manifest.RunnerLlamaCPP // Published Nimble is already labelled native.
			source.Format = manifest.FormatGGUF
			data, err := json.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			if err := manifest.WriteManifestData(name, data); err != nil {
				t.Fatal(err)
			}
			if err := manifest.WriteManifestData(model.ParseName("nimble-reference:original"), data); err != nil {
				t.Fatal(err)
			}
			var sibling manifest.Manifest
			if asList {
				ref, err := manifestReferenceForChild(source)
				if err != nil {
					t.Fatal(err)
				}
				sibling, err = manifest.NewManifestReference("sha256:"+strings.Repeat("e", 64), manifest.RunnerMLX, manifest.FormatSafetensors)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := manifest.WriteManifestList(name, []manifest.Manifest{sibling, ref}); err != nil {
					t.Fatal(err)
				}
			}
			before, err := loadSourceModelFromManifest(name, source)
			if err != nil {
				t.Fatal(err)
			}
			defer before.Close()
			if _, err := WaitLocalCompatibilityMigration(t.Context(), name, ""); err != nil {
				t.Fatal(err)
			}
			native, err := manifest.ParseNamedManifestForRunner(name, manifest.RunnerLlamaCPP)
			if err != nil {
				t.Fatal(err)
			}
			after, err := loadSourceModelFromManifest(name, native)
			if err != nil {
				t.Fatal(err)
			}
			defer after.Close()
			if got := after.GGUF.KeyValue("decision.type").String(); got != "nimble" {
				t.Fatalf("decision type=%q", got)
			}
			if (nimbleMigrator{}).NeedsMigration(after) {
				t.Fatal("converted model still requires migration")
			}
			tmpl := after.GGUF.KeyValue("tokenizer.chat_template.systemone").String()
			if !strings.Contains(tmpl, "Use {{data}} as data; café & <tags>.") || !strings.HasSuffix(tmpl, before.GGUF.KeyValue("tokenizer.chat_template").String()) {
				t.Fatal("system or chat template changed")
			}
			for i, layer := range source.Layers[1:] {
				if layer.Digest != native.Layers[i+1].Digest || layer.MediaType != native.Layers[i+1].MediaType {
					t.Fatal("ancillary layer contents changed")
				}
			}
			for _, tensor := range before.GGUF.TensorInfos() {
				converted := after.GGUF.TensorInfo(tensor.Name)
				if converted.Type != tensor.Type || !slices.Equal(converted.Shape, tensor.Shape) {
					t.Fatalf("tensor %s changed dtype/shape", tensor.Name)
				}
				a, err := io.ReadAll(io.NewSectionReader(before.GGUFData, before.GGUFDataOffset+int64(tensor.Offset), tensor.NumBytes()))
				if err != nil {
					t.Fatal(err)
				}
				b, err := io.ReadAll(io.NewSectionReader(after.GGUFData, after.GGUFDataOffset+int64(converted.Offset), converted.NumBytes()))
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(a, b) {
					t.Fatalf("tensor %s changed bytes", tensor.Name)
				}
			}
			if _, err := manifest.ParseNamedManifestForRunner(name, manifest.RunnerGGML); !errors.Is(err, manifest.ErrNoCompatibleManifest) {
				t.Fatalf("legacy child retained: %v", err)
			}
			first, err := manifest.ReadManifestData(name)
			if err != nil {
				t.Fatal(err)
			}
			if asList {
				var parent manifest.Manifest
				if err := json.Unmarshal(first, &parent); err != nil {
					t.Fatal(err)
				}
				if len(parent.Manifests) != 2 || parent.Manifests[0].BlobDigest() != sibling.BlobDigest() {
					t.Fatal("MLX sibling was not preserved")
				}
			}
			if _, err := WaitLocalCompatibilityMigration(t.Context(), name, ""); err != nil {
				t.Fatal(err)
			}
			second, err := manifest.ReadManifestData(name)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(first, second) {
				t.Fatal("second migration rewrote the manifest")
			}
			if _, err := os.Stat(before.GGUFPath); err != nil {
				t.Fatal("original blob removed:", err)
			}
		})
	}
}

func TestNimbleMigrationCustomTemplateIsNotReplaced(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("custom-nimble:latest")
	writeSourceManifest(t, name, sourceManifestInput{
		config:       model.ConfigV2{ModelFormat: "gguf", ModelFamily: "qwen35"},
		modelKV:      nimbleFixtureKV(),
		modelTensors: []*outTensor{fixtureTensor("output_norm.weight", gguf.TensorTypeF32, []uint64{4})},
		template:     "custom {{ .Prompt }}",
	})
	before, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if migrated, err := ensureLocalCompatibilityMigration(name, ""); !errors.Is(err, errUnsupportedFamily) || migrated != "" {
		t.Fatalf("migration=%v, error=%v", migrated, err)
	}
	after, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("custom template manifest changed")
	}
}

func TestNimbleMigrationRequires(t *testing.T) {
	for _, tc := range []struct {
		source, want string
		invalid      bool
	}{
		{source: "", want: "0.41.0"},
		{source: "0.35.0", want: "0.41.0"},
		{source: "v0.42.0", want: "v0.42.0"},
		{source: "0.42.0-rc1", want: "0.42.0-rc1"},
		{source: "invalid", invalid: true},
	} {
		t.Run(tc.source, func(t *testing.T) {
			src := fixtureSourceModel(t, nimbleFixtureKV(), []*outTensor{fixtureTensor("token_embd.weight", gguf.TensorTypeF16, []uint64{4})})
			src.Manifest = &manifest.Manifest{}
			src.Config.Requires = tc.source
			got, err := (nimbleMigrator{}).Migrate(src)
			if tc.invalid {
				if err == nil {
					t.Fatal("invalid minimum version accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.Requires != tc.want {
				t.Fatalf("requires=%q, want %q", got.Requires, tc.want)
			}
		})
	}
}
