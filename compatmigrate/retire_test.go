package compatmigrate

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestRetireConvertedModels(t *testing.T) {
	for _, scenario := range []string{"converted", "shared source", "missing output", "user list", "legacy only", "other selected child"} {
		t.Run(scenario, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			name := model.ParseName("old:latest")
			writeSourceManifest(t, name, sourceManifestInput{
				config:       model.ConfigV2{ModelFormat: "gguf", ModelFamily: "old"},
				modelKV:      outKV{"general.architecture": "old"},
				modelTensors: []*outTensor{fixtureTensor("old.weight", gguf.TensorTypeF16, []uint64{1, 8})},
			})
			original, err := manifest.ParseNamedManifest(name)
			if err != nil {
				t.Fatal(err)
			}
			if err := manifest.FillMetadata(original); err != nil {
				t.Fatal(err)
			}
			oldPath, err := manifest.BlobsPath(original.Layers[0].Digest)
			if err != nil {
				t.Fatal(err)
			}
			if scenario == "shared source" {
				data, err := manifest.ReadManifestData(name)
				if err != nil {
					t.Fatal(err)
				}
				if err := manifest.WriteManifestData(model.ParseName("other:custom"), data); err != nil {
					t.Fatal(err)
				}
			}
			if scenario != "legacy only" {
				convertedLayer := writeFixtureGGUFLayer(t, outKV{"general.architecture": "clean"}, []*outTensor{
					fixtureTensor("clean.weight", gguf.TensorTypeF16, []uint64{1, 8}),
				})
				child := *original
				child.Runner = manifest.RunnerLlamaCPP
				child.Format = manifest.FormatGGUF
				child.Layers = []manifest.Layer{convertedLayer}
				clean, err := manifestReferenceForChild(&child)
				if err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(&child)
				if err != nil {
					t.Fatal(err)
				}
				shadow, err := convertedLegacyShadowName(clean.BlobDigest())
				if err != nil {
					t.Fatal(err)
				}
				if err := manifest.WriteLegacyManifestData(shadow, data); err != nil {
					t.Fatal(err)
				}
				legacy, err := manifestReferenceForChild(original)
				if err != nil {
					t.Fatal(err)
				}
				foreign, err := manifest.NewManifestReference("sha256:"+strings.Repeat("f", 64), manifest.RunnerMLX, manifest.FormatSafetensors)
				if err != nil {
					t.Fatal(err)
				}
				refs := []manifest.Manifest{legacy, clean, foreign}
				// Reproduce the Phase 2 layout explicitly, without invoking a conversion.
				parentDigest, err := manifest.WriteManifestListPreserveLegacy(name, refs)
				if err != nil {
					t.Fatal(err)
				}
				if scenario != "user list" {
					if err := manifest.WriteLegacyAnchor(name, original, parentDigest, legacy.BlobDigest(), clean.BlobDigest()); err != nil {
						t.Fatal(err)
					}
				}
				if scenario == "missing output" {
					path, _ := manifest.BlobsPath(convertedLayer.Digest)
					if err := os.Remove(path); err != nil {
						t.Fatal(err)
					}
				}
			}
			before, err := manifest.ReadManifestData(name)
			if err != nil {
				t.Fatal(err)
			}
			if scenario == "other selected child" {
				if digest, err := retireConvertedModel(name, "sha256:"+strings.Repeat("f", 64)); err != nil || digest != "" {
					t.Fatalf("unrelated child redirected: %s, %v", digest, err)
				}
			} else {
				if err := RetireConvertedModels(); err != nil {
					t.Fatal(err)
				}
			}
			after, err := manifest.ReadManifestData(name)
			if err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "converted", "shared source":
				var parent manifest.Manifest
				if err := json.Unmarshal(after, &parent); err != nil {
					t.Fatal(err)
				}
				if len(parent.Manifests) != 2 || parent.Manifests[0].Runner != manifest.RunnerLlamaCPP || parent.Manifests[1].Runner != manifest.RunnerMLX {
					t.Fatalf("retired list lost replacement or foreign child: %s", after)
				}
				if _, err := manifest.ReadLegacyManifestData(name); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("legacy anchor remains: %v", err)
				}
			default:
				if !bytes.Equal(before, after) {
					t.Fatalf("%s unexpectedly modified", scenario)
				}
			}
			_, err = os.Stat(oldPath)
			if scenario == "converted" {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("unreferenced source remains: %v", err)
				}
			} else if err != nil {
				t.Fatalf("source was incorrectly reclaimed: %v", err)
			}
			if err := RetireConvertedModels(); err != nil {
				t.Fatalf("repeat retirement: %v", err)
			}
		})
	}
}
