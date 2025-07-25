package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

func createManifest(t *testing.T, path, name string) {
	t.Helper()

	p := filepath.Join(path, "manifests", name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(Manifest{}); err != nil {
		t.Fatal(err)
	}
}

func TestManifests(t *testing.T) {
	cases := map[string]struct {
		ps               []string
		wantValidCount   int
		wantInvalidCount int
	}{
		"empty": {},
		"single": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag"),
			},
			wantValidCount: 1,
		},
		"multiple": {
			ps: []string{
				filepath.Join("registry.ollama.ai", "library", "llama3", "latest"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_0"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_1"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q8_0"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_0"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_1"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q2_K"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_S"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_M"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_L"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_K_S"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_K_M"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_K_S"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_K_M"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q6_K"),
			},
			wantValidCount: 15,
		},
		"hidden": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag"),
				filepath.Join("host", "namespace", "model", ".hidden"),
			},
			wantValidCount:   1,
			wantInvalidCount: 1,
		},
		"subdir": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag", "one"),
				filepath.Join("host", "namespace", "model", "tag", "another", "one"),
			},
			wantInvalidCount: 2,
		},
		"upper tag": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "TAG"),
			},
			wantValidCount: 1,
		},
		"upper model": {
			ps: []string{
				filepath.Join("host", "namespace", "MODEL", "tag"),
			},
			wantValidCount: 1,
		},
		"upper namespace": {
			ps: []string{
				filepath.Join("host", "NAMESPACE", "model", "tag"),
			},
			wantValidCount: 1,
		},
		"upper host": {
			ps: []string{
				filepath.Join("HOST", "namespace", "model", "tag"),
			},
			wantValidCount: 1,
		},
	}

	for n, wants := range cases {
		t.Run(n, func(t *testing.T) {
			d := t.TempDir()
			t.Setenv("OLLAMA_MODELS", d)

			for _, p := range wants.ps {
				createManifest(t, d, p)
			}

			ms, err := Manifests(true)
			if err != nil {
				t.Fatal(err)
			}

			var ns []model.Name
			for k := range ms {
				ns = append(ns, k)
			}

			var gotValidCount, gotInvalidCount int
			for _, p := range wants.ps {
				n := model.ParseNameFromFilepath(p)
				if n.IsValid() {
					gotValidCount++
				} else {
					gotInvalidCount++
				}

				if !n.IsValid() && slices.Contains(ns, n) {
					t.Errorf("unexpected invalid name: %s", p)
				} else if n.IsValid() && !slices.Contains(ns, n) {
					t.Errorf("missing valid name: %s", p)
				}
			}

			if gotValidCount != wants.wantValidCount {
				t.Errorf("got valid count %d, want %d", gotValidCount, wants.wantValidCount)
			}

			if gotInvalidCount != wants.wantInvalidCount {
				t.Errorf("got invalid count %d, want %d", gotInvalidCount, wants.wantInvalidCount)
			}
		})
	}
}

func TestManifestSignatureIntegration(t *testing.T) {
	// Test Layer.IsSignature method
	t.Run("signature layer detection", func(t *testing.T) {
		tests := []struct {
			name      string
			layer     Layer
			expected  bool
		}{
			{
				name: "OMS signature layer",
				layer: Layer{
					MediaType: "application/vnd.oms.signature.v1+json",
					Digest:    "sha256:test",
				},
				expected: true,
			},
			{
				name: "model layer",
				layer: Layer{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    "sha256:test",
				},
				expected: false,
			},
			{
				name: "system layer",
				layer: Layer{
					MediaType: "application/vnd.ollama.image.system",
					Digest:    "sha256:test",
				},
				expected: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := tt.layer.IsSignature()
				if result != tt.expected {
					t.Errorf("Expected IsSignature()=%v, got %v", tt.expected, result)
				}
			})
		}
	})

	// Test Manifest.GetSignatureLayer method
	t.Run("get signature layer", func(t *testing.T) {
		manifest := &Manifest{
			Config: Layer{Digest: "sha256:config"},
			Layers: []Layer{
				{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    "sha256:model",
				},
				{
					MediaType: "application/vnd.oms.signature.v1+json",
					Digest:    "sha256:signature",
				},
				{
					MediaType: "application/vnd.ollama.image.system",
					Digest:    "sha256:system",
				},
			},
		}

		sigLayer := manifest.GetSignatureLayer()
		if sigLayer == nil {
			t.Fatal("Expected signature layer, got nil")
		}
		if sigLayer.MediaType != "application/vnd.oms.signature.v1+json" {
			t.Errorf("Expected OMS signature media type, got %s", sigLayer.MediaType)
		}
		if sigLayer.Digest != "sha256:signature" {
			t.Errorf("Expected signature digest, got %s", sigLayer.Digest)
		}
	})

	// Test manifest without signature layer
	t.Run("no signature layer", func(t *testing.T) {
		manifest := &Manifest{
			Config: Layer{Digest: "sha256:config"},
			Layers: []Layer{
				{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    "sha256:model",
				},
				{
					MediaType: "application/vnd.ollama.image.system",
					Digest:    "sha256:system",
				},
			},
		}

		sigLayer := manifest.GetSignatureLayer()
		if sigLayer != nil {
			t.Errorf("Expected nil signature layer, got %+v", sigLayer)
		}
	})
}

func TestManifestSignatureInfo(t *testing.T) {
	// Test SignatureInfo structure and JSON serialization
	t.Run("signature info serialization", func(t *testing.T) {
		sigInfo := &SignatureInfo{
			Format:       "oms-v1.0",
			SignatureURI: "sha256:abcd1234",
			Verified:     true,
			Signer:       "test@example.com",
			SignedAt:     time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC),
		}

		// Test JSON marshaling
		data, err := json.Marshal(sigInfo)
		if err != nil {
			t.Fatalf("Failed to marshal SignatureInfo: %v", err)
		}

		// Test JSON unmarshaling
		var unmarshaled SignatureInfo
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("Failed to unmarshal SignatureInfo: %v", err)
		}

		// Verify fields
		if unmarshaled.Format != sigInfo.Format {
			t.Errorf("Format mismatch: %s != %s", unmarshaled.Format, sigInfo.Format)
		}
		if unmarshaled.SignatureURI != sigInfo.SignatureURI {
			t.Errorf("SignatureURI mismatch: %s != %s", unmarshaled.SignatureURI, sigInfo.SignatureURI)
		}
		if unmarshaled.Verified != sigInfo.Verified {
			t.Errorf("Verified mismatch: %v != %v", unmarshaled.Verified, sigInfo.Verified)
		}
		if unmarshaled.Signer != sigInfo.Signer {
			t.Errorf("Signer mismatch: %s != %s", unmarshaled.Signer, sigInfo.Signer)
		}
	})
}

func TestManifestWithSignature(t *testing.T) {
	// Test complete manifest with signature
	t.Run("manifest with signature", func(t *testing.T) {
		manifest := &Manifest{
			SchemaVersion: 2,
			MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
			Config: Layer{
				MediaType: "application/vnd.docker.container.image.v1+json",
				Digest:    "sha256:config123",
				Size:      490,
			},
			Layers: []Layer{
				{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    "sha256:model123",
					Size:      1000000,
				},
				{
					MediaType: "application/vnd.ollama.image.system",
					Digest:    "sha256:system123", 
					Size:      68,
				},
				{
					MediaType: "application/vnd.oms.signature.v1+json",
					Digest:    "sha256:signature123",
					Size:      398,
				},
			},
			Signature: &SignatureInfo{
				Format:       "oms-v1.0",
				SignatureURI: "sha256:signature123",
				Verified:     true,
				Signer:       "test@example.com",
				SignedAt:     time.Now().UTC(),
			},
		}

		// Test JSON serialization of full manifest
		data, err := json.Marshal(manifest)
		if err != nil {
			t.Fatalf("Failed to marshal manifest: %v", err)
		}

		var unmarshaled Manifest
		if err := json.Unmarshal(data, &unmarshaled); err != nil {
			t.Fatalf("Failed to unmarshal manifest: %v", err)
		}

		// Verify signature info is preserved
		if unmarshaled.Signature == nil {
			t.Fatal("Signature info lost during JSON round-trip")
		}
		if unmarshaled.Signature.Format != manifest.Signature.Format {
			t.Error("Signature format mismatch after JSON round-trip")
		}
		if unmarshaled.Signature.Signer != manifest.Signature.Signer {
			t.Error("Signature signer mismatch after JSON round-trip")
		}

		// Verify signature layer is detected
		sigLayer := unmarshaled.GetSignatureLayer()
		if sigLayer == nil {
			t.Fatal("Signature layer not found after JSON round-trip")
		}
		if sigLayer.Digest != "sha256:signature123" {
			t.Error("Signature layer digest mismatch after JSON round-trip")
		}
	})
}

func TestManifestComputeModelDigest(t *testing.T) {
	// Test model digest computation with different layer configurations
	t.Run("model digest computation", func(t *testing.T) {
		manifest := &Manifest{
			Config: Layer{Digest: "sha256:config123"},
			Layers: []Layer{
				{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    "sha256:model123",
				},
				{
					MediaType: "application/vnd.ollama.image.system",
					Digest:    "sha256:system123",
				},
				{
					MediaType: "application/vnd.oms.signature.v1+json",
					Digest:    "sha256:signature123",
				},
			},
		}

		digest1, err := ComputeModelDigest(manifest)
		if err != nil {
			t.Fatalf("Failed to compute model digest: %v", err)
		}

		// Add another signature layer - digest should remain the same
		manifest.Layers = append(manifest.Layers, Layer{
			MediaType: "application/vnd.oms.signature.v1+json", 
			Digest:    "sha256:anothersig",
		})

		digest2, err := ComputeModelDigest(manifest)
		if err != nil {
			t.Fatalf("Failed to compute model digest with extra signature: %v", err)
		}

		if digest1 != digest2 {
			t.Errorf("Model digest changed after adding signature layer: %s != %s", digest1, digest2)
		}

		// Change a non-signature layer - digest should change
		manifest.Layers[0].Digest = "sha256:newmodel"
		digest3, err := ComputeModelDigest(manifest)
		if err != nil {
			t.Fatalf("Failed to compute model digest with changed layer: %v", err)
		}

		if digest1 == digest3 {
			t.Error("Model digest should change when non-signature layer changes")
		}
	})
}
