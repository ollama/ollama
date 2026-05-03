package model

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// fakeManifestWithConfig returns a ModelManifest whose config.json layer
// points at a real blob file in t.TempDir. Pass nil for configData to get
// a manifest without a config.json layer.
func fakeManifestWithConfig(t *testing.T, configData []byte) *manifest.ModelManifest {
	t.Helper()
	blobDir := filepath.Join(t.TempDir(), "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatalf("mkdir blobs: %v", err)
	}

	m := &manifest.Manifest{SchemaVersion: 2}
	if configData != nil {
		sum := sha256.Sum256(configData)
		hexSum := hex.EncodeToString(sum[:])
		blobPath := filepath.Join(blobDir, "sha256-"+hexSum)
		if err := os.WriteFile(blobPath, configData, 0o644); err != nil {
			t.Fatalf("write blob: %v", err)
		}
		m.Layers = []manifest.ManifestLayer{{
			MediaType: "application/vnd.ollama.image.json",
			Digest:    "sha256:" + hexSum,
			Size:      int64(len(configData)),
			Name:      "config.json",
		}}
	}
	return &manifest.ModelManifest{Manifest: m, BlobDir: blobDir}
}

func TestReadConfigQuantOverrides_NoConfig(t *testing.T) {
	m := fakeManifestWithConfig(t, nil)
	defaults, overrides, err := readConfigQuantOverrides(m)
	if err != nil || defaults.QuantType != "" || overrides != nil {
		t.Fatalf("got (%+v, %v, %v), want zero values", defaults, overrides, err)
	}
}

func TestReadConfigQuantOverrides_NoQuantizationBlock(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"hidden_size":2048}`))
	defaults, overrides, err := readConfigQuantOverrides(m)
	if err != nil || defaults.QuantType != "" || overrides != nil {
		t.Fatalf("got (%+v, %v, %v), want zero values", defaults, overrides, err)
	}
}

func TestReadConfigQuantOverrides_FlatQuantization(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization":{"group_size":16,"bits":4,"mode":"nvfp4"}}`))
	defaults, overrides, err := readConfigQuantOverrides(m)
	if err != nil {
		t.Fatal(err)
	}
	if defaults.QuantType != "NVFP4" || defaults.GroupSize != 16 || defaults.Bits != 4 || defaults.Mode != "nvfp4" {
		t.Errorf("defaults = %+v", defaults)
	}
	if overrides != nil {
		t.Errorf("overrides = %v, want nil", overrides)
	}
}

// Codex review #15744: a global config of {mode:"affine", bits:N} must not
// collapse to QuantType="AFFINE", because QuantizationParams("AFFINE") falls
// into the unknown-quant default (32, 8) and silently drops the bit intent.
func TestReadConfigQuantOverrides_AffineGlobalPreservesBits(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		wantType string
		wantBits int
	}{
		{"affine_4bit", `{"quantization":{"group_size":64,"bits":4,"mode":"affine"}}`, "INT4", 4},
		{"affine_8bit", `{"quantization":{"group_size":64,"bits":8,"mode":"affine"}}`, "INT8", 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := fakeManifestWithConfig(t, []byte(tc.body))
			defaults, _, err := readConfigQuantOverrides(m)
			if err != nil {
				t.Fatal(err)
			}
			if defaults.QuantType != tc.wantType {
				t.Errorf("QuantType = %q, want %q", defaults.QuantType, tc.wantType)
			}
			// Downstream check: QuantizationParams(QuantType) must round-trip
			// the bits intent. Otherwise Root code that derives params via
			// QuantType alone will silently drop bits.
			_, qbits, _ := QuantizationParams(defaults.QuantType)
			if qbits != tc.wantBits {
				t.Errorf("QuantizationParams(%q).bits = %d, want %d (bits intent lost)",
					defaults.QuantType, qbits, tc.wantBits)
			}
		})
	}
}

// Per-path overrides without an explicit "mode" default to "affine"; the
// QuantType derivation must apply the same mode+bits rule there too.
func TestReadConfigQuantOverrides_PerPathAffineOverridePreservesBits(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization":{
		"group_size":16,"bits":4,"mode":"nvfp4",
		"foo.gate":{"group_size":64,"bits":8}
	}}`))
	_, overrides, err := readConfigQuantOverrides(m)
	if err != nil {
		t.Fatal(err)
	}
	info := overrides["foo.gate.weight"]
	if info == nil || info.QuantType != "INT8" || info.Bits != 8 {
		t.Errorf("override = %+v, want QuantType=INT8 Bits=8", info)
	}
}

func TestReadConfigQuantOverrides_PerPathOverrideWithExplicitMode(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization":{
		"group_size":16,"bits":4,"mode":"nvfp4",
		"foo.mlp.gate":{"group_size":64,"bits":8,"mode":"affine"}
	}}`))
	_, overrides, err := readConfigQuantOverrides(m)
	if err != nil {
		t.Fatal(err)
	}
	info := overrides["foo.mlp.gate.weight"]
	if info == nil {
		t.Fatal("override for foo.mlp.gate.weight missing")
	}
	if info.QuantType != "INT8" || info.GroupSize != 64 || info.Bits != 8 || info.Mode != "affine" {
		t.Errorf("override = %+v, want QuantType=INT8 GroupSize=64 Bits=8 Mode=affine", info)
	}
}

func TestReadConfigQuantOverrides_PerPathOverrideOmittedModeIsAffine(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization":{
		"group_size":16,"bits":4,"mode":"nvfp4",
		"foo.mlp.gate":{"group_size":64,"bits":8}
	}}`))
	_, overrides, err := readConfigQuantOverrides(m)
	if err != nil {
		t.Fatal(err)
	}
	info := overrides["foo.mlp.gate.weight"]
	if info == nil || info.Mode != "affine" || info.QuantType != "INT8" || info.GroupSize != 64 || info.Bits != 8 {
		t.Errorf("override = %+v, want {INT8,64,8,affine}", info)
	}
}

func TestReadConfigQuantOverrides_QuantizationConfigAliasAccepted(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization_config":{"group_size":32,"bits":4,"mode":"mxfp4"}}`))
	defaults, _, err := readConfigQuantOverrides(m)
	if err != nil || defaults.Mode != "mxfp4" || defaults.GroupSize != 32 || defaults.Bits != 4 {
		t.Errorf("got %+v, err=%v", defaults, err)
	}
}

func TestReadConfigQuantOverrides_MultipleOverrides(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization":{
		"group_size":16,"bits":4,"mode":"nvfp4",
		"layers.0.gate":{"group_size":64,"bits":8},
		"layers.1.gate":{"group_size":64,"bits":8}
	}}`))
	_, overrides, err := readConfigQuantOverrides(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"layers.0.gate.weight", "layers.1.gate.weight"} {
		if overrides[path] == nil {
			t.Errorf("override for %q missing", path)
		}
	}
}

func TestReadConfigQuantOverrides_MalformedJSON(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{this is not valid json!`))
	defaults, overrides, err := readConfigQuantOverrides(m)
	if err != nil || defaults.QuantType != "" || overrides != nil {
		t.Fatalf("want silent fallback, got (%+v, %v, %v)", defaults, overrides, err)
	}
}

// TestRoot_OpenPopulatesFromConfigAndBlobs is the regression guard for the
// metadata/override path that caused the Qwen3.6 NVFP4 panic.
func TestRoot_OpenPopulatesFromConfigAndBlobs(t *testing.T) {
	m := fakeManifestWithConfig(t, []byte(`{"quantization":{
		"group_size":16,"bits":4,"mode":"nvfp4",
		"language_model.model.layers.0.mlp.gate":{"group_size":64,"bits":8}
	}}`))

	root, err := openFromManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if root.QuantType() != "NVFP4" || root.GroupSize() != 16 {
		t.Errorf("globals = (%q, %d), want (NVFP4, 16)", root.QuantType(), root.GroupSize())
	}
	info := root.TensorQuant("language_model.model.layers.0.mlp.gate.weight")
	if info == nil {
		t.Fatal("override not applied")
	}
	if info.Mode != "affine" || info.GroupSize != 64 || info.Bits != 8 {
		t.Errorf("gate override = %+v, want {affine,64,8}", info)
	}
}

func TestTensorQuantParams_ExplicitBitsMode(t *testing.T) {
	cases := []struct {
		name     string
		tq       *TensorQuantInfo
		wantGS   int
		wantBits int
		wantMode string
	}{
		{"mode_override_nvfp4", &TensorQuantInfo{QuantType: "INT4", GroupSize: 16, Bits: 4, Mode: "nvfp4"}, 16, 4, "nvfp4"},
		{"empty_quant_type_bits_mode_explicit", &TensorQuantInfo{GroupSize: 16, Bits: 4, Mode: "nvfp4"}, 16, 4, "nvfp4"},
		{"zero_fields_fallback_to_quant_type", &TensorQuantInfo{QuantType: "INT4"}, 64, 4, "affine"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gs, bits, mode, fromTensor := TensorQuantParams(0, 0, "", map[string]*TensorQuantInfo{"foo.weight": tc.tq}, "foo.weight")
			if gs != tc.wantGS || bits != tc.wantBits || mode != tc.wantMode || !fromTensor {
				t.Errorf("got (%d, %d, %q, %v), want (%d, %d, %q, true)", gs, bits, mode, fromTensor, tc.wantGS, tc.wantBits, tc.wantMode)
			}
		})
	}
}

func TestResolveLinearQuantParams_PerTensorOverridesGlobalViaBitsMode(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales)

	tq := map[string]*TensorQuantInfo{"foo.weight": {GroupSize: 64, Bits: 8, Mode: "affine"}}
	gs, bits, mode := ResolveLinearQuantParams(16, 4, "nvfp4", tq, "foo.weight", w, scales)
	if gs != 64 || bits != 8 || mode != "affine" {
		t.Errorf("got (%d, %d, %q), want (64, 8, affine)", gs, bits, mode)
	}
}

// TestResolveLinearQuantParams_InferenceSkippedWhenAffineFromTensorWithValidParams
// guards against shape inference accidentally overwriting a valid per-tensor
// entry when mode="affine".
func TestResolveLinearQuantParams_InferenceSkippedWhenAffineFromTensorWithValidParams(t *testing.T) {
	skipIfNoMLX(t)
	// Shapes that would infer (32, 4) if inference ran.
	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales)

	tq := map[string]*TensorQuantInfo{"foo.weight": {QuantType: "INT4", GroupSize: 64}}
	gs, bits, mode := ResolveLinearQuantParams(0, 0, "", tq, "foo.weight", w, scales)
	if gs != 64 || bits != 4 || mode != "affine" {
		t.Errorf("got (%d, %d, %q), want (64, 4, affine)", gs, bits, mode)
	}
}
