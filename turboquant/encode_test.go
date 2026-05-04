package turboquant

import (
	"fmt"
	"math"
	"testing"
)

// testOutlierPreset returns a copy of base with outlier split forced on
// for testing. PresetTQ3 / PresetTQ3K default to OutlierCount=0 because
// outlier-split alone (without asymmetric primary quantization) hurts
// decode throughput and PPL on tested models. Tests that exercise the
// outlier-split code path explicitly opt in via this helper.
func testOutlierPreset(base Preset, count int) Preset {
	out := base
	out.OutlierCount = count
	return out
}

// TestOutlierSplitBoundaries pins the single-block vs two-block boundary.
// dim == OutlierCount must produce a single block (no split).
// dim == OutlierCount+1 is the smallest two-block encoding.
// TestMemoryFormulaMatchesMarshalSize verifies that the two-block byte-count
// formula used in fs/ggml/ggml.go GraphSize matches the actual MarshalBinary
// output size. This catches formula drift whenever Block layout changes.
func TestMemoryFormulaMatchesMarshalSize(t *testing.T) {
	// The GraphSize formula replicated below assumes the two-block
	// outlier-split layout. All shipped tq* presets default to uniform
	// (OutlierCount=0) so force outlier variants here; the formula itself
	// is what's being pinned, not the default preset state.
	tq2Split := testOutlierPreset(PresetTQ2, 32)
	tq3Split := testOutlierPreset(PresetTQ3, 32)
	cases := []struct {
		preset Preset
		dim    int
	}{
		{tq2Split, 128},
		{tq3Split, 128},
		{tq3Split, 64},
	}
	for _, tc := range cases {
		vec := make([]float32, tc.dim)

		keyEncoded, err := EncodeKeyVector(vec, tc.preset)
		if err != nil {
			t.Fatalf("%s dim=%d key: %v", tc.preset.Name, tc.dim, err)
		}
		keyData, err := keyEncoded.MarshalBinary()
		if err != nil {
			t.Fatalf("%s dim=%d key marshal: %v", tc.preset.Name, tc.dim, err)
		}

		valueEncoded, err := EncodeValueVector(vec, tc.preset)
		if err != nil {
			t.Fatalf("%s dim=%d value: %v", tc.preset.Name, tc.dim, err)
		}
		valueData, err := valueEncoded.MarshalBinary()
		if err != nil {
			t.Fatalf("%s dim=%d value marshal: %v", tc.preset.Name, tc.dim, err)
		}

		// Memory formula for an outlier-split encoded vector. Each sub-block
		// carries a dim-wide ChannelBitmap (1 bit per original-vector
		// channel), not per-sub-block uint16 indices — a significant
		// per-vector saving for dim ≫ outlier count.
		const outlierCount = uint64(32)
		outlierBits := uint64(tc.preset.OutlierBits)
		regularKeyBits := uint64(tc.preset.KeyPrimaryBits)
		regularValueBits := uint64(tc.preset.ValueBits)
		dim := uint64(tc.dim)
		outlierData := (outlierCount*outlierBits + 7) / 8
		// QJL data is included only when QJLRowsDivisor > 0 (no ship preset
		// uses QJL — the formula stays so test-only QJL fixtures stay valid).
		qjlData := uint64(0)
		if tc.preset.QJLRowsDivisor > 0 {
			qjlData = (outlierCount + 7) / 8
		}
		bitmap := 2 * ((dim + 7) / 8) // one per sub-block
		// Fixed-overhead constant per encoded vector with outlier split: 10
		// bytes of EncodedVector header + 4 bytes blockLen per block (×2) +
		// 56 bytes per-block fixed header (×2). The 56 = 52 pre-Zero + 4 for
		// the new Zero float32 field.
		wantKey := 130 + bitmap + outlierData + ((dim-outlierCount)*regularKeyBits+7)/8 + qjlData
		wantValue := 130 + bitmap + outlierData + ((dim-outlierCount)*regularValueBits+7)/8

		if uint64(len(keyData)) != wantKey {
			t.Errorf("%s dim=%d: key MarshalBinary=%d bytes, formula=%d",
				tc.preset.Name, tc.dim, len(keyData), wantKey)
		}
		if uint64(len(valueData)) != wantValue {
			t.Errorf("%s dim=%d: value MarshalBinary=%d bytes, formula=%d",
				tc.preset.Name, tc.dim, len(valueData), wantValue)
		}
	}
}

func TestOutlierSplitBoundaries(t *testing.T) {
	for _, preset := range []Preset{testOutlierPreset(PresetTQ2, 32), testOutlierPreset(PresetTQ3, 32)} {
		atBoundary := pseudoRandomVector(preset.OutlierCount, 0xbabe)
		encoded, err := EncodeKeyVector(atBoundary, preset)
		if err != nil {
			t.Fatalf("%s dim=OutlierCount: %v", preset.Name, err)
		}
		if len(encoded.Blocks) != 1 {
			t.Errorf("%s dim=%d: got %d blocks, want 1 (no split at exact boundary)",
				preset.Name, preset.OutlierCount, len(encoded.Blocks))
		}
		if len(encoded.Blocks[0].ChannelBitmap) != 0 {
			t.Errorf("%s dim=%d: single-block should have no ChannelBitmap", preset.Name, preset.OutlierCount)
		}

		minSplit := pseudoRandomVector(preset.OutlierCount+1, 0xbabe)
		encoded2, err := EncodeKeyVector(minSplit, preset)
		if err != nil {
			t.Fatalf("%s dim=OutlierCount+1: %v", preset.Name, err)
		}
		if len(encoded2.Blocks) != 2 {
			t.Errorf("%s dim=%d: got %d blocks, want 2 (minimum outlier split)",
				preset.Name, preset.OutlierCount+1, len(encoded2.Blocks))
		}
		// Regular block has dim=1; verify it round-trips cleanly.
		data, err := encoded2.MarshalBinary()
		if err != nil {
			t.Fatalf("%s dim=OutlierCount+1 marshal: %v", preset.Name, err)
		}
		decoded, _, err := DecodeVector(data)
		if err != nil {
			t.Fatalf("%s dim=OutlierCount+1 decode: %v", preset.Name, err)
		}
		if len(decoded) != preset.OutlierCount+1 {
			t.Errorf("%s dim=OutlierCount+1: decoded len=%d want %d",
				preset.Name, len(decoded), preset.OutlierCount+1)
		}
	}
}

func TestEncodeDecodeRoundTripAcrossShapes(t *testing.T) {
	testCases := []struct {
		name   string
		values []float32
		preset Preset
		maxMSE float32
	}{
		{name: "small", values: []float32{0.25, -1.5, 3.25, 0.75, -0.5, 2.0, 1.0}, preset: PresetTQ3, maxMSE: 1.5},
		{name: "non-power-of-two", values: pseudoRandomVector(70, 2), preset: PresetTQ3, maxMSE: 3.0},
		{name: "multi-head-like", values: pseudoRandomVector(128, 3), preset: PresetTQ2, maxMSE: 5.0},
		{name: "constant", values: filledVector(31, 1.5), preset: PresetTQ2, maxMSE: 1.0},
		{name: "zero", values: filledVector(64, 0), preset: PresetTQ3, maxMSE: 0.01},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := EncodeVector(tc.values, tc.preset)
			if err != nil {
				t.Fatal(err)
			}
			data, err := encoded.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			decoded, preset, err := DecodeVector(data)
			if err != nil {
				t.Fatal(err)
			}
			if preset.Name != tc.preset.Name {
				t.Fatalf("preset = %q, want %q", preset.Name, tc.preset.Name)
			}
			if len(decoded) != len(tc.values) {
				t.Fatalf("decoded len = %d, want %d", len(decoded), len(tc.values))
			}

			stats := Compare(tc.values, decoded)
			if stats.MSE > tc.maxMSE {
				t.Fatalf("MSE = %v, want <= %v", stats.MSE, tc.maxMSE)
			}
		})
	}
}

func TestEncodeVectorDeterministicBytes(t *testing.T) {
	values := pseudoRandomVector(96, 0x4242)

	encodedA, err := EncodeVector(values, PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}
	encodedB, err := EncodeVector(values, PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}

	dataA, err := encodedA.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	dataB, err := encodedB.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	if string(dataA) != string(dataB) {
		t.Fatal("expected byte-identical encoding output")
	}
}

func TestEncodeKeyAndValueUseDifferentObjectives(t *testing.T) {
	values := pseudoRandomVector(32, 0x77)
	// QJL-enabled inline preset: ship presets set QJLRowsDivisor=0, so a
	// dedicated fixture is needed to exercise the residual-sketch code path.
	qjlOn := newPreset(100, "qjl_on", 3, 3, 1, 0x35c0ffee, 4, 0)
	keyEncoded, err := EncodeKeyVector(values, qjlOn)
	if err != nil {
		t.Fatal(err)
	}
	valueEncoded, err := EncodeValueVector(values, qjlOn)
	if err != nil {
		t.Fatal(err)
	}
	if keyEncoded.Blocks[0].Objective != uint8(objectiveProduct) {
		t.Fatalf("key objective = %d, want %d", keyEncoded.Blocks[0].Objective, objectiveProduct)
	}
	if valueEncoded.Blocks[0].Objective != uint8(objectiveMSE) {
		t.Fatalf("value objective = %d, want %d", valueEncoded.Blocks[0].Objective, objectiveMSE)
	}
	if keyEncoded.Blocks[0].QJLRows == 0 {
		t.Fatal("expected product-mode key rows to carry a residual sketch")
	}
	if valueEncoded.Blocks[0].QJLRows != 0 {
		t.Fatal("expected MSE value rows to omit a residual sketch")
	}
}

func TestPresetNames(t *testing.T) {
	// User-facing presets routable via OLLAMA_KV_CACHE_TYPE.
	for _, name := range []string{
		"tq2", "tq3", "tq4",
		"tq2k", "tq3k", "tq4k",
	} {
		preset, err := PresetByName(name)
		if err != nil {
			t.Fatalf("user-facing preset %q rejected: %v", name, err)
		}
		if preset.Name != name {
			t.Fatalf("preset %q resolved to %q", name, preset.Name)
		}
	}

	// Names dropped in the 6-preset consolidation must NOT resolve. The
	// equivalent ablations are now reachable via OLLAMA_TQ_DISABLE_OUTLIERS
	// and OLLAMA_TQ_DISABLE_ASYMMETRIC env vars.
	for _, name := range []string{
		"tq3a", "tq3ka", "tq3q", "tq3kq", "tq3qa", "tq3kqa",
		"tq2a", "tq2ka", "tq2q", "tq2kq", "tq2qa", "tq2kqa",
		"tq4a", "tq4ka", "tq4qa",
	} {
		if _, err := PresetByName(name); err == nil {
			t.Errorf("retired preset %q must not resolve via PresetByName", name)
		}
	}
}

// TestQJLDimMatchesPaperSpec verifies that a QJL-enabled preset (constructed
// inline — no shipping preset uses QJL) sketches d random projections,
// matching arXiv:2504.19874's spec. The shipping tq* presets set
// QJLRowsDivisor=0, so the math is exercised here by direct construction.
func TestQJLDimMatchesPaperSpec(t *testing.T) {
	tq2qjl := newPreset(110, "tq2_qjl", 2, 2, 1, 0x25c0ffee, 3, 0)
	tq3qjl := newPreset(100, "tq3_qjl", 3, 3, 1, 0x35c0ffee, 4, 0)
	cases := []struct {
		preset Preset
		dim    int
	}{
		{tq2qjl, 64},
		{tq2qjl, 128},
		{tq3qjl, 128},
		{tq3qjl, 256},
	}
	for _, tc := range cases {
		got := tc.preset.KeyQJLRows(tc.dim)
		if got != tc.dim {
			t.Errorf("%s: KeyQJLRows(%d) = %d, want %d (paper spec: d projections per d-dim vector)",
				tc.preset.Name, tc.dim, got, tc.dim)
		}
	}
}

// TestPaperMSEDistortionBound verifies that the MSE quantizer operates within
// the information-theoretic bounds from Theorem 3 of arXiv 2504.19874:
//
//	lower: D_mse >= 1/4^b
//	upper: D_mse <= (√3π/2) / 4^b ≈ 2.72 / 4^b
//
// Tested on 1000 random unit vectors at dim=128 (the primary validated head_dim).
func TestPaperMSEDistortionBound(t *testing.T) {
	const dim = 128
	const trials = 1000

	cases := []struct {
		preset Preset
		bits   int
	}{
		{PresetTQ2, PresetTQ2.ValueBits},
		{PresetTQ3, PresetTQ3.ValueBits},
	}

	for _, tc := range cases {
		t.Run(tc.preset.Name, func(t *testing.T) {
			lowerBound := 1.0 / math.Pow(4, float64(tc.bits))
			paperUpper := 2.72 / math.Pow(4, float64(tc.bits))

			var totalDistortion float64
			rng := splitmix64(0x1234567890abcdef)
			for range trials {
				// Random unit vector drawn from the uniform distribution on S^{d-1}.
				vec := make([]float32, dim)
				var norm2 float64
				for j := range vec {
					v := gaussianFloat64(&rng)
					vec[j] = float32(v)
					norm2 += v * v
				}
				norm := math.Sqrt(norm2)
				for j := range vec {
					vec[j] /= float32(norm)
				}

				encoded, err := EncodeValueVector(vec, tc.preset)
				if err != nil {
					t.Fatal(err)
				}
				data, err := encoded.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				decoded, _, err := DecodeVector(data)
				if err != nil {
					t.Fatal(err)
				}

				// D_mse = ||x - x_hat||^2 / ||x||^2 = ||x - x_hat||^2 (||x||=1)
				var distortion float64
				for j := range vec {
					d := float64(vec[j] - decoded[j])
					distortion += d * d
				}
				totalDistortion += distortion
			}
			avgDistortion := totalDistortion / float64(trials)

			t.Logf("%s D_mse = %.6f, paper bounds [%.6f, %.6f]",
				tc.preset.Name, avgDistortion, lowerBound, paperUpper)

			// Allow 1.5× headroom over the paper's upper bound to account for
			// finite-d effects and the Cartesian (non-PolarQuant) encoding path.
			if avgDistortion > paperUpper*1.5 {
				t.Fatalf("D_mse = %.6f exceeds paper upper bound × 1.5 (%.6f)",
					avgDistortion, paperUpper*1.5)
			}
		})
	}
}

// TestPaperProductUnbiasedness verifies that the product-objective estimator
// is near-unbiased: E[score(q, encoded_k) - dot(q, k)] ≈ 0. This is the
// central claim of Q_prod in arXiv 2504.19874.
func TestPaperProductUnbiasedness(t *testing.T) {
	const dim = 128
	const trials = 500
	// Allow 5% signed relative bias averaged over 500 trials.
	const maxRelBias = 0.05

	rng := splitmix64(0xdeadbeefcafe1234)
	var signedBias, rmsTrue float64
	for range trials {
		query := make([]float32, dim)
		key := make([]float32, dim)
		for j := range query {
			query[j] = float32(gaussianFloat64(&rng))
			key[j] = float32(gaussianFloat64(&rng))
		}

		encoded, err := EncodeKeyVector(key, PresetTQ3)
		if err != nil {
			t.Fatal(err)
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		decoded, _, err := DecodeVector(data)
		if err != nil {
			t.Fatal(err)
		}
		var estimated float32
		for j := range query {
			estimated += query[j] * decoded[j]
		}

		var trueDot float32
		for j := range query {
			trueDot += query[j] * key[j]
		}
		signedBias += float64(estimated - trueDot)
		rmsTrue += float64(trueDot * trueDot)
	}

	avgSignedBias := signedBias / float64(trials)
	rmsTrue = math.Sqrt(rmsTrue / float64(trials))
	relativeBias := math.Abs(avgSignedBias) / rmsTrue

	t.Logf("avg signed bias = %.6f, rms true dot = %.6f, relative bias = %.4f",
		avgSignedBias, rmsTrue, relativeBias)

	if relativeBias > maxRelBias {
		t.Fatalf("relative bias = %.4f, want <= %.4f (product estimator should be near-unbiased)",
			relativeBias, maxRelBias)
	}
}

// TestOutlierSplitMSEImproves verifies that encoding with the outlier-split
// strategy achieves lower MSE than uniform quantization at the same average bit
// rate. This is the core quality claim of §4.3 of arXiv 2504.19874.
func TestOutlierSplitMSEImproves(t *testing.T) {
	const dim = 128
	const trials = 200
	rng := splitmix64(0xfeedbabe12345678)

	for _, preset := range []Preset{testOutlierPreset(PresetTQ2, 32), testOutlierPreset(PresetTQ3, 32)} {
		t.Run(preset.Name, func(t *testing.T) {
			var splitMSE, uniformMSE float64
			for range trials {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = float32(gaussianFloat64(&rng))
				}

				// Outlier-split encoding (2-block, current path).
				splitEncoded, err := EncodeValueVector(vec, preset)
				if err != nil {
					t.Fatal(err)
				}
				splitData, err := splitEncoded.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				splitDecoded, _, err := DecodeVector(splitData)
				if err != nil {
					t.Fatal(err)
				}
				for j := range vec {
					d := float64(vec[j] - splitDecoded[j])
					splitMSE += d * d
				}

				// Uniform encoding: both sub-block sizes at regular bits, same total bits.
				// Encode the whole vector at regular bits to match the average bit rate.
				uniformBits := preset.ValueBits
				unifBlock, err := encodeSubBlock(vec, nil, preset, roleValue, objectiveMSE, uniformBits, preset.RotationSeed)
				if err != nil {
					t.Fatal(err)
				}
				codebook, _ := scalarCodebook(dim, uniformBits)
				rot := BuildRotation(dim, preset.RotationSeed)
				uIndices := unpackBits(unifBlock.RegularIndices, uniformBits, dim)
				unifRotated := make([]float32, dim)
				for j, idx := range uIndices {
					unifRotated[j] = dequantizeScalar(idx, codebook) * unifBlock.Scale
				}
				unifDecoded := ApplyInverseRotation(unifRotated, rot)
				for j := range vec {
					d := float64(vec[j] - unifDecoded[j])
					uniformMSE += d * d
				}
			}
			splitMSE /= float64(trials * dim)
			uniformMSE /= float64(trials * dim)
			t.Logf("%s: split MSE=%.6f  uniform MSE=%.6f", preset.Name, splitMSE, uniformMSE)
			if splitMSE >= uniformMSE {
				t.Errorf("outlier split MSE (%.6f) not lower than uniform MSE (%.6f)", splitMSE, uniformMSE)
			}
		})
	}
}

// TestOutlierSplitProductUnbiasedness verifies that the multi-block product
// estimator remains near-unbiased after the outlier split is applied.
func TestOutlierSplitProductUnbiasedness(t *testing.T) {
	const dim = 128
	const trials = 300
	const maxRelBias = 0.07

	outlierPreset := testOutlierPreset(PresetTQ3, 32)
	rng := splitmix64(0xabcdef0123456789)
	var signedBias, rmsTrue float64
	for range trials {
		query := make([]float32, dim)
		key := make([]float32, dim)
		for j := range query {
			query[j] = float32(gaussianFloat64(&rng))
			key[j] = float32(gaussianFloat64(&rng))
		}

		encoded, err := EncodeKeyVector(key, outlierPreset)
		if err != nil {
			t.Fatal(err)
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		decoded, _, err := DecodeVector(data)
		if err != nil {
			t.Fatal(err)
		}
		var estimated float32
		for j := range query {
			estimated += query[j] * decoded[j]
		}

		var trueDot float32
		for j := range query {
			trueDot += query[j] * key[j]
		}
		signedBias += float64(estimated - trueDot)
		rmsTrue += float64(trueDot * trueDot)
	}

	avgSignedBias := signedBias / float64(trials)
	rmsTrue = math.Sqrt(rmsTrue / float64(trials))
	relativeBias := math.Abs(avgSignedBias) / rmsTrue

	t.Logf("outlier-split avg signed bias = %.6f, rms true dot = %.6f, relative bias = %.4f",
		avgSignedBias, rmsTrue, relativeBias)
	if relativeBias > maxRelBias {
		t.Fatalf("relative bias = %.4f, want <= %.4f (multi-block estimator should be near-unbiased)",
			relativeBias, maxRelBias)
	}
}

func TestDistortionThresholds(t *testing.T) {
	tq2Mean, err := meanMSEForPreset(PresetTQ2)
	if err != nil {
		t.Fatal(err)
	}
	tq3Mean, err := meanMSEForPreset(PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}

	if tq2Mean > 5.5 {
		t.Fatalf("tq2 mean MSE = %v, want <= 5.5", tq2Mean)
	}
	if tq3Mean > 3.5 {
		t.Fatalf("tq3 mean MSE = %v, want <= 3.5", tq3Mean)
	}
	if tq3Mean > tq2Mean {
		t.Fatalf("tq3 mean MSE = %v, want <= tq2 mean MSE %v", tq3Mean, tq2Mean)
	}
}

// ── per-head uniform encoding ──────────────────────────────────────────────

func TestEncodeKeyPerHeadRoundTrip(t *testing.T) {
	const dim = 128
	const trials = 200

	for _, preset := range []Preset{PresetTQ2, PresetTQ3} {
		t.Run(preset.Name, func(t *testing.T) {
			rng := splitmix64(0xbeefcafe)
			var totalMSE float64
			var totalAbsErr, totalAbsDot float64

			for range trials {
				values := make([]float32, dim)
				query := make([]float32, dim)
				for j := range values {
					values[j] = float32(gaussianFloat64(&rng))
					query[j] = float32(gaussianFloat64(&rng))
				}

				packed, scale, err := EncodeKeyPerHead(values, preset)
				if err != nil {
					t.Fatal(err)
				}
				expectedBytes := (dim*preset.KeyPrimaryBits + 7) / 8
				if len(packed) != expectedBytes {
					t.Fatalf("packed len = %d, want %d", len(packed), expectedBytes)
				}

				dequantRot := DequantKeyPerHead(packed, scale, dim, preset.KeyPrimaryBits)

				// Verify MSE in rotated space (reconstruction quality).
				rotation := BuildRotation(dim, preset.RotationSeed)
				valuesRot := ApplyRotation(values, rotation)
				var mse float64
				for j := range valuesRot {
					d := float64(valuesRot[j] - dequantRot[j])
					mse += d * d
				}
				mse /= float64(dim)
				totalMSE += mse

				// Verify dot product in rotated space matches true dot product.
				queryRot := ApplyRotation(query, rotation)
				var estDot, trueDot float32
				for j := range queryRot {
					estDot += queryRot[j] * dequantRot[j]
				}
				for j := range query {
					trueDot += query[j] * values[j]
				}
				totalAbsErr += math.Abs(float64(estDot - trueDot))
				totalAbsDot += math.Abs(float64(trueDot))

				_ = scale
			}

			avgMSE := totalMSE / trials
			avgRelErr := totalAbsErr / (totalAbsDot + 1e-8)

			t.Logf("avg MSE = %.6f, avg relative dot error = %.4f", avgMSE, avgRelErr)

			// Uniform encoding (no outlier split, no QJL) has higher error than
			// full TQ. These thresholds validate the codec works, not paper quality.
			maxRelErr := 0.45
			if preset.KeyPrimaryBits >= 3 {
				maxRelErr = 0.25
			}
			if avgRelErr > maxRelErr {
				t.Fatalf("avg relative dot error = %.4f, want <= %.4f", avgRelErr, maxRelErr)
			}
		})
	}
}

// TestEncodeKeyPerHeadRoundTripLargeDims checks CPU reference round-trip
// quality at the head dims used by models outside the llama/qwen D=128 norm.
// These are regression targets for the non-128 code path: gemma3 global uses
// D=256, gemma4 global uses D=512. If any sub-test passes but the CUDA path
// produces different output at the same D, the bug is kernel-side; if a
// sub-test fails, the core TQ math (rotation + codebook) has a D-dependent
// bug.
func TestEncodeKeyPerHeadRoundTripLargeDims(t *testing.T) {
	const trials = 200

	for _, dim := range []int{256, 512} {
		for _, preset := range []Preset{PresetTQ2, PresetTQ3} {
			t.Run(fmt.Sprintf("dim%d_%s", dim, preset.Name), func(t *testing.T) {
				rng := splitmix64(0xbeefcafe ^ uint64(dim))
				var totalMSE, totalAbsErr, totalAbsDot float64

				for range trials {
					values := make([]float32, dim)
					query := make([]float32, dim)
					for j := range values {
						values[j] = float32(gaussianFloat64(&rng))
						query[j] = float32(gaussianFloat64(&rng))
					}

					packed, scale, err := EncodeKeyPerHead(values, preset)
					if err != nil {
						t.Fatal(err)
					}
					dequantRot := DequantKeyPerHead(packed, scale, dim, preset.KeyPrimaryBits)

					rotation := BuildRotation(dim, preset.RotationSeed)
					valuesRot := ApplyRotation(values, rotation)
					var mse float64
					for j := range valuesRot {
						d := float64(valuesRot[j] - dequantRot[j])
						mse += d * d
					}
					mse /= float64(dim)
					totalMSE += mse

					queryRot := ApplyRotation(query, rotation)
					var estDot, trueDot float32
					for j := range queryRot {
						estDot += queryRot[j] * dequantRot[j]
					}
					for j := range query {
						trueDot += query[j] * values[j]
					}
					totalAbsErr += math.Abs(float64(estDot - trueDot))
					totalAbsDot += math.Abs(float64(trueDot))
				}

				avgMSE := totalMSE / trials
				avgRelErr := totalAbsErr / (totalAbsDot + 1e-8)
				t.Logf("D=%d %s: avg MSE = %.6f, avg rel dot error = %.4f", dim, preset.Name, avgMSE, avgRelErr)

				maxRelErr := 0.45
				if preset.KeyPrimaryBits >= 3 {
					maxRelErr = 0.25
				}
				if avgRelErr > maxRelErr {
					t.Fatalf("D=%d %s: avg relative dot error = %.4f, want <= %.4f", dim, preset.Name, avgRelErr, maxRelErr)
				}
			})
		}
	}
}

// TestEncodeKeyPerHeadWithDCOffset checks whether TQ handles K vectors with
// a DC bias component. Models like qwen2/qwen2.5 have learned Q/K/V bias
// tensors; their K vectors have a non-zero mean that TQ's RMS-based scale
// normalization doesn't center out. This test adds a DC offset to random
// Gaussian samples and measures round-trip quality vs the non-biased case.
func TestEncodeKeyPerHeadWithDCOffset(t *testing.T) {
	const dim = 128
	const trials = 200

	offsets := []float32{0.0, 0.3, 1.0, 3.0}
	for _, offset := range offsets {
		for _, preset := range []Preset{PresetTQ3} {
			t.Run(fmt.Sprintf("offset%.1f_%s", offset, preset.Name), func(t *testing.T) {
				rng := splitmix64(0xbeefcafe)
				var totalAbsErr, totalAbsDot float64

				for range trials {
					values := make([]float32, dim)
					query := make([]float32, dim)
					for j := range values {
						values[j] = float32(gaussianFloat64(&rng)) + offset
						query[j] = float32(gaussianFloat64(&rng))
					}

					packed, scale, err := EncodeKeyPerHead(values, preset)
					if err != nil {
						t.Fatal(err)
					}
					dequantRot := DequantKeyPerHead(packed, scale, dim, preset.KeyPrimaryBits)

					rotation := BuildRotation(dim, preset.RotationSeed)
					queryRot := ApplyRotation(query, rotation)
					var estDot, trueDot float32
					for j := range queryRot {
						estDot += queryRot[j] * dequantRot[j]
					}
					for j := range query {
						trueDot += query[j] * values[j]
					}
					totalAbsErr += math.Abs(float64(estDot - trueDot))
					totalAbsDot += math.Abs(float64(trueDot))
				}

				avgRelErr := totalAbsErr / (totalAbsDot + 1e-8)
				t.Logf("offset=%.1f %s: avg rel dot error = %.4f", offset, preset.Name, avgRelErr)
			})
		}
	}
}

func TestEncodeKeyPerHeadZeroVector(t *testing.T) {
	values := make([]float32, 64)
	packed, scale, err := EncodeKeyPerHead(values, PresetTQ3)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 0 {
		t.Fatalf("expected zero scale for zero vector, got %v", scale)
	}
	dequant := DequantKeyPerHead(packed, scale, 64, PresetTQ3.KeyPrimaryBits)
	for i, v := range dequant {
		if v != 0 {
			t.Fatalf("dequant[%d] = %v, want 0", i, v)
		}
	}
}

func TestEncodeKeyPerHeadDimSizes(t *testing.T) {
	for _, dim := range []int{32, 64, 96, 128, 256} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			values := pseudoRandomVector(dim, uint64(dim))
			packed, scale, err := EncodeKeyPerHead(values, PresetTQ3)
			if err != nil {
				t.Fatal(err)
			}
			if scale <= 0 {
				t.Fatalf("expected positive scale for dim=%d", dim)
			}
			expectedBytes := (dim*PresetTQ3.KeyPrimaryBits + 7) / 8
			if len(packed) != expectedBytes {
				t.Fatalf("packed len = %d, want %d for dim=%d", len(packed), expectedBytes, dim)
			}
		})
	}
}

func TestEncodeKeyPerHeadDeterministic(t *testing.T) {
	for _, preset := range []Preset{PresetTQ2, PresetTQ3} {
		t.Run(preset.Name, func(t *testing.T) {
			vec := pseudoRandomVector(128, 0xcafe)
			packed1, scale1, err := EncodeKeyPerHead(vec, preset)
			if err != nil {
				t.Fatalf("first EncodeKeyPerHead: %v", err)
			}
			packed2, scale2, err := EncodeKeyPerHead(vec, preset)
			if err != nil {
				t.Fatalf("second EncodeKeyPerHead: %v", err)
			}
			if scale1 != scale2 {
				t.Fatalf("scale not deterministic: %f vs %f", scale1, scale2)
			}
			for i := range packed1 {
				if packed1[i] != packed2[i] {
					t.Fatalf("packed byte %d differs: %d vs %d", i, packed1[i], packed2[i])
				}
			}
		})
	}
}

// studentT3Float64 samples from Student-t with df=3 via Z / sqrt(ChiSq3/3).
// Student-t df=3 has finite mean/variance but heavy tails (kurtosis → ∞),
// which stresses quantizers that assume Gaussian inputs.
func studentT3Float64(rng *splitmix64) float64 {
	z := gaussianFloat64(rng)
	// ChiSq(3) = sum of 3 squared standard normals.
	var chisq float64
	for range 3 {
		n := gaussianFloat64(rng)
		chisq += n * n
	}
	return z / math.Sqrt(chisq/3.0)
}

// TestOutlierSplitVsUniformHeavyTailed verifies that the CPU reference
// outlier-split encoder beats the uniform encoder on heavy-tailed K
// vectors (Student-t df=3). This is the statistical regression test
// for the outlier-split algorithm: if someone breaks the split logic
// (e.g. misattributes channels to sub-blocks, uses the wrong scale),
// this test starts failing because outlier no longer helps.
//
// Acceptance: on Student-t df=3 inputs, outlier-split relative dot
// error must be <= 1.5x the uniform relative dot error. In practice
// outlier-split is substantially BETTER on heavy tails — the 1.5x
// bound is a floor to catch "outlier split broken" regressions, not
// the expected performance.
func TestOutlierSplitVsUniformHeavyTailed(t *testing.T) {
	const dim = 128
	const trials = 200

	presets := []Preset{
		testOutlierPreset(PresetTQ3, 32),
		testOutlierPreset(PresetTQ3K, 32),
	}
	for _, preset := range presets {
		t.Run(preset.Name, func(t *testing.T) {
			rng := splitmix64(0xc0de_babe_dead_beef)
			var uniformAbsErr, uniformAbsDot float64
			var outlierAbsErr, outlierAbsDot float64

			for trial := range trials {
				values := make([]float32, dim)
				query := make([]float32, dim)
				for j := range values {
					values[j] = float32(studentT3Float64(&rng))
					query[j] = float32(gaussianFloat64(&rng))
				}

				rotation := BuildRotation(dim, preset.RotationSeed)
				valuesRot := ApplyRotation(values, rotation)
				queryRot := ApplyRotation(query, rotation)

				var trueDot float32
				for j := range query {
					trueDot += query[j] * values[j]
				}

				// Uniform path.
				uPacked, uScale, err := EncodeKeyPerHead(values, preset)
				if err != nil {
					t.Fatalf("uniform encode: %v", err)
				}
				uDeq := DequantKeyPerHead(uPacked, uScale, dim, preset.KeyPrimaryBits)
				var uEstDot float32
				for j := range uDeq {
					uEstDot += queryRot[j] * uDeq[j]
				}
				uniformAbsErr += math.Abs(float64(uEstDot - trueDot))
				uniformAbsDot += math.Abs(float64(trueDot))

				// Outlier-split path.
				oEnc, err := EncodeKeyPerHeadOutlier(values, preset)
				if err != nil {
					t.Fatalf("outlier encode: %v", err)
				}
				oDeq := DequantKeyPerHeadOutlier(oEnc, preset, dim)
				var oEstDot float32
				for j := range oDeq {
					oEstDot += queryRot[j] * oDeq[j]
				}
				outlierAbsErr += math.Abs(float64(oEstDot - trueDot))
				outlierAbsDot += math.Abs(float64(trueDot))

				// Sanity: the outlier-split decode should still be close
				// enough to the rotated input that a round-trip MSE is
				// sensible. Catches frame-shifted reconstructions where
				// the dot happens to land right but the vector is wrong.
				var mse float64
				for j := range oDeq {
					d := float64(oDeq[j] - valuesRot[j])
					mse += d * d
				}
				mse /= float64(dim)
				if mse > 4.0 {
					t.Fatalf("trial %d: outlier-split rotated-space MSE = %.4f, reconstruction is garbage", trial, mse)
				}
			}

			uniformRelErr := uniformAbsErr / (uniformAbsDot + 1e-8)
			outlierRelErr := outlierAbsErr / (outlierAbsDot + 1e-8)
			t.Logf("%s Student-t df=3: uniform rel-dot-err=%.4f, outlier rel-dot-err=%.4f, ratio=%.3f",
				preset.Name, uniformRelErr, outlierRelErr, outlierRelErr/uniformRelErr)

			// Outlier-split must not be worse than 1.5x uniform. In
			// practice it should be substantially better.
			if outlierRelErr > 1.5*uniformRelErr {
				t.Fatalf("%s: outlier rel-dot-err %.4f > 1.5x uniform %.4f — outlier split broken",
					preset.Name, outlierRelErr, uniformRelErr)
			}
		})
	}
}

// TestOutlierPerHeadRoundTrip pins the CPU reference bit-exact round-trip
// at the shapes actually used by GPU kernels (headDim=128, 256). Encode
// → dequant → compare against ApplyRotation(values, rotation). This
// catches algorithmic drift (wrong codebook selection, off-by-one in
// slot mapping, index packing errors) even when the GPU path is broken.
func TestOutlierPerHeadRoundTrip(t *testing.T) {
	cases := []struct {
		name   string
		dim    int
		preset Preset
	}{
		{"d128_tq3", 128, testOutlierPreset(PresetTQ3, 32)},
		{"d128_tq3k", 128, testOutlierPreset(PresetTQ3K, 32)},
		{"d256_tq3k", 256, testOutlierPreset(PresetTQ3K, 32)}, // gemma3:1b global-layer headDim
		{"d128_tq2k", 128, testOutlierPreset(PresetTQ2K, 32)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const trials = 50
			rng := splitmix64(0xfeed_face_0000_0000 | uint64(tc.dim))
			var totalMSE float64

			for range trials {
				values := make([]float32, tc.dim)
				for j := range values {
					values[j] = float32(gaussianFloat64(&rng))
				}

				enc, err := EncodeKeyPerHeadOutlier(values, tc.preset)
				if err != nil {
					t.Fatalf("encode: %v", err)
				}
				if len(enc.OutlierIndices) != tc.preset.OutlierCount {
					t.Fatalf("outlier count = %d, want %d", len(enc.OutlierIndices), tc.preset.OutlierCount)
				}
				expRegularBytes := ((tc.dim-tc.preset.OutlierCount)*tc.preset.KeyPrimaryBits + 7) / 8
				if len(enc.RegularPacked) != expRegularBytes {
					t.Fatalf("regular packed len = %d, want %d", len(enc.RegularPacked), expRegularBytes)
				}
				expOutlierBytes := (tc.preset.OutlierCount*tc.preset.OutlierBits + 7) / 8
				if len(enc.OutlierPacked) != expOutlierBytes {
					t.Fatalf("outlier packed len = %d, want %d", len(enc.OutlierPacked), expOutlierBytes)
				}

				dec := DequantKeyPerHeadOutlier(enc, tc.preset, tc.dim)
				rotation := BuildRotation(tc.dim, tc.preset.RotationSeed)
				valuesRot := ApplyRotation(values, rotation)

				var mse float64
				for j := range valuesRot {
					d := float64(valuesRot[j] - dec[j])
					mse += d * d
				}
				mse /= float64(tc.dim)
				totalMSE += mse
			}

			avgMSE := totalMSE / trials
			t.Logf("%s: avg rotated-space round-trip MSE = %.4f", tc.name, avgMSE)
			// Lloyd-Max 3-bit on unit-variance Gaussian has MSE ~0.04 per
			// channel in rotated space. 0.15 is a loose bound that still
			// catches format bugs.
			if avgMSE > 0.15 {
				t.Fatalf("%s: round-trip MSE %.4f too large — encoder/dequant mismatch", tc.name, avgMSE)
			}
		})
	}
}
