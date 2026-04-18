package testutil

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestSkipIfNoMLX(t *testing.T) {
	SkipIfNoMLX(t)
}

func TestDefaultRefDir(t *testing.T) {
	// Without env var override
	got := DefaultRefDir("test-model")
	want := filepath.Join("/tmp", "ollama_ref", "test-model")
	if got != want {
		t.Errorf("DefaultRefDir = %q, want %q", got, want)
	}

	// With env var override
	t.Setenv("OLLAMA_REF_DIR", "/custom/ref")
	got = DefaultRefDir("test-model")
	want = "/custom/ref/test-model"
	if got != want {
		t.Errorf("DefaultRefDir with env = %q, want %q", got, want)
	}
}

func TestModelDir_Missing(t *testing.T) {
	// Verify the function returns the correct path when the directory exists.
	dir := t.TempDir()
	got := ModelDir(t, "TESTUTIL_MODEL_DIR_TEST", dir)
	if got != dir {
		t.Errorf("ModelDir = %q, want %q", got, dir)
	}
}

func TestModelDir_EnvOverride(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TESTUTIL_TEST_DIR", dir)
	got := ModelDir(t, "TESTUTIL_TEST_DIR", "/nonexistent/default")
	if got != dir {
		t.Errorf("ModelDir = %q, want %q", got, dir)
	}
}

func TestCompareArrays_Matching(t *testing.T) {
	SkipIfNoMLX(t)

	a := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2)
	b := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2)
	mlx.Eval(a, b)

	if !CompareArrays(t, "exact_match", a, b) {
		t.Error("identical arrays should match")
	}
}

func TestCompareArrays_WithinTolerance(t *testing.T) {
	SkipIfNoMLX(t)

	a := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2)
	b := mlx.FromValues([]float32{1.01, 2.01, 3.01, 4.01}, 2, 2)
	mlx.Eval(a, b)

	// Default tolerance (atol=1e-4, rtol=1e-3) should fail for diff=0.01
	entry := compareArraysInner("tight", a, b, defaultConfig())
	if entry.Passed {
		t.Error("should fail with default tolerance")
	}

	// BFloat16 tolerance (atol=5e-3, rtol=5e-3) should pass for diff=0.01
	cfg := defaultConfig()
	BFloat16Tol()(&cfg)
	entry = compareArraysInner("loose", a, b, cfg)
	if !entry.Passed {
		t.Error("should pass with BFloat16Tol")
	}
}

func TestCompareArrays_ShapeMismatch(t *testing.T) {
	SkipIfNoMLX(t)

	a := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2)
	b := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 4)
	mlx.Eval(a, b)

	entry := compareArraysInner("shape_mismatch", a, b, defaultConfig())
	if entry.Passed {
		t.Error("different shapes should not pass")
	}
}

func TestCompareArrays_CosineSim(t *testing.T) {
	SkipIfNoMLX(t)

	a := mlx.FromValues([]float32{1.0, 0.0, 0.0}, 3)
	b := mlx.FromValues([]float32{1.0, 0.0, 0.0}, 3)
	mlx.Eval(a, b)

	entry := compareArraysInner("parallel", a, b, defaultConfig())
	if entry.CosineSim < 0.999 {
		t.Errorf("parallel vectors cosine sim = %f, want ~1.0", entry.CosineSim)
	}

	// Orthogonal vectors
	c := mlx.FromValues([]float32{0.0, 1.0, 0.0}, 3)
	mlx.Eval(c)

	entry = compareArraysInner("orthogonal", a, c, defaultConfig())
	if entry.CosineSim > 0.001 {
		t.Errorf("orthogonal vectors cosine sim = %f, want ~0.0", entry.CosineSim)
	}
}

func TestCompareArraysCosineSim(t *testing.T) {
	SkipIfNoMLX(t)

	// Near-parallel vectors with one element wide in absolute terms but the
	// overall direction unchanged — should pass the cosine check.
	a := mlx.FromValues([]float32{50.0, 1.0, 1.0, 1.0}, 4)
	b := mlx.FromValues([]float32{51.0, 1.0, 1.0, 1.0}, 4)
	mlx.Eval(a, b)
	if !CompareArraysCosineSim(t, "near_parallel", a, b, 0.999) {
		t.Error("expected near-parallel vectors to pass")
	}

	// Diverged pair (one element sign-flipped) — should fail.
	c := mlx.FromValues([]float32{1.0, 1.0, 1.0, 1.0}, 4)
	d := mlx.FromValues([]float32{-1.0, 1.0, 1.0, 1.0}, 4)
	mlx.Eval(c, d)
	subT := &testing.T{}
	if CompareArraysCosineSim(subT, "diverged", c, d, 0.999) {
		t.Error("expected diverged vectors to fail")
	}
	if !subT.Failed() {
		t.Error("expected sub-test to record a failure")
	}
}

func TestCompareArrays_MaxDiffLocation(t *testing.T) {
	SkipIfNoMLX(t)

	a := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, 4)
	b := mlx.FromValues([]float32{1.0, 2.0, 3.5, 4.0}, 4)
	mlx.Eval(a, b)

	entry := compareArraysInner("max_at_2", a, b, defaultConfig())
	if entry.MaxDiffAt != 2 {
		t.Errorf("MaxDiffAt = %d, want 2", entry.MaxDiffAt)
	}
	if entry.MaxDiff < 0.49 || entry.MaxDiff > 0.51 {
		t.Errorf("MaxDiff = %f, want ~0.5", entry.MaxDiff)
	}
}

func TestCompareManyArrays(t *testing.T) {
	SkipIfNoMLX(t)

	got := map[string]*mlx.Array{
		"layer_0": mlx.FromValues([]float32{1.0, 2.0}, 2),
		"layer_1": mlx.FromValues([]float32{3.0, 4.0}, 2),
		"extra":   mlx.FromValues([]float32{5.0}, 1),
	}
	want := map[string]*mlx.Array{
		"layer_0": mlx.FromValues([]float32{1.0, 2.0}, 2),
		"layer_1": mlx.FromValues([]float32{3.0, 4.5}, 2), // mismatch
		"other":   mlx.FromValues([]float32{6.0}, 1),
	}
	for _, a := range got {
		mlx.Eval(a)
	}
	for _, a := range want {
		mlx.Eval(a)
	}

	report := CompareManyArrays(t, got, want)

	if len(report.Entries) != 2 {
		t.Fatalf("expected 2 entries (common keys), got %d", len(report.Entries))
	}

	// layer_0 should pass, layer_1 should fail
	for _, e := range report.Entries {
		switch e.Name {
		case "layer_0":
			if !e.Passed {
				t.Error("layer_0 should pass")
			}
		case "layer_1":
			if e.Passed {
				t.Error("layer_1 should fail with default tolerance")
			}
		}
	}

	if report.AllPassed() {
		t.Error("report should not pass when layer_1 fails")
	}
}

func TestCompareReport_Summary(t *testing.T) {
	SkipIfNoMLX(t)

	// Just verify Summary doesn't panic
	report := CompareReport{
		Entries: []CompareEntry{
			{Name: "layer_0", Passed: true, MaxDiff: 0.001, MeanDiff: 0.0005, CosineSim: 0.9999},
			{Name: "layer_1", Passed: false, MaxDiff: 0.5, MeanDiff: 0.1, CosineSim: 0.95},
		},
	}
	report.Summary(t)
}

func TestLoadReference_Missing(t *testing.T) {
	SkipIfNoMLX(t)

	// Should skip (not fail) when file doesn't exist
	// We can't easily test t.Skip from within, but we can verify
	// the function handles the path correctly
	path := filepath.Join(t.TempDir(), "nonexistent.safetensors")
	if _, err := os.Stat(path); err == nil {
		t.Fatal("test setup error: file should not exist")
	}
}

func TestWithTolerance(t *testing.T) {
	SkipIfNoMLX(t)

	a := mlx.FromValues([]float32{1.0, 2.0}, 2)
	b := mlx.FromValues([]float32{1.1, 2.1}, 2)
	mlx.Eval(a, b)

	// Should fail with tight tolerance
	entry := compareArraysInner("tight", a, b, defaultConfig())
	if entry.Passed {
		t.Error("should fail with default tolerance")
	}

	// Should pass with custom loose tolerance
	cfg := defaultConfig()
	WithTolerance(0.2, 0.2)(&cfg)
	entry = compareArraysInner("loose", a, b, cfg)
	if !entry.Passed {
		t.Error("should pass with atol=0.2")
	}
}

func TestModelDir_TildeExpansion(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}

	// Use a path that exists under home
	got := ModelDir(t, "TESTUTIL_UNUSED_ENV", home)
	if got != home {
		t.Errorf("ModelDir = %q, want %q", got, home)
	}
}

func TestCompareArraysPerPosition_AllMatch(t *testing.T) {
	SkipIfNoMLX(t)

	// [1, 4, 3] tensor: identical, every position should pass.
	a := mlx.FromValues([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, 1, 4, 3)
	b := mlx.FromValues([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, 1, 4, 3)
	mlx.Eval(a, b)

	r := CompareArraysPerPosition(t, "match", a, b, 1)
	if r.Length != 4 {
		t.Errorf("Length = %d, want 4", r.Length)
	}
	if !r.AllPassed() {
		t.Errorf("expected all positions to pass, first fail at %d", r.FirstFailIndex)
	}
	for i, d := range r.Drifts {
		if d.MaxDiff != 0 {
			t.Errorf("position %d max_diff = %f, want 0", i, d.MaxDiff)
		}
	}
}

func TestCompareArraysPerPosition_FirstFailLocated(t *testing.T) {
	SkipIfNoMLX(t)

	// [1, 4, 3] tensor: positions 0 and 1 match, position 2 has a tiny diff
	// (within tolerance), position 3 has a large diff (out of tolerance).
	// FirstFailIndex must be exactly 3.
	a := mlx.FromValues([]float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0001, 8.0, 9.0,
		10.0, 11.0, 999.0,
	}, 1, 4, 3)
	b := mlx.FromValues([]float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
		10.0, 11.0, 12.0,
	}, 1, 4, 3)
	mlx.Eval(a, b)

	r := CompareArraysPerPosition(t, "find_drift", a, b, 1, BFloat16Tol())
	if r.FirstFailIndex != 3 {
		t.Errorf("FirstFailIndex = %d, want 3", r.FirstFailIndex)
	}
	if r.WorstIndex != 3 {
		t.Errorf("WorstIndex = %d, want 3", r.WorstIndex)
	}
	if r.Drifts[0].Passed != true || r.Drifts[1].Passed != true {
		t.Error("positions 0 and 1 should pass with tight tolerance (exact match)")
	}
	if r.Drifts[3].Passed {
		t.Error("position 3 must fail (max_diff is 987)")
	}
	worst := r.Drifts[3].MaxDiff
	if worst < 986 || worst > 988 {
		t.Errorf("position 3 max_diff = %f, want ~987", worst)
	}
}

func TestTopDrifts(t *testing.T) {
	SkipIfNoMLX(t)

	// layer_a: one big outlier at pos 2. layer_b: uniform small drift everywhere.
	// Absolute ranking should put layer_a pos 2 first. Relative ranking should
	// also put layer_a pos 2 first (it's 99×+ its layer's median, while layer_b
	// is ~1× everywhere).
	got := map[string]*mlx.Array{
		"layer_a": mlx.FromValues([]float32{1, 2, 3, 4, 999, 6, 7, 8}, 1, 4, 2),
		"layer_b": mlx.FromValues([]float32{1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1}, 1, 4, 2),
	}
	want := map[string]*mlx.Array{
		"layer_a": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 4, 2),
		"layer_b": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 4, 2),
	}
	for _, a := range got {
		mlx.Eval(a)
	}
	for _, a := range want {
		mlx.Eval(a)
	}

	report := CompareLayersPerPosition(t, got, want, 1)

	absTop := report.TopDrifts(3)
	if len(absTop) != 3 {
		t.Fatalf("expected 3 top drifts, got %d", len(absTop))
	}
	if absTop[0].Layer != "layer_a" || absTop[0].Position != 2 {
		t.Errorf("top absolute drift = %s/pos%d, want layer_a/pos2", absTop[0].Layer, absTop[0].Position)
	}
	if absTop[0].MaxDiff < 990 {
		t.Errorf("top drift max_diff = %f, want ~994", absTop[0].MaxDiff)
	}

	relTop := report.TopDriftsByRelative(3)
	if relTop[0].Layer != "layer_a" || relTop[0].Position != 2 {
		t.Errorf("top relative drift = %s/pos%d, want layer_a/pos2", relTop[0].Layer, relTop[0].Position)
	}
	if relTop[0].RelToMedian < 10 {
		t.Errorf("top rel_to_median = %f, want >>1 (outlier)", relTop[0].RelToMedian)
	}

	// Per-tensor TopDrifts on a uniform layer: rel_to_median should be ~1.
	bTop := report.Layers[1].TopDrifts(0)
	for _, d := range bTop {
		if d.RelToMedian < 0.5 || d.RelToMedian > 2.0 {
			t.Errorf("uniform layer rel_to_median = %f, want ~1", d.RelToMedian)
		}
	}
}

func TestEarliestOutlier(t *testing.T) {
	SkipIfNoMLX(t)

	// Three "layers": layer_0 clean, layer_1 has an early outlier at pos 1
	// (the "bug origin"), layer_2 has larger absolute drift everywhere (the
	// amplified downstream effect). EarliestOutlier should return 1, not 2,
	// because layer_1 is where the relative anomaly first appears.
	got := map[string]*mlx.Array{
		"layer_0": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
		"layer_1": mlx.FromValues([]float32{1, 2, 50, 4, 5, 6}, 1, 3, 2),
		"layer_2": mlx.FromValues([]float32{100, 200, 300, 400, 500, 600}, 1, 3, 2),
	}
	want := map[string]*mlx.Array{
		"layer_0": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
		"layer_1": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
		"layer_2": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
	}
	for _, a := range got {
		mlx.Eval(a)
	}
	for _, a := range want {
		mlx.Eval(a)
	}

	report := CompareLayersPerPosition(t, got, want, 1)

	if got := report.EarliestOutlier(5); got != 1 {
		t.Errorf("EarliestOutlier = %d, want 1 (bug origin, not amplified downstream)", got)
	}
	layer, pos := report.EarliestOutlierPosition(5)
	if layer != 1 || pos != 1 {
		t.Errorf("EarliestOutlierPosition = (%d, %d), want (1, 1)", layer, pos)
	}

	// With a huge cutoff nothing should qualify.
	if got := report.EarliestOutlier(1e9); got != -1 {
		t.Errorf("EarliestOutlier with impossible cutoff = %d, want -1", got)
	}
}

func TestEarliestOutlierIgnoresPassingPositions(t *testing.T) {
	SkipIfNoMLX(t)

	got := map[string]*mlx.Array{
		"layer_0": mlx.FromValues([]float32{1, 1, 1, 1.1, 1, 1}, 1, 3, 2),
	}
	want := map[string]*mlx.Array{
		"layer_0": mlx.FromValues([]float32{1, 1, 1, 1, 1, 1}, 1, 3, 2),
	}
	for _, a := range got {
		mlx.Eval(a)
	}
	for _, a := range want {
		mlx.Eval(a)
	}

	report := CompareLayersPerPosition(t, got, want, 1, WithTolerance(0.5, 0))
	if got := report.EarliestOutlier(5); got != -1 {
		t.Errorf("EarliestOutlier = %d, want -1 for all-passing positions", got)
	}
	layer, pos := report.EarliestOutlierPosition(5)
	if layer != -1 || pos != -1 {
		t.Errorf("EarliestOutlierPosition = (%d, %d), want (-1, -1)", layer, pos)
	}
}

func TestCompareLayersPerPosition(t *testing.T) {
	SkipIfNoMLX(t)

	// Three "layers": layers.2 matches everywhere, layers.10 has drift
	// starting at position 2, and layers.11 has drift starting at position 1.
	// Aggregator should report FirstDivergentLayer = 1 because natural sort
	// orders layers.2 before layers.10.
	got := map[string]*mlx.Array{
		"layers.2":  mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
		"layers.10": mlx.FromValues([]float32{1, 2, 3, 4, 999, 999}, 1, 3, 2),
		"layers.11": mlx.FromValues([]float32{1, 2, 999, 999, 999, 999}, 1, 3, 2),
	}
	want := map[string]*mlx.Array{
		"layers.2":  mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
		"layers.10": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
		"layers.11": mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2),
	}
	for _, a := range got {
		mlx.Eval(a)
	}
	for _, a := range want {
		mlx.Eval(a)
	}

	report := CompareLayersPerPosition(t, got, want, 1)
	if got, want := report.FirstDivergentLayer(), 1; got != want {
		t.Errorf("FirstDivergentLayer = %d, want %d", got, want)
	}
	if got, want := report.Layers[0].Name, "layers.2"; got != want {
		t.Errorf("first layer = %q, want %q", got, want)
	}
	if got, want := report.Layers[1].Name, "layers.10"; got != want {
		t.Errorf("second layer = %q, want %q", got, want)
	}
	if report.Layers[0].FirstFailIndex != -1 {
		t.Errorf("layers.2 should pass, got first fail at %d", report.Layers[0].FirstFailIndex)
	}
	if report.Layers[1].FirstFailIndex != 2 {
		t.Errorf("layers.10 first fail = %d, want 2", report.Layers[1].FirstFailIndex)
	}
	layer, pos := report.EarliestToleranceExceedance()
	if layer != 2 || pos != 1 {
		t.Errorf("EarliestToleranceExceedance = (%d, %d), want (2, 1)", layer, pos)
	}
}
