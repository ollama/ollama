package testutil

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// PositionDrift summarizes how a single position-axis index of two tensors
// diverged. It's the per-position equivalent of CompareEntry.
type PositionDrift struct {
	Index    int     // index along the scanned axis (typically the sequence dimension)
	MaxDiff  float32 // max |got - want| across all elements at this position
	MeanDiff float32 // mean |got - want| at this position
	CosSim   float32 // cosine similarity between got and want at this position
	Passed   bool    // true if MaxDiff <= atol + rtol * |want| for every element
}

// PerPositionReport is the result of comparing two tensors with a position-aware
// scan. It's specifically designed to find the first index along an axis where
// drift exceeds tolerance, which is the most common diagnostic question for
// long-sequence / sliding-window / attention-mask bugs.
//
// FirstFailIndex is -1 if every position passed.
type PerPositionReport struct {
	Name           string
	Axis           int
	Length         int             // number of positions scanned
	Drifts         []PositionDrift // one entry per scanned position
	FirstFailIndex int             // -1 if all positions passed
	WorstIndex     int             // position with the largest max_diff
}

// AllPassed returns true if every position satisfied the tolerance.
func (r PerPositionReport) AllPassed() bool { return r.FirstFailIndex < 0 }

// Summary writes a compact human-readable trace to the test logger. Useful when
// you want to see how drift evolves across positions, not just where it starts.
// Prints every Nth position (auto-bucketed to ~20 rows) plus the first failing
// position and the worst position.
func (r PerPositionReport) Summary(t *testing.T) {
	t.Helper()
	if len(r.Drifts) == 0 {
		t.Logf("%s: no positions scored", r.Name)
		return
	}

	step := max(r.Length/20, 1)

	t.Logf("%s: scanned %d positions along axis %d", r.Name, r.Length, r.Axis)
	t.Logf("  %-8s %12s %12s %10s %s", "pos", "max_diff", "mean_diff", "cos_sim", "status")
	for i, d := range r.Drifts {
		isInteresting := i%step == 0 || i == r.Length-1 || i == r.FirstFailIndex || i == r.WorstIndex
		if !isInteresting {
			continue
		}
		status := "ok"
		if !d.Passed {
			status = "FAIL"
		}
		t.Logf("  %-8d %12.6f %12.6f %10.6f %s", d.Index, d.MaxDiff, d.MeanDiff, d.CosSim, status)
	}
	if r.FirstFailIndex >= 0 {
		t.Logf("  → first divergence at position %d", r.FirstFailIndex)
	}
	t.Logf("  → worst drift at position %d (max_diff=%.6f)", r.WorstIndex, r.Drifts[r.WorstIndex].MaxDiff)
}

// CompareArraysPerPosition compares two MLX arrays element-wise but reports
// drift broken down per index along a chosen axis. Returns a PerPositionReport
// with one entry per index. The first index where any element exceeds the
// tolerance is recorded as FirstFailIndex.
//
// Typical use: for a [B, N, H] hidden-state tensor, pass axis=1 to scan along
// the sequence dimension and find the first position where the model's output
// drifts from a reference. This is the diagnostic primitive for sliding-window,
// attention-mask, RoPE, and KV-cache bugs that manifest only after the input
// length crosses some threshold.
//
// Both got and want must have identical shapes. The function reduces along all
// non-axis dimensions to produce per-position statistics.
func CompareArraysPerPosition(t *testing.T, name string, got, want *mlx.Array, axis int, opts ...CompareOption) PerPositionReport {
	t.Helper()
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	gotDims := got.Dims()
	wantDims := want.Dims()
	report := PerPositionReport{
		Name:           name,
		Axis:           axis,
		FirstFailIndex: -1,
	}

	if len(gotDims) != len(wantDims) {
		t.Errorf("%s: shape rank mismatch: got %v, want %v", name, gotDims, wantDims)
		return report
	}
	for i := range gotDims {
		if gotDims[i] != wantDims[i] {
			t.Errorf("%s: shape mismatch at dim %d: got %v, want %v", name, i, gotDims, wantDims)
			return report
		}
	}
	if axis < 0 || axis >= len(gotDims) {
		t.Errorf("%s: axis %d out of range for shape %v", name, axis, gotDims)
		return report
	}

	gotF32 := got.AsType(mlx.DTypeFloat32)
	wantF32 := want.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotF32, wantF32)
	gotData := gotF32.Floats()
	wantData := wantF32.Floats()
	if len(gotData) != len(wantData) {
		t.Errorf("%s: tensor element count mismatch: got %d, want %d", name, len(gotData), len(wantData))
		return report
	}

	// Compute strides so we can iterate per-axis-index efficiently.
	rank := len(gotDims)
	strides := make([]int, rank)
	strides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * gotDims[i+1]
	}
	axisLen := gotDims[axis]
	report.Length = axisLen
	report.Drifts = make([]PositionDrift, axisLen)

	// For each index along axis, walk every element whose coordinates have
	// that index in the axis position and accumulate stats.
	axisStride := strides[axis]
	// Compute the size of one "slice" along axis: product of all other dims.
	sliceSize := 1
	for d, sz := range gotDims {
		if d == axis {
			continue
		}
		sliceSize *= sz
	}

	// Generate the sequence of element indices for a given axis index.
	// We iterate the cartesian product of non-axis dimensions and add
	// idx*axisStride.
	nonAxisDims := make([]int, 0, rank-1)
	nonAxisStrides := make([]int, 0, rank-1)
	for d := range rank {
		if d == axis {
			continue
		}
		nonAxisDims = append(nonAxisDims, gotDims[d])
		nonAxisStrides = append(nonAxisStrides, strides[d])
	}
	bases := make([]int, sliceSize)
	{
		coord := make([]int, len(nonAxisDims))
		for k := range bases {
			off := 0
			for i, c := range coord {
				off += c * nonAxisStrides[i]
			}
			bases[k] = off
			// increment coord (last dim varies fastest)
			for i := len(coord) - 1; i >= 0; i-- {
				coord[i]++
				if coord[i] < nonAxisDims[i] {
					break
				}
				coord[i] = 0
			}
		}
	}

	worstIdx := 0
	worstDiff := float32(-1)
	for idx := range axisLen {
		offset := idx * axisStride
		var (
			maxDiff  float32
			sumDiff  float32
			dot      float64
			gotNorm  float64
			wantNorm float64
			passed   = true
		)
		for _, base := range bases {
			pos := base + offset
			g := gotData[pos]
			w := wantData[pos]
			diff := float32(math.Abs(float64(g - w)))
			if diff > maxDiff {
				maxDiff = diff
			}
			sumDiff += diff
			dot += float64(g) * float64(w)
			gotNorm += float64(g) * float64(g)
			wantNorm += float64(w) * float64(w)
			if diff > cfg.atol+cfg.rtol*float32(math.Abs(float64(w))) {
				passed = false
			}
		}
		mean := sumDiff / float32(sliceSize)
		var cosSim float32
		if denom := math.Sqrt(gotNorm) * math.Sqrt(wantNorm); denom > 0 {
			cosSim = float32(dot / denom)
		}
		report.Drifts[idx] = PositionDrift{
			Index:    idx,
			MaxDiff:  maxDiff,
			MeanDiff: mean,
			CosSim:   cosSim,
			Passed:   passed,
		}
		if !passed && report.FirstFailIndex < 0 {
			report.FirstFailIndex = idx
		}
		if maxDiff > worstDiff {
			worstDiff = maxDiff
			worstIdx = idx
		}
	}
	report.WorstIndex = worstIdx
	return report
}

// LayerPositionReport is the per-(layer, position) result of comparing many
// layer-output tensors against references with position-axis breakdown.
type LayerPositionReport struct {
	Layers []PerPositionReport
}

// FirstDivergentLayer returns the index into Layers of the first layer where
// any position drifted, or -1 if all layers passed at all positions.
func (r LayerPositionReport) FirstDivergentLayer() int {
	for i, l := range r.Layers {
		if !l.AllPassed() {
			return i
		}
	}
	return -1
}

// EarliestToleranceExceedance returns the layer index and position for the
// earliest position that exceeded tolerance across the report. Ties keep report
// order. Returns (-1, -1) if every layer passed at all positions.
func (r LayerPositionReport) EarliestToleranceExceedance() (int, int) {
	layerIdx := -1
	pos := -1
	for i, l := range r.Layers {
		if l.FirstFailIndex < 0 {
			continue
		}
		if pos < 0 || l.FirstFailIndex < pos {
			layerIdx = i
			pos = l.FirstFailIndex
		}
	}
	return layerIdx, pos
}

// Summary prints a compact first tolerance-exceeding position trace across
// layers, answering the question "at each layer, where does drift start?"
// Useful for finding the layer where a position-dependent bug first manifests.
func (r LayerPositionReport) Summary(t *testing.T) {
	t.Helper()
	t.Logf("Per-layer first-divergence-position summary:")
	t.Logf("  %-32s %12s %12s %12s", "layer", "first_tol", "worst_pos", "worst_diff")
	for _, l := range r.Layers {
		first := "-"
		if l.FirstFailIndex >= 0 {
			first = fmt.Sprintf("%d", l.FirstFailIndex)
		}
		worstDiff := float32(0)
		if l.WorstIndex < len(l.Drifts) {
			worstDiff = l.Drifts[l.WorstIndex].MaxDiff
		}
		t.Logf("  %-32s %12s %12d %12.6f", l.Name, first, l.WorstIndex, worstDiff)
	}
	if first := r.FirstDivergentLayer(); first >= 0 {
		layer := r.Layers[first]
		t.Logf("  → first divergent layer in report order: %s, first tolerance-exceeding position: %d",
			layer.Name, layer.FirstFailIndex)
		if layerIdx, pos := r.EarliestToleranceExceedance(); layerIdx >= 0 && layerIdx != first {
			layer := r.Layers[layerIdx]
			t.Logf("  → earliest tolerance-exceeding position: %s, position %d", layer.Name, pos)
		}
	} else {
		t.Logf("  → all layers passed at all positions")
	}
}

// DriftRank is a single entry in a ranked drift listing. It identifies where
// the drift occurred (layer name + position index) and how large it was in
// both absolute terms and relative to the layer's own median drift.
//
// RelToMedian is the ratio max_diff / median(max_diff across all positions in
// this layer). A value of ~1 means the position is typical for the layer; a
// value of 50 means it's 50× the layer's median — a clear outlier regardless
// of absolute scale. This lets you find the *most anomalous* positions even
// in layers where everything has drifted.
type DriftRank struct {
	Layer       string
	Position    int
	MaxDiff     float32
	MeanDiff    float32
	CosSim      float32
	RelToMedian float32
}

// TopDrifts returns the top-n positions ranked by MaxDiff, descending.
// If n <= 0 or n > Length, all positions are returned. RelToMedian is
// computed against the median of this tensor's per-position MaxDiff.
func (r PerPositionReport) TopDrifts(n int) []DriftRank {
	if len(r.Drifts) == 0 {
		return nil
	}
	// Floor the median at a tiny epsilon so positions in layers with an
	// essentially-zero median (most positions perfect, one big outlier) still
	// produce a meaningful RelToMedian. Without the floor the outlier would
	// divide by zero and be ranked *below* layers where everything drifts
	// uniformly — exactly backwards from what we want.
	median := max(medianMaxDiff(r.Drifts), 1e-6)
	ranked := make([]DriftRank, len(r.Drifts))
	for i, d := range r.Drifts {
		rel := d.MaxDiff / median
		ranked[i] = DriftRank{
			Layer:       r.Name,
			Position:    d.Index,
			MaxDiff:     d.MaxDiff,
			MeanDiff:    d.MeanDiff,
			CosSim:      d.CosSim,
			RelToMedian: rel,
		}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].MaxDiff > ranked[j].MaxDiff })
	if n > 0 && n < len(ranked) {
		ranked = ranked[:n]
	}
	return ranked
}

// TopDrifts returns the top-n (layer, position) pairs ranked by MaxDiff across
// every layer in the report, descending. RelToMedian is computed against each
// layer's own median, so it's meaningful to compare across layers: a position
// with RelToMedian=50 in layer 3 and one with RelToMedian=2 in layer 17 tells
// you the layer-3 site is the real anomaly even if its absolute MaxDiff is
// smaller.
//
// This is the "iterate and squash" diagnostic: look at the top 10, fix the
// worst offender, re-run, repeat until the list is uniform noise.
func (r LayerPositionReport) TopDrifts(n int) []DriftRank {
	var all []DriftRank
	for _, l := range r.Layers {
		all = append(all, l.TopDrifts(0)...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].MaxDiff > all[j].MaxDiff })
	if n > 0 && n < len(all) {
		all = all[:n]
	}
	return all
}

// TopDriftsByRelative returns the top-n entries ranked by RelToMedian (how
// anomalous a position is within its own layer), descending. This surfaces
// positions that are outliers relative to their layer's baseline drift, which
// is often more diagnostic than raw MaxDiff when later layers have amplified
// earlier bugs into uniform large drift.
func (r LayerPositionReport) TopDriftsByRelative(n int) []DriftRank {
	var all []DriftRank
	for _, l := range r.Layers {
		all = append(all, l.TopDrifts(0)...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].RelToMedian > all[j].RelToMedian })
	if n > 0 && n < len(all) {
		all = all[:n]
	}
	return all
}

// EarliestOutlier walks layers in order and returns the index of the first
// layer that contains a tolerance-exceeding position whose RelToMedian exceeds
// relCutoff. Since drift compounds forward, the top-ranked absolute drifts
// always live in deep layers even when the bug originates earlier. This
// primitive answers the complementary question: *where did it start?*
//
// A good relCutoff is 5–10× — high enough to skip bf16 rounding noise, low
// enough to catch the first real anomaly before downstream amplification
// pushes every layer above the cutoff. Returns -1 if no layer qualifies.
//
// Caveat: if a bug corrupts a *majority* of positions in its origin layer,
// that layer's median is already elevated and the origin may not exceed the
// cutoff. In that case use FirstDivergentLayer (which triggers on any single
// failing position under the absolute tolerance) as the complementary signal.
func (r LayerPositionReport) EarliestOutlier(relCutoff float32) int {
	for i, l := range r.Layers {
		median := max(medianMaxDiff(l.Drifts), 1e-6)
		for _, d := range l.Drifts {
			if !d.Passed && d.MaxDiff/median >= relCutoff {
				return i
			}
		}
	}
	return -1
}

// EarliestOutlierPosition returns (layerIdx, position) of the first
// tolerance-exceeding (layer, position) pair whose RelToMedian exceeds
// relCutoff, scanning layers in order and positions in order within each layer.
// This is the single most useful drill-down pointer: "start your investigation
// here." Returns (-1, -1) if nothing qualifies.
func (r LayerPositionReport) EarliestOutlierPosition(relCutoff float32) (int, int) {
	for i, l := range r.Layers {
		median := max(medianMaxDiff(l.Drifts), 1e-6)
		for _, d := range l.Drifts {
			if !d.Passed && d.MaxDiff/median >= relCutoff {
				return i, d.Index
			}
		}
	}
	return -1, -1
}

// LogDriftRanks prints a ranked drift table to the test logger. Caller
// supplies the title so the same formatter works for absolute-ranked,
// relative-ranked, and per-tensor listings.
func LogDriftRanks(t *testing.T, title string, ranks []DriftRank) {
	t.Helper()
	t.Logf("%s", title)
	t.Logf("  %-24s %8s %12s %12s %10s %12s", "layer", "pos", "max_diff", "mean_diff", "cos_sim", "rel_to_med")
	for _, d := range ranks {
		t.Logf("  %-24s %8d %12.6f %12.6f %10.6f %12.2fx", d.Layer, d.Position, d.MaxDiff, d.MeanDiff, d.CosSim, d.RelToMedian)
	}
}

// medianMaxDiff returns the median of the per-position MaxDiff values.
// Uses a copy so the caller's slice order is preserved.
func medianMaxDiff(drifts []PositionDrift) float32 {
	if len(drifts) == 0 {
		return 0
	}
	vals := make([]float32, len(drifts))
	for i, d := range drifts {
		vals[i] = d.MaxDiff
	}
	slices.Sort(vals)
	mid := len(vals) / 2
	if len(vals)%2 == 1 {
		return vals[mid]
	}
	return (vals[mid-1] + vals[mid]) / 2
}

// CompareLayersPerPosition runs CompareArraysPerPosition over a set of named
// (got, want) tensor pairs and aggregates them. Layers are ordered by natural
// name sort so "layers.10" follows "layers.2", keeping the report deterministic
// and bisectable even when callers do not zero-pad layer numbers.
func CompareLayersPerPosition(t *testing.T, got, want map[string]*mlx.Array, axis int, opts ...CompareOption) LayerPositionReport {
	t.Helper()
	var keys []string
	for k := range want {
		if _, ok := got[k]; ok {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		return naturalLess(keys[i], keys[j])
	})

	report := LayerPositionReport{}
	for _, k := range keys {
		report.Layers = append(report.Layers, CompareArraysPerPosition(t, k, got[k], want[k], axis, opts...))
	}
	return report
}

func naturalLess(a, b string) bool {
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		ac, bc := a[i], b[j]
		if asciiDigit(ac) && asciiDigit(bc) {
			ai, bj := i, j
			for i < len(a) && asciiDigit(a[i]) {
				i++
			}
			for j < len(b) && asciiDigit(b[j]) {
				j++
			}
			an := trimLeadingZeroes(a[ai:i])
			bn := trimLeadingZeroes(b[bj:j])
			if len(an) != len(bn) {
				return len(an) < len(bn)
			}
			if an != bn {
				return an < bn
			}
			continue
		}
		if ac != bc {
			return ac < bc
		}
		i++
		j++
	}
	return len(a) < len(b)
}

func asciiDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

func trimLeadingZeroes(s string) string {
	for len(s) > 1 && s[0] == '0' {
		s = s[1:]
	}
	return s
}
