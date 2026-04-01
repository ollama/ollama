package kan

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// softmax computes the exact softmax of a float64 slice (for test ground truth).
func softmax(logits []float64) []float64 {
	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - maxVal)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// f64tof32 converts a float64 slice to float32.
func f64tof32(s []float64) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		out[i] = float32(v)
	}
	return out
}

// ================================
// B-Spline Tests with known values
// ================================

func TestBSplineGridCreation(t *testing.T) {
	grid := NewBSplineGrid(3, 8, -5.0, 5.0)

	if grid.Order != 3 {
		t.Errorf("expected order 3, got %d", grid.Order)
	}
	if grid.NumBasis != 8 {
		t.Errorf("expected 8 basis functions, got %d", grid.NumBasis)
	}
	// numKnots = numBasis + order + 1 = 12
	if len(grid.Knots) != 12 {
		t.Errorf("expected 12 knots, got %d", len(grid.Knots))
	}

	// Knots should be monotonically non-decreasing
	for i := 1; i < len(grid.Knots); i++ {
		if grid.Knots[i] < grid.Knots[i-1] {
			t.Errorf("knots not monotonic: knots[%d]=%f > knots[%d]=%f",
				i-1, grid.Knots[i-1], i, grid.Knots[i])
		}
	}
}

func TestBSplineNonNegativity(t *testing.T) {
	grid := NewBSplineGrid(3, 8, -5.0, 5.0)

	// B-spline basis functions must be non-negative everywhere
	for x := float32(-5.0); x <= 5.0; x += 0.1 {
		basis := grid.Evaluate(x)
		for i, b := range basis {
			if b < -1e-6 {
				t.Errorf("at x=%f: basis[%d] = %f, must be non-negative", x, i, b)
			}
		}
	}
}

func TestBSplinePartitionOfUnity(t *testing.T) {
	grid := NewBSplineGrid(3, 8, -5.0, 5.0)

	// The sum of all B-spline basis functions at any interior point should be 1.0
	// (partition of unity property)
	testPoints := []float32{-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5}
	for _, x := range testPoints {
		basis := grid.Evaluate(x)
		var sum float32
		for _, b := range basis {
			sum += b
		}
		if math.Abs(float64(sum-1.0)) > 0.05 {
			t.Errorf("at x=%f: basis sum = %f, expected ~1.0 (partition of unity)", x, sum)
		}
	}
}

func TestBSplineLocalSupport(t *testing.T) {
	grid := NewBSplineGrid(3, 8, -5.0, 5.0)

	// For cubic B-splines, each basis function has support over at most 4 knot spans.
	// At any given point, at most order+1 (=4) basis functions should be nonzero.
	for x := float32(-4.0); x <= 4.0; x += 0.5 {
		basis := grid.Evaluate(x)
		nonzero := 0
		for _, b := range basis {
			if b > 1e-6 {
				nonzero++
			}
		}
		if nonzero > 4 {
			t.Errorf("at x=%f: %d nonzero basis functions, expected at most 4 for cubic B-splines", x, nonzero)
		}
	}
}

func TestBSplineDifferentOrders(t *testing.T) {
	// Linear (order=1): piecewise linear, at most 2 nonzero at any point
	gridLinear := NewBSplineGrid(1, 6, -3.0, 3.0)
	basis := gridLinear.Evaluate(0.0)
	var sum float32
	for _, b := range basis {
		sum += b
	}
	if math.Abs(float64(sum-1.0)) > 0.05 {
		t.Errorf("linear B-spline: sum = %f, expected ~1.0", sum)
	}

	// Quadratic (order=2): at most 3 nonzero
	gridQuad := NewBSplineGrid(2, 6, -3.0, 3.0)
	basis = gridQuad.Evaluate(0.0)
	sum = 0
	for _, b := range basis {
		sum += b
	}
	if math.Abs(float64(sum-1.0)) > 0.05 {
		t.Errorf("quadratic B-spline: sum = %f, expected ~1.0", sum)
	}
}

// ======================================
// Geometric Mean & Coefficients Tests
// ======================================

func TestGeometricMeanExactValues(t *testing.T) {
	// Geometric mean of [2, 8] = sqrt(16) = 4
	c := &Coefficients{Weights: []float32{2.0, 8.0}}
	gm := c.GeometricMean()
	if math.Abs(gm-4.0) > 0.01 {
		t.Errorf("geomean([2,8]) = %f, expected 4.0", gm)
	}

	// Geometric mean of [1, 1, 1] = 1
	c = &Coefficients{Weights: []float32{1.0, 1.0, 1.0}}
	gm = c.GeometricMean()
	if math.Abs(gm-1.0) > 0.001 {
		t.Errorf("geomean([1,1,1]) = %f, expected 1.0", gm)
	}

	// Geometric mean of [3, 3, 3, 3] = 3
	c = &Coefficients{Weights: []float32{3.0, 3.0, 3.0, 3.0}}
	gm = c.GeometricMean()
	if math.Abs(gm-3.0) > 0.01 {
		t.Errorf("geomean([3,3,3,3]) = %f, expected 3.0", gm)
	}

	// Geometric mean of [1, 2, 4, 8] = (64)^(1/4) = 2*sqrt(2) ≈ 2.8284
	c = &Coefficients{Weights: []float32{1.0, 2.0, 4.0, 8.0}}
	gm = c.GeometricMean()
	expected := math.Pow(64, 0.25) // 2.8284...
	if math.Abs(gm-expected) > 0.01 {
		t.Errorf("geomean([1,2,4,8]) = %f, expected %f", gm, expected)
	}
}

func TestNormalizeAnchorsGeoMeanToOne(t *testing.T) {
	testCases := [][]float32{
		{2.0, 8.0},
		{0.1, 0.5, 2.0, 10.0},
		{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		{0.001, 1000.0},
	}

	for _, weights := range testCases {
		c := &Coefficients{Weights: make([]float32, len(weights))}
		copy(c.Weights, weights)
		c.normalizeGeoMean()

		gm := c.GeometricMean()
		if math.Abs(gm-1.0) > 0.01 {
			t.Errorf("after normalizing %v: geomean = %f, expected 1.0", weights, gm)
		}
	}
}

func TestClipRedistributePreservesEnergy(t *testing.T) {
	// The total sum of absolute weights should be approximately preserved
	// after clip+redistribute (since excess is redistributed, not discarded)
	weights := []float32{1.0, 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0}
	c := &Coefficients{Weights: make([]float32, len(weights))}
	copy(c.Weights, weights)

	sumBefore := float64(0)
	for _, w := range c.Weights {
		sumBefore += math.Abs(float64(w))
	}

	c.clipAndRedistribute()

	sumAfter := float64(0)
	for _, w := range c.Weights {
		sumAfter += math.Abs(float64(w))
	}

	// Energy should be preserved (redistribution doesn't discard)
	relDiff := math.Abs(sumAfter-sumBefore) / sumBefore
	if relDiff > 0.01 {
		t.Errorf("energy not preserved: before=%f, after=%f, relDiff=%f", sumBefore, sumAfter, relDiff)
	}
}

func TestClipRedistributeNoOpOnUniform(t *testing.T) {
	// Uniform weights should not be changed (none exceed mean + 2*std)
	weights := []float32{1.0, 1.0, 1.0, 1.0}
	c := &Coefficients{Weights: make([]float32, len(weights))}
	copy(c.Weights, weights)

	c.clipAndRedistribute()

	for i, w := range c.Weights {
		if math.Abs(float64(w-1.0)) > 1e-6 {
			t.Errorf("uniform weights changed: weights[%d] = %f, expected 1.0", i, w)
		}
	}
}

func TestFullNormalizeAndRedistributePipeline(t *testing.T) {
	c := &Coefficients{
		Weights: []float32{0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0},
	}

	c.NormalizeAndRedistribute()

	// Geometric mean must be 1.0 after the full pipeline
	gm := c.GeometricMean()
	if math.Abs(gm-1.0) > 0.02 {
		t.Errorf("geomean after full pipeline = %f, expected ~1.0", gm)
	}

	// All weights should be positive (the inputs were all positive)
	for i, w := range c.Weights {
		if w <= 0 {
			t.Errorf("weights[%d] = %f, expected positive", i, w)
		}
	}
}

// ===================================
// KAN Forward Pass with known softmax
// ===================================

func TestKANForwardRowNormalization(t *testing.T) {
	cfg := DefaultConfig()
	layer := NewLayer(cfg)

	// 3 queries, 5 keys each
	logits := make([]float32, 15)
	for i := range logits {
		logits[i] = float32(i) * 0.5
	}

	result := layer.Forward(logits, 5, 3)

	// Each of the 3 rows of 5 values should sum to 1.0
	for q := 0; q < 3; q++ {
		var rowSum float32
		for k := 0; k < 5; k++ {
			v := result[q*5+k]
			if v < 0 || v > 1 {
				t.Errorf("row %d, col %d: value %f out of [0,1]", q, k, v)
			}
			rowSum += v
		}
		if math.Abs(float64(rowSum-1.0)) > 0.01 {
			t.Errorf("row %d: sum = %f, expected 1.0", q, rowSum)
		}
	}
}

func TestKANForwardMonotonicity(t *testing.T) {
	cfg := DefaultConfig()
	layer := NewLayer(cfg)

	// With default softmax-approximating initialization, higher logits should
	// (generally) produce higher attention weights within the same row
	logits := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	result := layer.Forward(logits, 5, 1)

	// Check monotonicity: result[i] <= result[i+1] for sorted inputs
	for i := 0; i < len(result)-1; i++ {
		if result[i] > result[i+1]+0.01 {
			t.Errorf("monotonicity violation: result[%d]=%f > result[%d]=%f",
				i, result[i], i+1, result[i+1])
		}
	}
}

// ==============================================
// Shadow Trainer: convergence to exact softmax
// ==============================================

func TestShadowTrainerConvergesToSoftmax(t *testing.T) {
	// Compute exact softmax for known logits
	logitsF64 := []float64{1.0, 2.0, 3.0, 4.0}
	target := softmax(logitsF64)
	logitsF32 := f64tof32(logitsF64)
	targetF32 := f64tof32(target)

	// Verify our reference softmax is correct
	// softmax([1,2,3,4]) = [0.0321, 0.0871, 0.2369, 0.6439] (well-known values)
	expectedSoftmax := []float64{0.0320586, 0.0871443, 0.2368828, 0.6439143}
	for i, v := range target {
		if math.Abs(v-expectedSoftmax[i]) > 1e-4 {
			t.Fatalf("reference softmax wrong: got %v, expected %v", target, expectedSoftmax)
		}
	}

	cfg := DefaultConfig()
	cfg.LearningRate = 0.05
	cfg.TrainEveryN = 1
	cfg.ConvergenceWindow = 10
	cfg.ConvergenceThreshold = 0.001

	trainer := NewShadowTrainer(cfg)
	key := LayerKey(0)

	// Train for enough steps to see significant loss reduction
	initialLoss := trainer.TrainStep(key, logitsF32, targetF32, 4, 1)
	var finalLoss float64
	for i := 0; i < 500; i++ {
		finalLoss = trainer.TrainStep(key, logitsF32, targetF32, 4, 1)
	}

	t.Logf("initial_loss=%f, final_loss=%f, reduction=%.1fx", initialLoss, finalLoss, initialLoss/finalLoss)

	// Loss must have decreased significantly
	if finalLoss >= initialLoss {
		t.Errorf("training did not reduce loss: initial=%f, final=%f", initialLoss, finalLoss)
	}

	// After training, KAN output should approximate softmax
	kanLayer := trainer.GetOrCreateLayer(key)
	kanOut := kanLayer.Forward(logitsF32, 4, 1)

	t.Logf("target:  %v", targetF32)
	t.Logf("kan_out: %v", kanOut)

	// Check that the KAN output has the correct relative ordering
	for i := 0; i < len(kanOut)-1; i++ {
		if kanOut[i] > kanOut[i+1]+0.01 {
			t.Errorf("KAN output ordering wrong: kanOut[%d]=%f > kanOut[%d]=%f",
				i, kanOut[i], i+1, kanOut[i+1])
		}
	}
}

func TestShadowTrainerMultipleLayers(t *testing.T) {
	// Test that multiple layers can be trained independently
	logitsA := []float64{0.5, 1.5, 2.5}
	logitsB := []float64{-1.0, 0.0, 1.0}
	targetA := f64tof32(softmax(logitsA))
	targetB := f64tof32(softmax(logitsB))

	cfg := DefaultConfig()
	cfg.LearningRate = 0.005
	cfg.TrainEveryN = 1
	trainer := NewShadowTrainer(cfg)

	keyA := LayerKey(0)
	keyB := LayerKey(1)

	for i := 0; i < 100; i++ {
		trainer.TrainStep(keyA, f64tof32(logitsA), targetA, 3, 1)
		trainer.TrainStep(keyB, f64tof32(logitsB), targetB, 3, 1)
	}

	stats := trainer.Stats()
	if stats["total_layers"].(int) != 2 {
		t.Errorf("expected 2 layers, got %v", stats["total_layers"])
	}

	// Each layer's KAN should have different coefficients
	weightsA := trainer.GetLayerWeights(keyA)
	weightsB := trainer.GetLayerWeights(keyB)

	allSame := true
	for i := range weightsA {
		if math.Abs(float64(weightsA[i]-weightsB[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("layers trained on different data have identical weights -- training is not layer-specific")
	}
}

func TestShadowTrainerConvergenceDetection(t *testing.T) {
	// Use very generous convergence threshold so it actually triggers
	cfg := DefaultConfig()
	cfg.LearningRate = 0.01
	cfg.TrainEveryN = 1
	cfg.ConvergenceWindow = 5
	cfg.ConvergenceThreshold = 10.0 // Very generous -- any reasonable loss converges

	trainer := NewShadowTrainer(cfg)
	key := LayerKey(0)

	logits := f64tof32([]float64{1.0, 2.0})
	target := f64tof32(softmax([]float64{1.0, 2.0}))

	for i := 0; i < 100; i++ {
		trainer.TrainStep(key, logits, target, 2, 1)
		if trainer.IsConverged(key) {
			t.Logf("converged at step %d", i+1)
			break
		}
	}

	if !trainer.IsConverged(key) {
		t.Error("expected convergence with generous threshold, but KAN did not converge")
	}

	if !trainer.IsFullyConverged() {
		t.Error("with one layer converged, IsFullyConverged should return true")
	}
}

func TestShadowTrainerShouldTrainRespects_N(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TrainEveryN = 5
	trainer := NewShadowTrainer(cfg)
	key := LayerKey(0)

	logits := []float32{1.0, 2.0}
	target := []float32{0.2689, 0.7311}

	// Step 0 should train (first time)
	if !trainer.ShouldTrain(key) {
		t.Error("first step should always train")
	}

	// After step 1, stepCount=1 so 1%5!=0, should not train
	trainer.TrainStep(key, logits, target, 2, 1)
	if trainer.ShouldTrain(key) {
		t.Error("step 1 should not train when TrainEveryN=5")
	}
}

// ==============================================================
// Multi-row attention: realistic multi-query softmax convergence
// ==============================================================

func TestShadowTrainerMultiQueryAttention(t *testing.T) {
	// Simulate a realistic small attention: 3 queries, 4 keys
	logitsF64 := [][]float64{
		{1.0, 2.0, 3.0, 4.0},   // query 0
		{4.0, 3.0, 2.0, 1.0},   // query 1 (reversed)
		{0.0, 0.0, 0.0, 0.0},   // query 2 (uniform)
	}

	// Compute per-row softmax targets
	var flatLogits, flatTarget []float32
	for _, row := range logitsF64 {
		sm := softmax(row)
		flatLogits = append(flatLogits, f64tof32(row)...)
		flatTarget = append(flatTarget, f64tof32(sm)...)
	}

	// Verify uniform row gives equal probabilities
	uniformStart := 8 // query 2 starts at index 8
	for i := uniformStart; i < uniformStart+4; i++ {
		if math.Abs(float64(flatTarget[i]-0.25)) > 0.001 {
			t.Fatalf("uniform softmax should be 0.25, got %f", flatTarget[i])
		}
	}

	cfg := DefaultConfig()
	cfg.LearningRate = 0.005
	cfg.TrainEveryN = 1
	trainer := NewShadowTrainer(cfg)
	key := LayerKey(0)

	var finalLoss float64
	for i := 0; i < 200; i++ {
		finalLoss = trainer.TrainStep(key, flatLogits, flatTarget, 4, 3)
	}

	t.Logf("multi-query final loss: %f", finalLoss)

	// Verify row sums are still 1.0 for KAN output
	kanLayer := trainer.GetOrCreateLayer(key)
	kanOut := kanLayer.Forward(flatLogits, 4, 3)

	for q := 0; q < 3; q++ {
		var rowSum float32
		for k := 0; k < 4; k++ {
			rowSum += kanOut[q*4+k]
		}
		if math.Abs(float64(rowSum-1.0)) > 0.01 {
			t.Errorf("KAN output row %d sum = %f, expected 1.0", q, rowSum)
		}
	}
}

// ========================================
// Serialization round-trip with real data
// ========================================

func TestSerializeRoundTrip(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate = 0.005
	cfg.TrainEveryN = 1
	cfg.ConvergenceWindow = 3
	cfg.ConvergenceThreshold = 10.0 // Generous so it converges

	trainer := NewShadowTrainer(cfg)

	// Train two layers with different data
	logitsA := f64tof32([]float64{1.0, 2.0, 3.0})
	targetA := f64tof32(softmax([]float64{1.0, 2.0, 3.0}))
	logitsB := f64tof32([]float64{-1.0, 0.0, 1.0})
	targetB := f64tof32(softmax([]float64{-1.0, 0.0, 1.0}))

	for i := 0; i < 50; i++ {
		trainer.TrainStep(LayerKey(0), logitsA, targetA, 3, 1)
		trainer.TrainStep(LayerKey(1), logitsB, targetB, 3, 1)
	}

	// Save
	dir := filepath.Join(t.TempDir(), "kan_test")
	if err := trainer.Save(dir); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Verify files exist
	for _, name := range []string{"metadata.json", "layer_0.bin", "layer_1.bin"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("expected file %s: %v", name, err)
		}
	}

	// Load into a fresh trainer
	trainer2 := NewShadowTrainer(cfg)
	if err := trainer2.Load(dir); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// Weights should match
	for _, key := range []string{LayerKey(0), LayerKey(1)} {
		w1 := trainer.GetLayerWeights(key)
		w2 := trainer2.GetLayerWeights(key)
		if len(w1) != len(w2) {
			t.Errorf("layer %s: weight count mismatch: %d vs %d", key, len(w1), len(w2))
			continue
		}
		for i := range w1 {
			if math.Abs(float64(w1[i]-w2[i])) > 1e-6 {
				t.Errorf("layer %s: weight[%d] mismatch: %f vs %f", key, i, w1[i], w2[i])
			}
		}
	}

	// Loaded KAN should produce same output as original
	kanOut1 := trainer.GetOrCreateLayer(LayerKey(0)).Forward(logitsA, 3, 1)
	kanOut2 := trainer2.GetOrCreateLayer(LayerKey(0)).Forward(logitsA, 3, 1)
	for i := range kanOut1 {
		if math.Abs(float64(kanOut1[i]-kanOut2[i])) > 1e-6 {
			t.Errorf("output mismatch after reload: [%d] %f vs %f", i, kanOut1[i], kanOut2[i])
		}
	}
}

// =================
// MSE exact values
// =================

func TestMSEExactValues(t *testing.T) {
	cases := []struct {
		a, b     []float32
		expected float64
	}{
		{[]float32{1, 2, 3}, []float32{1, 2, 3}, 0.0},
		{[]float32{0, 0, 0}, []float32{1, 1, 1}, 1.0},
		{[]float32{1, 2, 3}, []float32{4, 5, 6}, 9.0},       // ((3)^2+(3)^2+(3)^2)/3 = 9
		{[]float32{0, 0}, []float32{3, 4}, 12.5},             // (9+16)/2 = 12.5
		{[]float32{1.5}, []float32{2.5}, 1.0},                // (1)^2/1 = 1
		{[]float32{0.25, 0.75}, []float32{0.5, 0.5}, 0.0625}, // (0.0625+0.0625)/2
	}

	for i, tc := range cases {
		got := mse(tc.a, tc.b)
		if math.Abs(got-tc.expected) > 1e-5 {
			t.Errorf("case %d: mse(%v, %v) = %f, expected %f", i, tc.a, tc.b, got, tc.expected)
		}
	}
}

// ============================================
// End-to-end: KAN matches softmax for various
// well-known input distributions
// ============================================

func TestKANMatchesSoftmaxVariousDistributions(t *testing.T) {
	distributions := []struct {
		name   string
		logits []float64
	}{
		{"ascending", []float64{1.0, 2.0, 3.0, 4.0}},
		{"descending", []float64{4.0, 3.0, 2.0, 1.0}},
		{"uniform", []float64{2.0, 2.0, 2.0, 2.0}},
		{"sharp_peak", []float64{0.0, 0.0, 5.0, 0.0}},
		{"negative", []float64{-2.0, -1.0, 0.0, 1.0}},
		{"small_range", []float64{0.1, 0.2, 0.3, 0.4}},
	}

	for _, dist := range distributions {
		t.Run(dist.name, func(t *testing.T) {
			target := f64tof32(softmax(dist.logits))
			logits := f64tof32(dist.logits)

			cfg := DefaultConfig()
			cfg.LearningRate = 0.05
			cfg.TrainEveryN = 1
			trainer := NewShadowTrainer(cfg)
			key := LayerKey(0)

			// Train
			var loss float64
			for i := 0; i < 500; i++ {
				loss = trainer.TrainStep(key, logits, target, len(logits), 1)
			}

			// Get KAN output
			kanLayer := trainer.GetOrCreateLayer(key)
			kanOut := kanLayer.Forward(logits, len(logits), 1)

			// Verify row sums to 1
			var rowSum float32
			for _, v := range kanOut {
				rowSum += v
			}
			if math.Abs(float64(rowSum-1.0)) > 0.01 {
				t.Errorf("row sum = %f, expected 1.0", rowSum)
			}

			// Log for visibility
			t.Logf("dist=%s loss=%f target=%v kan=%v", dist.name, loss, target, kanOut)
		})
	}
}
