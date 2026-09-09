package llm

import "testing"

const (
	mib = 1024 * 1024
	gib = 1024 * mib
)

func testKey() CalibrationKey {
	return CalibrationKey{Model: "/models/test.gguf", ModelSize: 1234, NumParallel: 1, NumGPU: 1}
}

// withinMiB keeps the assertions readable: these are memory sizes, and agreement to the
// byte is neither achievable from a least-squares fit nor required by the caller.
func withinMiB(got, want uint64, mibs float64) bool {
	diff := float64(got) - float64(want)
	if diff < 0 {
		diff = -diff
	}
	return diff <= mibs*mib
}

func TestCalibrationFallsBackToPriorWithoutSamples(t *testing.T) {
	c := NewVRAMCalibration()

	got, calibrated := c.Predict(testKey(), 8192, 10*gib, 1024)
	if calibrated {
		t.Error("reported a calibrated prediction with no samples recorded")
	}
	if want := uint64(10*gib + 8192*1024); got != want {
		t.Errorf("prior not returned unchanged: got=%d want=%d", got, want)
	}
}

func TestCalibrationFitsTwoSamples(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	// 20 GiB of weights plus exactly 2048 bytes per token.
	c.Record(key, 8192, 20*gib+8192*2048)
	c.Record(key, 32768, 20*gib+32768*2048)

	for _, numCtx := range []int{4096, 16384, 65536, 262144} {
		got, calibrated := c.Predict(key, numCtx, 999*gib, 1)
		if !calibrated {
			t.Fatalf("ctx %d: fell back to the prior despite two samples", numCtx)
		}
		want := uint64(20*gib + numCtx*2048)
		if !withinMiB(got, want, 1) {
			t.Errorf("ctx %d: got=%d want=%d", numCtx, got, want)
		}
	}
}

// The prior is deliberately absurd in the test above and here: a calibrated prediction
// must not be influenced by it at all.
func TestCalibrationSingleSampleUsesPriorSlopeThroughMeasuredPoint(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()
	c.Record(key, 8192, 30*gib)

	// One sample fixes the intercept; the slope has to come from the prior.
	got, calibrated := c.Predict(key, 16384, 999*gib, 4096)
	if !calibrated {
		t.Fatal("a single sample should still beat the prior")
	}
	if want := uint64(30*gib + 8192*4096); got != want {
		t.Errorf("got=%d want=%d", got, want)
	}

	// And it must extrapolate downwards as well, without underflowing.
	got, _ = c.Predict(key, 1024, 999*gib, 4096)
	if want := uint64(30*gib - 7168*4096); got != want {
		t.Errorf("downward: got=%d want=%d", got, want)
	}
	if got, _ := c.Predict(key, 1, 0, 1<<62); got != 0 {
		t.Errorf("expected underflow to clamp at 0, got %d", got)
	}
}

func TestCalibrationKeyChangeInvalidates(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()
	c.Record(key, 8192, 20*gib+8192*2048)
	c.Record(key, 32768, 20*gib+32768*2048)

	// Each of these describes a load whose memory use is not comparable, so each must miss
	// the samples above rather than silently applying them.
	changed := map[string]func(*CalibrationKey){
		"model":           func(k *CalibrationKey) { k.Model = "/models/other.gguf" },
		"model size":      func(k *CalibrationKey) { k.ModelSize = 5678 },
		"projector":       func(k *CalibrationKey) { k.Projectors = "/blobs/sha256-abc" },
		"kv cache type":   func(k *CalibrationKey) { k.KVCacheType = "q8_0" },
		"flash attention": func(k *CalibrationKey) { k.FlashAttention = true },
		"num batch":       func(k *CalibrationKey) { k.NumBatch = 512 },
		"num parallel":    func(k *CalibrationKey) { k.NumParallel = 4 },
		"num gpu":         func(k *CalibrationKey) { k.NumGPU = 2 },
	}

	for name, mutate := range changed {
		t.Run(name, func(t *testing.T) {
			other := testKey()
			mutate(&other)
			if _, calibrated := c.Predict(other, 8192, 10*gib, 1024); calibrated {
				t.Errorf("%s changed but the old samples were still applied", name)
			}
		})
	}

	if _, calibrated := c.Predict(key, 8192, 10*gib, 1024); !calibrated {
		t.Error("the original key stopped resolving to its own samples")
	}
}

func TestCalibrationRecordReplacesSameContext(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	c.Record(key, 8192, 40*gib)
	c.Record(key, 8192, 50*gib) // the model changed underneath, or the first load was odd
	c.Record(key, 16384, 50*gib+8192*2048)

	got, calibrated := c.Predict(key, 8192, 999*gib, 1)
	if !calibrated {
		t.Fatal("expected a calibrated prediction")
	}
	if !withinMiB(got, 50*gib, 1) {
		t.Errorf("stale sample was not replaced: got=%d want≈%d", got, uint64(50*gib))
	}
}

func TestCalibrationIgnoresUnusableInput(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	c.Record(key, 0, 40*gib)  // no context length to attribute it to
	c.Record(key, 8192, 0)    // a load that reported nothing
	c.Record(key, -1, 40*gib) // nonsense

	if _, calibrated := c.Predict(key, 8192, 10*gib, 1024); calibrated {
		t.Error("unusable samples were recorded")
	}

	// A nil store is usable and simply never calibrates, so callers need no nil check.
	var nilStore *VRAMCalibration
	nilStore.Record(key, 8192, 40*gib)
	if got, calibrated := nilStore.Predict(key, 8192, 10*gib, 1024); calibrated || got != uint64(10*gib+8192*1024) {
		t.Errorf("nil store: got=%d calibrated=%v", got, calibrated)
	}
}

func TestCalibrationRejectsNegativeSlope(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	// Memory does not shrink as context grows. Samples that say otherwise describe
	// something this model cannot express, so the prior is safer than the line.
	c.Record(key, 8192, 40*gib)
	c.Record(key, 32768, 30*gib)

	got, calibrated := c.Predict(key, 262144, 50*gib, 1024)
	if calibrated {
		t.Error("extrapolated a negative slope instead of falling back")
	}
	if want := uint64(50*gib + 262144*1024); got != want {
		t.Errorf("got=%d want=%d", got, want)
	}
}

func TestCalibrationBoundsSampleCount(t *testing.T) {
	c := NewVRAMCalibration()
	key := testKey()

	for i := 1; i <= maxCalibrationSamples+5; i++ {
		c.Record(key, i*1024, uint64(20*gib+i*1024*2048))
	}

	c.mu.Lock()
	n := len(c.samples[key])
	c.mu.Unlock()
	if n != maxCalibrationSamples {
		t.Errorf("sample count not bounded: got=%d want=%d", n, maxCalibrationSamples)
	}

	// Dropping the oldest must not disturb the fit, since the line is the same.
	got, calibrated := c.Predict(key, 100*1024, 999*gib, 1)
	if !calibrated {
		t.Fatal("expected a calibrated prediction")
	}
	if want := uint64(20*gib + 100*1024*2048); !withinMiB(got, want, 1) {
		t.Errorf("got=%d want=%d", got, want)
	}
}

// TestCalibrationAgainstMeasuredCurve uses VRAM actually measured on a two-card host, to
// check that calibrating from the two cheapest loads predicts the expensive ones. The
// metadata prediction is 7.05 GiB low at the longest context because it models one KV
// cache and this architecture allocates two; the point of calibrating is that the shortfall
// does not have to be understood to be corrected.
func TestCalibrationAgainstMeasuredCurve(t *testing.T) {
	measured := []struct {
		numCtx int
		vram   float64 // GiB, from llama-server's own buffer accounting
	}{
		{8192, 76.84},
		{32768, 77.71},
		{131072, 81.41},
		{262144, 86.35},
	}

	c := NewVRAMCalibration()
	key := testKey()
	for _, m := range measured[:2] {
		c.Record(key, m.numCtx, uint64(m.vram*gib))
	}

	for _, m := range measured[2:] {
		got, calibrated := c.Predict(key, m.numCtx, 76*gib, 10138)
		if !calibrated {
			t.Fatalf("ctx %d: expected a calibrated prediction", m.numCtx)
		}
		want := uint64(m.vram * gib)
		if !withinMiB(got, want, 600) {
			t.Errorf("ctx %d: got=%.2f GiB want=%.2f GiB (off by %.0f MiB)",
				m.numCtx, float64(got)/gib, m.vram, (float64(got)-float64(want))/mib)
		}
	}
}
