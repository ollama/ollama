package lfm3audio

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestFFTMatchesDFT(t *testing.T) {
	for _, n := range []int{1, 2, 12, 400, 7 * 9} {
		plan := newFFTPlan(n)
		in := make([]complex128, n)
		for i := range in {
			in[i] = complex(math.Sin(float64(i*i)*0.37)+0.1*float64(i%7), math.Cos(float64(i)*1.3))
		}
		out := make([]complex128, n)
		plan.transform(out, in, 1, plan.factors, make([]complex128, n))
		for k := range n {
			var want complex128
			for j := range n {
				want += in[j] * cmplx.Rect(1, -2*math.Pi*float64(j*k%n)/float64(n))
			}
			if d := cmplx.Abs(out[k] - want); d > 1e-9*float64(n) {
				t.Fatalf("n=%d bin %d: got %v want %v", n, k, out[k], want)
			}
		}
	}
}

func TestFrameAndTokenCounts(t *testing.T) {
	cfg := Config{NFFT: 400, HopLength: 160, EncoderStride: 8}
	// Counts from the publisher feature extractor on the bring-up clips.
	for _, tc := range []struct{ samples, frames, tokens int }{
		{0, 1, 1},
		{300, 1, 1},
		{400, 1, 1},
		{401, 2, 1},
		{560, 2, 1},
		{16000, 99, 13},
		{16123, 100, 13},
		{64466, 402, 51},
		{160000, 999, 125},
		{608681, 3803, 476},
		{3043405, 19020, 2378},
	} {
		if got := cfg.Frames(tc.samples); got != tc.frames {
			t.Errorf("Frames(%d) = %d, want %d", tc.samples, got, tc.frames)
		}
		if got := cfg.TokenCount(tc.samples); got != tc.tokens {
			t.Errorf("TokenCount(%d) = %d, want %d", tc.samples, got, tc.tokens)
		}
	}
}

func defaultConfig() Config {
	return Config{SampleRate: 16000, NumMelBins: 128, NFFT: 400, HopLength: 160, WinLength: 400, LogFloor: -12, NormOffset: 4, NormDivisor: 4, EncoderStride: 8}
}

func TestMelFiltersMatchReference(t *testing.T) {
	// Entries and total of the publisher's Slaney filter bank, [bin, mel].
	filters := slaneyMelFilters(201, 128, 16000)
	for _, tc := range []struct {
		bin, mel int
		want     float64
	}{
		{3, 4, 0.037121958997910855},
		{57, 76, 0.017146784337107264},
		{97, 97, 8.431751844295866e-05},
		{131, 110, 0.004396729194795724},
		{162, 119, 0.0047705221623043255},
	} {
		if got := filters[tc.mel*201+tc.bin]; math.Abs(got-tc.want) > 1e-8*math.Abs(tc.want)+1e-12 {
			t.Errorf("filter[bin %d, mel %d] = %.12g, want %.12g", tc.bin, tc.mel, got, tc.want)
		}
	}
	var sum float64
	for _, v := range filters {
		sum += v
	}
	if math.Abs(sum-3.1909855285406996) > 1e-7 {
		t.Errorf("filter sum = %.12g", sum)
	}
}

func TestFeaturesMatchReference(t *testing.T) {
	// A tone pair over a deterministic broadband floor keeps every bin well
	// above float32 rounding noise in the reference.
	samples := make([]float32, 4123)
	for i := range samples {
		n := float64(i)
		samples[i] = float32(0.5*math.Sin(2*math.Pi*440*n/16000) + 0.25*math.Sin(2*math.Pi*3000*n/16000+0.3) +
			0.02*(float64((i*7919)%1009)/1009-0.5))
	}
	f := NewFeatureExtractor(defaultConfig()).Extract(samples)
	if f.Frames != 25 || f.Tokens != 4 || len(f.Data) != 128*25 {
		t.Fatalf("frames=%d tokens=%d len=%d", f.Frames, f.Tokens, len(f.Data))
	}
	for _, tc := range []struct {
		mel, frame int
		want       float64
	}{
		{0, 0, -0.35767197608947754},
		{5, 0, -0.4665696620941162},
		{17, 3, 1.3373782634735107},
		{40, 12, 0.10283339023590088},
		{64, 24, 0.4370649456977844},
		{100, 7, -0.4064192771911621},
		{127, 24, 0.18086832761764526},
		{90, 18, -0.39610719680786133},
	} {
		if got := float64(f.Data[tc.mel*f.Frames+tc.frame]); math.Abs(got-tc.want) > 2e-5 {
			t.Errorf("feature[mel %d, frame %d] = %.7f, want %.7f", tc.mel, tc.frame, got, tc.want)
		}
	}
	for frame, want := range map[int]float64{0: -22.422245383262634, 12: -26.513341903686523, 24: 68.51431185007095} {
		var sum float64
		for m := range 128 {
			sum += float64(f.Data[m*f.Frames+frame])
		}
		if math.Abs(sum-want) > 1e-3 {
			t.Errorf("frame %d sum = %.6f, want %.6f", frame, sum, want)
		}
	}
}

func TestFeaturesSilenceHitsFloor(t *testing.T) {
	f := NewFeatureExtractor(defaultConfig()).Extract(make([]float32, 1000))
	for i, v := range f.Data {
		if v != -2 {
			t.Fatalf("feature %d = %v, want the floor -2", i, v)
		}
	}
}
