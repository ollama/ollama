package lfm3audio

import (
	"errors"
	"math"
	"math/cmplx"
	"runtime"
	"sync"

	"github.com/ollama/ollama/mlxrunner/model/audio"
)

// Features is one clip's log-mel spectrogram, mel-major like the reference
// input_features: Data[m*Frames+f] is mel bin m of frame f.
type Features struct {
	Data   []float32
	Frames int
	Tokens int
}

// Frames returns the spectrogram frame count for a clip of n samples. Clips
// are right-padded to a whole frame, so trailing samples are never dropped.
func (c Config) Frames(n int) int {
	if n < c.NFFT {
		return 1
	}
	return 1 + (n-c.NFFT+c.HopLength-1)/c.HopLength
}

// TokenCount returns the number of encoder outputs, and so of audio
// placeholder tokens, for a clip of n samples.
func (c Config) TokenCount(n int) int {
	return (c.Frames(n) + c.EncoderStride - 1) / c.EncoderStride
}

// FeatureExtractor computes log-mel features on the CPU.
type FeatureExtractor struct {
	cfg     Config
	window  []float32
	filters []float64 // [mel, bin]
	bins    int
	plan    *fftPlan
}

func NewFeatureExtractor(cfg Config) *FeatureExtractor {
	bins := cfg.NFFT/2 + 1
	window := make([]float32, cfg.WinLength)
	for i := range window {
		window[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(cfg.WinLength)))
	}
	return &FeatureExtractor{
		cfg:     cfg,
		window:  window,
		filters: slaneyMelFilters(bins, cfg.NumMelBins, float64(cfg.SampleRate)),
		bins:    bins,
		plan:    newFFTPlan(cfg.NFFT),
	}
}

// FromWAV decodes a WAV clip, mixes it to mono by averaging channels,
// resamples it to the model rate, and extracts its features.
func (e *FeatureExtractor) FromWAV(data []byte) (Features, error) {
	samples, rate, err := audio.Decode(data)
	if err != nil {
		return Features{}, err
	}
	samples = audio.Resample(samples, rate, e.cfg.SampleRate)
	if len(samples) == 0 {
		return Features{}, errors.New("audio is empty")
	}
	return e.Extract(samples), nil
}

// Extract computes the features of mono samples at the model rate.
func (e *FeatureExtractor) Extract(samples []float32) Features {
	c := e.cfg
	frames := c.Frames(len(samples))
	padded := samples
	if need := (frames-1)*c.HopLength + c.NFFT; need > len(samples) {
		padded = make([]float32, need)
		copy(padded, samples)
	}

	mels := c.NumMelBins
	out := make([]float32, mels*frames)
	workers := min(runtime.GOMAXPROCS(0), (frames+63)/64)
	per := (frames + workers - 1) / workers
	var wg sync.WaitGroup
	for w := range workers {
		lo, hi := w*per, min(frames, (w+1)*per)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			e.frames(padded, out, frames, lo, hi)
		}()
	}
	wg.Wait()
	return Features{Data: out, Frames: frames, Tokens: (frames + c.EncoderStride - 1) / c.EncoderStride}
}

func (e *FeatureExtractor) frames(samples, out []float32, frames, lo, hi int) {
	c := e.cfg
	in := make([]complex128, c.NFFT)
	spectrum := make([]complex128, c.NFFT)
	scratch := make([]complex128, c.NFFT)
	power := make([]float64, e.bins)
	for f := lo; f < hi; f++ {
		frame := samples[f*c.HopLength:]
		for i := range c.NFFT {
			in[i] = complex(float64(frame[i])*float64(e.window[i]), 0)
		}
		e.plan.transform(spectrum, in, 1, e.plan.factors, scratch)
		for k := range e.bins {
			re, im := real(spectrum[k]), imag(spectrum[k])
			power[k] = re*re + im*im
		}
		for m := range c.NumMelBins {
			row := e.filters[m*e.bins : (m+1)*e.bins]
			var mel float64
			for k, w := range row {
				mel += w * power[k]
			}
			v := c.LogFloor
			if mel > 0 {
				v = max(math.Log10(mel), c.LogFloor)
			}
			out[m*frames+f] = float32((v + c.NormOffset) / c.NormDivisor)
		}
	}
}

// slaneyMelFilters builds [mel, bin] triangular filters on the Slaney mel
// scale from 0 Hz to Nyquist, each normalized to unit area in Hz.
func slaneyMelFilters(bins, mels int, sampleRate float64) []float64 {
	const (
		breakHz  = 1000.0
		breakMel = 15.0
	)
	logStep := math.Log(6.4) / 27
	hzToMel := func(hz float64) float64 {
		if hz < breakHz {
			return 3 * hz / 200
		}
		return breakMel + math.Log(hz/breakHz)/logStep
	}
	melToHz := func(mel float64) float64 {
		if mel < breakMel {
			return 200 * mel / 3
		}
		return breakHz * math.Exp(logStep*(mel-breakMel))
	}

	nyquist := sampleRate / 2
	top := hzToMel(nyquist)
	edges := make([]float64, mels+2)
	for i := range edges {
		edges[i] = melToHz(top * float64(i) / float64(mels+1))
	}
	filters := make([]float64, mels*bins)
	for m := range mels {
		left, center, right := edges[m], edges[m+1], edges[m+2]
		area := 2 / (right - left)
		for k := range bins {
			hz := nyquist * float64(k) / float64(bins-1)
			rising := (hz - left) / (center - left)
			falling := (right - hz) / (right - center)
			if v := min(rising, falling); v > 0 {
				filters[m*bins+k] = v * area
			}
		}
	}
	return filters
}

// fftPlan is a recursive mixed-radix FFT for one length; the STFT size of
// 400 is not a power of two.
type fftPlan struct {
	n       int
	factors []int
	twiddle []complex128
}

func newFFTPlan(n int) *fftPlan {
	p := &fftPlan{n: n, twiddle: make([]complex128, n)}
	for k := range n {
		p.twiddle[k] = cmplx.Rect(1, -2*math.Pi*float64(k)/float64(n))
	}
	for rest, f := n, 2; rest > 1; {
		for rest%f != 0 {
			f++
		}
		p.factors = append(p.factors, f)
		rest /= f
	}
	return p
}

// transform writes the DFT of in[0], in[stride], ... into out, whose length
// is the product of factors. scratch needs room for the largest factor.
func (p *fftPlan) transform(out, in []complex128, stride int, factors []int, scratch []complex128) {
	n := len(out)
	if n == 1 {
		out[0] = in[0]
		return
	}
	radix := factors[0]
	m := n / radix
	for r := range radix {
		p.transform(out[r*m:(r+1)*m], in[r*stride:], stride*radix, factors[1:], scratch)
	}
	step := p.n / n
	sub := scratch[:radix]
	for k := range m {
		for r := range radix {
			sub[r] = out[r*m+k]
		}
		for q := range radix {
			i := k + q*m
			var sum complex128
			for r := range radix {
				sum += sub[r] * p.twiddle[(r*i*step)%p.n]
			}
			out[i] = sum
		}
	}
}
