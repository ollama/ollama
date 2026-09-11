package gemma4

import (
	"errors"
	"math"
	"math/cmplx"
	"sync"

	"github.com/ollama/ollama/x/mlxrunner/model/audio"
)

// Audio front-end constants from the reference Gemma4AudioFeatureExtractor
// and processor config; none are exposed in config.json.
const (
	audioSampleRate = 16000
	audioMelBins    = 128
	audioFrameLen   = 320 // 20 ms
	audioHopLen     = 160 // 10 ms
	audioFFTLen     = 512
	audioMelFloor   = 1e-3
	audioMaxFreq    = 8000.0

	// Clips longer than 30 s are encoded as independent chunks of at most
	// 30 s, one soft-token run each; the reference extractor truncates at
	// this length instead.
	audioChunkSeconds = 30
)

// audioChunk is one span of the clip: its encoder input rows — log-mel
// frames for the conformer, one raw waveform frame per soft token for the
// unified embedder — and the soft-token count they reduce to.
type audioChunk struct {
	data      []float32 // [frames, len/frames]
	frames    int
	numTokens int
}

// processAudio decodes an audio container into per-chunk log-mel features,
// resampled to 16 kHz.
func processAudio(data []byte) ([]audioChunk, error) {
	samples, rate, err := audio.Decode(data)
	if err != nil {
		return nil, err
	}
	samples = audio.Resample(samples, rate, audioSampleRate)

	// One frame needs frameLen+1 samples after the semicausal left pad.
	if len(samples) < audioFrameLen+1-audioFrameLen/2 {
		return nil, errors.New("audio too short")
	}

	var chunks []audioChunk
	for _, chunk := range audio.Split(samples, audioSampleRate, audioChunkSeconds) {
		mel, frames := melSpectrogram(chunk)
		tokens := frames
		for range 2 {
			tokens = (tokens-1)/2 + 1
		}
		chunks = append(chunks, audioChunk{data: mel, frames: frames, numTokens: tokens})
	}
	return chunks, nil
}

// processUnifiedAudio decodes audio for the encoder-free variant: raw
// 16 kHz samples in fixed-length frames, one soft token per frame, the
// final partial frame zero-padded. No 30 s chunking; length is bounded by
// the decode duration cap and the context.
func processUnifiedAudio(data []byte, samplesPerToken int) ([]float32, int, error) {
	samples, rate, err := audio.Decode(data)
	if err != nil {
		return nil, 0, err
	}
	samples = audio.Resample(samples, rate, audioSampleRate)
	if len(samples) == 0 {
		return nil, 0, errors.New("audio too short")
	}

	tokens := (len(samples) + samplesPerToken - 1) / samplesPerToken
	frames := make([]float32, tokens*samplesPerToken)
	copy(frames, samples)
	return frames, tokens, nil
}

// audioWindow is the periodic Hann window, rounded to float32 like the
// reference before it multiplies the frame.
var audioWindow = sync.OnceValue(func() []float32 {
	w := make([]float32, audioFrameLen)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/audioFrameLen))
	}
	return w
})

// melSpectrogram computes [frames, audioMelBins] log-mel features over the
// samples with semicausal padding: frameLen/2 zeros are prepended so the
// first frame is centered at t=0, and every frame's window covers only real
// audio past that — the reference's trailing pad-and-mask produces the same
// valid frames.
func melSpectrogram(samples []float32) ([]float32, int) {
	pad := audioFrameLen / 2
	if len(samples)+pad < audioFrameLen+1 {
		return nil, 0
	}
	frames := (len(samples)+pad-(audioFrameLen+1))/audioHopLen + 1

	padded := make([]float32, pad+len(samples))
	copy(padded[pad:], samples)

	window := audioWindow()
	filters := audioMelFilters()
	numBins := audioFFTLen/2 + 1

	out := make([]float32, frames*audioMelBins)
	fftBuf := make([]complex128, audioFFTLen)
	mags := make([]float64, numBins)
	for f := range frames {
		frame := padded[f*audioHopLen:]
		for i := range audioFrameLen {
			fftBuf[i] = complex(float64(frame[i]*window[i]), 0)
		}
		for i := audioFrameLen; i < audioFFTLen; i++ {
			fftBuf[i] = 0
		}
		fft(fftBuf)
		for k := range numBins {
			mags[k] = cmplx.Abs(fftBuf[k])
		}

		for m := range audioMelBins {
			var mel float64
			for k := range numBins {
				mel += mags[k] * filters[m*numBins+k]
			}
			out[f*audioMelBins+m] = float32(math.Log(mel + audioMelFloor))
		}
	}
	return out, frames
}

// audioMelFilters builds the HTK-scale triangular filterbank as [mel, bin]
// weights, mirroring the reference mel_filter_bank (no normalization).
var audioMelFilters = sync.OnceValue(func() []float64 {
	hzToMel := func(f float64) float64 { return 2595 * math.Log10(1+f/700) }
	melToHz := func(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }

	melMax := hzToMel(audioMaxFreq)
	corners := make([]float64, audioMelBins+2)
	for i := range corners {
		corners[i] = melToHz(float64(i) * melMax / float64(audioMelBins+1))
	}

	numBins := audioFFTLen/2 + 1
	filters := make([]float64, audioMelBins*numBins)
	for m := range audioMelBins {
		left, center, right := corners[m], corners[m+1], corners[m+2]
		for k := range numBins {
			freq := float64(k) * audioSampleRate / audioFFTLen
			v := min((freq-left)/(center-left), (right-freq)/(right-center))
			if v > 0 {
				filters[m*numBins+k] = v
			}
		}
	}
	return filters
})

// fft performs an in-place Cooley-Tukey radix-2 FFT.
func fft(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}

	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}

	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		for start := 0; start < n; start += size {
			wn := complex(1, 0)
			for k := range half {
				t := wn * x[start+k+half]
				x[start+k+half] = x[start+k] - t
				x[start+k] = x[start+k] + t
				wn *= w
			}
		}
	}
}
