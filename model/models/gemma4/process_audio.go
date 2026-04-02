package gemma4

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/cmplx"
)

// Audio preprocessing constants.
const (
	audioSampleRate    = 16000
	melBins            = 128
	frameLengthMs      = 20.0
	hopLengthMs        = 10.0
	minFrequency       = 0.0
	maxFrequency       = 8000.0
	melFloor           = 1e-3
	maxAudioSoftTokens = 750
)

// Computed from the above constants.
var (
	frameLength = int(math.Round(audioSampleRate * frameLengthMs / 1000.0)) // 320
	hopLength   = int(math.Round(audioSampleRate * hopLengthMs / 1000.0))   // 160
)

// decodeWAV extracts mono float32 PCM samples from a WAV file, resampled to 16kHz.
func decodeWAV(data []byte) ([]float32, error) {
	if len(data) < 12 {
		return nil, fmt.Errorf("WAV file too short")
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAV file")
	}

	var audioFormat uint16
	var numChannels, sampleRate, bitsPerSample int
	var audioData []byte
	foundFmt := false

	offset := 12
	for offset+8 <= len(data) {
		chunkID := string(data[offset : offset+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		chunkData := data[offset+8 : min(offset+8+chunkSize, len(data))]

		switch chunkID {
		case "fmt ":
			if len(chunkData) < 16 {
				return nil, fmt.Errorf("fmt chunk too short")
			}
			audioFormat = binary.LittleEndian.Uint16(chunkData[0:2])
			numChannels = int(binary.LittleEndian.Uint16(chunkData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunkData[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(chunkData[14:16]))
			if audioFormat == 0xFFFE && len(chunkData) >= 26 {
				audioFormat = binary.LittleEndian.Uint16(chunkData[24:26])
			}
			foundFmt = true
		case "data":
			audioData = chunkData
		}

		offset += 8 + chunkSize
		if chunkSize%2 != 0 {
			offset++
		}
	}

	if !foundFmt {
		return nil, fmt.Errorf("no fmt chunk found in WAV file")
	}
	if audioFormat != 1 && audioFormat != 3 {
		return nil, fmt.Errorf("unsupported WAV format: %d (need PCM=1 or float=3)", audioFormat)
	}
	if audioData == nil {
		return nil, fmt.Errorf("no data chunk found in WAV file")
	}

	samples := decodeWAVSamples(audioData, audioFormat, bitsPerSample, numChannels)
	if sampleRate != audioSampleRate {
		samples = resampleLinear(samples, sampleRate, audioSampleRate)
	}
	return samples, nil
}

func decodeWAVSamples(data []byte, format uint16, bits, channels int) []float32 {
	bytesPerSample := bits / 8
	totalSamples := len(data) / (bytesPerSample * channels)
	mono := make([]float32, totalSamples)

	for i := range totalSamples {
		var sum float64
		for ch := range channels {
			off := (i*channels + ch) * bytesPerSample
			if off+bytesPerSample > len(data) {
				break
			}
			switch {
			case format == 1 && bits == 16:
				v := int16(binary.LittleEndian.Uint16(data[off : off+2]))
				sum += float64(v) / 32768.0
			case format == 1 && bits == 32:
				v := int32(binary.LittleEndian.Uint32(data[off : off+4]))
				sum += float64(v) / 2147483648.0
			case format == 1 && bits == 24:
				v := int32(data[off]) | int32(data[off+1])<<8 | int32(data[off+2])<<16
				if v&0x800000 != 0 {
					v |= ^0xFFFFFF
				}
				sum += float64(v) / 8388608.0
			case format == 3 && bits == 32:
				v := math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4]))
				sum += float64(v)
			case format == 1 && bits == 8:
				sum += (float64(data[off]) - 128.0) / 128.0
			}
		}
		mono[i] = float32(sum / float64(channels))
	}
	return mono
}

func resampleLinear(samples []float32, fromRate, toRate int) []float32 {
	n := int(float64(len(samples)) / float64(fromRate) * float64(toRate))
	out := make([]float32, n)
	for i := range n {
		pos := float64(i) * float64(len(samples)-1) / float64(n-1)
		idx := int(pos)
		frac := float32(pos - float64(idx))
		if idx+1 < len(samples) {
			out[i] = samples[idx]*(1-frac) + samples[idx+1]*frac
		} else {
			out[i] = samples[idx]
		}
	}
	return out
}

// computeMelSpectrogram computes the log mel spectrogram from PCM samples.
// Returns shape [numFrames, melBins] as float32 slice, and numFrames.
func computeMelSpectrogram(samples []float32) ([]float32, int) {
	fftLen := 1
	for fftLen < frameLength {
		fftLen <<= 1
	}
	fftLen *= 2 // fft_overdrive=True

	// Hanning-nonzero window.
	window := make([]float64, frameLength)
	arg := math.Pi * 2.0 / float64(frameLength)
	for i := range frameLength {
		window[i] = 0.5 - 0.5*math.Cos(arg*(float64(i)+0.5))
	}

	numFreqBins := fftLen/2 + 1
	melFilters := buildMelFilterBank(numFreqBins, melBins, minFrequency, maxFrequency, audioSampleRate)

	frameSizeForUnfold := frameLength + 1
	numFrames := (len(samples) - frameSizeForUnfold) / hopLength
	if numFrames <= 0 {
		return nil, 0
	}

	result := make([]float32, numFrames*melBins)
	fftInput := make([]complex128, fftLen)

	for f := range numFrames {
		start := f * hopLength
		for i := range frameLength {
			fftInput[i] = complex(float64(samples[start+i])*window[i], 0)
		}
		for i := frameLength; i < fftLen; i++ {
			fftInput[i] = 0
		}

		fft(fftInput)

		for m := range melBins {
			var melVal float64
			for k := range numFreqBins {
				mag := cmplx.Abs(fftInput[k])
				melVal += mag * float64(melFilters[k*melBins+m])
			}
			if melVal < melFloor {
				melVal = melFloor
			}
			result[f*melBins+m] = float32(math.Log(melVal))
		}
	}

	return result, numFrames
}

func buildMelFilterBank(numFreqBins, numMels int, fMin, fMax float64, sr int) []float32 {
	hzToMel := func(f float64) float64 {
		return 2595.0 * math.Log10(1.0+f/700.0)
	}
	melToHz := func(m float64) float64 {
		return 700.0 * (math.Pow(10.0, m/2595.0) - 1.0)
	}

	melMin := hzToMel(fMin)
	melMax := hzToMel(fMax)

	melPts := make([]float64, numMels+2)
	for i := range melPts {
		melPts[i] = melMin + float64(i)*(melMax-melMin)/float64(numMels+1)
	}
	filterFreqs := make([]float64, numMels+2)
	for i, m := range melPts {
		filterFreqs[i] = melToHz(m)
	}

	fftFreqs := make([]float64, numFreqBins)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * float64(sr) / float64(2*(numFreqBins-1))
	}

	filters := make([]float32, numFreqBins*numMels)
	for m := range numMels {
		fLeft := filterFreqs[m]
		fCenter := filterFreqs[m+1]
		fRight := filterFreqs[m+2]
		for k := range numFreqBins {
			f := fftFreqs[k]
			var v float64
			if f >= fLeft && f <= fCenter && fCenter > fLeft {
				v = (f - fLeft) / (fCenter - fLeft)
			} else if f > fCenter && f <= fRight && fRight > fCenter {
				v = (fRight - f) / (fRight - fCenter)
			}
			if v > 0 {
				filters[k*numMels+m] = float32(v)
			}
		}
	}
	return filters
}

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
		halfSize := size / 2
		w := complex(math.Cos(2*math.Pi/float64(size)), -math.Sin(2*math.Pi/float64(size)))
		for start := 0; start < n; start += size {
			wn := complex(1, 0)
			for k := range halfSize {
				t := wn * x[start+k+halfSize]
				x[start+k+halfSize] = x[start+k] - t
				x[start+k] = x[start+k] + t
				wn *= w
			}
		}
	}
}

// isAudioData checks if the data starts with WAV magic bytes.
func isAudioData(data []byte) bool {
	return len(data) >= 12 && string(data[0:4]) == "RIFF" && string(data[8:12]) == "WAVE"
}
