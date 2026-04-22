package nemotronh

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/cmplx"
)

const (
	parakeetHopLength         = 160
	parakeetNFFT              = 512
	parakeetWinLength         = 400
	parakeetPreemphasis       = 0.97
	parakeetLogZeroGuardValue = 1.0 / (1 << 24)
	parakeetNormalizeEps      = 1e-5
)

func isAudioData(data []byte) bool {
	return len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE"
}

func decodeWAV(data []byte, targetSampleRate int) ([]float32, error) {
	if len(data) < 12 {
		return nil, fmt.Errorf("WAV file too short")
	}
	if !isAudioData(data) {
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
		chunkEnd := min(offset+8+chunkSize, len(data))
		chunkData := data[offset+8 : chunkEnd]

		switch chunkID {
		case "fmt ":
			if len(chunkData) < 16 {
				return nil, fmt.Errorf("fmt chunk too short")
			}
			audioFormat = binary.LittleEndian.Uint16(chunkData[0:2])
			numChannels = int(binary.LittleEndian.Uint16(chunkData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunkData[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(chunkData[14:16]))
			if audioFormat == 0xfffe && len(chunkData) >= 26 {
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
	if numChannels <= 0 {
		return nil, fmt.Errorf("invalid WAV channel count: %d", numChannels)
	}

	samples := decodeWAVSamples(audioData, audioFormat, bitsPerSample, numChannels)
	if sampleRate != targetSampleRate {
		samples = resampleLinear(samples, sampleRate, targetSampleRate)
	}
	return samples, nil
}

func decodeWAVSamples(data []byte, format uint16, bits, channels int) []float32 {
	bytesPerSample := bits / 8
	if bytesPerSample <= 0 || channels <= 0 {
		return nil
	}
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
					v |= ^0xffffff
				}
				sum += float64(v) / 8388608.0
			case format == 3 && bits == 32:
				sum += float64(math.Float32frombits(binary.LittleEndian.Uint32(data[off : off+4])))
			case format == 1 && bits == 8:
				sum += (float64(data[off]) - 128.0) / 128.0
			}
		}
		mono[i] = float32(sum / float64(channels))
	}
	return mono
}

func resampleLinear(samples []float32, fromRate, toRate int) []float32 {
	if fromRate <= 0 || toRate <= 0 || len(samples) == 0 {
		return samples
	}
	n := int(float64(len(samples)) / float64(fromRate) * float64(toRate))
	if n <= 1 {
		return slicesCloneOne(samples)
	}
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

func slicesCloneOne(samples []float32) []float32 {
	if len(samples) == 0 {
		return nil
	}
	return []float32{samples[0]}
}

func computeParakeetMelSpectrogram(samples []float32, extractor *AudioFeatureExtractor, opts *AudioOptions) ([]float32, int, int, error) {
	if len(samples) == 0 {
		return nil, 0, 0, fmt.Errorf("audio too short to encode")
	}
	if opts == nil {
		opts = defaultAudioOptions()
	}

	melBins := opts.melBins
	freqBins := parakeetNFFT/2 + 1
	window, melFilters := extractor.windowAndFilters(melBins, freqBins, opts.sampleRate)
	if len(window) != parakeetWinLength {
		return nil, 0, 0, fmt.Errorf("invalid Parakeet window length: %d", len(window))
	}
	if len(melFilters) != melBins*freqBins {
		return nil, 0, 0, fmt.Errorf("invalid Parakeet mel filter shape: %d", len(melFilters))
	}

	emphasized := make([]float32, len(samples))
	emphasized[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		emphasized[i] = samples[i] - parakeetPreemphasis*samples[i-1]
	}

	frames := len(samples)/parakeetHopLength + 1
	validFrames := max(1, len(samples)/parakeetHopLength)
	if validFrames > frames {
		validFrames = frames
	}

	result := make([]float32, frames*melBins)
	fftInput := make([]complex128, parakeetNFFT)
	winOffset := (parakeetNFFT - parakeetWinLength) / 2
	centerPad := parakeetNFFT / 2

	for frame := range frames {
		for i := range parakeetNFFT {
			fftInput[i] = 0
		}
		for i := range parakeetWinLength {
			src := frame*parakeetHopLength + i + winOffset - centerPad
			if src >= 0 && src < len(emphasized) {
				fftInput[i+winOffset] = complex(float64(emphasized[src])*float64(window[i]), 0)
			}
		}

		fft(fftInput)

		for mel := range melBins {
			var v float64
			filterOffset := mel * freqBins
			for freq := range freqBins {
				mag := cmplx.Abs(fftInput[freq])
				v += float64(melFilters[filterOffset+freq]) * mag * mag
			}
			result[frame*melBins+mel] = float32(math.Log(v + parakeetLogZeroGuardValue))
		}
	}

	for mel := range melBins {
		var sum float64
		for frame := range validFrames {
			sum += float64(result[frame*melBins+mel])
		}
		mean := sum / float64(validFrames)

		var variance float64
		for frame := range validFrames {
			d := float64(result[frame*melBins+mel]) - mean
			variance += d * d
		}
		denom := max(1, validFrames-1)
		std := math.Sqrt(variance / float64(denom))

		for frame := range frames {
			idx := frame*melBins + mel
			if frame >= validFrames {
				result[idx] = 0
				continue
			}
			result[idx] = float32((float64(result[idx]) - mean) / (std + parakeetNormalizeEps))
		}
	}

	return result, frames, validFrames, nil
}

func defaultParakeetWindow() []float32 {
	window := make([]float32, parakeetWinLength)
	for i := range window {
		window[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(parakeetWinLength-1)))
	}
	return window
}

func buildSlaneyMelFilterBank(numFreqBins, numMels int, sampleRate int) []float32 {
	hzToMel := func(f float64) float64 {
		if f < 1000 {
			return 3 * f / 200
		}
		return 15 + math.Log(f/1000)*27/math.Log(6.4)
	}
	melToHz := func(m float64) float64 {
		if m < 15 {
			return 200 * m / 3
		}
		return 1000 * math.Exp(math.Log(6.4)*(m-15)/27)
	}

	minMel := hzToMel(0)
	maxMel := hzToMel(float64(sampleRate) / 2)
	mels := make([]float64, numMels+2)
	freqs := make([]float64, numMels+2)
	for i := range mels {
		mels[i] = minMel + (maxMel-minMel)*float64(i)/float64(numMels+1)
		freqs[i] = melToHz(mels[i])
	}

	fftFreqs := make([]float64, numFreqBins)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * float64(sampleRate) / float64(parakeetNFFT)
	}

	filters := make([]float32, numMels*numFreqBins)
	for mel := range numMels {
		left, center, right := freqs[mel], freqs[mel+1], freqs[mel+2]
		enorm := 2.0 / (right - left)
		for freq, fftFreq := range fftFreqs {
			var lower, upper float64
			if center > left {
				lower = (fftFreq - left) / (center - left)
			}
			if right > center {
				upper = (right - fftFreq) / (right - center)
			}
			v := math.Max(0, math.Min(lower, upper))
			filters[mel*numFreqBins+freq] = float32(v * enorm)
		}
	}
	return filters
}

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
