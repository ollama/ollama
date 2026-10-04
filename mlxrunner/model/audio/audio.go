package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
)

// maxAudioSeconds caps decoded clip duration. The declared rate is
// untrusted; this is what bounds Resample's output length.
const maxAudioSeconds = 600

// Decode decodes a supported audio container into mono PCM samples at
// the container's declared sample rate, downmixing multiple channels by
// averaging. The supported containers are decided here so all models accept
// the same formats; WAV is the only one today. Clips longer than
// maxAudioSeconds are rejected. Models resample to their own rate with
// Resample.
func Decode(data []byte) ([]float32, int, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errors.New("unrecognized audio format")
	}
	samples, rate, err := decodeWAV(data)
	if err != nil {
		return nil, 0, err
	}
	if len(samples) > maxAudioSeconds*rate {
		return nil, 0, fmt.Errorf("audio longer than %d seconds", maxAudioSeconds)
	}
	return samples, rate, nil
}

func decodeWAV(data []byte) ([]float32, int, error) {
	var format uint16
	var channels, rate, bits int
	var pcm []byte
	foundFmt := false

	offset := 12
	for offset+8 <= len(data) {
		id := string(data[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		body := data[offset+8 : min(offset+8+size, len(data))]

		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, errors.New("wav: fmt chunk too short")
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			rate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
			// WAVE_FORMAT_EXTENSIBLE carries the real format code in the
			// extension's subformat GUID.
			if format == 0xfffe && len(body) >= 26 {
				format = binary.LittleEndian.Uint16(body[24:26])
			}
			foundFmt = true
		case "data":
			pcm = body
		}

		// Chunks are word-aligned.
		offset += 8 + size + size%2
	}

	if !foundFmt {
		return nil, 0, errors.New("wav: no fmt chunk")
	}
	if pcm == nil {
		return nil, 0, errors.New("wav: no data chunk")
	}
	if channels < 1 || rate < 1 {
		return nil, 0, fmt.Errorf("wav: invalid fmt: %d channels at %d Hz", channels, rate)
	}

	var sample func([]byte) float64
	switch {
	case format == 1 && bits == 8:
		sample = func(b []byte) float64 { return (float64(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		sample = func(b []byte) float64 {
			return float64(int16(binary.LittleEndian.Uint16(b))) / 32768
		}
	case format == 1 && bits == 24:
		sample = func(b []byte) float64 {
			v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
			return float64(v<<8>>8) / 8388608
		}
	case format == 1 && bits == 32:
		sample = func(b []byte) float64 {
			return float64(int32(binary.LittleEndian.Uint32(b))) / 2147483648
		}
	case format == 3 && bits == 32:
		sample = func(b []byte) float64 {
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
		}
	default:
		return nil, 0, fmt.Errorf("wav: unsupported format %d with %d bits", format, bits)
	}

	stride := bits / 8
	n := len(pcm) / (stride * channels)
	mono := make([]float32, n)
	for i := range n {
		var sum float64
		for c := range channels {
			sum += sample(pcm[(i*channels+c)*stride:])
		}
		mono[i] = float32(sum / float64(channels))
	}
	return mono, rate, nil
}

const (
	resampleLobes   = 16  // sinc zero crossings per side of the kernel
	resampleRolloff = 0.9 // cutoff as a fraction of the lower Nyquist
	resampleBeta    = 9.0 // Kaiser shape, roughly 80 dB of stopband
)

// resampleKernel tabulates the Kaiser-windowed sinc over half its support;
// weights interpolate linearly between entries.
var resampleKernel = sync.OnceValue(func() []float64 {
	const size = 8192
	i0 := besselI0(resampleBeta)
	kernel := make([]float64, size+1)
	for i := range kernel {
		u := float64(i) / size
		kernel[i] = sinc(resampleLobes*u) * besselI0(resampleBeta*math.Sqrt(1-u*u)) / i0
	}
	return kernel
})

// Resample converts samples between rates with a Kaiser-windowed sinc
// kernel evaluated at each output sample's fractional source position. The
// cutoff sits below the lower of the two Nyquist frequencies, so the same
// kernel low-passes when downsampling and rejects images when upsampling.
// Weights are normalized per output sample, which keeps unity gain where
// the clip edges truncate the kernel.
func Resample(samples []float32, from, to int) []float32 {
	if from == to || len(samples) < 2 {
		return samples
	}

	ratio := float64(to) / float64(from)
	scale := resampleRolloff * min(1, ratio)
	halfWidth := resampleLobes / scale
	kernel := resampleKernel()

	n := int((int64(len(samples))*int64(to) + int64(from) - 1) / int64(from))
	out := make([]float32, n)
	for i := range out {
		center := float64(i) / ratio
		lo := max(int(math.Ceil(center-halfWidth)), 0)
		hi := min(int(math.Floor(center+halfWidth)), len(samples)-1)

		var acc, norm float64
		for j := lo; j <= hi; j++ {
			pos := math.Abs(float64(j)-center) / halfWidth * float64(len(kernel)-1)
			idx := int(pos)
			if idx >= len(kernel)-1 {
				continue
			}
			w := kernel[idx] + (kernel[idx+1]-kernel[idx])*(pos-float64(idx))
			acc += w * float64(samples[j])
			norm += w
		}
		if norm != 0 {
			out[i] = float32(acc / norm)
		}
	}
	return out
}

// Cuts go at the lowest-energy window of splitWindowMillis found within
// splitSearchSeconds around each chunk's even share of the clip.
const (
	splitSearchSeconds = 4
	splitWindowMillis  = 100
)

// Split divides samples into the fewest chunks of at most maxSeconds each,
// sized evenly, with each cut moved to the quietest point near its even
// share so a boundary lands on a pause where the clip allows. Chunks are
// subslices of samples.
func Split(samples []float32, rate, maxSeconds int) [][]float32 {
	maxChunk := rate * maxSeconds
	span := rate * splitSearchSeconds
	window := max(rate*splitWindowMillis/1000, 1)
	var chunks [][]float32
	for k := (len(samples) + maxChunk - 1) / maxChunk; k > 1; k-- {
		target := len(samples) / k
		// The rest of the clip must still fit in k-1 chunks.
		lo := max(target-span/2, len(samples)-(k-1)*maxChunk)
		hi := min(target+span/2, maxChunk)
		cut := lo + quietestPoint(samples[lo:hi], window)
		chunks = append(chunks, samples[:cut])
		samples = samples[cut:]
	}
	return append(chunks, samples)
}

// quietestPoint returns the center of the lowest-energy window in x,
// the latest one on ties.
func quietestPoint(x []float32, window int) int {
	if len(x) <= window {
		return len(x) / 2
	}
	var energy float64
	for _, v := range x[:window] {
		energy += float64(v) * float64(v)
	}
	best, bestEnergy := 0, energy
	for i := window; i < len(x); i++ {
		energy += float64(x[i])*float64(x[i]) - float64(x[i-window])*float64(x[i-window])
		if energy <= bestEnergy {
			best, bestEnergy = i-window+1, energy
		}
	}
	return best + window/2
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// besselI0 is the zeroth-order modified Bessel function of the first kind,
// by power series.
func besselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 64; k++ {
		term *= x * x / (4 * float64(k) * float64(k))
		sum += term
		if term < 1e-16*sum {
			break
		}
	}
	return sum
}
