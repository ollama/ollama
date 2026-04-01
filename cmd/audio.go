package cmd

import (
	"encoding/binary"
	"sync"
	"time"
)

const (
	audioSampleRate = 16000
	audioChannels   = 1
	audioFrameSize  = 1024 // samples per callback
)

// AudioRecorder captures audio from the default microphone.
// Platform-specific capture is provided by audioStream (audio_darwin.go, etc.).
type AudioRecorder struct {
	stream          audioStream
	mu              sync.Mutex
	samples         []float32
	started         time.Time
	MaxChunkSeconds int // hard split limit in seconds; 0 means use default
}

// audioStream is the platform-specific audio capture interface.
type audioStream interface {
	// Start begins capturing. Samples are delivered via the callback.
	Start(callback func(samples []float32)) error
	// Stop ends capturing and releases resources.
	Stop() error
}

// NewAudioRecorder creates a recorder ready to capture from the default mic.
func NewAudioRecorder() (*AudioRecorder, error) {
	stream, err := newAudioStream(audioSampleRate, audioChannels, audioFrameSize)
	if err != nil {
		return nil, err
	}
	return &AudioRecorder{stream: stream}, nil
}

// Start begins capturing audio from the microphone.
func (r *AudioRecorder) Start() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.samples = make([]float32, 0, audioSampleRate*30) // preallocate ~30s
	r.started = time.Now()

	return r.stream.Start(func(samples []float32) {
		r.mu.Lock()
		r.samples = append(r.samples, samples...)
		r.mu.Unlock()
	})
}

// Stop ends the recording and returns the duration.
func (r *AudioRecorder) Stop() (time.Duration, error) {
	r.mu.Lock()
	dur := time.Since(r.started)
	r.mu.Unlock()

	if r.stream != nil {
		r.stream.Stop()
	}

	return dur, nil
}

// Duration returns how long the current recording has been running.
func (r *AudioRecorder) Duration() time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.started.IsZero() {
		return 0
	}
	return time.Since(r.started)
}

// Chunking constants for live transcription.
const (
	chunkTargetSamples     = 8 * audioSampleRate // 8s — start yielding when silence found
	chunkMinSamples        = 5 * audioSampleRate // start scanning for silence at 5s
	defaultMaxAudioSeconds = 28                  // default hard split (just under typical 30s model cap)
	silenceWindow          = 800                 // 50ms RMS window
)

func (r *AudioRecorder) maxChunk() int {
	if r.MaxChunkSeconds > 0 {
		return r.MaxChunkSeconds * audioSampleRate
	}
	return defaultMaxAudioSeconds * audioSampleRate
}

// TakeChunk checks if there are enough accumulated samples to yield a chunk.
// If so, it splits at the best silence boundary, removes the consumed samples
// from the buffer, and returns the chunk as WAV bytes. Returns nil if not enough
// audio has accumulated yet.
func (r *AudioRecorder) TakeChunk() []byte {
	r.mu.Lock()
	n := len(r.samples)
	if n < chunkMinSamples {
		r.mu.Unlock()
		return nil
	}

	maxSamples := r.maxChunk()

	if n < chunkTargetSamples && n < maxSamples {
		r.mu.Unlock()
		return nil
	}

	limit := n
	if limit > maxSamples {
		limit = maxSamples
	}

	splitAt := limit
	bestEnergy := float64(1e30)

	scanStart := limit - silenceWindow
	scanEnd := chunkMinSamples
	for pos := scanStart; pos >= scanEnd; pos -= silenceWindow / 2 {
		end := pos + silenceWindow
		if end > n {
			end = n
		}
		var sumSq float64
		for _, s := range r.samples[pos:end] {
			sumSq += float64(s) * float64(s)
		}
		rms := sumSq / float64(end-pos)
		if rms < bestEnergy {
			bestEnergy = rms
			splitAt = pos + silenceWindow/2
		}
	}

	chunk := make([]float32, splitAt)
	copy(chunk, r.samples[:splitAt])
	remaining := make([]float32, n-splitAt)
	copy(remaining, r.samples[splitAt:])
	r.samples = remaining
	r.mu.Unlock()

	return encodeWAV(chunk, audioSampleRate, audioChannels)
}

// FlushWAV returns any remaining samples as WAV, clearing the buffer.
func (r *AudioRecorder) FlushWAV() []byte {
	r.mu.Lock()
	samples := r.samples
	r.samples = nil
	r.mu.Unlock()

	if len(samples) == 0 {
		return nil
	}
	return encodeWAV(samples, audioSampleRate, audioChannels)
}

// WAV encodes the captured samples as a WAV file in memory.
func (r *AudioRecorder) WAV() ([]byte, error) {
	r.mu.Lock()
	samples := make([]float32, len(r.samples))
	copy(samples, r.samples)
	r.mu.Unlock()

	if len(samples) == 0 {
		return nil, errNoAudio
	}

	return encodeWAV(samples, audioSampleRate, audioChannels), nil
}

// encodeWAV produces a 16-bit PCM WAV file from float32 samples.
func encodeWAV(samples []float32, sampleRate, channels int) []byte {
	numSamples := len(samples)
	bitsPerSample := 16
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	dataSize := numSamples * blockAlign

	buf := make([]byte, 44+dataSize)

	copy(buf[0:4], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:8], uint32(36+dataSize))
	copy(buf[8:12], "WAVE")

	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16)
	binary.LittleEndian.PutUint16(buf[20:22], 1)
	binary.LittleEndian.PutUint16(buf[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(buf[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(buf[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(buf[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(buf[34:36], uint16(bitsPerSample))

	copy(buf[36:40], "data")
	binary.LittleEndian.PutUint32(buf[40:44], uint32(dataSize))

	offset := 44
	for _, s := range samples {
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		val := int16(s * 32767)
		binary.LittleEndian.PutUint16(buf[offset:offset+2], uint16(val))
		offset += 2
	}

	return buf
}
