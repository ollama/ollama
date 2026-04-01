package cmd

/*
#cgo LDFLAGS: -framework CoreAudio -framework AudioToolbox
#include <AudioToolbox/AudioQueue.h>
#include <string.h>

// Callback context passed to AudioQueue.
typedef struct {
	int ready;  // set to 1 when a buffer is filled
} AQContext;

// C callback — re-enqueues the buffer so recording continues.
// Not static — must be visible to the linker for Go's function pointer.
void aqInputCallback(
	void *inUserData,
	AudioQueueRef inAQ,
	AudioQueueBufferRef inBuffer,
	const AudioTimeStamp *inStartTime,
	UInt32 inNumberPacketDescriptions,
	const AudioStreamPacketDescription *inPacketDescs)
{
	// Re-enqueue the buffer immediately so recording continues.
	AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, NULL);
}
*/
import "C"

import (
	"fmt"
	"math"
	"sync"
	"time"
)

var errNoAudio = fmt.Errorf("no audio recorded")

const numAQBuffers = 3

type coreAudioStream struct {
	queue    C.AudioQueueRef
	buffers  [numAQBuffers]C.AudioQueueBufferRef
	mu       sync.Mutex
	callback func(samples []float32)
	running  bool
	pollDone chan struct{}

	sampleRate int
	channels   int
	frameSize  int
}

func newAudioStream(sampleRate, channels, frameSize int) (audioStream, error) {
	return &coreAudioStream{
		sampleRate: sampleRate,
		channels:   channels,
		frameSize:  frameSize,
	}, nil
}

func (s *coreAudioStream) Start(callback func(samples []float32)) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.callback = callback

	// Set up audio format: 16-bit signed integer PCM, mono, 16kHz.
	var format C.AudioStreamBasicDescription
	format.mSampleRate = C.Float64(s.sampleRate)
	format.mFormatID = C.kAudioFormatLinearPCM
	format.mFormatFlags = C.kLinearPCMFormatFlagIsSignedInteger | C.kLinearPCMFormatFlagIsPacked
	format.mBitsPerChannel = 16
	format.mChannelsPerFrame = C.UInt32(s.channels)
	format.mBytesPerFrame = 2 * C.UInt32(s.channels)
	format.mFramesPerPacket = 1
	format.mBytesPerPacket = format.mBytesPerFrame

	// Create the audio queue.
	var status C.OSStatus
	status = C.AudioQueueNewInput(
		&format,
		C.AudioQueueInputCallback(C.aqInputCallback),
		nil,               // user data
		C.CFRunLoopRef(0), // NULL run loop — use internal thread
		C.CFStringRef(0),  // NULL run loop mode
		0,                 // flags
		&s.queue,
	)
	if status != 0 {
		return fmt.Errorf("AudioQueueNewInput failed: %d", status)
	}

	// Allocate and enqueue buffers.
	bufferBytes := C.UInt32(s.frameSize * int(format.mBytesPerFrame))
	for i := range s.buffers {
		status = C.AudioQueueAllocateBuffer(s.queue, bufferBytes, &s.buffers[i])
		if status != 0 {
			C.AudioQueueDispose(s.queue, C.true)
			return fmt.Errorf("AudioQueueAllocateBuffer failed: %d", status)
		}
		status = C.AudioQueueEnqueueBuffer(s.queue, s.buffers[i], 0, nil)
		if status != 0 {
			C.AudioQueueDispose(s.queue, C.true)
			return fmt.Errorf("AudioQueueEnqueueBuffer failed: %d", status)
		}
	}

	// Start recording.
	status = C.AudioQueueStart(s.queue, nil)
	if status != 0 {
		C.AudioQueueDispose(s.queue, C.true)
		return fmt.Errorf("AudioQueueStart failed: %d", status)
	}

	s.running = true
	s.pollDone = make(chan struct{})

	// Poll buffers for data. AudioQueue re-enqueues in the C callback,
	// so we read the data out periodically.
	go s.pollLoop()

	return nil
}

func (s *coreAudioStream) pollLoop() {
	defer close(s.pollDone)

	// Read at roughly frameSize intervals.
	interval := time.Duration(float64(s.frameSize) / float64(s.sampleRate) * float64(time.Second))
	if interval < 10*time.Millisecond {
		interval = 10 * time.Millisecond
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		s.mu.Lock()
		if !s.running {
			s.mu.Unlock()
			return
		}

		// Read available data from each buffer.
		for i := range s.buffers {
			buf := s.buffers[i]
			if buf.mAudioDataByteSize > 0 {
				numSamples := int(buf.mAudioDataByteSize) / 2 // 16-bit samples
				if numSamples > 0 {
					raw := (*[1 << 28]int16)(buf.mAudioData)[:numSamples:numSamples]
					floats := make([]float32, numSamples)
					for j, v := range raw {
						floats[j] = float32(v) / float32(math.MaxInt16)
					}
					s.callback(floats)
				}
				buf.mAudioDataByteSize = 0
			}
		}
		s.mu.Unlock()
	}
}

func (s *coreAudioStream) Stop() error {
	s.mu.Lock()
	s.running = false
	queue := s.queue
	s.mu.Unlock()

	if queue != nil {
		C.AudioQueueStop(queue, C.true)
		C.AudioQueueDispose(queue, C.true)
	}

	if s.pollDone != nil {
		<-s.pollDone
	}

	return nil
}
