package cmd

import (
	"fmt"
	"math"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

var errNoAudio = fmt.Errorf("no audio recorded")

// WASAPI COM GUIDs
var (
	iidIMMDeviceEnumerator = guid{0xA95664D2, 0x9614, 0x4F35, [8]byte{0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}}
	clsidMMDeviceEnumerator = guid{0xBCDE0395, 0xE52F, 0x467C, [8]byte{0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}}
	iidIAudioClient        = guid{0x1CB9AD4C, 0xDBFA, 0x4C32, [8]byte{0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}}
	iidIAudioCaptureClient = guid{0xC8ADBD64, 0xE71E, 0x48A0, [8]byte{0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17}}
)

type guid struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

// WAVEFORMATEX structure
type waveFormatEx struct {
	FormatTag      uint16
	Channels       uint16
	SamplesPerSec  uint32
	AvgBytesPerSec uint32
	BlockAlign     uint16
	BitsPerSample  uint16
	CbSize         uint16
}

const (
	wavePCM         = 1
	eCapture        = 1
	eConsole        = 0
	audclntSharemode = 0 // AUDCLNT_SHAREMODE_SHARED
	audclntStreamflagsEventcallback = 0x00040000

	coinitMultithreaded = 0x0
	clsctxAll           = 0x17

	reftimesPerSec    = 10000000 // 100ns units per second
	reftimesPerMillis = 10000
)

var (
	ole32    = syscall.NewLazyDLL("ole32.dll")
	coInit   = ole32.NewProc("CoInitializeEx")
	coCreate = ole32.NewProc("CoCreateInstance")
)

type wasapiStream struct {
	mu       sync.Mutex
	callback func(samples []float32)
	running  bool
	done     chan struct{}

	sampleRate int
	channels   int
	frameSize  int

	// COM interfaces (stored as uintptr for syscall)
	enumerator uintptr
	device     uintptr
	client     uintptr
	capture    uintptr
}

func newAudioStream(sampleRate, channels, frameSize int) (audioStream, error) {
	return &wasapiStream{
		sampleRate: sampleRate,
		channels:   channels,
		frameSize:  frameSize,
	}, nil
}

func (s *wasapiStream) Start(callback func(samples []float32)) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.callback = callback

	// Initialize COM
	hr, _, _ := coInit.Call(0, uintptr(coinitMultithreaded))
	// S_OK or S_FALSE (already initialized) are both fine
	if hr != 0 && hr != 1 {
		return fmt.Errorf("CoInitializeEx failed: 0x%08x", hr)
	}

	// Create device enumerator
	hr, _, _ = coCreate.Call(
		uintptr(unsafe.Pointer(&clsidMMDeviceEnumerator)),
		0,
		uintptr(clsctxAll),
		uintptr(unsafe.Pointer(&iidIMMDeviceEnumerator)),
		uintptr(unsafe.Pointer(&s.enumerator)),
	)
	if hr != 0 {
		return fmt.Errorf("CoCreateInstance(MMDeviceEnumerator) failed: 0x%08x", hr)
	}

	// Get default capture device
	// IMMDeviceEnumerator::GetDefaultAudioEndpoint is vtable index 4
	hr = comCall(s.enumerator, 4, uintptr(eCapture), uintptr(eConsole), uintptr(unsafe.Pointer(&s.device)))
	if hr != 0 {
		return fmt.Errorf("GetDefaultAudioEndpoint failed: 0x%08x", hr)
	}

	// Activate IAudioClient
	// IMMDevice::Activate is vtable index 3
	hr = comCall(s.device, 3,
		uintptr(unsafe.Pointer(&iidIAudioClient)),
		uintptr(clsctxAll),
		0,
		uintptr(unsafe.Pointer(&s.client)),
	)
	if hr != 0 {
		return fmt.Errorf("IMMDevice::Activate failed: 0x%08x", hr)
	}

	// Set up format: 16-bit PCM mono 16kHz
	format := waveFormatEx{
		FormatTag:      wavePCM,
		Channels:       uint16(s.channels),
		SamplesPerSec:  uint32(s.sampleRate),
		BitsPerSample:  16,
		BlockAlign:     uint16(2 * s.channels),
		AvgBytesPerSec: uint32(s.sampleRate * 2 * s.channels),
		CbSize:         0,
	}

	// Initialize audio client
	// IAudioClient::Initialize is vtable index 3
	bufferDuration := int64(reftimesPerSec) // 1 second buffer
	hr = comCall(s.client, 3,
		uintptr(audclntSharemode),
		0, // stream flags
		uintptr(bufferDuration),
		0, // periodicity (0 = use default)
		uintptr(unsafe.Pointer(&format)),
		0, // audio session GUID (NULL = default)
	)
	if hr != 0 {
		return fmt.Errorf("IAudioClient::Initialize failed: 0x%08x", hr)
	}

	// Get capture client
	// IAudioClient::GetService is vtable index 8
	hr = comCall(s.client, 8,
		uintptr(unsafe.Pointer(&iidIAudioCaptureClient)),
		uintptr(unsafe.Pointer(&s.capture)),
	)
	if hr != 0 {
		return fmt.Errorf("IAudioClient::GetService failed: 0x%08x", hr)
	}

	// Start capture
	// IAudioClient::Start is vtable index 6
	hr = comCall(s.client, 6)
	if hr != 0 {
		return fmt.Errorf("IAudioClient::Start failed: 0x%08x", hr)
	}

	s.running = true
	s.done = make(chan struct{})
	go s.captureLoop()

	return nil
}

func (s *wasapiStream) captureLoop() {
	defer close(s.done)

	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		s.mu.Lock()
		if !s.running {
			s.mu.Unlock()
			return
		}

		// Read available packets
		for {
			var data uintptr
			var numFrames uint32
			var flags uint32

			// IAudioCaptureClient::GetBuffer is vtable index 3
			hr := comCall(s.capture, 3,
				uintptr(unsafe.Pointer(&data)),
				uintptr(unsafe.Pointer(&numFrames)),
				uintptr(unsafe.Pointer(&flags)),
				0, // device position (not needed)
				0, // QPC position (not needed)
			)
			if hr != 0 || numFrames == 0 {
				break
			}

			// Convert int16 samples to float32
			samples := make([]float32, numFrames*uint32(s.channels))
			raw := (*[1 << 28]int16)(unsafe.Pointer(data))[:len(samples):len(samples)]
			for i, v := range raw {
				samples[i] = float32(v) / float32(math.MaxInt16)
			}

			s.callback(samples)

			// IAudioCaptureClient::ReleaseBuffer is vtable index 4
			comCall(s.capture, 4, uintptr(numFrames))
		}

		s.mu.Unlock()
	}
}

func (s *wasapiStream) Stop() error {
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()

	if s.done != nil {
		<-s.done
	}

	// IAudioClient::Stop is vtable index 7
	if s.client != 0 {
		comCall(s.client, 7)
	}

	// Release COM interfaces (IUnknown::Release is vtable index 2)
	if s.capture != 0 {
		comCall(s.capture, 2)
	}
	if s.client != 0 {
		comCall(s.client, 2)
	}
	if s.device != 0 {
		comCall(s.device, 2)
	}
	if s.enumerator != 0 {
		comCall(s.enumerator, 2)
	}

	return nil
}

// comCall invokes a COM method by vtable index.
func comCall(obj uintptr, method uintptr, args ...uintptr) uintptr {
	vtable := *(*uintptr)(unsafe.Pointer(obj))
	fn := *(*uintptr)(unsafe.Pointer(vtable + method*unsafe.Sizeof(uintptr(0))))

	// Build syscall args: first arg is always 'this' pointer
	callArgs := make([]uintptr, 1+len(args))
	callArgs[0] = obj
	copy(callArgs[1:], args)

	var hr uintptr
	switch len(callArgs) {
	case 1:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0])
	case 2:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0], callArgs[1])
	case 3:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0], callArgs[1], callArgs[2])
	case 4:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0], callArgs[1], callArgs[2], callArgs[3])
	case 5:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0], callArgs[1], callArgs[2], callArgs[3], callArgs[4])
	case 6:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0], callArgs[1], callArgs[2], callArgs[3], callArgs[4], callArgs[5])
	case 7:
		hr, _, _ = syscall.SyscallN(fn, callArgs[0], callArgs[1], callArgs[2], callArgs[3], callArgs[4], callArgs[5], callArgs[6])
	default:
		hr, _, _ = syscall.SyscallN(fn, callArgs...)
	}
	return hr
}
