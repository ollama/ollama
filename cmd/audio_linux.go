package cmd

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

// Function pointer types for ALSA functions loaded at runtime.
typedef int (*pcm_open_fn)(void**, const char*, int, int);
typedef int (*pcm_simple_fn)(void*);
typedef long (*pcm_readi_fn)(void*, void*, unsigned long);
typedef int (*hw_malloc_fn)(void**);
typedef void (*hw_free_fn)(void*);
typedef int (*hw_any_fn)(void*, void*);
typedef int (*hw_set_int_fn)(void*, void*, int);
typedef int (*hw_set_uint_fn)(void*, void*, unsigned int);
typedef int (*hw_set_rate_fn)(void*, void*, unsigned int*, int*);
typedef int (*hw_set_period_fn)(void*, void*, unsigned long*, int*);
typedef int (*hw_apply_fn)(void*, void*);
typedef const char* (*strerror_fn)(int);

// Trampoline functions — call dynamically loaded ALSA symbols.
static int alsa_pcm_open(void* fn, void** h, const char* name, int stream, int mode) {
	return ((pcm_open_fn)fn)(h, name, stream, mode);
}
static int alsa_pcm_close(void* fn, void* h) { return ((pcm_simple_fn)fn)(h); }
static int alsa_pcm_prepare(void* fn, void* h) { return ((pcm_simple_fn)fn)(h); }
static int alsa_pcm_drop(void* fn, void* h) { return ((pcm_simple_fn)fn)(h); }
static long alsa_pcm_readi(void* fn, void* h, void* buf, unsigned long frames) {
	return ((pcm_readi_fn)fn)(h, buf, frames);
}
static int alsa_hw_malloc(void* fn, void** p) { return ((hw_malloc_fn)fn)(p); }
static void alsa_hw_free(void* fn, void* p) { ((hw_free_fn)fn)(p); }
static int alsa_hw_any(void* fn, void* h, void* p) { return ((hw_any_fn)fn)(h, p); }
static int alsa_hw_set_access(void* fn, void* h, void* p, int v) { return ((hw_set_int_fn)fn)(h, p, v); }
static int alsa_hw_set_format(void* fn, void* h, void* p, int v) { return ((hw_set_int_fn)fn)(h, p, v); }
static int alsa_hw_set_channels(void* fn, void* h, void* p, unsigned int v) { return ((hw_set_uint_fn)fn)(h, p, v); }
static int alsa_hw_set_rate(void* fn, void* h, void* p, unsigned int* v, int* d) { return ((hw_set_rate_fn)fn)(h, p, v, d); }
static int alsa_hw_set_period(void* fn, void* h, void* p, unsigned long* v, int* d) { return ((hw_set_period_fn)fn)(h, p, v, d); }
static int alsa_hw_apply(void* fn, void* h, void* p) { return ((hw_apply_fn)fn)(h, p); }
static const char* alsa_strerror(void* fn, int e) { return ((strerror_fn)fn)(e); }
*/
import "C"

import (
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"
)

var errNoAudio = fmt.Errorf("no audio recorded")

const (
	sndPCMStreamCapture       = 1
	sndPCMAccessRWInterleaved = 3
	sndPCMFormatS16LE         = 2
)

var (
	alsaLoadErr error
	alsaOnce    sync.Once
	alsa        alsaFuncs
)

type alsaFuncs struct {
	pcmOpen, pcmClose, pcmPrepare, pcmDrop, pcmReadi       unsafe.Pointer
	hwMalloc, hwFree, hwAny                                 unsafe.Pointer
	hwSetAccess, hwSetFormat, hwSetChannels                  unsafe.Pointer
	hwSetRate, hwSetPeriod, hwApply                          unsafe.Pointer
	strerror                                                 unsafe.Pointer
}

func loadALSA() {
	var lib unsafe.Pointer
	for _, name := range []string{"libasound.so.2", "libasound.so"} {
		cName := C.CString(name)
		lib = C.dlopen(cName, C.RTLD_NOW)
		C.free(unsafe.Pointer(cName))
		if lib != nil {
			break
		}
	}
	if lib == nil {
		alsaLoadErr = fmt.Errorf("audio capture unavailable: libasound.so not found")
		return
	}

	sym := func(name string) unsafe.Pointer {
		cName := C.CString(name)
		defer C.free(unsafe.Pointer(cName))
		return C.dlsym(lib, cName)
	}

	syms := []struct {
		ptr  *unsafe.Pointer
		name string
	}{
		{&alsa.pcmOpen, "snd_pcm_open"},
		{&alsa.pcmClose, "snd_pcm_close"},
		{&alsa.pcmPrepare, "snd_pcm_prepare"},
		{&alsa.pcmDrop, "snd_pcm_drop"},
		{&alsa.pcmReadi, "snd_pcm_readi"},
		{&alsa.hwMalloc, "snd_pcm_hw_params_malloc"},
		{&alsa.hwFree, "snd_pcm_hw_params_free"},
		{&alsa.hwAny, "snd_pcm_hw_params_any"},
		{&alsa.hwSetAccess, "snd_pcm_hw_params_set_access"},
		{&alsa.hwSetFormat, "snd_pcm_hw_params_set_format"},
		{&alsa.hwSetChannels, "snd_pcm_hw_params_set_channels"},
		{&alsa.hwSetRate, "snd_pcm_hw_params_set_rate_near"},
		{&alsa.hwSetPeriod, "snd_pcm_hw_params_set_period_size_near"},
		{&alsa.hwApply, "snd_pcm_hw_params"},
		{&alsa.strerror, "snd_strerror"},
	}

	for _, s := range syms {
		*s.ptr = sym(s.name)
		if *s.ptr == nil {
			alsaLoadErr = fmt.Errorf("audio capture unavailable: missing %s in libasound", s.name)
			return
		}
	}
}

func alsaError(code C.int) string {
	if alsa.strerror == nil {
		return fmt.Sprintf("error %d", code)
	}
	return C.GoString(C.alsa_strerror(alsa.strerror, code))
}

type alsaStream struct {
	handle   unsafe.Pointer
	mu       sync.Mutex
	callback func(samples []float32)
	running  bool
	done     chan struct{}

	sampleRate int
	channels   int
	frameSize  int
}

func newAudioStream(sampleRate, channels, frameSize int) (audioStream, error) {
	alsaOnce.Do(loadALSA)
	if alsaLoadErr != nil {
		return nil, alsaLoadErr
	}
	return &alsaStream{
		sampleRate: sampleRate,
		channels:   channels,
		frameSize:  frameSize,
	}, nil
}

func (s *alsaStream) Start(callback func(samples []float32)) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.callback = callback

	cName := C.CString("default")
	defer C.free(unsafe.Pointer(cName))

	rc := C.alsa_pcm_open(alsa.pcmOpen, (*unsafe.Pointer)(unsafe.Pointer(&s.handle)), cName, C.int(sndPCMStreamCapture), 0)
	if rc < 0 {
		return fmt.Errorf("snd_pcm_open: %s", alsaError(rc))
	}

	var hwParams unsafe.Pointer
	C.alsa_hw_malloc(alsa.hwMalloc, (*unsafe.Pointer)(unsafe.Pointer(&hwParams)))
	defer C.alsa_hw_free(alsa.hwFree, hwParams)

	C.alsa_hw_any(alsa.hwAny, s.handle, hwParams)

	if rc = C.alsa_hw_set_access(alsa.hwSetAccess, s.handle, hwParams, C.int(sndPCMAccessRWInterleaved)); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("set access: %s", alsaError(rc))
	}
	if rc = C.alsa_hw_set_format(alsa.hwSetFormat, s.handle, hwParams, C.int(sndPCMFormatS16LE)); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("set format: %s", alsaError(rc))
	}
	if rc = C.alsa_hw_set_channels(alsa.hwSetChannels, s.handle, hwParams, C.uint(s.channels)); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("set channels: %s", alsaError(rc))
	}

	rate := C.uint(s.sampleRate)
	if rc = C.alsa_hw_set_rate(alsa.hwSetRate, s.handle, hwParams, &rate, nil); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("set rate: %s", alsaError(rc))
	}

	periodSize := C.ulong(s.frameSize)
	if rc = C.alsa_hw_set_period(alsa.hwSetPeriod, s.handle, hwParams, &periodSize, nil); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("set period: %s", alsaError(rc))
	}

	if rc = C.alsa_hw_apply(alsa.hwApply, s.handle, hwParams); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("apply hw params: %s", alsaError(rc))
	}

	if rc = C.alsa_pcm_prepare(alsa.pcmPrepare, s.handle); rc < 0 {
		C.alsa_pcm_close(alsa.pcmClose, s.handle)
		return fmt.Errorf("prepare: %s", alsaError(rc))
	}

	s.running = true
	s.done = make(chan struct{})
	go s.captureLoop(int(periodSize))

	return nil
}

func (s *alsaStream) captureLoop(periodSize int) {
	defer close(s.done)

	buf := make([]int16, periodSize*s.channels)

	for {
		s.mu.Lock()
		if !s.running {
			s.mu.Unlock()
			return
		}
		handle := s.handle
		s.mu.Unlock()

		frames := C.alsa_pcm_readi(alsa.pcmReadi, handle, unsafe.Pointer(&buf[0]), C.ulong(periodSize))
		if frames < 0 {
			C.alsa_pcm_prepare(alsa.pcmPrepare, handle)
			continue
		}
		if frames == 0 {
			time.Sleep(5 * time.Millisecond)
			continue
		}

		numSamples := int(frames) * s.channels
		floats := make([]float32, numSamples)
		for i := 0; i < numSamples; i++ {
			floats[i] = float32(buf[i]) / float32(math.MaxInt16)
		}

		s.mu.Lock()
		if s.callback != nil {
			s.callback(floats)
		}
		s.mu.Unlock()
	}
}

func (s *alsaStream) Stop() error {
	s.mu.Lock()
	s.running = false
	handle := s.handle
	s.handle = nil
	s.mu.Unlock()

	if s.done != nil {
		<-s.done
	}

	if handle != nil {
		C.alsa_pcm_drop(alsa.pcmDrop, handle)
		C.alsa_pcm_close(alsa.pcmClose, handle)
	}

	return nil
}
