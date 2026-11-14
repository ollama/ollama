package video

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework Foundation
#include <stdlib.h>
#include "avfoundation_darwin.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

// maxSampleRate is the highest common studio rate, as for audio input.
const maxSampleRate = 384000

// avFile decodes with AVFoundation, which uses the system's hardware
// decoders and codec licenses.
type avFile struct {
	mu    sync.Mutex // guards asset against Close during a decode
	asset unsafe.Pointer

	meta     Meta
	rotation int
	rate     int
	channels int
}

func cError(err *C.char) error {
	if err == nil {
		return errors.New("video: decoding failed")
	}
	defer C.free(unsafe.Pointer(err))
	return fmt.Errorf("video: %s", C.GoString(err))
}

func openNative(path, mime string) (File, error) {
	cpath, cmime := C.CString(path), C.CString(mime)
	defer C.free(unsafe.Pointer(cpath))
	defer C.free(unsafe.Pointer(cmime))
	var m C.ov_meta
	var cerr *C.char
	asset := C.ov_open(cpath, cmime, &m, &cerr)
	if asset == nil {
		return nil, fmt.Errorf("%w: %w", errUnsupported, cError(cerr))
	}
	f := &avFile{asset: asset, rotation: int(m.rotation), rate: int(m.sample_rate), channels: int(m.channels)}
	f.meta = Meta{
		Frames:   int(m.frames),
		Duration: float64(m.duration),
		Width:    int(m.width),
		Height:   int(m.height),
		Audio:    m.audio != 0,
	}
	if f.rotation == 90 || f.rotation == 270 {
		f.meta.Width, f.meta.Height = f.meta.Height, f.meta.Width
	}
	if f.meta.Frames <= 0 || !(f.meta.Duration > 0) || math.IsInf(f.meta.Duration, 0) || f.meta.Width <= 0 || f.meta.Height <= 0 {
		f.Close()
		return nil, fmt.Errorf("%w: no decodable frames", errUnsupported)
	}
	f.meta.FPS = float64(f.meta.Frames) / f.meta.Duration
	runtime.SetFinalizer(f, (*avFile).Close)
	return f, nil
}

func (f *avFile) Meta() Meta { return f.meta }

func (f *avFile) Frames(ctx context.Context, indices []int, fn func(Frame) error) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.asset == nil {
		return errors.New("video: file closed")
	}
	for i, idx := range indices {
		if idx < 0 || (i > 0 && idx < indices[i-1]) {
			return fmt.Errorf("video: frame index %d out of order or range", idx)
		}
	}
	var cerr *C.char
	reader := C.ov_frames_begin(f.asset, &cerr)
	if reader == nil {
		return cError(cerr)
	}
	defer C.ov_frames_end(reader)

	next := 0
	for index := 0; next < len(indices); index++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		want := indices[next] == index
		var pix *C.uint8_t
		var w, h, stride C.int
		switch C.ov_frames_next(reader, boolInt(want), &pix, &w, &h, &stride, &cerr) {
		case 0:
			return fmt.Errorf("video: frame %d past the last frame (%d)", indices[next], index)
		case -1:
			return cError(cerr)
		}
		if !want {
			continue
		}
		img := bgraToNRGBA(unsafe.Slice((*byte)(pix), int(stride)*int(h)), int(w), int(h), int(stride), f.rotation)
		C.free(unsafe.Pointer(pix))
		if b := img.Bounds(); b.Dx() != f.meta.Width || b.Dy() != f.meta.Height {
			return fmt.Errorf("video: frame %d is %dx%d, the stream says %dx%d", index, b.Dx(), b.Dy(), f.meta.Width, f.meta.Height)
		}
		// Repeated indices share the decoded frame.
		for next < len(indices) && indices[next] == index {
			if err := fn(Frame{Index: index, Image: img}); err != nil {
				return err
			}
			next++
		}
	}
	return nil
}

func (f *avFile) PCM(maxSeconds int) ([]float32, int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.asset == nil {
		return nil, 0, errors.New("video: file closed")
	}
	if !f.meta.Audio {
		return nil, 0, nil
	}
	if f.rate > maxSampleRate {
		return nil, 0, fmt.Errorf("video: unsupported soundtrack sample rate %d Hz, the maximum is %d Hz", f.rate, maxSampleRate)
	}
	var samples *C.float
	var count C.int64_t
	var cerr *C.char
	switch C.ov_pcm(f.asset, C.int(f.channels), C.int64_t(int64(maxSeconds)*int64(f.rate)), &samples, &count, &cerr) {
	case 0:
	case -2:
		return nil, 0, fmt.Errorf("video: soundtrack longer than %d seconds", maxSeconds)
	default:
		return nil, 0, cError(cerr)
	}
	defer C.free(unsafe.Pointer(samples))
	mono := make([]float32, int(count))
	copy(mono, unsafe.Slice((*float32)(unsafe.Pointer(samples)), int(count)))
	return mono, f.rate, nil
}

func (f *avFile) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.asset != nil {
		C.ov_close(f.asset)
		f.asset = nil
	}
	return nil
}

func boolInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// bgraToNRGBA converts BGRA rows to NRGBA, rotating clockwise by degrees
// (a multiple of 90) to the display orientation.
func bgraToNRGBA(src []byte, w, h, stride, degrees int) *image.NRGBA {
	dw, dh := w, h
	if degrees == 90 || degrees == 270 {
		dw, dh = h, w
	}
	dst := image.NewNRGBA(image.Rect(0, 0, dw, dh))
	for y := range h {
		row := src[y*stride:]
		for x := range w {
			var dx, dy int
			switch degrees {
			case 90:
				dx, dy = h-1-y, x
			case 180:
				dx, dy = w-1-x, h-1-y
			case 270:
				dx, dy = y, w-1-x
			default:
				dx, dy = x, y
			}
			s, d := row[4*x:], dst.Pix[dy*dst.Stride+4*dx:]
			d[0], d[1], d[2], d[3] = s[2], s[1], s[0], 255
		}
	}
	return dst
}
