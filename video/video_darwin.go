package video

/*
#cgo LDFLAGS: -framework AVFoundation -framework CoreMedia -framework CoreGraphics -framework CoreVideo -framework Foundation -framework ImageIO -framework UniformTypeIdentifiers
#include <stdlib.h>
#include <stdint.h>

// Extract frames from a video file using AVFoundation.
// Returns JPEG data for each frame concatenated, with offsets/sizes in the out arrays.
// Audio is extracted as 16kHz mono PCM (int16).
int extract_video_frames(
	const char* path,
	int max_frames,
	int extract_audio,
	// Frame output: caller provides buffers, function fills them
	uint8_t** frame_data,    // out: array of frame JPEG pointers (caller frees each)
	int* frame_sizes,        // out: array of frame JPEG sizes
	int* num_frames,         // out: actual number of frames extracted
	// Audio output
	uint8_t** audio_data,    // out: PCM int16 data (caller frees)
	int* audio_size          // out: PCM data size in bytes
);

void free_ptr(void* p);
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"unsafe"
)

func extract(path string, opts Options) (*Result, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	maxFrames := C.int(opts.MaxFrames)
	extractAudio := C.int(0)
	if opts.ExtractAudio {
		extractAudio = 1
	}

	// Allocate output arrays
	frameData := make([]*C.uint8_t, opts.MaxFrames)
	frameSizes := make([]C.int, opts.MaxFrames)
	var numFrames C.int

	var audioData *C.uint8_t
	var audioSize C.int

	rc := C.extract_video_frames(
		cPath,
		maxFrames,
		extractAudio,
		(**C.uint8_t)(unsafe.Pointer(&frameData[0])),
		(*C.int)(unsafe.Pointer(&frameSizes[0])),
		&numFrames,
		&audioData,
		&audioSize,
	)
	if rc != 0 {
		return nil, fmt.Errorf("video extraction failed (code %d)", rc)
	}

	result := &Result{}

	// Copy frame data to Go slices and free C memory
	for i := 0; i < int(numFrames); i++ {
		if frameData[i] != nil && frameSizes[i] > 0 {
			size := int(frameSizes[i])
			data := C.GoBytes(unsafe.Pointer(frameData[i]), C.int(size))
			result.Frames = append(result.Frames, data)
			C.free_ptr(unsafe.Pointer(frameData[i]))
		}
	}

	// Copy audio data and wrap in WAV header
	if audioData != nil && audioSize > 0 {
		pcm := C.GoBytes(unsafe.Pointer(audioData), audioSize)
		C.free_ptr(unsafe.Pointer(audioData))
		result.Audio = wrapPCMAsWAV(pcm, 16000, 1, 16)
	}

	return result, nil
}

// wrapPCMAsWAV wraps raw PCM int16 data in a WAV header.
func wrapPCMAsWAV(pcm []byte, sampleRate, channels, bitsPerSample int) []byte {
	var buf bytes.Buffer
	dataSize := len(pcm)
	fileSize := 36 + dataSize

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, int32(fileSize))
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, int32(16)) // chunk size
	binary.Write(&buf, binary.LittleEndian, int16(1))  // PCM format
	binary.Write(&buf, binary.LittleEndian, int16(channels))
	binary.Write(&buf, binary.LittleEndian, int32(sampleRate))
	byteRate := sampleRate * channels * bitsPerSample / 8
	binary.Write(&buf, binary.LittleEndian, int32(byteRate))
	blockAlign := channels * bitsPerSample / 8
	binary.Write(&buf, binary.LittleEndian, int16(blockAlign))
	binary.Write(&buf, binary.LittleEndian, int16(bitsPerSample))

	// data chunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, int32(dataSize))
	buf.Write(pcm)

	return buf.Bytes()
}
