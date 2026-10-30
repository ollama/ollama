// Package video reads video files as sampled frames and a mono soundtrack.
//
// Models own sampling: they read Meta, choose frame indices, and pull just
// those frames in one sequential pass.
package video

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"os"
	"slices"
)

// errUnsupported reports a container or codec no backend on this system can
// decode.
var errUnsupported = errors.New("video: unsupported format")

// Meta describes a video. Width and Height are the display size, after any
// rotation in the container.
type Meta struct {
	Frames   int
	FPS      float64 // average rate, Frames / Duration
	Duration float64 // seconds
	Width    int
	Height   int
	Audio    bool
}

// Frame is one decoded frame at display size and orientation.
type Frame struct {
	Index int
	Image *image.NRGBA
}

// File is an opened video.
type File interface {
	Meta() Meta

	// Frames decodes the frames at indices, which must be ascending
	// (repeats share one decode), calling fn in order.
	Frames(ctx context.Context, indices []int, fn func(Frame) error) error

	// PCM decodes the soundtrack downmixed to mono at its native rate,
	// failing once it runs past maxSeconds. It returns nil samples when the
	// video has no audio.
	PCM(maxSeconds int) ([]float32, int, error)

	Close() error
}

// Open opens the video at path.
func Open(path string) (File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	head := make([]byte, 12)
	_, err = io.ReadFull(f, head)
	f.Close()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errUnsupported, err)
	}
	mime := mimeType(head)
	if mime == "" {
		return nil, errUnsupported
	}
	return openNative(path, mime)
}

// notVideoBrands are major brands of files that share the MP4 container
// without being video: HEIF and AVIF images and sequences, MPEG-4 audio,
// JPEG 2000 and Canon raw stills.
var notVideoBrands = []string{
	"avif", "avis", "heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs", "mif1", "mif2", "msf1",
	"M4A ", "M4B ", "M4P ", "F4A ", "F4B ",
	"jp2 ", "jpx ", "jpm ", "crx ",
}

// mimeType names the container that data starts with, or "".
func mimeType(data []byte) string {
	if len(data) < 12 || !bytes.Equal(data[4:8], []byte("ftyp")) {
		return ""
	}
	brand := string(data[8:12])
	switch {
	case slices.Contains(notVideoBrands, brand):
		return ""
	case brand == "qt  ":
		return "video/quicktime"
	}
	return "video/mp4"
}

// IsVideo reports whether data starts like a supported video container,
// MP4 or QuickTime: true for an "isom" or "qt  " file, false for HEIC,
// AVIF or M4A ones.
func IsVideo(data []byte) bool { return mimeType(data) != "" }
