// Package video extracts frames and audio from video files.
//
// Video files are decomposed into JPEG image frames and a WAV audio track.
// The frames and audio can then be fed into a multimodal model's vision
// and audio encoders respectively.
//
// Platform-specific implementations:
//   - macOS: AVFoundation (system framework, zero external deps)
//   - Windows: Media Foundation (pure Go via syscall, zero external deps)
//   - Linux: shells out to ffmpeg (must be installed)
package video

import (
	"fmt"
	"net/http"
	"strings"
)

// Result holds extracted frames and optional audio from a video file.
type Result struct {
	Frames [][]byte // JPEG-encoded image frames in temporal order
	Audio  []byte   // WAV 16kHz mono audio (nil if no audio track)
}

// Options controls frame extraction behavior.
type Options struct {
	MaxFrames    int  // Max frames to extract (0 = default 16)
	ExtractAudio bool // Whether to extract the audio track
}

// Extract reads a video file and returns extracted frames and audio.
func Extract(path string, opts Options) (*Result, error) {
	if opts.MaxFrames <= 0 {
		opts.MaxFrames = 4
	}
	if opts.MaxFrames > 64 {
		opts.MaxFrames = 64
	}
	return extract(path, opts)
}

// IsVideo returns true if the content type indicates a video file.
func IsVideo(data []byte) bool {
	if len(data) < 512 {
		return false
	}
	ct := http.DetectContentType(data[:512])
	return IsVideoContentType(ct)
}

// IsVideoContentType returns true if the MIME type is a video type.
func IsVideoContentType(contentType string) bool {
	return strings.HasPrefix(contentType, "video/")
}

// VideoExtensions lists recognized video file extensions.
var VideoExtensions = []string{
	".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv",
}

// IsVideoExtension returns true if the extension (with dot) is a video format.
func IsVideoExtension(ext string) bool {
	ext = strings.ToLower(ext)
	for _, v := range VideoExtensions {
		if ext == v {
			return true
		}
	}
	return false
}

// ErrFFmpegNotFound is returned on Linux when ffmpeg is not installed.
var ErrFFmpegNotFound = fmt.Errorf("video support requires ffmpeg; install it with: sudo apt install ffmpeg (Debian/Ubuntu) or sudo dnf install ffmpeg (Fedora/RHEL)")
