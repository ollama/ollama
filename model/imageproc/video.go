package imageproc

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"log/slog"
	"sync"
	"time"
)

// VideoExtractionConfig holds configuration for video frame extraction
type VideoExtractionConfig struct {
	// FPS specifies how many frames per second to extract from the video
	FPS float64

	// Quality specifies the JPEG quality for extracted frames (1-31, lower is better quality)
	Quality int

	// MaxFrames limits the maximum number of frames to extract (0 = no limit)
	MaxFrames int

	// Timeout specifies the maximum time allowed for ffmpeg extraction
	Timeout time.Duration
}

// DefaultVideoConfig returns sensible defaults for video extraction
func DefaultVideoConfig() VideoExtractionConfig {
	return VideoExtractionConfig{
		FPS:       1.0, // 1 frame per second
		Quality:   2,   // High quality JPEG
		MaxFrames: 0,   // No limit
		Timeout:   60 * time.Second,
	}
}

var (
	embeddedFFmpegAvailable bool
	checkOnce               sync.Once
)

// ExtractVideoFrames extracts frames from video data.
//
// This function automatically tries embedded FFmpeg libs first (if built with -tags ffmpeg,cgo),
// via go-astiav (handled by video_astiav.go) then falls back to system ffmpeg binary
// if embedded is unavailable or fails.
//
// Parameters:
//   - videoData: Raw video file bytes (any format supported by ffmpeg)
//   - config: Configuration for frame extraction (use DefaultVideoConfig() for defaults)
//
// Returns:
//   - []image.Image: Slice of extracted frames
//   - error: Any error encountered during extraction
//
// Example:
//
//	config := imageproc.DefaultVideoConfig()
//	config.FPS = 2.0  // Extract 2 frames per second
//	frames, err := imageproc.ExtractVideoFrames(videoData, config)
func ExtractVideoFrames(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	if len(videoData) == 0 {
		return nil, fmt.Errorf("video data is empty")
	}

	// Check if embedded FFmpeg is available (once per process)
	checkOnce.Do(func() {
		embeddedFFmpegAvailable = checkEmbeddedFFmpeg()
		if embeddedFFmpegAvailable {
			slog.Debug("using embedded FFmpeg for video processing")
		} else {
			slog.Debug("embedded FFmpeg libs unavailable, will use system ffmpeg binary if available")
		}
	})

	// Try embedded FFmpeg first
	if embeddedFFmpegAvailable {
		frames, err := extractVideoFramesImpl(videoData, config)
		if err == nil {
			return frames, nil
		}
		// Log warning but continue to fallback
		slog.Warn("embedded FFmpeg libs failed, falling back to system ffmpeg binary", "error", err)
	}

	// Fallback to system ffmpeg binary (handled by video_fallback.go)
	return extractVideoFramesImpl(videoData, config)
}
