package imageproc

import (
	"fmt"
	"image"
	"time"
)

// VideoExtractionConfig holds configuration for video frame extraction
type VideoExtractionConfig struct {
	// FPS specifies the frame rate for extraction (e.g., 1.0 = 1 frame per second)
	FPS float64

	// Quality specifies JPEG quality for extracted frames (1-31, lower is better, 2 is high quality)
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

// ExtractVideoFrames extracts frames from video data.
//
// Uses embedded FFmpeg if built with -tags ffmpeg,cgo (via go-astiav),
// otherwise uses system ffmpeg binary.
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

	return extractVideoFramesImpl(videoData, config)
}
