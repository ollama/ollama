//go:build !ffmpeg || !cgo

package imageproc

import "image"

// extractVideoFramesImpl uses system ffmpeg when embedded FFmpeg is not available.
func extractVideoFramesImpl(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	return extractVideoFramesSystem(videoData, config)
}

// checkEmbeddedFFmpeg returns false when built without embedded FFmpeg
func checkEmbeddedFFmpeg() bool {
	return false
}
