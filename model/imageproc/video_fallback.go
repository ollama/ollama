//go:build !ffmpeg || !cgo
// +build !ffmpeg !cgo

package imageproc

import (
	"bytes"
	"fmt"
	"image"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// extractVideoFramesImpl falls back to system ffmpeg binary when embedded FFmpeg is not available.
// This implementation is used when building without -tags ffmpeg,cgo.
func extractVideoFramesImpl(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	// Check if system ffmpeg is available
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, fmt.Errorf("video support unavailable: ffmpeg not found in PATH and embedded FFmpeg not enabled (build with -tags ffmpeg,cgo to use embedded FFmpeg)")
	}

	// Create temporary directory for video processing
	tempDir, err := os.MkdirTemp("", "ollama-video-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tempDir)

	// Write video data to temporary file
	videoPath := filepath.Join(tempDir, "input.mp4")
	if err := os.WriteFile(videoPath, videoData, 0o600); err != nil {
		return nil, fmt.Errorf("failed to write video file: %w", err)
	}

	// Prepare ffmpeg command
	// -i: input file
	// -vf fps=X: extract X frames per second
	// -vsync 0: disable frame synchronization (extract all frames)
	// -q:v N: quality (1-31, lower is better quality, 2 is high quality)
	// output_%04d.jpg: output pattern
	framePattern := filepath.Join(tempDir, "frame_%04d.jpg")
	args := []string{
		"-i", videoPath,
		"-vf", fmt.Sprintf("fps=%.2f", config.FPS),
		"-vsync", "0",
		"-q:v", fmt.Sprintf("%d", config.Quality),
	}

	// Add frame limit if specified
	if config.MaxFrames > 0 {
		args = append(args, "-frames:v", fmt.Sprintf("%d", config.MaxFrames))
	}

	args = append(args, framePattern)

	cmd := exec.Command("ffmpeg", args...)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	// Create a context with timeout if specified
	if config.Timeout > 0 {
		timer := time.AfterFunc(config.Timeout, func() {
			if cmd.Process != nil {
				cmd.Process.Kill()
			}
		})
		defer timer.Stop()
	}

	// Run ffmpeg
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg extraction failed: %w (stderr: %s)", err, stderr.String())
	}

	// Read extracted frames
	frameFiles, err := filepath.Glob(filepath.Join(tempDir, "frame_*.jpg"))
	if err != nil {
		return nil, fmt.Errorf("failed to list extracted frame files: %w", err)
	}

	if len(frameFiles) == 0 {
		return nil, fmt.Errorf("no frames extracted from video (video may be corrupted or unsupported format)")
	}

	// Load frames as images
	frames := make([]image.Image, 0, len(frameFiles))
	for _, framePath := range frameFiles {
		frameData, err := os.ReadFile(framePath)
		if err != nil {
			return nil, fmt.Errorf("failed to read frame %s: %w", framePath, err)
		}

		img, _, err := image.Decode(bytes.NewReader(frameData))
		if err != nil {
			return nil, fmt.Errorf("failed to decode frame %s: %w", framePath, err)
		}

		frames = append(frames, img)
	}

	return frames, nil
}

// checkEmbeddedFFmpeg returns false when embedded FFmpeg is not available
func checkEmbeddedFFmpeg() bool {
	return false
}
