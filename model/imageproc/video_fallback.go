package imageproc

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// extractVideoFramesSystem uses system ffmpeg binary for video frame extraction.
// NO build tags - this is always available so video_astiav.go can use it as fallback.
func extractVideoFramesSystem(videoData []byte, config VideoExtractionConfig) ([]image.Image, error) {
	// Check if system ffmpeg is available
	ffmpegPath, err := exec.LookPath("ffmpeg")
	if err != nil {
		slog.Error("System ffmpeg not found in PATH")
		return nil, fmt.Errorf("video support unavailable: ffmpeg not found in PATH")
	}

	slog.Debug("Using system ffmpeg for video extraction", "path", ffmpegPath, "size", len(videoData))

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
	framePattern := filepath.Join(tempDir, "frame_%04d.jpg")
	args := []string{
		"-i", videoPath,
		"-vf", fmt.Sprintf("fps=%.2f", config.FPS),
		"-vsync", "0",
		"-q:v", fmt.Sprintf("%d", config.Quality),
	}

	if config.MaxFrames > 0 {
		args = append(args, "-frames:v", fmt.Sprintf("%d", config.MaxFrames))
	}
	args = append(args, framePattern)

	cmd := exec.Command("ffmpeg", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if config.Timeout > 0 {
		timer := time.AfterFunc(config.Timeout, func() {
			if cmd.Process != nil {
				cmd.Process.Kill()
			}
		})
		defer timer.Stop()
	}

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg extraction failed: %w (stderr: %s)", err, stderr.String())
	}

	frameFiles, err := filepath.Glob(filepath.Join(tempDir, "frame_*.jpg"))
	if err != nil {
		return nil, fmt.Errorf("failed to list extracted frame files: %w", err)
	}

	if len(frameFiles) == 0 {
		return nil, fmt.Errorf("no frames extracted from video")
	}

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

	slog.Debug("System ffmpeg extraction completed", "num_frames", len(frames))
	return frames, nil
}
