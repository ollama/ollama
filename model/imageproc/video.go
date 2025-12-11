package imageproc

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"os/exec"
	"path/filepath"
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

// ExtractVideoFrames extracts frames from video data using ffmpeg.
//
// This function:
// - Creates a temporary directory for processing
// - Writes the video data to a temporary file
// - Uses ffmpeg to extract frames at the specified FPS
// - Loads the extracted frames as image.Image objects
// - Cleans up temporary files
//
// The function requires ffmpeg to be installed and available in PATH.
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
	// Validate input
	if len(videoData) == 0 {
		return nil, fmt.Errorf("video data is empty")
	}

	// Check if ffmpeg is available
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, fmt.Errorf("ffmpeg is not installed or not in PATH: %w", err)
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
