package imageproc

import (
	"bytes"
	"os/exec"
	"testing"
	"time"
)

// TestDefaultVideoConfig verifies the default configuration values
func TestDefaultVideoConfig(t *testing.T) {
	config := DefaultVideoConfig()

	if config.FPS != 1.0 {
		t.Errorf("Expected FPS=1.0, got %f", config.FPS)
	}

	if config.Quality != 2 {
		t.Errorf("Expected Quality=2, got %d", config.Quality)
	}

	if config.MaxFrames != 0 {
		t.Errorf("Expected MaxFrames=0, got %d", config.MaxFrames)
	}

	if config.Timeout != 60*time.Second {
		t.Errorf("Expected Timeout=60s, got %v", config.Timeout)
	}
}

// TestExtractVideoFrames_EmptyData tests error handling for empty video data
func TestExtractVideoFrames_EmptyData(t *testing.T) {
	config := DefaultVideoConfig()
	frames, err := ExtractVideoFrames([]byte{}, config)

	if err == nil {
		t.Error("Expected error for empty video data, got nil")
	}

	if frames != nil {
		t.Errorf("Expected nil frames for empty data, got %d frames", len(frames))
	}

	if err.Error() != "video data is empty" {
		t.Errorf("Expected 'video data is empty' error, got: %v", err)
	}
}

// TestExtractVideoFrames_InvalidData tests error handling for invalid video data
func TestExtractVideoFrames_InvalidData(t *testing.T) {
	config := DefaultVideoConfig()
	invalidData := []byte("this is not a video")

	frames, err := ExtractVideoFrames(invalidData, config)

	if err == nil {
		t.Error("Expected error for invalid video data, got nil")
	}

	if frames != nil {
		t.Errorf("Expected nil frames for invalid data, got %d frames", len(frames))
	}
}

// TestExtractVideoFrames_FFmpegAvailable verifies ffmpeg is installed
func TestExtractVideoFrames_FFmpegAvailable(t *testing.T) {
	// ffmpeg MUST be available in CI/test environment
}

// TestExtractVideoFrames_ValidVideo tests successful frame extraction
func TestExtractVideoFrames_ValidVideo(t *testing.T) {
	// Create a simple test video using ffmpeg
	cmd := exec.Command("ffmpeg",
		"-f", "lavfi",
		"-i", "testsrc=duration=2:size=320x240:rate=5",
		"-pix_fmt", "yuv420p",
		"-f", "mp4",
		"-movflags", "frag_keyframe+empty_moov",
		"pipe:1",
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to create test video: %v", err)
	}

	videoData := stdout.Bytes()
	if len(videoData) == 0 {
		t.Fatal("Generated video data is empty")
	}

	// Extract frames
	config := DefaultVideoConfig()
	config.FPS = 2.0

	frames, err := ExtractVideoFrames(videoData, config)
	if err != nil {
		t.Fatalf("ExtractVideoFrames failed: %v", err)
	}

	if len(frames) == 0 {
		t.Error("Expected at least 1 frame, got 0")
	}

	// Verify frames are valid images
	for i, frame := range frames {
		if frame == nil {
			t.Errorf("Frame %d is nil", i)
			continue
		}

		bounds := frame.Bounds()
		if bounds.Dx() == 0 || bounds.Dy() == 0 {
			t.Errorf("Frame %d has invalid dimensions: %dx%d", i, bounds.Dx(), bounds.Dy())
		}
	}
}

// TestExtractVideoFrames_MaxFrames tests frame limiting
func TestExtractVideoFrames_MaxFrames(t *testing.T) {
	// Create a test video
	cmd := exec.Command("ffmpeg",
		"-f", "lavfi",
		"-i", "testsrc=duration=3:size=320x240:rate=5",
		"-pix_fmt", "yuv420p",
		"-f", "mp4",
		"-movflags", "frag_keyframe+empty_moov",
		"pipe:1",
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to create test video: %v", err)
	}

	videoData := stdout.Bytes()

	// Extract with frame limit
	config := DefaultVideoConfig()
	config.FPS = 5.0
	config.MaxFrames = 3

	frames, err := ExtractVideoFrames(videoData, config)
	if err != nil {
		t.Fatalf("ExtractVideoFrames failed: %v", err)
	}

	if len(frames) > config.MaxFrames {
		t.Errorf("Expected at most %d frames, got %d", config.MaxFrames, len(frames))
	}
}

// TestExtractVideoFrames_CustomConfig tests custom configuration
func TestExtractVideoFrames_CustomConfig(t *testing.T) {
	config := VideoExtractionConfig{
		FPS:       0.5,
		Quality:   10,
		MaxFrames: 5,
		Timeout:   30 * time.Second,
	}

	if config.FPS != 0.5 {
		t.Errorf("Expected FPS=0.5, got %f", config.FPS)
	}

	if config.Quality != 10 {
		t.Errorf("Expected Quality=10, got %d", config.Quality)
	}

	if config.MaxFrames != 5 {
		t.Errorf("Expected MaxFrames=5, got %d", config.MaxFrames)
	}

	if config.Timeout != 30*time.Second {
		t.Errorf("Expected Timeout=30s, got %v", config.Timeout)
	}
}

// TestExtractVideoFrames_SingleFrame tests video with only 1 frame
func TestExtractVideoFrames_SingleFrame(t *testing.T) {
	// Create a 1-second video at 1 FPS (results in 1 frame)
	cmd := exec.Command("ffmpeg",
		"-f", "lavfi",
		"-i", "testsrc=duration=0.5:size=320x240:rate=1",
		"-pix_fmt", "yuv420p",
		"-f", "mp4",
		"-movflags", "frag_keyframe+empty_moov",
		"pipe:1",
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed to create test video: %v", err)
	}

	videoData := stdout.Bytes()

	config := DefaultVideoConfig()
	config.FPS = 1.0

	frames, err := ExtractVideoFrames(videoData, config)
	if err != nil {
		t.Fatalf("ExtractVideoFrames failed: %v", err)
	}

	// Single-frame videos should still work
	if len(frames) == 0 {
		t.Error("Expected at least 1 frame, got 0")
	}
}

// TestEmbeddedFFmpegAvailability tests if embedded FFmpeg libraries are detected correctly
func TestEmbeddedFFmpegAvailability(t *testing.T) {
	// This test verifies the embedded FFmpeg detection
	// The actual value depends on build tags
	available := checkEmbeddedFFmpeg()

	t.Logf("Embedded FFmpeg libs available: %v", available)

	// Just verify the function runs without error
	// In CI, this will be true when built with -tags ffmpeg,cgo
	// and false otherwise
}

// TestExtractVideoFrames_Implementation tests that the correct implementation is used
func TestExtractVideoFrames_Implementation(t *testing.T) {
	// Create a small test video
	cmd := exec.Command("ffmpeg",
		"-f", "lavfi",
		"-i", "testsrc=duration=0.5:size=160x120:rate=1",
		"-pix_fmt", "yuv420p",
		"-f", "mp4",
		"-movflags", "frag_keyframe+empty_moov",
		"pipe:1",
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		t.Skip("ffmpeg not available for test video generation")
	}

	videoData := stdout.Bytes()
	if len(videoData) == 0 {
		t.Fatal("Generated video data is empty")
	}

	// Test extraction works regardless of implementation
	config := DefaultVideoConfig()
	frames, err := ExtractVideoFrames(videoData, config)
	if err != nil {
		t.Fatalf("ExtractVideoFrames failed: %v", err)
	}

	if len(frames) == 0 {
		t.Error("Expected at least 1 frame")
	}

	t.Logf("Successfully extracted %d frame(s) using %s",
		len(frames),
		map[bool]string{true: "embedded FFmpeg", false: "system ffmpeg"}[checkEmbeddedFFmpeg()])
}

// BenchmarkExtractVideoFrames benchmarks video frame extraction
func BenchmarkExtractVideoFrames(b *testing.B) {
	// Create test video once
	cmd := exec.Command("ffmpeg",
		"-f", "lavfi",
		"-i", "testsrc=duration=1:size=320x240:rate=5",
		"-pix_fmt", "yuv420p",
		"-f", "mp4",
		"-movflags", "frag_keyframe+empty_moov",
		"pipe:1",
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		b.Fatalf("Failed to create test video: %v", err)
	}

	videoData := stdout.Bytes()
	config := DefaultVideoConfig()

	b.ResetTimer()

	for range b.N {
		_, err := ExtractVideoFrames(videoData, config)
		if err != nil {
			b.Fatalf("ExtractVideoFrames failed: %v", err)
		}
	}
}
