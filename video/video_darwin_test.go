package video

import (
	"os"
	"testing"
)

func TestExtractRealVideo(t *testing.T) {
	// Skip if test video doesn't exist (CI environments)
	testVideo := ".tmp/test_video.mp4"
	// Try repo root
	if _, err := os.Stat(testVideo); err != nil {
		testVideo = "../.tmp/test_video.mp4"
		if _, err := os.Stat(testVideo); err != nil {
			t.Skip("test video not available")
		}
	}

	result, err := Extract(testVideo, Options{MaxFrames: 2, ExtractAudio: true})
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if len(result.Frames) == 0 {
		t.Fatal("no frames extracted")
	}
	if len(result.Frames) > 2 {
		t.Errorf("expected at most 2 frames, got %d", len(result.Frames))
	}

	// Verify frames are valid JPEG
	for i, frame := range result.Frames {
		if len(frame) < 2 || frame[0] != 0xFF || frame[1] != 0xD8 {
			t.Errorf("frame %d is not valid JPEG (first bytes: %x)", i, frame[:min(4, len(frame))])
		}
	}

	// Verify audio is valid WAV
	if result.Audio == nil {
		t.Error("expected audio but got nil")
	} else if len(result.Audio) < 44 {
		t.Error("audio WAV too short")
	} else if string(result.Audio[:4]) != "RIFF" || string(result.Audio[8:12]) != "WAVE" {
		t.Error("audio is not valid WAV")
	}

	t.Logf("Extracted %d frames, audio %d bytes", len(result.Frames), len(result.Audio))
}
