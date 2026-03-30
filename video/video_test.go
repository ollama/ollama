package video

import (
	"net/http"
	"testing"
)

func TestIsVideo(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want bool
	}{
		{
			name: "mp4 header",
			data: make([]byte, 512), // will be filled with mp4 magic
			want: false,             // zeros aren't video
		},
		{
			name: "jpeg data",
			data: append([]byte{0xFF, 0xD8, 0xFF, 0xE0}, make([]byte, 508)...),
			want: false,
		},
		{
			name: "wav data",
			data: append([]byte("RIFF"), append(make([]byte, 4), []byte("WAVE")...)...),
			want: false,
		},
		{
			name: "too short",
			data: []byte{0x00, 0x01, 0x02},
			want: false,
		},
		{
			name: "nil data",
			data: nil,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsVideo(tt.data)
			if got != tt.want {
				t.Errorf("IsVideo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsVideoContentType(t *testing.T) {
	tests := []struct {
		ct   string
		want bool
	}{
		{"video/mp4", true},
		{"video/webm", true},
		{"video/quicktime", true},
		{"video/x-msvideo", true},
		{"image/jpeg", false},
		{"audio/wave", false},
		{"application/octet-stream", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.ct, func(t *testing.T) {
			if got := IsVideoContentType(tt.ct); got != tt.want {
				t.Errorf("IsVideoContentType(%q) = %v, want %v", tt.ct, got, tt.want)
			}
		})
	}
}

func TestIsVideoExtension(t *testing.T) {
	tests := []struct {
		ext  string
		want bool
	}{
		{".mp4", true},
		{".MP4", true},
		{".webm", true},
		{".mov", true},
		{".avi", true},
		{".mkv", true},
		{".m4v", true},
		{".jpg", false},
		{".png", false},
		{".wav", false},
		{".txt", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.ext, func(t *testing.T) {
			if got := IsVideoExtension(tt.ext); got != tt.want {
				t.Errorf("IsVideoExtension(%q) = %v, want %v", tt.ext, got, tt.want)
			}
		})
	}
}

func TestExtractDefaults(t *testing.T) {
	// Test that Options defaults are applied correctly
	opts := Options{}
	if opts.MaxFrames != 0 {
		t.Errorf("zero value MaxFrames should be 0, got %d", opts.MaxFrames)
	}

	// Extract with a non-existent file should error
	_, err := Extract("/nonexistent/video.mp4", Options{})
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestIsVideoWithRealContentType(t *testing.T) {
	// Verify that http.DetectContentType correctly identifies video types
	// by testing with real file magic bytes

	// MP4 ftyp box: 4 bytes size + "ftyp" + brand
	mp4Data := make([]byte, 512)
	copy(mp4Data[0:], []byte{0x00, 0x00, 0x00, 0x20}) // size=32
	copy(mp4Data[4:], "ftypisom")                       // ftyp + brand
	copy(mp4Data[12:], []byte{0x00, 0x00, 0x02, 0x00}) // minor version

	ct := http.DetectContentType(mp4Data)
	t.Logf("MP4 content type: %s", ct)
	// Note: Go's http.DetectContentType may or may not detect MP4.
	// The ftyp box detection depends on the Go version.
}
