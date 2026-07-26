package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMessagesContainAudio(t *testing.T) {
	// Valid WAV header: "RIFF" + 4 bytes size + "WAVE"
	wavData := append([]byte("RIFF"), 0, 0, 0, 0)
	wavData = append(wavData, []byte("WAVE")...)
	wavData = append(wavData, make([]byte, 20)...) // padding

	// Valid MP3 header (ID3 tag)
	mp3Data := append([]byte("ID3"), make([]byte, 20)...)

	// PNG image data (not audio)
	pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	pngData = append(pngData, make([]byte, 20)...)

	tests := []struct {
		name     string
		messages []api.Message
		want     bool
	}{
		{
			name:     "no messages",
			messages: nil,
			want:     false,
		},
		{
			name: "text only",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			want: false,
		},
		{
			name: "image only",
			messages: []api.Message{
				{Role: "user", Content: "Describe this.", Images: []api.ImageData{pngData}},
			},
			want: false,
		},
		{
			name: "wav audio",
			messages: []api.Message{
				{Role: "user", Content: "Transcribe this.", Images: []api.ImageData{wavData}},
			},
			want: true,
		},
		{
			name: "mp3 audio",
			messages: []api.Message{
				{Role: "user", Content: "Transcribe this.", Images: []api.ImageData{mp3Data}},
			},
			want: true,
		},
		{
			name: "mixed image and audio",
			messages: []api.Message{
				{Role: "user", Content: "Describe this.", Images: []api.ImageData{pngData, wavData}},
			},
			want: true,
		},
		{
			name: "audio in earlier message",
			messages: []api.Message{
				{Role: "system", Content: "You are a transcription assistant."},
				{Role: "user", Content: "First clip.", Images: []api.ImageData{wavData}},
				{Role: "user", Content: "Now explain it."},
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := messagesContainAudio(tt.messages)
			if got != tt.want {
				t.Errorf("messagesContainAudio() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestImagesContainAudio(t *testing.T) {
	wavData := append([]byte("RIFF"), 0, 0, 0, 0)
	wavData = append(wavData, []byte("WAVE")...)
	wavData = append(wavData, make([]byte, 20)...)

	pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	pngData = append(pngData, make([]byte, 20)...)

	tests := []struct {
		name   string
		images []api.ImageData
		want   bool
	}{
		{
			name:   "nil images",
			images: nil,
			want:   false,
		},
		{
			name:   "empty images",
			images: []api.ImageData{},
			want:   false,
		},
		{
			name:   "only images",
			images: []api.ImageData{pngData},
			want:   false,
		},
		{
			name:   "wav file",
			images: []api.ImageData{wavData},
			want:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := imagesContainAudio(tt.images)
			if got != tt.want {
				t.Errorf("imagesContainAudio() = %v, want %v", got, tt.want)
			}
		})
	}
}
