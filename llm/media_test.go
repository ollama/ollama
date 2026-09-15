package llm

import "testing"

func TestDetectMediaKind(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want MediaKind
	}{
		{"wav", []byte("RIFF\x00\x00\x00\x00WAVEfmt "), MediaKindAudio},
		{"mp3 id3", []byte("ID3\x03\x00\x00\x00\x00\x00\x00"), MediaKindAudio},
		{"mp4", append([]byte{0x00, 0x00, 0x00, 0x18}, []byte("ftypmp42")...), MediaKindVideo},
		{"webm", []byte{0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x02, 0x03, 0x04}, MediaKindVideo},
		{"avi", []byte("RIFF\x00\x00\x00\x00AVI LIST"), MediaKindVideo},
		{"png", []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\x00"), MediaKindImage},
		{"jpeg", []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46}, MediaKindImage},
		{"unknown", []byte("not a media file"), MediaKindUnknown},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DetectMediaKind(tc.data); got != tc.want {
				t.Errorf("DetectMediaKind(%q) = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}
