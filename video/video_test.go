package video

import "testing"

func TestIsVideo(t *testing.T) {
	for _, tc := range []struct {
		data []byte
		want bool
	}{
		{[]byte("\x00\x00\x00\x20ftypisom"), true},
		{[]byte("\x00\x00\x00\x14ftypqt  "), true},
		{[]byte("\x00\x00\x00\x1cftypM4V "), true},
		{[]byte("\x00\x00\x00\x1cftypavif"), false}, // still images share the container
		{[]byte("\x00\x00\x00\x18ftypheic"), false},
		{[]byte("\x00\x00\x00\x18ftypmif1"), false},
		{[]byte("\x00\x00\x00\x20ftypM4A "), false}, // so does MPEG-4 audio
		{[]byte("\x00\x00\x00\x20ftypM4B "), false},
		{[]byte("\x00\x00\x00\x14ftypjp2 "), false},
		{[]byte("\x00\x00\x00\x18ftypcrx "), false},
		{[]byte{0x1a, 0x45, 0xdf, 0xa3, 0, 0, 0, 0, 0, 0, 0, 0}, false}, // WebM: no decoder
		{[]byte("RIFF\x00\x00\x00\x00AVI "), false},
		{[]byte("RIFF\x00\x00\x00\x00WAVE"), false},
		{[]byte("\xff\xd8\xff\xe0\x00\x10JFIF\x00"), false},
		{[]byte("ftyp"), false},
	} {
		if got := IsVideo(tc.data); got != tc.want {
			t.Errorf("IsVideo(%q) = %v, want %v", tc.data, got, tc.want)
		}
	}
}
