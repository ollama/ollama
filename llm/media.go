package llm

import (
	"bytes"
	"net/http"
	"strings"
)

func NewMediaData(id int, data []byte) MediaData {
	return MediaData{
		Data: data,
		ID:   id,
		Kind: DetectMediaKind(data),
	}
}

func DetectMediaKind(data []byte) MediaKind {
	if _, ok := AudioFormat(data); ok {
		return MediaKindAudio
	}
	if strings.HasPrefix(http.DetectContentType(data), "image/") {
		return MediaKindImage
	}
	return MediaKindUnknown
}

func AudioFormat(data []byte) (string, bool) {
	if len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")) {
		return "wav", true
	}
	if len(data) >= 3 && bytes.Equal(data[:3], []byte("ID3")) {
		return "mp3", true
	}
	if len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0 {
		return "mp3", true
	}
	return "", false
}
