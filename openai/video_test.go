package openai

import (
	"encoding/base64"
	"testing"
)

func TestFromChatRequestVideo(t *testing.T) {
	data := base64.StdEncoding.EncodeToString([]byte("video-bytes"))
	for _, input := range []any{"data:video/mp4;base64," + data, map[string]any{"url": "data:video/webm;base64," + data}} {
		req, err := FromChatRequest(ChatCompletionRequest{Messages: []Message{{Role: "user", Content: []any{
			map[string]any{"type": "text", "text": "before"},
			map[string]any{"type": "video_url", "video_url": input},
			map[string]any{"type": "text", "text": "after"},
		}}}})
		if err != nil {
			t.Fatal(err)
		}
		if len(req.Messages) != 3 || req.Messages[0].Content != "before" || req.Messages[2].Content != "after" || len(req.Messages[1].Videos) != 1 || string(req.Messages[1].Videos[0]) != "video-bytes" {
			t.Fatalf("video or content order lost: %#v", req.Messages)
		}
	}
	for _, input := range []any{nil, 1, map[string]any{}, "", "https://example.com/clip.mp4", "file:///clip.mp4", "data:text/plain;base64," + data, "data:video/mp4;base64,!!!", "data:video/mp4;base64,"} {
		_, err := FromChatRequest(ChatCompletionRequest{Messages: []Message{{Role: "user", Content: []any{map[string]any{"type": "video_url", "video_url": input}}}}})
		if err == nil {
			t.Errorf("accepted invalid video input %#v", input)
		}
	}
}
