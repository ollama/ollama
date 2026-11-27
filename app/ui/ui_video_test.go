//go:build windows || darwin

package ui

import (
	"bytes"
	"testing"

	"github.com/ollama/ollama/app/store"
)

func TestBuildChatRequestVideoAttachments(t *testing.T) {
	for _, filename := range []string{"clip.mp4", "CLIP.MOV", "clip.webm", "clip.mkv", "clip.avi"} {
		t.Run(filename, func(t *testing.T) {
			video := []byte{0, 1, 255, 2}
			chat := &store.Chat{Messages: []store.Message{{Role: "user", Content: "describe the clip", Attachments: []store.File{
				{Filename: filename, Data: video},
				{Filename: "image.png", Data: []byte("image")},
				{Filename: "notes.txt", Data: []byte("notes")},
			}}}}
			request, err := (&Server{}).buildChatRequest(chat, "qwen3-vl", nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			message := request.Messages[0]
			if len(message.Videos) != 1 || !bytes.Equal(message.Videos[0], video) {
				t.Fatalf("video bytes lost: %#v", message.Videos)
			}
			if len(message.Images) != 1 || string(message.Images[0]) != "image" {
				t.Fatalf("image bytes lost: %#v", message.Images)
			}
			if message.Content != "describe the clip\n--- File: notes.txt ---\nnotes\n--- End of notes.txt ---" {
				t.Fatalf("binary video became text: %q", message.Content)
			}
		})
	}
}
