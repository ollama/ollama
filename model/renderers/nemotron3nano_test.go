package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestNemotron3NanoRenderer_Images(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
		expected string
	}{
		{
			name: "single image inserts placeholder",
			msgs: []api.Message{
				{Role: "user", Content: "Describe this image.", Images: []api.ImageData{api.ImageData("img1")}},
			},
			expected: "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n[img-0] Describe this image.<|im_end|>\n\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "generic image placeholder is rewritten",
			msgs: []api.Message{
				{Role: "user", Content: "[img]Describe this image.", Images: []api.ImageData{api.ImageData("img1")}},
			},
			expected: "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n[img-0]Describe this image.<|im_end|>\n\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "image offsets increment across messages",
			msgs: []api.Message{
				{Role: "user", Content: "Describe the first image.", Images: []api.ImageData{api.ImageData("img1")}},
				{Role: "assistant", Content: "It shows something."},
				{Role: "user", Content: "Compare these.", Images: []api.ImageData{api.ImageData("img2"), api.ImageData("img3")}},
			},
			expected: "\n\n\n<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n[img-0] Describe the first image.<|im_end|>\n<|im_start|>assistant\n<think></think>It shows something.<|im_end|>\n<|im_start|>user\n[img-1][img-2] Compare these.<|im_end|>\n\n<|im_start|>assistant\n<think>\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			renderer := &Nemotron3NanoRenderer{}
			rendered, err := renderer.Render(tt.msgs, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.expected, rendered); diff != "" {
				t.Fatalf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
