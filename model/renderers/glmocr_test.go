package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestGlmOcrRenderer_Images(t *testing.T) {
	tests := []struct {
		name     string
		renderer *GlmOcrRenderer
		messages []api.Message
		expected string
	}{
		{
			name:     "use_img_tags_single_image",
			renderer: &GlmOcrRenderer{useImgTags: true},
			messages: []api.Message{
				{
					Role:    "user",
					Content: "Describe this image.",
					Images:  []api.ImageData{api.ImageData("img1")},
				},
			},
			expected: "[gMASK]<sop><|user|>\n[img-0] Describe this image.<|assistant|>\n",
		},
		{
			name:     "use_img_tags_multiple_images",
			renderer: &GlmOcrRenderer{useImgTags: true},
			messages: []api.Message{
				{
					Role:    "user",
					Content: "Describe these images.",
					Images:  []api.ImageData{api.ImageData("img1"), api.ImageData("img2")},
				},
			},
			expected: "[gMASK]<sop><|user|>\n[img-0][img-1] Describe these images.<|assistant|>\n",
		},
		{
			name:     "multi_turn_increments_image_offset",
			renderer: &GlmOcrRenderer{useImgTags: true},
			messages: []api.Message{
				{
					Role:    "user",
					Content: "First image",
					Images:  []api.ImageData{api.ImageData("img1")},
				},
				{
					Role:    "assistant",
					Content: "Processed.",
				},
				{
					Role:    "user",
					Content: "Second image",
					Images:  []api.ImageData{api.ImageData("img2")},
				},
			},
			expected: "[gMASK]<sop><|user|>\n[img-0] First image<|assistant|>\n<think></think>\nProcessed.\n<|user|>\n[img-1] Second image<|assistant|>\n",
		},
		{
			name:     "default_no_img_tags",
			renderer: &GlmOcrRenderer{},
			messages: []api.Message{
				{
					Role:    "user",
					Content: "No image tags expected.",
					Images:  []api.ImageData{api.ImageData("img1")},
				},
			},
			expected: "[gMASK]<sop><|user|>\nNo image tags expected.<|assistant|>\n",
		},
		{
			name:     "no_images_content_unchanged",
			renderer: &GlmOcrRenderer{useImgTags: true},
			messages: []api.Message{
				{
					Role:    "user",
					Content: "Text only message.",
				},
			},
			expected: "[gMASK]<sop><|user|>\nText only message.<|assistant|>\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.renderer.Render(tt.messages, nil, nil)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Fatalf("Render() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
