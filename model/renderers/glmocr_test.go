package renderers

import (
	"strings"
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

// ResolveThinking fills Default:false when the client omits think. That must not
// inject /nothink or an empty generation <think></think> — those markers derail
// glm-ocr table recognition (see #18810).
func TestGlmOcrRenderer_NoNothinkWhenThinkingOff(t *testing.T) {
	messages := []api.Message{
		{
			Role:    "user",
			Content: "Table Recognition:",
			Images:  []api.ImageData{api.ImageData("img1")},
		},
	}
	want := "[gMASK]<sop><|user|>\n[img-0] Table Recognition:<|assistant|>\n"

	tests := []struct {
		name  string
		think *api.ThinkValue
	}{
		{name: "think_omitted", think: nil},
		{name: "think_false", think: &api.ThinkValue{Value: false}},
		{name: "think_true", think: &api.ThinkValue{Value: true}},
	}

	renderer := &GlmOcrRenderer{useImgTags: true}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(messages, nil, tt.think)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatalf("Render() mismatch (-want +got):\n%s", diff)
			}
			if strings.Contains(got, "/nothink") {
				t.Fatalf("prompt unexpectedly contains /nothink: %q", got)
			}
			if strings.HasSuffix(got, "<think></think>\n") {
				t.Fatalf("prompt unexpectedly ends with empty think block: %q", got)
			}
		})
	}
}
