package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLFM3RendererPrompt(t *testing.T) {
	cases := []struct {
		name     string
		messages []api.Message
		think    *api.ThinkValue
		want     string
	}{
		{
			name:     "thinking default",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
			want:     "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>",
		},
		{
			name:     "thinking disabled",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
			think:    &api.ThinkValue{Value: false},
			want:     "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>",
		},
		{
			name: "previous reasoning removed",
			messages: []api.Message{
				{Role: "user", Content: "Calculate"},
				{Role: "assistant", Thinking: "Four.", Content: "4"},
				{Role: "user", Content: "Again"},
			},
			want: "<|startoftext|><|im_start|>user\nCalculate<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>4<|im_end|>\n" +
				"<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>",
		},
		{
			name:     "image placeholder",
			messages: []api.Message{{Role: "user", Content: "Describe it.", Images: []api.ImageData{[]byte{1}}}},
			want: "<|startoftext|><|im_start|>user\n<|vision_start|><|image|><|vision_end|>Describe it." +
				"<|im_end|>\n<|im_start|>assistant\n<think>",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := (&LFM3Renderer{}).Render(tc.messages, nil, tc.think)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Errorf("prompt = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLFM3RendererImageTag(t *testing.T) {
	for _, tc := range []struct {
		name    string
		role    string
		content string
		images  int
		want    string
	}{
		{name: "one image", role: "user", content: "Describe it.", images: 1, want: "[img-0]Describe it."},
		{name: "two images", role: "user", content: "Compare them.", images: 2, want: "[img-0][img-1]Compare them."},
		{name: "explicit markers", role: "user", content: "A [img] B [img] C", images: 2, want: "A [img-0] B [img-1] C"},
		{name: "indexed marker", role: "user", content: "Describe [img-0]", images: 1, want: "Describe [img-0]"},
		{name: "indexed markers", role: "user", content: "Compare [img-0] and [img-1]", images: 2, want: "Compare [img-0] and [img-1]"},
		{name: "indexed tool image", role: "tool", content: "Found [img-0]", images: 1, want: "<tool_response>\nFound [img-0]\n</tool_response>"},
		{name: "publisher marker", role: "user", content: "A " + lfm3Image + " B", images: 1, want: "A [img-0] B"},
		{name: "tool images", role: "tool", content: "Found them.", images: 2, want: "<tool_response>\n[img-0]\n[img-1]\nFound them.\n</tool_response>"},
		{name: "tool image with preformatted text", role: "tool", content: "<tool_response>\nOK\n</tool_response>", images: 1, want: "<tool_response>\n[img-0]\n<tool_response>\nOK\n</tool_response>\n</tool_response>"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			images := make([]api.ImageData, tc.images)
			for i := range images {
				images[i] = []byte{byte(i + 1)}
			}
			got, err := (&LFM3Renderer{useImgTags: true}).Render(
				[]api.Message{{Role: tc.role, Content: tc.content, Images: images}}, nil, nil,
			)
			if err != nil {
				t.Fatal(err)
			}
			want := "<|startoftext|><|im_start|>" + tc.role + "\n" + tc.want + "<|im_end|>\n<|im_start|>assistant\n<think>"
			if got != want {
				t.Errorf("prompt = %q, want %q", got, want)
			}
		})
	}
}

func TestLFM3RendererImageTagsAcrossMessages(t *testing.T) {
	messages := []api.Message{
		{Role: "system", Content: "Inspect [img-0].", Images: []api.ImageData{[]byte{1}}},
		{Role: "user", Content: "Compare [img-1] and [img-2].", Images: []api.ImageData{[]byte{2}, []byte{3}}},
		{Role: "assistant", Content: "Here.", Images: []api.ImageData{[]byte{4}}},
	}
	got, err := (&LFM3Renderer{useImgTags: true}).Render(messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|startoftext|><|im_start|>system\nInspect [img-0].<|im_end|>\n" +
		"<|im_start|>user\nCompare [img-1] and [img-2].<|im_end|>\n" +
		"<|im_start|>assistant\n<think></think>[img-3]Here."
	if got != want {
		t.Errorf("prompt = %q, want %q", got, want)
	}
}
