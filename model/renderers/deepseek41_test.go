package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestDeepSeekV41TextPrompts(t *testing.T) {
	chat := &api.ThinkValue{Value: false}
	for _, test := range []struct {
		name     string
		messages []api.Message
		want     string
	}{
		{
			name:     "single turn",
			messages: []api.Message{{Role: "user", Content: "hello"}},
			want:     "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>",
		},
		{
			name: "system and turns",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "q1"},
				{Role: "assistant", Content: "a1"},
				{Role: "user", Content: "q2"},
			},
			want: "<｜begin▁of▁sentence｜><｜System｜>You are helpful.<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜User｜>q2<｜Assistant｜></think>",
		},
		{
			name: "midconversation system",
			messages: []api.Message{
				{Role: "user", Content: "q1"},
				{Role: "assistant", Content: "a1"},
				{Role: "system", Content: "New rule"},
			},
			want: "<｜begin▁of▁sentence｜><｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜System｜>New rule<｜Assistant｜></think>",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := RenderWithRenderer("deepseek-v41", test.messages, nil, chat)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("prompt = %q, want %q", got, test.want)
			}
		})
	}
}

func TestDeepSeekV41DefaultThinkingPrompt(t *testing.T) {
	got, err := RenderWithRenderer("deepseek-v41", []api.Message{{Role: "user", Content: "hello"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>hello<｜Assistant｜><think>"
	if got != want {
		t.Fatalf("prompt = %q, want %q", got, want)
	}
}

func TestDeepSeekV41ToolResultTurn(t *testing.T) {
	got, err := RenderWithRenderer("deepseek-v41", []api.Message{{Role: "tool", Content: "result"}}, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatal(err)
	}
	if got != "<｜begin▁of▁sentence｜><｜User｜><tool_result>result</tool_result><｜Assistant｜></think>" {
		t.Fatalf("tool result prompt = %q", got)
	}
}

func TestDeepSeekV41ImagePlacement(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "inspect", Images: []api.ImageData{{1}}},
		{Role: "assistant", Content: "done"},
		{Role: "user", Content: "compare [img] and [img]", Images: []api.ImageData{{2}, {3}}},
	}
	got, err := RenderWithRenderer("deepseek-v41", messages, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatal(err)
	}
	want := "<｜begin▁of▁sentence｜><｜User｜>inspect\n\n[img-0]<｜Assistant｜></think>done<｜end▁of▁sentence｜><｜User｜>compare \n\n[img-1]\n\n and \n\n[img-2]<｜Assistant｜></think>"
	if got != want {
		t.Fatalf("image prompt = %q, want %q", got, want)
	}
}

func TestDeepSeekV41KeepsExplicitImageTag(t *testing.T) {
	got, err := RenderWithRenderer("deepseek-v41", []api.Message{{Role: "user", Content: "Look [img-0]", Images: []api.ImageData{{1}}}}, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatal(err)
	}
	want := "<｜begin▁of▁sentence｜><｜User｜>Look \n\n[img-0]<｜Assistant｜></think>"
	if got != want {
		t.Fatalf("prompt = %q, want %q", got, want)
	}
}

func TestDeepSeekV41CompletesPartialImageTags(t *testing.T) {
	got, err := RenderWithRenderer("deepseek-v41", []api.Message{{
		Role: "user", Content: "Look [img-0] and [img-", Images: []api.ImageData{{1}, {2}},
	}}, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatal(err)
	}
	want := "<｜begin▁of▁sentence｜><｜User｜>Look \n\n[img-0]\n\n and [img-\n\n[img-1]<｜Assistant｜></think>"
	if got != want {
		t.Fatalf("prompt = %q, want %q", got, want)
	}
}

func TestDeepSeekV41RejectsLiteralImageToken(t *testing.T) {
	_, err := RenderWithRenderer("deepseek-v41", []api.Message{{Role: "user", Content: "<｜deepseek_image｜>", Images: []api.ImageData{{1}}}}, nil, &api.ThinkValue{Value: false})
	if err == nil {
		t.Fatal("literal image token in text must be rejected")
	}
}

func TestDeepSeekV41RejectsAmbiguousToolName(t *testing.T) {
	tool := api.Tool{Type: "function", Function: api.ToolFunction{Name: "a::b::c"}}
	if _, err := RenderWithRenderer("deepseek-v41", []api.Message{{Role: "user", Content: "hello"}}, []api.Tool{tool}, nil); err == nil {
		t.Fatal("tool name with two namespace separators accepted")
	}
}
