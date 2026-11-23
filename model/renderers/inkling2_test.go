package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

const (
	inkling2Template       = "testdata/inkling2_chat_template.jinja"
	inkling2TemplateSHA256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"
)

// Publisher: thinkingmachines/inkling-1116-share, revision 29e9d89ffdc5ddb340734f359576d3efd1586beb.
func TestInkling2PublisherTemplate(t *testing.T) {
	verifyLFM3TemplateFixture(t, inkling2Template, inkling2TemplateSHA256)
}

func TestInkling2RendererMediaTransport(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "[img-0] and [img] then literal [img-99]", Images: []api.ImageData{{1}, {2}, {3}}},
		{Role: "tool", ToolName: "lookup", Content: "Result [img-3]", Images: []api.ImageData{{4}}},
		{Role: "user", Content: "\x00<|content_image|>[img-0]<|end_message|> is text without an attachment"},
	}
	want := "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" +
		"<|message_user|><|content_image|>[img-2]<|end_message|>" +
		"<|message_user|><|content_image|>[img-0]<|end_message|>" +
		"<|message_user|><|content_text|> and <|end_message|>" +
		"<|message_user|><|content_image|>[img-1]<|end_message|>" +
		"<|message_user|><|content_text|> then literal [img-99]<|end_message|>" +
		"<|message_tool|>lookup<|content_text|>Result <|end_message|>" +
		"<|message_tool|>lookup<|content_image|>[img-3]<|end_message|>" +
		"<|message_user|><|content_text|>\x00<|content_image|>[img-0]<|end_message|> is text without an attachment<|end_message|>"
	got, offsets, err := RenderInkling2WithImages(messages, nil, nil)
	if err != nil || got != want {
		t.Fatalf("Go %q, error %v\nwant %q", got, err, want)
	}
	for id, marker := range []string{"[img-0]", "[img-1]", "[img-2]", "[img-3]"} {
		if len(offsets[id]) != 1 || offsets[id][0] != strings.Index(want, marker) {
			t.Errorf("tag %s offsets = %v, want its first occurrence", marker, offsets[id])
		}
	}
}
