package pplxdecider

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
)

func TestScorePromptUsesPublisherImageMarkers(t *testing.T) {
	previous := renderers.RenderImgTags
	renderers.RenderImgTags = true
	defer func() { renderers.RenderImgTags = previous }()

	m := &Model{config: config{Codes: []string{"A"}}}
	input := llm.ScoreRequest{
		System: "system", State: "state",
		Images: []api.ImageData{{1}, {2}},
	}
	row := llm.ScoreRow{Question: &llm.ScoreQuestion{Instructions: "question", Options: []string{"option"}}}
	prompt, _, err := m.scorePrompt(input, row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "<|im_start|>user\n"+imageMarker+imageMarker+"State:\nstate") {
		t.Errorf("publisher image markers missing from user message: %q", prompt)
	}
	if strings.Contains(prompt, "[img-") {
		t.Errorf("server image placeholders leaked into publisher prompt: %q", prompt)
	}
}

func TestScorePromptLocatesAttachedImageAfterSystemMarkers(t *testing.T) {
	m := &Model{config: config{Codes: []string{"A"}}}
	input := llm.ScoreRequest{
		System: "literal " + imageMarker,
		State:  imageMarker + " state",
		Images: []api.ImageData{{1}},
	}
	row := llm.ScoreRow{Question: &llm.ScoreQuestion{Instructions: "question", Options: []string{"option"}}}
	prompt, imageStart, err := m.scorePrompt(input, row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt[:imageStart], imageMarker) {
		t.Fatalf("system marker missing before attached image: %q", prompt)
	}
	if !strings.HasPrefix(prompt[imageStart:], imageMarker+"State:\n"+input.State) {
		t.Fatalf("attached image start = %d in %q", imageStart, prompt)
	}
}
