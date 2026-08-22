package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

// Segment is a piece of a rendered prompt. Control segments carry template
// markup whose special-token literals are meant to be parsed as special
// tokens. Content segments carry user- or tool-supplied message data that
// must be tokenized verbatim, so that special-token-like literals in it
// (e.g. "</think>") are not encoded as control tokens.
type Segment struct {
	Text    string
	Content bool
}

// SegmentRenderer is implemented by renderers that can distinguish template
// markup from message content in their rendered output.
type SegmentRenderer interface {
	RenderSegments(messages []api.Message, tools []api.Tool, think *api.ThinkValue) ([]Segment, error)
}

// RenderSegmentsWithRenderer renders messages with the named renderer and
// returns the prompt as control/content segments. It returns nil segments
// (and no error) if the renderer does not support segmented rendering.
func RenderSegmentsWithRenderer(name string, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) ([]Segment, error) {
	renderer := rendererForName(name)
	if renderer == nil {
		return nil, nil
	}

	sr, ok := renderer.(SegmentRenderer)
	if !ok {
		return nil, nil
	}

	return sr.RenderSegments(msgs, tools, think)
}

// JoinSegments concatenates segments back into a flat prompt string.
func JoinSegments(segments []Segment) string {
	var sb strings.Builder
	for _, s := range segments {
		sb.WriteString(s.Text)
	}
	return sb.String()
}

// segmentBuilder accumulates a rendered prompt as segments, merging adjacent
// writes of the same kind.
type segmentBuilder struct {
	segments []Segment
}

func (b *segmentBuilder) write(text string, content bool) {
	if text == "" {
		return
	}
	if n := len(b.segments); n > 0 && b.segments[n-1].Content == content {
		b.segments[n-1].Text += text
		return
	}
	b.segments = append(b.segments, Segment{Text: text, Content: content})
}

// control appends template markup.
func (b *segmentBuilder) control(text string) {
	b.write(text, false)
}

// content appends user- or tool-supplied data.
func (b *segmentBuilder) content(text string) {
	b.write(text, true)
}

func (b *segmentBuilder) Segments() []Segment {
	return b.segments
}
