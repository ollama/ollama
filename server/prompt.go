package server

import (
	"bytes"
	"context"
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool) (prompt string, images []llm.ImageData, _ error) {
	var system []api.Message
	// always include the last message
	n := len(msgs) - 1
	// in reverse, find all messages that fit into context window
	for i := n - 1; i >= 0; i-- {
		system = make([]api.Message, 0)
		for j := range i {
			if msgs[j].Role == "system" {
				system = append(system, msgs[j])
			}
		}

		var b bytes.Buffer
		if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[i:]...), Tools: tools}); err != nil {
			return "", nil, err
		}

		s, err := tokenize(ctx, b.String())
		if err != nil {
			return "", nil, err
		}

		c := len(s)
		if m.ProjectorPaths != nil {
			for _, m := range msgs[i:] {
				// images are represented as 768 sized embeddings
				// TODO: get embedding length from project metadata
				c += 768 * len(m.Images)
			}
		}

		if c > opts.NumCtx {
			slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[i:]))
			break
		} else {
			n = i
		}
	}

	// truncate any messages that do not fit into the context window
	var b bytes.Buffer
	if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[n:]...), Tools: tools}); err != nil {
		return "", nil, err
	}

	for _, m := range msgs[n:] {
		for _, i := range m.Images {
			images = append(images, llm.ImageData{
				ID:   len(images),
				Data: i,
			})
		}
	}

	return b.String(), images, nil
}
