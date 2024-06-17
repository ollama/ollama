package server

import (
	"bytes"
	"context"
	"log/slog"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
)

func chatPrompt(ctx context.Context, r *runnerRef, msgs []api.Message) (prompt string, images []llm.ImageData, _ error) {
	// extract system messages which should always be included
	var system []api.Message
	msgs = slices.DeleteFunc(msgs, func(m api.Message) bool {
		if m.Role == "system" {
			system = append(system, m)
			return true
		}

		return false
	})

	if len(system) == 0 && r.model.System != "" {
		// add model system prompt since it wasn't provided
		system = append(system, api.Message{Role: "system", Content: r.model.System})
	}

	n := len(msgs) - 1
	for i := n - 1; i >= 0; i-- {
		var b bytes.Buffer
		if err := r.model.Template.Execute(&b, template.Values{Messages: append(system, msgs[i:]...)}); err != nil {
			return "", nil, err
		}

		s, err := r.llama.Tokenize(ctx, b.String())
		if err != nil {
			return "", nil, err
		}

		c := len(s)
		if r.model.ProjectorPaths != nil {
			for _, m := range msgs[i:] {
				// TODO: get image embedding length from project metadata
				c += 768 * len(m.Images)
			}
		}

		if c > r.NumCtx {
			slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[i:]))
			break
		} else {
			n = i
		}
	}

	var b bytes.Buffer
	if err := r.model.Template.Execute(&b, template.Values{Messages: append(system, msgs[n:]...)}); err != nil {
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
