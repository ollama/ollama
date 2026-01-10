package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
// It also returns numKeep, the number of tokens in system messages + tools that should be protected from truncation.
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, images []llm.ImageData, numKeep int, _ error) {
	var system []api.Message

	// TODO: Ideally we would compute this from the projector metadata but some pieces are implementation dependent
	// Clip images are represented as 768 tokens, each an embedding
	imageNumTokens := 768

	n := len(msgs) - 1
	// in reverse, find all messages that fit into context window
	for i := n; i >= 0; i-- {
		// always include the last message
		if i == n {
			continue
		}

		system = make([]api.Message, 0)
		for j := range i {
			if msgs[j].Role == "system" {
				system = append(system, msgs[j])
			}
		}

		p, err := renderPrompt(m, append(system, msgs[i:]...), tools, think)
		if err != nil {
			return "", nil, 0, err
		}

		s, err := tokenize(ctx, p)
		if err != nil {
			return "", nil, 0, err
		}

		ctxLen := len(s)
		if m.ProjectorPaths != nil {
			for _, m := range msgs[i:] {
				ctxLen += imageNumTokens * len(m.Images)
			}
		}

		if truncate && ctxLen > opts.NumCtx {
			slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[i:]))
			break
		} else {
			n = i
		}
	}

	currMsgIdx := n

	for cnt, msg := range msgs[currMsgIdx:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return "", nil, 0, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			imgData := llm.ImageData{
				ID:   len(images),
				Data: i,
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}

			images = append(images, imgData)
		}
		msgs[currMsgIdx+cnt].Content = prefix + prompt
	}

	// truncate any messages that do not fit into the context window
	p, err := renderPrompt(m, append(system, msgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, 0, err
	}

	// Compute numKeep: tokens for system messages + tools that should be protected from truncation
	// Re-collect all system messages from the entire conversation
	allSystemMsgs := make([]api.Message, 0)
	for _, msg := range msgs {
		if msg.Role == "system" {
			allSystemMsgs = append(allSystemMsgs, msg)
		}
	}
	protectedPrompt, err := renderPrompt(m, allSystemMsgs, tools, think)
	if err != nil {
		return "", nil, 0, err
	}

	protectedTokens, err := tokenize(ctx, protectedPrompt)
	if err != nil {
		return "", nil, 0, err
	}

	numKeep = len(protectedTokens)

	// Error if system+tools leaves less than 100 tokens for conversation
	if numKeep > 0 && numKeep > opts.NumCtx-100 {
		return "", nil, 0, fmt.Errorf("system prompt and tools (%d tokens) exceed context length (%d) minus required buffer (100 tokens)", numKeep, opts.NumCtx)
	}

	// Cap numKeep to ensure at least 200 tokens can be generated
	if opts.NumCtx > 200 {
		numKeep = min(numKeep, opts.NumCtx-200)
	}

	return p, images, numKeep, nil
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if m.Config.Renderer != "" {
		rendered, err := renderers.RenderWithRenderer(m.Config.Renderer, msgs, tools, think)
		if err != nil {
			return "", err
		}
		return rendered, nil
	}

	var b bytes.Buffer
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := m.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", err
	}
	return b.String(), nil
}
