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
	"github.com/ollama/ollama/types/model"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and media that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, media []llm.MediaData, _ error) {
	var system []api.Message

	// TODO: This is only a truncation heuristic; llama-server handles the
	// actual image/media inputs. Replace this with projector/model-aware media
	// token accounting so image history is neither over-packed nor over-trimmed.
	// Clip images are represented as 768 tokens, each an embedding.
	imageNumTokens := 768

	lastMsgIdx := len(msgs) - 1
	currMsgIdx := 0

	if truncate {
		// Start with all messages and remove from the front until it fits in context
		for i := 0; i <= lastMsgIdx; i++ {
			// Collect system messages from the portion we're about to skip
			system = make([]api.Message, 0)
			for j := range i {
				if msgs[j].Role == "system" {
					system = append(system, msgs[j])
				}
			}

			p, err := renderPrompt(m, append(system, msgs[i:]...), tools, think)
			if err != nil {
				return "", nil, err
			}

			s, err := tokenize(ctx, p)
			if err != nil {
				return "", nil, err
			}

			ctxLen := len(s)
			if m.ProjectorPaths != nil {
				for _, msg := range msgs[i:] {
					ctxLen += imageNumTokens * (len(msg.Images) + len(msg.Videos))
				}
			}

			if ctxLen <= opts.NumCtx {
				currMsgIdx = i
				break
			}

			// Must always include at least the last message
			if i == lastMsgIdx {
				currMsgIdx = lastMsgIdx
				break
			}
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[currMsgIdx:]))
	}

	renderMsgs, media, err := mediaTaggedMessages(m, msgs, currMsgIdx, false, m.Config.Renderer == "")
	if err != nil {
		return "", nil, err
	}

	// truncate any messages that do not fit into the context window
	p, err := renderPrompt(m, append(system, renderMsgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, err
	}

	return p, media, nil
}

func mediaTaggedMessages(m *Model, msgs []api.Message, start int, clearMedia, tagContent bool) ([]api.Message, []llm.MediaData, error) {
	renderMsgs := slices.Clone(msgs)
	var media []llm.MediaData

	for cnt, msg := range renderMsgs[start:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return nil, nil, errors.New("this model only supports one image while more than one image requested")
		}

		messageMedia := llm.MessageFromAPI(msg).Media
		for i := range messageMedia {
			messageMedia[i].ID = len(media)
			media = append(media, messageMedia[i])
		}
		if tagContent {
			renderMsgs[start+cnt].Content = mediaTaggedContent(msg.Content, messageMedia)
		}

		if clearMedia {
			renderMsgs[start+cnt].Images = nil
			renderMsgs[start+cnt].Videos = nil
		}
	}

	return renderMsgs, media, nil
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if m.Config.Renderer != "" {
		rendererName := resolveRendererName(m)
		rendered, err := renderers.RenderWithRenderer(rendererName, msgs, tools, think)
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

// Keep the existing image-named markers for every media kind. The runner expands
// a video slot into timed frames; Go renderers must never expand it themselves.
func mediaTaggedContent(content string, media []llm.MediaData) string {
	var prefix strings.Builder
	for _, item := range media {
		tag := fmt.Sprintf("[img-%d]", item.ID)
		if strings.Contains(content, tag) {
			continue
		}
		if strings.Contains(content, "[img]") {
			content = strings.Replace(content, "[img]", tag, 1)
		} else {
			prefix.WriteString(tag)
		}
	}
	return prefix.String() + content
}

func validateVideoInput(m *Model, messages []api.Message) error {
	if !slices.ContainsFunc(messages, func(msg api.Message) bool { return len(msg.Videos) > 0 }) {
		return nil
	}
	if m.IsMLX() || slices.Contains(m.Config.ModelFamilies, "mllama") {
		return fmt.Errorf("%w: video input requires a supported multimodal GGUF model, such as qwen3-vl or qwen2.5vl", errCapabilities)
	}
	return m.CheckCapabilities(model.CapabilityVision)
}
