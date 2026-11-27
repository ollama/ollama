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

// firstFittingStart returns the smallest start index i in [0, last] whose
// context length fits in numCtx, or last if no start index fits (so the
// caller still keeps at least the last message). It checks i == 0 first —
// the common case where everything fits — and then binary-searches the
// remainder, so truncating a long history costs O(log N) render/tokenize
// evaluations instead of one per dropped message (see #19942).
//
// ctxLen must be non-increasing in i: dropping messages from the front must
// not grow the rendered prompt. Chat templates render each message's content
// once plus fixed per-message separators, so this holds; a template that
// renders more text for fewer messages would get a valid cut that is not
// necessarily the earliest fitting one.
func firstFittingStart(ctxLen func(i int) (int, error), last, numCtx int) (int, error) {
	if last <= 0 {
		return 0, nil
	}
	if n, err := ctxLen(0); err != nil {
		return 0, err
	} else if n <= numCtx {
		return 0, nil
	}
	best := -1
	for lo, hi := 1, last; lo <= hi; {
		mid := (lo + hi) / 2
		n, err := ctxLen(mid)
		if err != nil {
			return 0, err
		}
		if n <= numCtx {
			best = mid
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	if best == -1 {
		return last, nil
	}
	return best, nil
}

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
		// Find the earliest start index whose rendered prompt fits in
		// context, binary-searching the cut point instead of
		// re-rendering and re-tokenizing the whole remaining history
		// once per dropped message (see #19942).
		ctxLen := func(i int) (int, error) {
			// Collect system messages from the portion we're about to skip
			sys := make([]api.Message, 0)
			for j := range i {
				if msgs[j].Role == "system" {
					sys = append(sys, msgs[j])
				}
			}

			p, err := renderPrompt(m, append(sys, msgs[i:]...), tools, think)
			if err != nil {
				return 0, err
			}

			s, err := tokenize(ctx, p)
			if err != nil {
				return 0, err
			}

			n := len(s)
			if m.ProjectorPaths != nil {
				for _, msg := range msgs[i:] {
					n += imageNumTokens * len(msg.Images)
				}
			}
			return n, nil
		}

		idx, err := firstFittingStart(ctxLen, lastMsgIdx, opts.NumCtx)
		if err != nil {
			return "", nil, err
		}
		currMsgIdx = idx

		system = make([]api.Message, 0)
		for j := range currMsgIdx {
			if msgs[j].Role == "system" {
				system = append(system, msgs[j])
			}
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[currMsgIdx:]))
	}

	renderMsgs, media, err := imageTaggedMessages(m, msgs, currMsgIdx, false)
	if err != nil {
		return "", nil, err
	}

	renderMsgs = append(system, renderMsgs[currMsgIdx:]...)
	if len(media) > 0 && m.IsMLX() && m.Config.Renderer == "inkling2" {
		prompt, offsets, err := renderers.RenderInkling2WithImages(renderMsgs, tools, think)
		if err != nil {
			return "", nil, err
		}
		for i := range media {
			media[i].TagOffsets = offsets[media[i].ID]
		}
		return prompt, media, nil
	}

	p, err := renderPrompt(m, renderMsgs, tools, think)
	if err != nil {
		return "", nil, err
	}

	return p, media, nil
}

func imageTaggedMessages(m *Model, msgs []api.Message, start int, clearImages bool) ([]api.Message, []llm.MediaData, error) {
	renderMsgs := slices.Clone(msgs)
	var media []llm.MediaData

	for cnt, msg := range renderMsgs[start:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return nil, nil, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			mediaData := llm.NewMediaData(len(media), i)
			media = append(media, mediaData)

			if m.Config.Renderer != "" {
				continue
			}

			// The prompt marker is still image-named for compatibility with
			// existing templates and llama-server media marker replacement.
			imgTag := fmt.Sprintf("[img-%d]", mediaData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}

		if m.Config.Renderer == "" {
			renderMsgs[start+cnt].Content = prefix + prompt
		}
		if clearImages {
			renderMsgs[start+cnt].Images = nil
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
