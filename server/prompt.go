package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"slices"
	"strings"

	// Image formats imageTokenCost can size from the header; anything else
	// (llama.cpp's stb decoder also accepts e.g. bmp/tga) falls back to the
	// worst-case estimate.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and media that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, media []llm.MediaData, _ error) {
	var system []api.Message

	// This is only a truncation heuristic; llama-server handles the actual
	// image/media inputs. Costs are per-arch and, where the arch's resize
	// math is replicated in llm.ImageTokensForSize, per-image-size.
	imageNumTokens := imageTokenCosts(m, opts, msgs)

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
			for _, t := range imageNumTokens[i:] {
				ctxLen += t
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

	renderMsgs, media, err := imageTaggedMessages(m, msgs, currMsgIdx, false)
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

// visionTokenArch returns the architecture used for image-token accounting
// and whether the model has an image input path at all.
//
// Vision models come in two layouts: a separate projector layer
// (ProjectorPaths), or — for the llm.InlineVisionArch architectures — vision
// tensors inline in the main GGUF with no projector layer at all, which the
// runner self-references as --mmproj. The latter used to be invisible to the
// ProjectorPaths check here, so their images were counted as zero tokens and
// multi-image chats could pass the context-fit check yet overflow
// llama-server's context.
//
// The arch gate can over-charge a text-only variant of an inline-vision arch
// (e.g. a gemma3 GGUF without vision tensors): the charge only applies to
// requests that actually attach images, which such a model cannot process
// anyway, and checking real vision capability here would mean reopening the
// GGUF on every request.
func visionTokenArch(m *Model) (string, bool) {
	if m == nil {
		return "", false
	}

	arch := m.Config.ModelFamily
	if !llm.InlineVisionArch(arch) {
		for _, family := range m.Config.ModelFamilies {
			if llm.InlineVisionArch(family) {
				arch = family
				break
			}
		}
	}

	if len(m.ProjectorPaths) == 0 && !llm.InlineVisionArch(arch) {
		return "", false
	}

	return arch, true
}

// maxImageTokens returns the worst-case per-image token cost the chat
// truncation heuristics charge against the context window, or 0 for a model
// with no image input path.
func maxImageTokens(m *Model, opts *api.Options) int {
	if opts == nil {
		return 0
	}

	arch, ok := visionTokenArch(m)
	if !ok {
		return 0
	}

	return llm.MaxImageTokens(arch, *opts)
}

// imageTokenCosts returns, per message, the token cost the truncation
// heuristics charge for that message's images: the size-aware per-arch cost
// when the image header decodes and the arch's preprocessing is replicated in
// llm.ImageTokensForSize, and the flat per-arch worst case otherwise. The
// slice is all zeros when the model has no image input path.
func imageTokenCosts(m *Model, opts *api.Options, msgs []api.Message) []int {
	costs := make([]int, len(msgs))

	worstCase := maxImageTokens(m, opts)
	if worstCase == 0 {
		return costs
	}
	arch, _ := visionTokenArch(m)

	for i, msg := range msgs {
		for _, img := range msg.Images {
			costs[i] += imageTokenCost(arch, *opts, img, worstCase)
		}
	}

	return costs
}

// imageTokenCost returns the token cost of one image. Dimensions come from
// the image header alone (image.DecodeConfig); formats Go cannot identify
// fall back to worstCase, which over-trims rather than overflows.
func imageTokenCost(arch string, opts api.Options, img api.ImageData, worstCase int) int {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(img))
	if err != nil {
		return worstCase
	}

	if n, ok := llm.ImageTokensForSize(arch, opts, cfg.Width, cfg.Height); ok {
		return n
	}
	return worstCase
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
