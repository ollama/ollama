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
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, images []llm.ImageData, _ error) {
	// TODO: Ideally we would compute this from the projector metadata but some pieces are implementation dependent
	// Clip images are represented as 768 tokens, each an embedding
	imageNumTokens := 768

	lastMsgIdx := len(msgs) - 1
	currMsgIdx := 0

	// TODO: chatPrompt tokenizes here only to count tokens for the context
	// window check, then discards the token IDs. The final rendered prompt
	// (produced at the bottom of this function) is sent as text to the runner
	// process, which re-tokenizes it from scratch in runner/ollamarunner via
	// tokenizer.Tokenizer.Encode. This means every chat request tokenizes the
	// full prompt at least twice using the same Go byte-pair encoding codepath.
	// Passing the token IDs from this step directly to the runner (e.g. by
	// adding a Tokens field to llm.CompletionRequest) would eliminate the
	// redundant second tokenization entirely.

	// fitsContext reports whether the rendered prompt starting from message
	// index i (with system messages from msgs[:i] prepended) fits within the
	// context window. Token count is monotonically non-increasing as i grows,
	// since increasing i only drops non-system messages.
	fitsContext := func(i int) (bool, error) {
		var sys []api.Message
		for j := range i {
			if msgs[j].Role == "system" {
				sys = append(sys, msgs[j])
			}
		}

		p, err := renderPrompt(m, append(sys, msgs[i:]...), tools, think)
		if err != nil {
			return false, err
		}

		s, err := tokenize(ctx, p)
		if err != nil {
			return false, err
		}

		ctxLen := len(s)
		if m.ProjectorPaths != nil {
			for _, msg := range msgs[i:] {
				ctxLen += imageNumTokens * len(msg.Images)
			}
		}

		return ctxLen <= opts.NumCtx, nil
	}

	if truncate {
		// Try the full prompt first — this is the common case where
		// everything fits and requires only a single tokenize call.
		fits, err := fitsContext(0)
		if err != nil {
			return "", nil, err
		}

		if !fits && lastMsgIdx > 0 {
			// Binary search for the smallest i in [1, lastMsgIdx] where
			// the prompt fits. If nothing fits, converges to lastMsgIdx
			// which is always included as a last resort.
			low, high := 1, lastMsgIdx
			for low < high {
				mid := low + (high-low)/2
				fits, err := fitsContext(mid)
				if err != nil {
					return "", nil, err
				}
				if fits {
					high = mid
				} else {
					low = mid + 1
				}
			}
			currMsgIdx = low
		}
	}

	// Collect system messages from the portion we skipped
	var system []api.Message
	for j := range currMsgIdx {
		if msgs[j].Role == "system" {
			system = append(system, msgs[j])
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[currMsgIdx:]))
	}

	for cnt, msg := range msgs[currMsgIdx:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return "", nil, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			imgData := llm.ImageData{
				ID:   len(images),
				Data: i,
			}
			images = append(images, imgData)

			if m.Config.Renderer != "" {
				continue
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}
		msgs[currMsgIdx+cnt].Content = prefix + prompt
	}

	// truncate any messages that do not fit into the context window
	p, err := renderPrompt(m, append(system, msgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, err
	}

	return p, images, nil
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
