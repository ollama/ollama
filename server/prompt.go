package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server/imageproc"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

var errTooManyImages = errors.New("vision model only supports a single image per message")

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool) (prompt string, images []llm.ImageData, _ error) {
	var system []api.Message

	isMllama := checkMllamaModelFamily(m)

	n := len(msgs) - 1
	// in reverse, find all messages that fit into context window
	for i := n; i >= 0; i-- {
		if isMllama && len(msgs[i].Images) > 1 {
			return "", nil, errTooManyImages
		}

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

		var b bytes.Buffer
		if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[i:]...), Tools: tools}); err != nil {
			return "", nil, err
		}

		s, err := tokenize(ctx, b.String())
		if err != nil {
			return "", nil, err
		}

		ctxLen := len(s)
		if m.ProjectorPaths != nil {
			for _, m := range msgs[i:] {
				// images are represented as 768 sized embeddings
				// TODO: get embedding length from project metadata
				ctxLen += 768 * len(m.Images)
			}
		}

		if ctxLen > opts.NumCtx {
			slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[i:]))
			break
		} else {
			n = i
		}
	}

	currMsgIdx := n

	if isMllama {
		lastMsgIdx := len(msgs) - 1
		for i := lastMsgIdx; i >= currMsgIdx; i-- {
			if len(msgs[i].Images) > 0 {
				data, aspectRatioID, err := imageproc.Preprocess(msgs[i].Images[0])
				if err != nil {
					return "", nil, err
				}

				buf := new(bytes.Buffer)
				err = binary.Write(buf, binary.LittleEndian, data)
				if err != nil {
					return "", nil, err
				}

				imgData := llm.ImageData{
					Data:          buf.Bytes(),
					AspectRatioID: aspectRatioID,
				}

				msgs[i].Content = strings.TrimSpace("<|image|>" + msgs[i].Content)
				images = append(images, imgData)
				break
			}
		}
	} else {
		for cnt, msg := range msgs[currMsgIdx:] {
			prefix := ""
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
			msgs[currMsgIdx+cnt].Content = strings.TrimSpace(prefix + " " + prompt)
		}
	}

	// truncate any messages that do not fit into the context window
	var b bytes.Buffer
	if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[currMsgIdx:]...), Tools: tools}); err != nil {
		return "", nil, err
	}

	return b.String(), images, nil
}

func checkMllamaModelFamily(m *Model) bool {
	for _, arch := range m.Config.ModelFamilies {
		if arch == "mllama" {
			return true
		}
	}
	return false
}
