package server

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/applefm"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// appleFMModelName is the name Apple's on-device Foundation Model is listed as.
const appleFMModelName = "apple-foundation-model"

var appleFMAvailable = applefm.Available

// registerAppleFoundationModel adds the on-device model to the local model
// list when the operating system provides it. The manifest only holds a
// config, since the weights are managed by macOS.
func registerAppleFoundationModel(ctx context.Context) error {
	if err := appleFMAvailable(ctx); err != nil {
		slog.Debug("apple foundation model not registered", "error", err)
		return nil
	}

	config := model.ConfigV2{
		ModelFormat:  applefm.ModelFormat,
		ModelFamily:  applefm.ModelFormat,
		Capabilities: []string{string(model.CapabilityCompletion), string(model.CapabilityVision)},
		ContextLen:   applefm.ContextLength,
	}
	b, err := json.Marshal(config)
	if err != nil {
		return err
	}
	layer, err := manifest.NewLayer(bytes.NewReader(b), manifest.MediaTypeImageConfig)
	if err != nil {
		return err
	}

	name := model.ParseName(appleFMModelName)
	if mf, err := manifest.ParseNamedManifest(name); err == nil && mf.Config.Digest == layer.Digest {
		return nil
	}

	slog.Info("registering apple foundation model", "model", name.DisplayShortest())
	return manifest.WriteManifestWithMetadata(name, layer, nil, applefm.RunnerName, applefm.ModelFormat)
}

// completeWithChat sends msgs to r.Chat in place of a rendered prompt, for
// runners that take messages rather than a prompt.
func completeWithChat(r llm.LlamaServer, msgs []api.Message) func(context.Context, llm.CompletionRequest, func(llm.CompletionResponse)) error {
	return func(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
		return r.Chat(ctx, llm.ChatRequest{Messages: msgs, Format: req.Format, Options: req.Options}, func(cr llm.ChatResponse) {
			fn(llm.CompletionResponse{
				Content:            cr.Message.Content,
				Done:               cr.Done,
				DoneReason:         cr.DoneReason,
				PromptEvalCount:    cr.PromptEvalCount,
				PromptEvalDuration: cr.PromptEvalDuration,
				EvalCount:          cr.EvalCount,
				EvalDuration:       cr.EvalDuration,
			})
		})
	}
}
